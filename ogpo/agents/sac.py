import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import Actor, Value, LogParam
from ogpo.networks.goal_net import GoalConditionedNetwork
from ogpo.agents.modules.dist_helper import hl_gauss_target
from ogpo.agents.modules.q_helper import (
    sample_mip_q_values,
    sample_mip_q_ensemble_values,
    compute_mip_q_predictions,
    compute_mip_q_ensemble_predictions,
)


class SACAgent(flax.struct.PyTreeNode):
    """Soft Actor-Critic (SAC) agent.

    The actor is shared between all Q functions of different chunk lengths.
    The actor always predicts action chunks, and is shared between all Q functions.
    There are chunked critics and non-chunked critics.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss. Supports both standard ensemble Q and MIP-Q critics."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first action

        # Sample next actions with current policy
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        use_mip_q = self.config.get('use_mip_q', False)
        if use_mip_q:
            num_M = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            target_critic_fn = self.network.select('target_critic')
            bsz = batch['next_observations'].shape[0]
            rng, noise_rng = jax.random.split(rng)
            if self.config.get('mip_q_ensemble', False):
                noise_sample = jax.random.uniform(
                    noise_rng, (bsz, 1), minval=-noise_scale, maxval=noise_scale,
                )
                next_qs = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=batch['next_observations'],
                    actions=next_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                )  # (M, B)
            else:
                noise_samples = jax.random.uniform(
                    noise_rng, (num_M, bsz, 1), minval=-noise_scale, maxval=noise_scale,
                )
                next_qs = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=batch['next_observations'],
                    actions=next_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                )  # (N, B)
        else:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], actions=next_actions,
            )
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        # Get temperature for entropy bonus
        temp = self.network.select('temp')()

        # SAC target with optional entropy bonus
        if self.config.get('entropy_backup', False):
            target_q = batch['rewards'][..., -1] + \
                (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * \
                (next_q - temp * next_log_probs)
        else:
            target_q = batch['rewards'][..., -1] + \
                (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        if use_mip_q:
            critic_base_fn = self.network.select('critic')

            def critic_fn_with_params(obs, actions, scalar_input, time, is_encoded, return_logits):
                return critic_base_fn(
                    obs,
                    actions=actions,
                    scalar_input=scalar_input,
                    time=time,
                    params=grad_params,
                    is_encoded=is_encoded,
                    return_logits=return_logits,
                )

            rng, mip_rng = jax.random.split(rng)
            if self.config.get('mip_q_ensemble', False):
                q_0, q = compute_mip_q_ensemble_predictions(
                    critic_fn=critic_fn_with_params,
                    observations=batch['observations'],
                    actions=batch_actions,
                    target_q=target_q,
                    rng=mip_rng,
                    noise_scale=noise_scale,
                    mip_t_star=mip_t_star,
                    return_logits=False,
                )  # (M, B), (M, B)
            else:
                q_0, q = compute_mip_q_predictions(
                    critic_fn=critic_fn_with_params,
                    observations=batch['observations'],
                    actions=batch_actions,
                    target_q=target_q,
                    rng=mip_rng,
                    noise_scale=noise_scale,
                    mip_t_star=mip_t_star,
                    num_ensemble_members=num_M,
                    return_logits=False,
                )  # (N, B), (N, B)
            td_err_0 = jnp.square(q_0 - target_q[None, :])
            td_err_t = jnp.square(q   - target_q[None, :])
            critic_loss = ((td_err_0 + td_err_t) * batch['valid'][..., -1]).mean()
        else:
            # Current Q-values (standard ensemble path)
            q = self.network.select('critic')(
                batch['observations'], actions=batch_actions, params=grad_params,
            )
            critic_loss = jnp.mean((q - target_q) ** 2 * batch['valid'][..., -1])

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'next_log_probs_mean': next_log_probs.mean(),
            'temp': temp
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss with integrated temperature loss."""
        # Sample actions with current policy
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Get Q-values for sampled actions (MIP-Q two-step or plain critic forward)
        if self.config.get('use_mip_q', False):
            num_M = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            critic_fn = self.network.select('critic')
            bsz = batch['observations'].shape[0]
            rng, noise_rng = jax.random.split(rng)
            if self.config.get('mip_q_ensemble', False):
                noise_sample = jax.random.uniform(
                    noise_rng, (bsz, 1), minval=-noise_scale, maxval=noise_scale,
                )
                qs = sample_mip_q_ensemble_values(
                    critic_fn=critic_fn,
                    observations=batch['observations'],
                    actions=actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                )  # (M, B)
            else:
                noise_samples = jax.random.uniform(
                    noise_rng, (num_M, bsz, 1), minval=-noise_scale, maxval=noise_scale,
                )
                qs = sample_mip_q_values(
                    critic_fn=critic_fn,
                    observations=batch['observations'],
                    actions=actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                )  # (N, B)
        else:
            qs = self.network.select('critic')(batch['observations'], actions=actions)

        if self.config['q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)

        # Get temperature
        temp = self.network.select('temp')()

        # SAC actor loss: maximize Q - temperature * entropy
        actor_loss = jnp.mean((temp * log_probs - q) * batch['valid'][..., -1])

        # Temperature loss
        temp_param = self.network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs * batch['valid'][..., -1]).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + temp_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(),
            'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(),
            'temp': temp
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        """Update the agent over a batch of transitions.

        Extra args (batch_success, success_flag) accepted for runner
        compatibility but are unused — SAC has no success-buffer logic.
        """
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def bc_loss(self, batch, grad_params, rng):
        """Compute BC loss: negative log-prob of dataset actions under the actor."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        # Clip actions away from ±1 to avoid atanh(±1)=±inf in TanhNormal log_prob.
        batch_actions = jnp.clip(batch_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_probs = dist.log_prob(batch_actions)
        bc_loss = -(log_probs * batch['valid'][..., -1]).mean()
        return bc_loss, {'bc_loss': bc_loss, 'bc_log_probs': log_probs.mean()}

    @jax.jit
    def bc_update(self, batch):
        """Update actor via behavioral cloning."""
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.bc_loss(batch, grad_params, rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info = {f'offline_agent/{k}': v for k, v in info.items()}
        return self.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _q_warmup_update(agent, batch):
        """Train critic only via TD learning (no actor update)."""
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.critic_loss(batch, grad_params, rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Batch Q warmup: critic-only updates over a scan."""
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def batch_critic_only_update(self, batch):
        """Critic-only updates (used by online_rl_runner when utd_q > utd_pi)."""
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
        **kwargs,
    ):
        """Sample stochastic actions from the actor policy."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, **kwargs):
        """Deterministic action selection (policy mean).

        SAC has no flow model; this returns the deterministic policy mean.
        Provided so the BC/online_rl_runner eval path works unchanged.
        """
        dist = self.network.select('actor')(observations)
        if hasattr(dist, 'mode'):
            return dist.mode()
        return dist.mean()

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new SAC agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        # Promote to ≥2D for network init. main.py passes single-example
        # tensors (shape `(D,)`) via Dataset.sample(()); plain Value tolerates
        # 1D inputs but ValueMIPEnsemble's tile path requires a batch dim.
        if ex_observations.ndim == 1:
            ex_observations = ex_observations[None]
        if ex_actions.ndim == 1:
            ex_actions = ex_actions[None]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        if config.get('use_mip_q', False):
            if config.get('mip_q_ensemble', False):
                from ogpo.networks import ValueMIPEnsemble
                critic_def = ValueMIPEnsemble(
                    hidden_dims=config['value_hidden_dims'],
                    num_ensembles=config.get('num_ensemble_members', 10),
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config.get('num_bins', 256),
                    q_min=config.get('q_min', None),
                    q_max=config.get('q_max', None),
                )
                print(f"[MIP-Q] SAC using ValueMIPEnsemble (M={config.get('num_ensemble_members', 10)}, "
                      f"noise={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)})")
            else:
                from ogpo.networks import ValueMIP
                critic_def = ValueMIP(
                    hidden_dims=config['value_hidden_dims'],
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config.get('num_bins', 256),
                    q_min=config.get('q_min', None),
                    q_max=config.get('q_max', None),
                )
                print(f"[MIP-Q] SAC using ValueMIP (N={config.get('num_ensemble_members', 10)} noise samples, "
                      f"noise={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)})")
        else:
            critic_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config.get('num_bins', 256),
                q_min=config.get('q_min', None),
                q_max=config.get('q_max', None),
            )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,
            state_dependent_std=True,
        )

        # Wrap with GoalConditionedNetwork if goal conditioning is enabled
        if config.get('goal_conditioning_mode'):
            goal_dim = config['no_of_subtasks']
            goal_embed_dim = config.get('goal_embed_dim', 32)
            actor_def = GoalConditionedNetwork(
                inner=actor_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_actor', True),
            )
            critic_def = GoalConditionedNetwork(
                inner=critic_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_critic', True),
            )
            print(f"[GC] Wrapped actor (use_goals={config.get('goal_conditioned_actor', True)}) "
                  f"and critic (use_goals={config.get('goal_conditioned_critic', True)}) "
                  f"with GoalConditionedNetwork (goal_dim={goal_dim}, embed_dim={goal_embed_dim})")

        temp_def = LogParam(init_value=config.get('init_temperature', 1.0))
        # Build init kwargs per network. MIP-Q critics need a `rng` kwarg
        # for noise sampling during init (otherwise raise "must provide
        # scalar_input or rng"). Standard Value critics ignore it.
        critic_init_kw = dict(observations=ex_observations, actions=full_actions)
        target_critic_init_kw = dict(critic_init_kw)
        if config.get('use_mip_q', False):
            init_rng, critic_init_rng, target_critic_init_rng = jax.random.split(init_rng, 3)
            critic_init_kw['rng'] = critic_init_rng
            target_critic_init_kw['rng'] = target_critic_init_rng

        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            critic=(critic_def, critic_init_kw),
            target_critic=(copy.deepcopy(critic_def), target_critic_init_kw),
            temp=(temp_def, ()),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        # Set target entropy if not specified (using action_dim / 2)
        if config.get('target_entropy') is None:
            config['target_entropy'] = -float(full_action_dim) / 2

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sac',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            num_qs=10,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            # SAC-specific
            init_temperature=1.0,
            target_entropy=ml_collections.config_dict.placeholder(float),
            # Critic loss
            critic_loss_type='mse',
            num_bins=256,
            entropy_backup=False,
            # Goal conditioning
            goal_conditioning_mode=ml_collections.config_dict.placeholder(str),
            goal_conditioned_actor=True,
            goal_conditioned_critic=True,
            goal_embed_dim=32,
            no_of_subtasks=ml_collections.config_dict.placeholder(int),
            # MIP-Q (off by default)
            use_mip_q=False,
            mip_q_ensemble=True,
            mip_q_noise_scale=1.0,
            mip_q_t_star=0.9,
            num_ensemble_members=10,
            # Required by runner but unused by SAC
            best_of_n=1,
        )
    )
    return config
