"""SAC + QIPM with parallel safety subtask for obstacle avoidance.

A near-fork of ogpo/agents/sac_qipm.py specialized for the
``sac_qipm_avoid_obstacle`` algorithm. Two per-subtask Q-heads:
  - Head 0: task_done       (standard QIPM TD: +5 done, -1 not done, capped at +15)
  - Head 1: collision_occurred (inverted: -5 done, +1 not done, floored at -15)

Both subtasks ratchet up monotonically, so the existing QIPM machinery
(``compute_qipm_td_target``, the per-subtask Value head, the replay buffer
template) is reused unchanged for head 0; head 1 uses a private mirror helper
``_compute_safety_td_target`` defined in this file.

The actor loss bypasses ``_get_sequential_gate_weights`` entirely and uses a
parallel sum ``q_total = q_task + safety_coeff * q_safety``.

This file is intentionally self-contained: no inheritance from SACQIPMAgent
to avoid Flax PyTreeNode subclassing pitfalls.
"""
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
from ogpo.agents.modules.qipm_helper import compute_qipm_td_target


def _compute_safety_td_target(
    subtask_status: jnp.ndarray,
    next_q: jnp.ndarray,
    masks: jnp.ndarray,
    discount,
    horizon_length: int,
) -> jnp.ndarray:
    """Soft-margin CMDP cost-critic TD target for the safety subtask.

    The wrapper stores a CONTINUOUS proximity cost in subtask_status[..., 1]:

        cost = 1.0                              if inside the obstacle
             = (safe_margin - sd) / safe_margin if 0 < signed_dist < safe_margin
             = 0.0                              if signed_dist >= safe_margin

    The Q-value backup is the standard discounted Bellman with a per-step cost:

        Q_safety(s, a) = -5 * cost(s)  +  γ^H * mask * Q_safety(s', a')

    There is NO absorbing terminal value and NO floor. This is the standard
    CMDP cost-critic formulation, but with a smooth proximity penalty rather
    than a binary inside/outside flag. The smooth signal lets the actor feel
    a "push away from the rectangle" gradient even before it touches the
    obstacle, which is the difference between learning the *shape* of the
    safe region and just learning "if I'm in, leave."

    Args:
        subtask_status: (B, horizon_length, 1) per-step continuous cost in [0, 1].
        next_q: (1, B) aggregated next Q-values for the safety head.
        masks: (B,) or (B, horizon_length) termination masks.
        discount: scalar float (or per-subtask array).
        horizon_length: action chunking horizon.

    Returns:
        TD targets of shape (1, B).
    """
    if masks.ndim > 1:
        masks = masks[..., -1]

    disc_factor = jnp.asarray(discount) ** horizon_length
    if disc_factor.ndim > 0:
        disc_factor = disc_factor[:, None]  # (S, 1)

    # (S=1, B) per-step proximity cost from the last timestep of the chunk
    cost = subtask_status[..., -1, :].T.astype(jnp.float32)

    rewards = -5.0 * cost
    target_q = rewards + disc_factor * masks * next_q
    return target_q


class SACQIPMAvoidObstacleAgent(flax.struct.PyTreeNode):
    """SAC agent with parallel task + safety per-subtask Q-functions."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Critic loss with task head (standard QIPM) + safety head (inverted)."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next actions
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        # Next Q-values: (S=2, M, B)
        next_qs = self.network.select('target_critic')(
            batch['next_observations'], actions=next_actions)

        # Aggregate over ensemble: (S, M, B) -> (S, B)
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=1)
        else:
            next_q = jnp.mean(next_qs, axis=1)

        # Split subtask_status into task vs safety along the subtask axis.
        # batch['subtask_status'] is (B, H, S=2)
        task_status = batch['subtask_status'][..., 0:1]
        safety_status = batch['subtask_status'][..., 1:2]

        # Per-head TD targets, each (1, B)
        target_q_task = compute_qipm_td_target(
            subtask_status=task_status,
            next_q=next_q[0:1],
            masks=batch['masks'],
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
            no_of_subtasks=1,
        )
        target_q_safety = _compute_safety_td_target(
            subtask_status=safety_status,
            next_q=next_q[1:2],
            masks=batch['masks'],
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
        )
        target_q = jnp.concatenate([target_q_task, target_q_safety], axis=0)  # (2, B)

        # Current Q-values: (2, M, B)
        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params)

        # MSE loss against target broadcast across ensemble
        td_err = jnp.square(q - target_q[:, None, :])
        critic_loss = (td_err * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_task_mean': q[0].mean(),
            'q_safety_mean': q[1].mean(),
            'target_q_task_mean': target_q[0].mean(),
            'target_q_safety_mean': target_q[1].mean(),
            'next_log_probs_mean': next_log_probs.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """SAC actor loss with parallel q_total = q_task + safety_coeff * q_safety."""
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Per-subtask Q-values for sampled actions: (2, M, B)
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        if self.config['q_agg'] == 'min':
            q_per_subtask = jnp.min(qs, axis=1)  # (2, B)
        else:
            q_per_subtask = jnp.mean(qs, axis=1)  # (2, B)

        safety_coeff = self.config['safety_coeff']
        q_task = q_per_subtask[0]
        q_safety = q_per_subtask[1]
        q_total = q_task + safety_coeff * q_safety  # (B,)

        # Temperature
        temp = self.network.select('temp')()

        # SAC actor loss: minimize (temp * log_pi - Q_total)
        actor_loss = jnp.mean((temp * log_probs - q_total) * batch['valid'][..., -1])

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
            'q_total_mean': q_total.mean(),
            'q_task_actor_mean': q_task.mean(),
            'q_safety_actor_mean': q_safety.mean(),
            'temp': temp,
            'safety_coeff': safety_coeff,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
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
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
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
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # --- BC methods (actor-only) ---

    def bc_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_actions = jnp.clip(batch_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_probs = dist.log_prob(batch_actions)
        bc_loss = -(log_probs * batch['valid'][..., -1]).mean()
        return bc_loss, {'bc_loss': bc_loss, 'bc_log_probs': log_probs.mean()}

    @jax.jit
    def bc_update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.bc_loss(batch, grad_params, rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info = {f'offline_agent/{k}': v for k, v in info.items()}
        return self.replace(network=new_network, rng=new_rng), info

    # --- Q warmup (critic-only, per-subtask TD with the same parallel logic) ---

    @staticmethod
    def _q_warmup_update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.critic_loss(batch, grad_params, rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # --- Per-subtask Q-values (for QIPM eval) ---

    @jax.jit
    def get_per_subtask_q_values(self, observations, actions):
        flat_actions = actions.reshape(observations.shape[0], -1)
        qs = self.network.select('critic')(observations, actions=flat_actions)  # (2, M, B)
        if self.config['q_agg'] == 'min':
            return jnp.min(qs, axis=1)  # (2, B)
        return jnp.mean(qs, axis=1)  # (2, B)

    # --- Action selection ---

    @jax.jit
    def sample_actions(self, observations, rng=None, **kwargs):
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, **kwargs):
        dist = self.network.select('actor')(observations)
        if hasattr(dist, 'mode'):
            return dist.mode()
        return dist.mean()

    # --- Creation ---

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        no_of_subtasks = config['no_of_subtasks']
        assert no_of_subtasks == 2, (
            f"sac_qipm_avoid_obstacle requires no_of_subtasks=2 (task + safety), "
            f"got {no_of_subtasks}"
        )

        # Critic with 2 output heads (shared backbone)
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config.get('num_bins', 256),
            q_min=config.get('q_min', None),
            q_max=config.get('q_max', None),
            num_subtasks=no_of_subtasks,
        )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,
            state_dependent_std=True,
        )

        # Optional GoalConditionedNetwork wrapper (kept for parity with sac_qipm)
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

        temp_def = LogParam(init_value=config.get('init_temperature', 1.0))
        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
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

        if config.get('target_entropy') is None:
            config['target_entropy'] = -float(full_action_dim) / 2

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sac_qipm_avoid_obstacle',
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
            # QIPM-style: forced to 2 (task + safety)
            no_of_subtasks=2,
            use_dense_rewards=False,
            normalize_rewards=False,
            # Parallel-sum actor weighting
            safety_coeff=1.0,
            # Goal conditioning kept for parity with sac_qipm; usually disabled
            goal_conditioning_mode=ml_collections.config_dict.placeholder(str),
            goal_conditioned_actor=True,
            goal_conditioned_critic=True,
            goal_embed_dim=32,
            # Runner compat
            best_of_n=1,
        )
    )
    return config
