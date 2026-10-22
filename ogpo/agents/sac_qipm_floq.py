"""SAC + QIPM with FloQ flow-matching critic.

A near-fork of `sac_qipm.py` with the critic architecture swapped from
`Value` (MSE-trained MLP ensemble) to `FloQCritic` (flow-matching velocity
field + distilled MLP head). Same QIPM TD target synthesis, same per-subtask
sequential gating in the actor loss, same `qipm_runner.py` runner — only the
critic forward + critic loss change.

The actor loss is **unchanged** because `network.select('critic')(obs, action)`
still returns `(S, M, B)` from the distilled head, which is the same shape as
the existing `Value(critic_loss_type='mse', num_subtasks=…)` returned. So the
QIPM gating logic, sequential gate weights, temp loss are all binary-identical
to `sac_qipm`.

The critic loss is replaced with a two-loss flow-matching update via the
helpers in `ogpo/agents/modules/floq_helpers.py`.

Self-contained: no inheritance from SACQIPMAgent to avoid Flax PyTreeNode
subclassing pitfalls (matches the pattern used by sac_qipm_avoid_obstacle).
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
from ogpo.networks import Actor, LogParam
from ogpo.networks.floq_critic import FloQCritic
from ogpo.networks.goal_net import GoalConditionedNetwork
from ogpo.agents.modules.floq_helpers import (
    compute_floq_qipm_td_target,
    floq_critic_loss_components,
)


class SACQIPMFloqAgent(flax.struct.PyTreeNode):
    """SAC agent with per-subtask FloQ flow-matching critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_sequential_gate_weights(self, subtask_status):
        """Compute sequential gating weights from ground truth subtask status.

        Verbatim from sac_qipm.SACQIPMAgent._get_sequential_gate_weights.
        Supports the same 4 gating modes (goal_myopic, goal_future,
        grandparent_myopic, grandparent_future) plus the legacy gate_on_self
        mode. See sac_qipm.py for full docstring.
        """
        done = subtask_status.T  # (S, B)

        if self.config.get('gate_on_self', False):
            gate_weights = 1.0 - done
            gate_weights = gate_weights.at[-1, :].set(1.0)
            return gate_weights

        gating_mode = self.config.get('advantage_gating_mode', 'grandparent_future')

        active_idx = jnp.argmin(done, axis=0)
        active_mask = jax.nn.one_hot(active_idx, done.shape[0]).T

        parent_done = jnp.concatenate([
            jnp.full((1, done.shape[1]), 1.0),
            done[:-1, :]
        ], axis=0)

        if gating_mode == 'goal_myopic':
            gate_weights = active_mask * (1.0 - done)
        elif gating_mode == 'goal_future':
            gate_weights = 1.0 - done
        elif gating_mode == 'grandparent_myopic':
            gate_weights = active_mask * (1.0 - parent_done)
        elif gating_mode == 'grandparent_future':
            gate_weights = 1.0 - parent_done
            gate_weights = gate_weights.at[-1, :].set(1.0)
        else:
            assert False, (
                f"Unknown advantage_gating_mode: {gating_mode}. "
                f"Must be 'goal_myopic', 'goal_future', 'grandparent_myopic', or 'grandparent_future'."
            )

        return gate_weights

    def critic_loss(self, batch, grad_params, rng):
        """FloQ-based critic loss: flow matching + distillation, both per-subtask.

        Drop-in for sac_qipm.critic_loss but with a flow-matching velocity
        field as the critic backbone. The TD target is computed from the
        TARGET velocity field's Euler integration on (s', a'), then plumbed
        through the existing compute_qipm_td_target for per-subtask reward
        synthesis. The training loss is two components, both per-subtask:
        flow-matching MSE on the velocity field + distillation MSE on the
        fast distilled head with stop_gradient on the integrated flow Q.
        """
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next actions (same as sac_qipm)
        rng, sample_rng, td_rng, loss_rng = jax.random.split(rng, 4)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        # Per-subtask TD targets via TARGET velocity field flow integration.
        target_q = compute_floq_qipm_td_target(
            network=self.network,
            batch=batch,
            next_actions=next_actions,
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
            no_of_subtasks=self.config['no_of_subtasks'],
            q_agg=self.config['q_agg'],
            rng=td_rng,
        )  # (S, B)

        # Two-loss FloQ critic update on the ONLINE velocity field + distilled head.
        total_loss, metrics = floq_critic_loss_components(
            network=self.network,
            grad_params=grad_params,
            batch=batch,
            qipm_td_target=target_q,
            actions=batch_actions,
            no_of_subtasks=self.config['no_of_subtasks'],
            flow_num_ensembles=self.config['floq']['flow_num_ensembles'],
            distill_weight=self.config['floq']['distill_weight'],
            noise_min=self.config['floq']['noise_min'],
            noise_max=self.config['floq']['noise_max'],
            rng=loss_rng,
        )

        metrics['next_log_probs_mean'] = next_log_probs.mean()
        metrics['target_q_mean'] = target_q.mean()
        return total_loss, metrics

    def actor_loss(self, batch, grad_params, rng):
        """SAC actor loss with sequentially-gated per-subtask Q-values.

        UNCHANGED from sac_qipm.actor_loss. The distilled head returns
        (S, M, B) just like the existing Value, so the gating math is identical.
        """
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Per-subtask Q-values for sampled actions: (S, M, B)
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        # Aggregate ensemble: (S, M, B) -> (S, B)
        if self.config['q_agg'] == 'min':
            q_per_subtask = jnp.min(qs, axis=1)
        else:
            q_per_subtask = jnp.mean(qs, axis=1)

        # Sequential gating with ground truth dones
        subtask_status = batch.get('subtask_status')
        if subtask_status is not None and subtask_status.ndim == 3:
            subtask_status = subtask_status[:, -1, :]  # (B, H, S) -> (B, S)
        gate_weights = self._get_sequential_gate_weights(subtask_status)  # (S, B)

        # Gated Q-total: sum_s [Q_s * gate_s]
        q_gated = q_per_subtask * gate_weights  # (S, B)
        q_total = q_gated.sum(axis=0)  # (B,)

        temp = self.network.select('temp')()

        actor_loss = jnp.mean((temp * log_probs - q_total) * batch['valid'][..., -1])

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
            'q_gated_mean': q_total.mean(),
            'temp': temp,
            'gate_weights_mean': gate_weights.mean(),
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
        """EMA update for target params. Verbatim from sac_qipm."""
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

    # --- BC methods (actor-only, unchanged from sac_qipm) ---

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

    # --- Q warmup (critic-only) ---

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
        qs = self.network.select('critic')(observations, actions=flat_actions)  # (S, M, B)
        if self.config['q_agg'] == 'min':
            return jnp.min(qs, axis=1)  # (S, B)
        return jnp.mean(qs, axis=1)  # (S, B)

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

        # FloQ critic config: derive q_min/q_max/noise_min/noise_max from the
        # ACTUAL realized TD target range under QIPM, NOT the naive r/(1-γ)
        # ceiling. This is critical: QIPM caps each per-subtask TD target at
        # `V_subgoal=15` from above (see qipm_helper.py:69), so the velocity
        # field's `z_1 = TD_target` lives in roughly [-100, +15], not the plain
        # SAC range [-100, +500]. Using the naive ceiling produces bins that
        # are 6x too wide and a Gaussian return-embedding that's ~2x wider than
        # the entire useful Q range — i.e., a velocity field that can't read
        # its own z_t input.
        floq_cfg = dict(config.get('floq', {}))
        discount = config['discount']
        r_min = floq_cfg.get('r_min', -1.0)
        # V_subgoal must match the constant hardcoded in
        # ogpo/agents/modules/qipm_helper.py:69. If that file changes, this
        # default needs to change too.
        v_subgoal = floq_cfg.get('v_subgoal', 15.0)
        q_offset = floq_cfg.get('q_offset', 5.0)            # Q-unit safety margin
        noise_coverage = floq_cfg.get('noise_coverage', 0.2)

        q_min = (r_min / (1.0 - discount)) - q_offset       # ≈ -100 - 5 = -105
        q_max = v_subgoal + q_offset                         # 15 + 5 = 20
        noise_min = noise_coverage * q_min                   # ≈ -21
        noise_max = noise_coverage * q_max                   # ≈ +4

        # Persist the resolved values back into the agent config so the
        # critic_loss helpers can read them at runtime.
        floq_cfg['q_min'] = float(q_min)
        floq_cfg['q_max'] = float(q_max)
        floq_cfg['noise_min'] = float(noise_min)
        floq_cfg['noise_max'] = float(noise_max)
        config['floq'] = floq_cfg

        critic_def = FloQCritic(
            hidden_dims=tuple(floq_cfg.get('hidden_dims', (512, 512, 512, 512))),
            flow_num_ensembles=int(floq_cfg.get('flow_num_ensembles', 2)),
            layer_norm=bool(floq_cfg.get('layer_norm', True)),
            embed_time=bool(floq_cfg.get('embed_time', True)),
            time_embed_dim=int(floq_cfg.get('time_embed_dim', 64)),
            use_prob_embed=bool(floq_cfg.get('use_prob_embed', True)),
            return_embed_bins=int(floq_cfg.get('return_embed_bins', 51)),
            sigma=float(floq_cfg.get('sigma', 16.0)),
            q_min=float(q_min),
            q_max=float(q_max),
            distill_hidden_dims=tuple(floq_cfg.get('distill_hidden_dims', (512, 512, 512, 512))),
            distill_num_ensembles=int(floq_cfg.get('distill_num_ensembles', 10)),
            critic_flow_steps=int(floq_cfg.get('critic_flow_steps', 4)),
            noise_min=float(noise_min),
            noise_max=float(noise_max),
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
            print(f"[GC] Wrapped FloQ actor and critic with GoalConditionedNetwork "
                  f"(goal_dim={goal_dim}, embed_dim={goal_embed_dim})")

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
            agent_name='sac_qipm_floq',
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
            num_qs=10,                # only used to satisfy common.yaml expectations
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            init_temperature=1.0,
            target_entropy=ml_collections.config_dict.placeholder(float),
            critic_loss_type='mse',   # the distilled head uses MSE; FloQ training is separate
            num_bins=256,
            entropy_backup=False,
            no_of_subtasks=4,
            use_dense_rewards=True,
            normalize_rewards=False,
            gate_on_self=False,
            goal_conditioning_mode=ml_collections.config_dict.placeholder(str),
            goal_conditioned_actor=True,
            goal_conditioned_critic=True,
            goal_embed_dim=32,
            advantage_gating_mode='grandparent_future',
            best_of_n=1,
            # FloQ-specific (overridable from yaml/CLI as agent.floq.*)
            floq=dict(
                hidden_dims=(512, 512, 512, 512),
                flow_num_ensembles=2,
                layer_norm=True,
                embed_time=True,
                time_embed_dim=64,
                use_prob_embed=True,
                return_embed_bins=51,
                sigma=16.0,
                distill_hidden_dims=(512, 512, 512, 512),
                distill_num_ensembles=10,
                critic_flow_steps=4,
                noise_samples=1,
                noise_coverage=0.2,            # 0.2 * QIPM q_max ≈ +4 noise headroom
                r_min=-1.0,
                v_subgoal=15.0,                # MUST match qipm_helper.py:69 V_subgoal
                q_offset=5.0,                  # Q-unit safety margin on both ends of the bin support
                distill_weight=1.0,
                q_agg='mean',
            ),
        )
    )
    return config
