"""SAC + QIPM: SAC with per-subtask Q-functions and sequential gating.

Combines SAC's entropy-regularized actor-critic with QIPM's:
- Shared-backbone critic with per-subtask output heads
- Per-subtask TD targets (reward=+5 done, -1 not done, capped at V_subgoal=15)
- Sequential gating: subtask i gated by done(i-1)
- Ground truth subtask_status for gating (no learned done predictor)
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
from ogpo.agents.modules.q_helper import (
    sample_mip_q_values,
    sample_mip_q_ensemble_values,
    compute_mip_q_predictions,
    compute_mip_q_ensemble_predictions,
)


def _slice_actor_obs(observations, actor_obs_dim):
    """Slice the actor's view of the observation to its leading actor_obs_dim
    dims (None = full obs). sac_qipm's opt-in ``critic_joint_proprio`` widens
    obs with joint proprioception for the CRITICS (q/qdot carry velocity and
    arm configuration that the eef-centric keys lack), but the offline demos'
    joint dims are recorded under the ORIGINAL (OSC) controller while eval
    runs under the mink controller — 7-DoF redundancy resolution and velocity
    profiles differ, so an actor conditioned on them degrades as BC fits
    those features (observed 2026-07-27 in safe_qipm: sub0 0.89 -> 0.11 over
    BC 40k..130k; see docs/changes/2026-07-27-actor-obs-slice/). The proprio
    keys are APPENDED to the obs, so a prefix slice removes exactly them.
    Idempotent (a prefix of a prefix). Goal conditioning appends AFTER
    proprio and would be sliced away — asserted off in create() when
    actor_obs_dim is set.
    """
    if actor_obs_dim is None:
        return observations
    return observations[..., :actor_obs_dim]


class SACQIPMAgent(flax.struct.PyTreeNode):
    """SAC agent with per-subtask Q-functions and sequential gating."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_sequential_gate_weights(self, subtask_status):
        """Compute sequential gating weights from ground truth subtask status.

        Legacy: gate_on_self=True uses 1-done with final subtask=1.

        Supports 4 gating modes via config['advantage_gating_mode']:
        - goal_myopic: active_mask * (1 - done_i)
        - goal_future: 1 - done_i
        - grandparent_myopic: active_mask * (1 - parent_done), last subtask = 1
        - grandparent_future (default): 1 - parent_done, last subtask = 1

        Args:
            subtask_status: (B, S) ground truth done flags.

        Returns:
            (S, B) gating weights.
        """
        done = subtask_status.T  # (S, B)

        # Legacy gate_on_self mode: 1 - done_i with final subtask = 1
        if self.config.get('gate_on_self', False):
            gate_weights = 1.0 - done
            gate_weights = gate_weights.at[-1, :].set(1.0)
            return gate_weights

        gating_mode = self.config.get('advantage_gating_mode', 'grandparent_future')

        # Active goal: first undone subtask per batch element
        active_idx = jnp.argmin(done, axis=0)  # (B,)
        active_mask = jax.nn.one_hot(active_idx, done.shape[0]).T  # (S, B)

        # Parent done: d_{i-1} with d_{-1} = 1 (subtask 0 prerequisite trivially done)
        parent_done = jnp.concatenate([
            jnp.full((1, done.shape[1]), 1.0),
            done[:-1, :]
        ], axis=0)  # (S, B)

        if gating_mode == 'goal_myopic':
            gate_weights = active_mask * (1.0 - done)  # (S, B)
        elif gating_mode == 'goal_future':
            gate_weights = 1.0 - done  # (S, B)
        elif gating_mode == 'grandparent_myopic':
            gate_weights = active_mask * (1.0 - parent_done)  # (S, B)
        elif gating_mode == 'grandparent_future':
            gate_weights = 1.0 - parent_done  # (S, B)
            gate_weights = gate_weights.at[-1, :].set(1.0)
        else:
            assert False, (
                f"Unknown advantage_gating_mode: {gating_mode}. "
                f"Must be 'goal_myopic', 'goal_future', 'grandparent_myopic', or 'grandparent_future'."
            )

        return gate_weights

    def critic_loss(self, batch, grad_params, rng):
        """Per-subtask critic loss using QIPM TD targets. Supports both standard ensemble Q and MIP-Q critics."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next actions (the actor sees the sliced obs; critics keep
        # the full width — see _slice_actor_obs)
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(
            _slice_actor_obs(batch['next_observations'], self.config['actor_obs_dim']))
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        S = self.config['no_of_subtasks']
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
                    noise_rng, (bsz, S), minval=-noise_scale, maxval=noise_scale,
                )
                next_qs = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=batch['next_observations'],
                    actions=next_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                )  # (S, M, B)
            else:
                noise_samples = jax.random.uniform(
                    noise_rng, (num_M, bsz, S), minval=-noise_scale, maxval=noise_scale,
                )
                next_qs = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=batch['next_observations'],
                    actions=next_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                )  # (S, N, B)
        else:
            # Next Q-values: (S, M, B)
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], actions=next_actions,
            )

        # Aggregate over ensemble/noise axis (axis=1) -> (S, B)
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=1)
        else:
            next_q = jnp.mean(next_qs, axis=1)

        # Per-subtask TD targets: (S, B)
        target_q = compute_qipm_td_target(
            subtask_status=batch['subtask_status'],
            next_q=next_q,
            masks=batch['masks'],
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
            no_of_subtasks=S,
        )

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
                )  # (S, M, B), (S, M, B)
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
                )  # (S, N, B), (S, N, B)
            td_err_0 = jnp.square(q_0 - target_q[:, None, :])
            td_err_t = jnp.square(q   - target_q[:, None, :])
            critic_loss = ((td_err_0 + td_err_t) * batch['valid'][..., -1]).mean()
        else:
            # Current Q-values (standard ensemble path): (S, M, B)
            q = self.network.select('critic')(
                batch['observations'], actions=batch_actions, params=grad_params,
            )
            td_err = jnp.square(q - target_q[:, None, :])
            critic_loss = (td_err * batch['valid'][..., -1]).mean()

        # --- Per-subgoal TD error logging (for wandb diagnostics) ---
        # All td_err* tensors have shape (S, M, B). Reduce over (M, B) with valid-mask
        # weighting so the per-subgoal scalar matches the formulation of critic_loss.
        v = batch['valid'][..., -1].astype(jnp.float32)  # (B,)
        if use_mip_q:
            # td_err_0 + td_err_t both (S, M, B); ensemble dim is axis=1.
            ens_dim = td_err_0.shape[1]
            denom = v.sum() * ens_dim + 1e-8
            td_err_0_per_s = (td_err_0 * v).sum(axis=(1, 2)) / denom  # (S,)
            td_err_t_per_s = (td_err_t * v).sum(axis=(1, 2)) / denom  # (S,)
            td_err_per_s = td_err_0_per_s + td_err_t_per_s            # (S,)
        else:
            ens_dim = td_err.shape[1]
            denom = v.sum() * ens_dim + 1e-8
            td_err_per_s = (td_err * v).sum(axis=(1, 2)) / denom      # (S,)

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'next_log_probs_mean': next_log_probs.mean(),
            'td_err_total': td_err_per_s.sum(),
        }
        for i in range(S):
            info[f'td_err_subtask_{i}'] = td_err_per_s[i]

        if use_mip_q:
            info['td_err0_total'] = td_err_0_per_s.sum()
            info['td_err_t_total'] = td_err_t_per_s.sum()
            for i in range(S):
                info[f'td_err0_subtask_{i}'] = td_err_0_per_s[i]
                info[f'td_err_t_subtask_{i}'] = td_err_t_per_s[i]

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """SAC actor loss with sequentially-gated per-subtask Q-values."""
        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(
            _slice_actor_obs(batch['observations'], self.config['actor_obs_dim']),
            params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Per-subtask Q-values for sampled actions (MIP-Q two-step or plain critic forward)
        if self.config.get('use_mip_q', False):
            S = self.config['no_of_subtasks']
            num_M = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            critic_fn = self.network.select('critic')
            bsz = batch['observations'].shape[0]
            rng, noise_rng = jax.random.split(rng)
            if self.config.get('mip_q_ensemble', False):
                noise_sample = jax.random.uniform(
                    noise_rng, (bsz, S), minval=-noise_scale, maxval=noise_scale,
                )
                qs = sample_mip_q_ensemble_values(
                    critic_fn=critic_fn,
                    observations=batch['observations'],
                    actions=actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                )  # (S, M, B)
            else:
                noise_samples = jax.random.uniform(
                    noise_rng, (num_M, bsz, S), minval=-noise_scale, maxval=noise_scale,
                )
                qs = sample_mip_q_values(
                    critic_fn=critic_fn,
                    observations=batch['observations'],
                    actions=actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                )  # (S, N, B)
        else:
            # Per-subtask Q-values for sampled actions: (S, M, B)
            qs = self.network.select('critic')(batch['observations'], actions=actions)

        # Aggregate ensemble/noise axis (axis=1) -> (S, B)
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

        # Temperature
        temp = self.network.select('temp')()

        # SAC actor loss: minimize (temp * log_pi - Q_gated)
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
            'q_gated_mean': q_total.mean(),
            'temp': temp,
            'gate_weights_mean': gate_weights.mean(),
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

    # --- BC methods (same as SAC, actor-only) ---

    def bc_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_actions = jnp.clip(batch_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        dist = self.network.select('actor')(
            _slice_actor_obs(batch['observations'], self.config['actor_obs_dim']),
            params=grad_params)
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

    # --- Q warmup (critic-only, per-subtask TD) ---

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

    @jax.jit
    def batch_critic_only_update(self, batch):
        """Critic-only updates (used by online_rl_runner when utd_q > utd_pi)."""
        agent, infos = jax.lax.scan(self._q_warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # --- Per-subtask Q-values (for QIPM eval) ---

    @jax.jit
    def get_per_subtask_q_values(self, observations, actions):
        """Return per-subtask Q-values aggregated over ensemble.

        Args:
            observations: (B, obs_dim)
            actions: (B, H, action_dim) action chunks

        Returns:
            (S, B) per-subtask Q-values.
        """
        flat_actions = actions.reshape(observations.shape[0], -1)
        if self.config.get('use_mip_q', False):
            S = self.config['no_of_subtasks']
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = observations.shape[0]
            critic_fn = self.network.select('critic')
            if self.config.get('mip_q_ensemble', False):
                # Deterministic eval: zero noise
                noise_sample = jnp.zeros((batch_size, S))
                qs = sample_mip_q_ensemble_values(
                    critic_fn=critic_fn,
                    observations=observations,
                    actions=flat_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                )  # (S, M, B)
            else:
                num_M = self.config.get('num_ensemble_members', 10)
                noise_samples = jnp.zeros((num_M, batch_size, S))
                qs = sample_mip_q_values(
                    critic_fn=critic_fn,
                    observations=observations,
                    actions=flat_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                )  # (S, N, B)
        else:
            qs = self.network.select('critic')(observations, actions=flat_actions)  # (S, M, B)
        if self.config['q_agg'] == 'min':
            return jnp.min(qs, axis=1)  # (S, B)
        return jnp.mean(qs, axis=1)  # (S, B)

    # --- Action selection ---

    @jax.jit
    def sample_actions(self, observations, rng=None, **kwargs):
        # Rollout entry point: receives full-width env obs; the actor's view
        # is sliced here (critics elsewhere keep the full width).
        dist = self.network.select('actor')(
            _slice_actor_obs(observations, self.config['actor_obs_dim']))
        actions = dist.sample(seed=rng)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, **kwargs):
        # Deterministic (ODE-style) eval entry point: same slice as
        # sample_actions.
        dist = self.network.select('actor')(
            _slice_actor_obs(observations, self.config['actor_obs_dim']))
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

        # actor_obs_dim compatibility guards (before any network setup):
        # combinations that would silently mis-train under the actor slice
        # must fail here instead.
        if config['actor_obs_dim'] is not None:
            assert not config.get('goal_conditioning_mode'), (
                "actor_obs_dim is incompatible with goal conditioning: the "
                "goal vector is appended AFTER the proprio dims, so the "
                "actor's prefix slice would silently strip the goals"
            )

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

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        no_of_subtasks = config['no_of_subtasks']

        # Critic: standard per-subtask Value, or MIP-Q variant when use_mip_q is set.
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
                    num_subtasks=no_of_subtasks,
                )
                print(f"[MIP-Q] SAC-QIPM using ValueMIPEnsemble (S={no_of_subtasks}, M={config.get('num_ensemble_members', 10)}, "
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
                    num_subtasks=no_of_subtasks,
                )
                print(f"[MIP-Q] SAC-QIPM using ValueMIP (S={no_of_subtasks}, N={config.get('num_ensemble_members', 10)} noise samples, "
                      f"noise={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)})")
        else:
            # Critic with per-subtask output heads (shared backbone)
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
        # for noise sampling during init.
        critic_init_kw = dict(observations=ex_observations, actions=full_actions)
        target_critic_init_kw = dict(critic_init_kw)
        if config.get('use_mip_q', False):
            init_rng, critic_init_rng, target_critic_init_rng = jax.random.split(init_rng, 3)
            critic_init_kw['rng'] = critic_init_rng
            target_critic_init_kw['rng'] = target_critic_init_rng

        # Actor-side nets are BUILT at the sliced width so that any obs path
        # bypassing the _slice_actor_obs funnel fails loudly with a shape
        # error instead of silently consuming the full observation. Critics
        # init on the full ex_observations above.
        _aod = config['actor_obs_dim']
        if _aod is not None:
            assert 0 < _aod <= ex_observations.shape[-1], (
                f"actor_obs_dim={_aod} must be in (0, obs_dim="
                f"{ex_observations.shape[-1]}]"
            )
        ex_actor_observations = _slice_actor_obs(ex_observations, _aod)

        network_info = dict(
            actor=(actor_def, (ex_actor_observations,)),
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

        if config.get('target_entropy') is None:
            config['target_entropy'] = -float(full_action_dim) / 2

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sac_qipm',
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
            # QIPM-specific
            no_of_subtasks=4,
            use_dense_rewards=True,
            normalize_rewards=False,
            gate_on_self=False,
            # Critic-side joint proprio, opt-in (YAML wins — this dict is dead
            # code; see configs/algos/sac_qipm.yaml)
            critic_joint_proprio=False,
            actor_obs_dim=ml_collections.config_dict.placeholder(int),
            # Goal conditioning
            goal_conditioning_mode=ml_collections.config_dict.placeholder(str),
            goal_conditioned_actor=True,
            goal_conditioned_critic=True,
            goal_embed_dim=32,
            advantage_gating_mode='grandparent_future',
            # MIP-Q (off by default)
            use_mip_q=False,
            mip_q_ensemble=True,
            mip_q_noise_scale=1.0,
            mip_q_t_star=0.9,
            num_ensemble_members=10,
            # Runner compat
            best_of_n=1,
        )
    )
    return config
