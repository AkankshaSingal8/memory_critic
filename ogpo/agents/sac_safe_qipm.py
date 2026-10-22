"""SAC + Q-IPM with always-active safety Q-heads (shared backbone).

A near-fork of ogpo/agents/sac_qipm.py for the ``sac_safe_qipm`` algorithm —
the SAC-line counterpart of ogpo/agents/safe_qipm.py, replicating its design
decisions exactly. The per-subtask critic is widened from S_task heads to
S_task + no_of_safety_subtasks heads on the same shared backbone:

  - Heads [0, S_task): task subgoals — standard QIPM TD (+5 done / -1 pending,
    capped at V_subgoal), gate-weighted in the SAC actor objective.
  - Heads [S_task, S): safety subgoals — CMDP cost critics trained with
    ``_compute_safety_td_target`` (reward = -1 * unsafe-steps-in-chunk, plain
    Bellman backup, zero upper cap, no lower cap, no absorbing state,
    terminals SELECTED rather than multiplied by a 0/1 mask), ALWAYS
    ACTIVE in the actor objective: never gated by task progress.

Scalarization in the actor objective:

    Q_total = sum_task gate_s * Q_s + safety_coeff * sum_safety Q_k

Task gates come from sac_qipm's LOCAL ``_get_sequential_gate_weights`` (kept
verbatim — including its known goal_future divergence from the shared
qipm_helper.compute_gate_weights (no last-row forcing), the legacy
gate_on_self mode, and no DAG support), fed TASK columns only: both gate
implementations force-set rows by position, so safety columns must never
enter them.

The env wrapper appends the per-step safety costs as trailing columns of
``subtask_status`` (see RobomimicLowdimWrapper.configure_safety_status in
envs/robomimic_utils.py), so ``batch['subtask_status']`` is
(B, H, S_task + no_of_safety_subtasks). Two sources, chosen by the config key
``safety_head_mode`` — the math below is identical either way, only K changes:

  - "single": one column from ``info['unsafe_step']``, the aggregate unsafe bit.
  - "multi": K columns from ``info['safety_violations']``, one per safety
    CONSTRAINT (joint_limits / velocity / collision / workspace / keepout),
    giving one always-active cost head per constraint. See
    configs/algos/sac_safe_qipm_multihead.yaml.

NOTE: in "multi" an all-zero safety row DOES mean the step was safe.
``info['unsafe_step']`` is exactly ``any(info['safety_violations'])`` — both are
decided from the REQUEST ``q_des`` before the safety filter runs, so no cost
mass can escape the heads. Because detection is demand-side, the rows describe
what the policy ASKED for and are identical whether or not the shield
prevented it — the property that makes the experimental conditions comparable.

``no_of_subtasks`` keeps meaning the TASK count everywhere; the total head
count is derived at the seams, never stored. v1 restrictions (no MIP-Q / no
goal conditioning; K >= 1; a categorical critic's support grid must reach the
safety cost floor; offline_ratio == 0 is asserted at dispatch in main.py) are
asserted fail-fast in ``create()``.

This file is intentionally self-contained: no inheritance from SACQIPMAgent
to avoid Flax PyTreeNode subclassing pitfalls.
"""
import copy
from typing import Any

import chex
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


def _compute_safety_td_target(
    subtask_status: jnp.ndarray,
    next_q: jnp.ndarray,
    masks: jnp.ndarray,
    discount,
    horizon_length: int,
) -> jnp.ndarray:
    """CMDP cost-critic TD target for the always-active safety heads.

    The env wrapper appends per-step unsafe flags (0/1, or graded costs in
    [0, 1]) as the trailing columns of ``subtask_status``; unlike task flags
    they are instantaneous, NOT ratcheting. The per-chunk cost is the SUM of
    the flags over the H-step chunk ("-1 per unsafe step") and the backup is
    a plain discounted Bellman equation:

        Q_safety(s, a) = min(0, -1 * sum_H(unsafe) + gamma_safety^H * Q'_safety)

    with the bootstrap term SELECTED off on terminals rather than multiplied
    by a 0/1 mask (2026-08-02; see the terminal-select note below).

    No V_subgoal cap and no absorbing terminal value: safety never
    "completes", so the head is an expected discounted future cost, not a
    milestone value. The only bound is the ZERO UPPER CAP (2026-07-27): the
    true cost-Q is <= 0 identically, so positive targets can only be
    bootstrap overestimation and are clamped; the negative side stays
    uncapped (bounded anyway by -H/(1-gamma_safety^H)). Chunks that straddle
    an episode boundary need no masking here — the critic loss multiplies by
    valid[..., -1], which is 0 for
    exactly those chunks (ogpo/utils/datasets.py sample_sequence).

    Adapted from sac_qipm_avoid_obstacle._compute_safety_td_target with two
    deliberate differences: sum over the chunk instead of the last timestep
    (a binary flag firing mid-chunk would otherwise be dropped), and
    coefficient -1 instead of -5 (the scalarization weight lives in
    safety_coeff).

    Args:
        subtask_status: (B, horizon_length, K) per-step unsafe flags.
        next_q: (K, B) aggregated next Q-values for the safety heads.
        masks: (B,) or (B, horizon_length) termination masks.
        discount: scalar or (K,) per-safety-head discount factors.
        horizon_length: action chunking horizon.

    Returns:
        TD targets of shape (K, B).
    """
    chex.assert_shape(subtask_status, (None, horizon_length, None))
    if masks.ndim > 1:
        masks = masks[..., -1]

    disc_factor = jnp.asarray(discount) ** horizon_length
    if disc_factor.ndim > 0:
        disc_factor = disc_factor[:, None]  # (K, 1)

    # (K, B) unsafe-step count summed over the chunk
    unsafe_count = subtask_status.sum(axis=-2).T.astype(jnp.float32)

    rewards = -1.0 * unsafe_count
    # Terminal select, NOT the multiplicative blend `masks * next_q`
    # (2026-08-02, backported from _compute_safety_hjb_target): IEEE gives
    # 0.0*inf = 0.0*nan = nan, so one non-finite element of next_q on a chunk
    # the mask was supposed to erase would nan the whole batch's td_loss
    # through the mean. Nothing checks next_q for finiteness. Bit-identical to
    # the old form for finite next_q: mask=1 -> rewards + disc*next_q,
    # mask=0 -> rewards, exactly as before. masks are binary here
    # (1 - float(terminated), ogpo/runners/qipm_runner.py).
    target_q = jnp.where(
        masks[None, :] > 0, rewards + disc_factor * next_q, rewards
    )
    # Zero upper cap (2026-07-27): a cost critic's true Q is <= 0 identically
    # (rewards are never positive), so any positive target is bootstrap
    # overestimation on OOD actions — the 2026-07-26 runs logged
    # q_subtask_4_max up to +141. No LOWER cap: large negative values are
    # legitimate discounted cost (see docs/changes/2026-07-27-safety-td-zero-cap/).
    return jnp.minimum(target_q, 0.0)


def _slice_actor_obs(observations, actor_obs_dim):
    """Slice the actor's view of the observation to its leading actor_obs_dim
    dims (None = full obs). The safe_qipm family widens obs with joint
    proprioception for the SAFETY CRITICS by default (critic_joint_proprio,
    opt-out available since 2026-08-07; q/qdot are what the constraint
    surfaces are functions of), but the offline demos' joint dims are recorded
    under the ORIGINAL (OSC) controller while eval runs under the mink
    controller — 7-DoF redundancy resolution and velocity profiles differ, so
    an actor conditioned on them degrades as BC fits those features (observed
    2026-07-27: sub0 0.89 -> 0.11 over BC 40k..130k in all four v2 runs; see
    docs/changes/2026-07-27-actor-obs-slice/). The proprio keys are APPENDED
    to the obs, so a prefix slice removes exactly them. Idempotent (a prefix
    of a prefix; actor_obs_dim == full width is a value-identity, the opt-out
    geometry). Goal conditioning appends AFTER proprio and would be sliced
    away — asserted off in sac_safe_qipm v1.
    """
    if actor_obs_dim is None:
        return observations
    return observations[..., :actor_obs_dim]


class SACSafeQIPMAgent(flax.struct.PyTreeNode):
    """SAC + Q-IPM with always-active safety Q-heads on a shared-backbone critic."""

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

        S_task = self.config['no_of_subtasks']
        n_safety = self.config['no_of_safety_subtasks']
        S = S_task + n_safety  # total head count (derived at seams, never stored)
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

        # Per-subtask TD targets: task heads use QIPM synthesis (+5/-1, capped
        # at V_subgoal) with the SCALAR discount (this fork has no
        # subtask_discounts plumbing — deliberate, matching sac_qipm); safety
        # heads use the plain CMDP cost backup via _compute_safety_td_target
        # (zero upper cap, no lower cap, no absorbing state) with
        # safety_discount. Both helpers squeeze masks internally.
        target_q_task = compute_qipm_td_target(
            subtask_status=batch['subtask_status'][..., :S_task],
            next_q=next_q[:S_task],
            masks=batch['masks'],
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
            no_of_subtasks=S_task,
        )
        target_q_safety = _compute_safety_td_target(
            subtask_status=batch['subtask_status'][..., S_task:],
            next_q=next_q[S_task:],
            masks=batch['masks'],
            discount=jnp.full((n_safety,), self.config['safety_discount']),
            horizon_length=self.config['horizon_length'],
        )
        target_q = jnp.concatenate([target_q_task, target_q_safety], axis=0)  # (S, B)

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

        # Per-head Q statistics (parity with safe_qipm.get_td_loss — the
        # q_subtask_{i}_max metric is what caught the 2026-07-26 safety-head
        # collapse's +141 overshoot). Uniform naming across task and safety
        # heads, no safety_ prefix (rename considered and declined for
        # safe_qipm; same decision here).
        q_per_subtask_mean = q.mean(axis=(1, 2))  # (S,)
        q_per_subtask_min = q.min(axis=(1, 2))    # (S,)
        q_per_subtask_max = q.max(axis=(1, 2))    # (S,)

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
            info[f'q_subtask_{i}_mean'] = q_per_subtask_mean[i]
            info[f'q_subtask_{i}_min'] = q_per_subtask_min[i]
            info[f'q_subtask_{i}_max'] = q_per_subtask_max[i]

        if use_mip_q:
            info['td_err0_total'] = td_err_0_per_s.sum()
            info['td_err_t_total'] = td_err_t_per_s.sum()
            for i in range(S):
                info[f'td_err0_subtask_{i}'] = td_err_0_per_s[i]
                info[f'td_err_t_subtask_{i}'] = td_err_t_per_s[i]

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """SAC actor loss: gated task Q-sum plus always-active safety Q-sum."""
        S_task = self.config['no_of_subtasks']
        n_safety = self.config['no_of_safety_subtasks']
        safety_coeff = self.config['safety_coeff']

        rng, sample_rng = jax.random.split(rng)
        dist = self.network.select('actor')(
            _slice_actor_obs(batch['observations'], self.config['actor_obs_dim']),
            params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        # Per-subtask Q-values for sampled actions (MIP-Q two-step or plain critic forward)
        if self.config.get('use_mip_q', False):
            S = S_task + n_safety
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

        # Sequential gating with ground truth dones — TASK columns only. Both
        # gate implementations force-set rows by position (the local helper
        # sets its last row to 1 in two modes), so safety columns must never
        # enter them. Safety heads are added un-gated (always active) below.
        subtask_status = batch.get('subtask_status')
        if subtask_status is not None and subtask_status.ndim == 3:
            subtask_status = subtask_status[:, -1, :]  # (B, H, S) -> (B, S)
        gate_weights = self._get_sequential_gate_weights(
            subtask_status[:, :S_task])  # (S_task, B)

        # Scalarization: Q_total = sum_task gate_s*Q_s + safety_coeff * sum_safety Q_k
        q_gated = q_per_subtask[:S_task] * gate_weights  # (S_task, B)
        q_total = (
            q_gated.sum(axis=0)
            + safety_coeff * q_per_subtask[S_task:].sum(axis=0)
        )  # (B,)

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
            # q_gated_mean is the full scalarized total (gated task sum +
            # safety_coeff * safety sum); key kept for dashboard continuity.
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
            observations: (B, obs_dim) full-width obs (critic view; no slice)
            actions: (B, H, action_dim) action chunks

        Returns:
            (S_task + K, B) raw per-head Q-values — task heads then safety
            heads. Eval consumers slice [:no_of_subtasks]
            (ogpo/utils/evaluation_qipm.py); safety heads are excluded from
            eval metrics in v1.
        """
        flat_actions = actions.reshape(observations.shape[0], -1)
        if self.config.get('use_mip_q', False):
            S = self.config['no_of_subtasks'] + self.config['no_of_safety_subtasks']
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

        # --- v1 restrictions: combinations unverified with safety heads are
        # rejected fail-fast (docs/changes/2026-08-10-sac-safe-qipm-safety-heads) ---
        assert config['no_of_safety_subtasks'] >= 1, (
            "sac_safe_qipm requires no_of_safety_subtasks >= 1; use "
            "algo=sac_qipm for a pure task-head agent"
        )
        assert not config.get('use_mip_q', False), \
            "sac_safe_qipm v1 does not support use_mip_q"
        assert not config.get('goal_conditioning_mode'), \
            "sac_safe_qipm v1 does not support goal conditioning"

        # Categorical-critic support-grid guard (2026-08-02, ported from
        # safe_qipm). Deliberately NOT safe_qipm_hjb's blanket
        # `critic_loss_type == 'mse'`: that assert exists because positive HJ
        # margins are unrepresentable when q_max = 0, and q_max = 0 is exactly
        # RIGHT for cost heads (the true cost-Q is <= 0 identically). The real
        # exposure here is the other edge — a hlgauss support grid is
        # linspace(q_min, q_max, num_bins) SHARED across every head, so if
        # q_min stops above the safety head's floor the head saturates in the
        # bottom bin and simply stops reporting how much cost it has
        # accumulated, with no error.
        if config['critic_loss_type'] != 'mse':
            _safety_floor = -float(config['horizon_length']) / (
                1.0 - float(config['safety_discount']) ** config['horizon_length']
            )
            _q_min = config.get('q_min', None)
            assert _q_min is not None and float(_q_min) <= _safety_floor, (
                f"critic_loss_type={config['critic_loss_type']!r} regresses onto a "
                f"support grid shared by all heads, but its lower edge q_min="
                f"{_q_min} does not reach the safety heads' floor "
                f"-horizon_length/(1 - safety_discount**horizon_length) = "
                f"{_safety_floor:.1f} (horizon_length="
                f"{config['horizon_length']}, safety_discount="
                f"{config['safety_discount']}). The cost heads would saturate "
                f"in the bottom bin instead of tracking accumulated cost. Use "
                f"critic_loss_type='mse', or lower q_min to at most "
                f"{_safety_floor:.1f} (it is derived from `discount` in "
                f"ogpo/utils/main_helper.get_q_bounds)."
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
        no_of_safety_subtasks = config['no_of_safety_subtasks']
        total_subtasks = no_of_subtasks + no_of_safety_subtasks
        print("[sac_safe_qipm] Creating shared-backbone critic: "
              f"{no_of_subtasks} task + "
              f"{no_of_safety_subtasks} safety heads")

        # Critic: standard per-subtask Value, or MIP-Q variant when use_mip_q
        # is set (dead in v1 — asserted off above; widened to total_subtasks
        # for consistency).
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
                    num_subtasks=total_subtasks,
                )
                print(f"[MIP-Q] SAC-Safe-QIPM using ValueMIPEnsemble (S={total_subtasks}, M={config.get('num_ensemble_members', 10)}, "
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
                    num_subtasks=total_subtasks,
                )
                print(f"[MIP-Q] SAC-Safe-QIPM using ValueMIP (S={total_subtasks}, N={config.get('num_ensemble_members', 10)} noise samples, "
                      f"noise={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)})")
        else:
            # Critic with per-subtask output heads (shared backbone):
            # task heads [0, S_task) then safety heads [S_task, S).
            critic_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config.get('num_bins', 256),
                q_min=config.get('q_min', None),
                q_max=config.get('q_max', None),
                num_subtasks=total_subtasks,
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
            agent_name='sac_safe_qipm',
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
            # Always-active safety Q-heads (YAML wins — this dict is dead
            # code; see configs/algos/sac_safe_qipm.yaml)
            no_of_safety_subtasks=1,
            safety_coeff=1.0,
            safety_discount=0.99,
            safety_head_mode='single',
            critic_joint_proprio=True,
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
