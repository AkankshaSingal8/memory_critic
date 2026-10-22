"""Helpers for the FloQ + QIPM critic training step.

Pure functions called from `SACQIPMFloqAgent.critic_loss`. They:

  1. Compute per-subtask QIPM TD targets using the *target* velocity field's
     Euler-integrated flow Q on (s', a') as the bootstrap value (instead of a
     standard MLP forward as `sac_qipm` does).
  2. Compute the two-loss FloQ critic update:
       - Flow-matching loss on the velocity field's predicted velocity vs the
         linear interpolation target velocity (z_1 - z_0).
       - Distillation loss on the distilled critic vs a stop-gradient'd flow Q
         from integrating the *online* velocity field.

These two helpers compose so the new agent's critic_loss is essentially a
4-line wrapper that delegates to them. The shared `compute_qipm_td_target`
helper from `ogpo/agents/modules/qipm_helper.py` is reused unchanged.
"""
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from ogpo.agents.modules.qipm_helper import compute_qipm_td_target


def compute_floq_qipm_td_target(
    network,
    batch: Dict[str, jnp.ndarray],
    next_actions: jnp.ndarray,
    discount: float,
    horizon_length: int,
    no_of_subtasks: int,
    q_agg: str,
    rng: jnp.ndarray,
) -> jnp.ndarray:
    """Compute QIPM-style per-subtask TD targets via the TARGET velocity field.

    Pipeline:
      1. Sample independent noise per subtask: noise_ratios shape (B, S).
      2. Run target velocity field's Euler integration on (s', a') via
         `network.select('target_critic').integrate_flow(...)` -> (E, B, S).
      3. Aggregate over ensemble dim (axis=0) -> (B, S), transpose -> (S, B).
      4. Pass into the existing `compute_qipm_td_target` to apply per-subtask
         reward synthesis (+5/-1, V_subgoal cap, terminal fix) and produce
         the final (S, B) TD targets.

    The shape gotcha to remember (validation note 3): `compute_qipm_td_target`
    expects next_q in (S, B), while `integrate_flow` returns (E, B, S).
    Aggregate over axis=0 then transpose.

    Returns:
        td_targets: (S, B) per-subtask TD targets.
    """
    B = batch["observations"].shape[0]
    S = no_of_subtasks

    # Independent noise per subtask. Shared across ensemble members (acorn pattern).
    noise_ratios = jax.random.uniform(rng, (B, S))

    # Run target velocity field's Euler integration on (s', a').
    # Use ModuleDict.call_submodule (flax_utils.py:52) to dispatch the
    # `integrate_flow` method on the inner FloQCritic — `network.select(...).
    # apply(method='integrate_flow')` would crash because TrainState routes
    # method dispatch through the OUTER ModuleDict which has no integrate_flow.
    next_flow_q = network(
        'target_critic',
        'integrate_flow',
        batch['next_observations'],
        next_actions,
        noise_ratios,
        method='call_submodule',
    )  # (E, B, S)

    # Aggregate over ensemble.
    if q_agg == 'min':
        next_q_per_subtask = jnp.min(next_flow_q, axis=0)  # (B, S)
    else:
        next_q_per_subtask = jnp.mean(next_flow_q, axis=0)  # (B, S)

    # Transpose to QIPM helper's (S, B) convention.
    next_q_qipm = next_q_per_subtask.T  # (S, B)

    # Apply QIPM per-subtask reward synthesis and bootstrap.
    target_q = compute_qipm_td_target(
        subtask_status=batch['subtask_status'],
        next_q=next_q_qipm,
        masks=batch['masks'],
        discount=discount,
        horizon_length=horizon_length,
        no_of_subtasks=no_of_subtasks,
    )  # (S, B)

    return target_q


def floq_critic_loss_components(
    network,
    grad_params: Any,
    batch: Dict[str, jnp.ndarray],
    qipm_td_target: jnp.ndarray,  # (S, B) from compute_floq_qipm_td_target
    actions: jnp.ndarray,         # (B, full_action_dim) — chunk-flattened by caller
    no_of_subtasks: int,
    flow_num_ensembles: int,
    distill_weight: float,
    noise_min: float,
    noise_max: float,
    rng: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute (flow_matching_loss + distill_weight * distillation_loss).

    Single integration of the online velocity field on (s, a) is reused for
    the distillation target — no separate forward needed. The flow-matching
    loss path computes z_t directly via interpolation (no flow integration).

    Loss A — Flow matching:
      z_0 ~ Uniform[noise_min, noise_max]^(B, S), independent per subtask
      z_1 = qipm_td_target.T  -> (B, S)
      t   ~ Uniform[0, 1]      -> (B, 1)
      z_t = (1 - t) * z_0 + t * z_1  -> (B, S) -> tile to (E, B, S)
      target_v = z_1 - z_0             -> tile to (E, B, S)
      pred_v = velocity_field(s, a, z_t, t; params=grad_params)  -> (E, B, S)
      flow_loss = mean(MSE(pred_v, target_v) * valid_mask)

    Loss B — Distillation:
      noise_ratios ~ Uniform[0, 1]^(B, S)
      flow_q_full = integrate_flow(s, a; params=grad_params)  -> (E, B, S)
      flow_q_target = stop_gradient(flow_q_full.mean(axis=0))  -> (B, S)
      distilled_q = distilled(s, a; params=grad_params)        -> (S, M, B) or (M, B)
      For S>1: distilled mean over ensemble axis=1 -> (S, B), transpose -> (B, S)
      For S=1: distilled mean over ensemble axis=0 -> (B,), unsqueeze -> (B, 1)
      distill_loss = mean(MSE(distilled_q_mean, flow_q_target) * valid_mask)

    Returns:
        (total_loss, metrics_dict)
    """
    B = batch["observations"].shape[0]
    S = no_of_subtasks
    E = flow_num_ensembles

    valid_mask = batch['valid'][..., -1]  # (B,) — last horizon step's validity

    rng, noise_rng_a, time_rng, noise_rng_b = jax.random.split(rng, 4)

    # =========================================================================
    # Loss A — Flow matching (no flow integration needed for the loss path)
    # =========================================================================
    # Sample independent z_0 per subtask in the noise range.
    noise_ratios_flow = jax.random.uniform(noise_rng_a, (B, S))
    z_0 = noise_min * (1 - noise_ratios_flow) + noise_max * noise_ratios_flow  # (B, S)
    z_1 = qipm_td_target.T  # (S, B) -> (B, S)

    # Flow time t shared across batch (one t per minibatch is fine; we sample
    # a single time for all elements like acorn does).
    t_flow = jax.random.uniform(time_rng, (B, 1))  # (B, 1)

    # Interpolate. Broadcast t over the S dim.
    z_t = (1.0 - t_flow) * z_0 + t_flow * z_1  # (B, S)
    target_v = z_1 - z_0                        # (B, S)

    # Tile to ensemble dim for the velocity field forward.
    z_t_ens = jnp.tile(z_t[None], [E, 1, 1])           # (E, B, S)
    target_v_ens = jnp.tile(target_v[None], [E, 1, 1])  # (E, B, S)

    pred_v = network(
        'critic',
        'velocity',
        batch['observations'],
        actions,
        z_t_ens,
        t_flow,
        method='call_submodule',
        params=grad_params,
    )  # (E, B, S)

    flow_se = jnp.square(pred_v - target_v_ens)  # (E, B, S)
    # Apply valid mask along batch dim. Mean over E and S to keep scale aligned
    # with how OGPO's other critic losses compute MSE.
    flow_loss = jnp.mean(flow_se * valid_mask[None, :, None])

    # =========================================================================
    # Loss B — Distillation (one integration of the online velocity field)
    # =========================================================================
    noise_ratios_distill = jax.random.uniform(noise_rng_b, (B, S))
    flow_q_full = network(
        'critic',
        'integrate_flow',
        batch['observations'],
        actions,
        noise_ratios_distill,
        method='call_submodule',
        params=grad_params,
    )  # (E, B, S)
    # Mean over ensemble + stop_gradient. This is the distillation target.
    flow_q_target = jax.lax.stop_gradient(jnp.mean(flow_q_full, axis=0))  # (B, S)

    # Distilled critic forward (no method= -> __call__ -> distilled head).
    distilled_q = network.select('critic')(
        batch['observations'],
        actions=actions,
        params=grad_params,
    )  # (S, M, B) when S>1, (M, B) when S=1

    # Aggregate ensemble dim and align to (B, S) for the MSE.
    if S > 1:
        # distilled_q: (S, M, B) -> mean over M -> (S, B) -> transpose -> (B, S)
        distilled_q_mean = jnp.mean(distilled_q, axis=1).T  # (B, S)
    else:
        # distilled_q: (M, B) -> mean over M -> (B,) -> unsqueeze -> (B, 1)
        distilled_q_mean = jnp.mean(distilled_q, axis=0)[:, None]  # (B, 1)

    distill_se = jnp.square(distilled_q_mean - flow_q_target)  # (B, S)
    distill_loss = jnp.mean(distill_se * valid_mask[:, None])

    total_loss = flow_loss + distill_weight * distill_loss

    metrics = {
        'critic_loss': total_loss,
        'floq_flow_loss': flow_loss,
        'floq_distill_loss': distill_loss,
        'flow_q_mean': flow_q_target.mean(),
        'distilled_q_mean': distilled_q_mean.mean(),
        'td_target_mean': qipm_td_target.mean(),
        'q_mean': distilled_q.mean(),
        'q_max': distilled_q.max(),
        'q_min': distilled_q.min(),
    }
    return total_loss, metrics
