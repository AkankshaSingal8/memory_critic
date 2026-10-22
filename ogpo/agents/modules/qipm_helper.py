"""Q-IPM helper functions: per-subtask TD targets, advantages, done predictor
loss, and the safe-family safety-head aggregation shared by agents/safe_qipm.py
and agents/modules/fql_helper.py."""

from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
import numpy as np


def compute_gate_weights(
    subtask_status_last: jnp.ndarray,
    gating_mode: str = 'grandparent_future',
    dag: Optional[Dict[int, list]] = None,
) -> jnp.ndarray:
    """Per-subtask gate weights for Q-scalarization.

    Lifted verbatim from ``OGPOQIPMAgent._compute_advantages`` so that
    both qipm and qc_qipm share a single source of truth for the gating rules.

    Args:
        subtask_status_last: (B, S) — gold done flags at the end of the
            current chunk (1 = done, 0 = pending).
        gating_mode: One of
            - 'goal_myopic'         : only the active (first undone) subtask
            - 'goal_future'         : every undone subtask contributes
            - 'grandparent_myopic'  : only the active subtask, weighted by
                                      (1 - parent_done)
            - 'grandparent_future'  : every subtask, weighted by
                                      (1 - parent_done)
            - 'last_only'           : only the final subtask contributes
                                      (gate = [0, ..., 0, 1] always, ignores
                                      subtask_status and dag).
            For the DAG variant, only 'goal_myopic', 'goal_future',
            'grandparent_future' are supported.
        dag: Optional {subtask_idx: [prereq_indices]}. When provided, uses
            DAG-based prerequisite gating instead of the sequential rule.

    Returns:
        ``(S, B)`` gate weights. Non-zero entries indicate subtasks that
        should contribute to the scalarized Q (or advantage).
    """
    # (B, S) -> (S, B)
    done_probs_t = subtask_status_last.T
    num_subtasks = done_probs_t.shape[0]
    batch_size = done_probs_t.shape[1]

    if gating_mode == 'last_only':
        gate_weights = jnp.zeros((num_subtasks, batch_size))
        gate_weights = gate_weights.at[-1, :].set(1.0)
        return gate_weights

    if dag is not None:
        # Compute per-subtask all_prereqs_done: (S, B)
        all_prereqs_done_list = []
        for subtask_idx in range(num_subtasks):
            prereqs = dag.get(subtask_idx, [])
            if prereqs:
                prereq_dones = jnp.stack([done_probs_t[p] for p in prereqs], axis=0)
                apd = prereq_dones.prod(axis=0)  # (B,) — AND logic
            else:
                apd = jnp.ones(batch_size)
            all_prereqs_done_list.append(apd)
        all_prereqs_done = jnp.stack(all_prereqs_done_list, axis=0)  # (S, B)

        if gating_mode == 'goal_myopic':
            # DAG frontier only: prereqs done AND self not done
            gate_weights = all_prereqs_done * (1.0 - done_probs_t)
        elif gating_mode == 'goal_future':
            # All undone subtasks contribute (no DAG ordering used)
            gate_weights = 1.0 - done_probs_t
        elif gating_mode == 'grandparent_future':
            # Weighted by how far prereqs are from being met
            gate_weights = 1.0 - all_prereqs_done
        else:
            assert False, f"Unknown advantage_gating_mode for DAG: {gating_mode}"

        # Final subtask always contributes (ultimate goal)
        gate_weights = gate_weights.at[-1, :].set(1.0)
        return gate_weights

    # Sequential gating: subtask i is gated by done(i-1)
    # Active goal: first undone subtask per batch element
    active_idx = jnp.argmin(done_probs_t, axis=0)  # (B,)
    active_mask = jax.nn.one_hot(active_idx, num_subtasks).T  # (S, B)

    # Parent done: d_{i-1} with d_{-1} = 1 (subtask 0 prerequisite trivially done)
    parent_done = jnp.concatenate([
        jnp.full((1, batch_size), 1.0),
        done_probs_t[:-1, :]
    ], axis=0)  # (S, B)

    if gating_mode == 'goal_myopic':
        # Only active goal contributes, weighted by 1 - d_i
        gate_weights = active_mask * (1.0 - done_probs_t)  # (S, B)
    elif gating_mode == 'goal_future':
        # All subtasks contribute, weighted by 1 - d_i
        gate_weights = 1.0 - done_probs_t  # (S, B)
        gate_weights = gate_weights.at[-1, :].set(1.0)
    elif gating_mode == 'grandparent_myopic':
        # Only active goal contributes, weighted by 1 - d_{i-1}
        gate_weights = active_mask * (1.0 - parent_done)  # (S, B)
        gate_weights = gate_weights.at[-1, :].set(1.0)
    elif gating_mode == 'grandparent_future':
        # All subtasks contribute, weighted by 1 - d_{i-1}
        gate_weights = 1.0 - parent_done  # (S, B)
        gate_weights = gate_weights.at[-1, :].set(1.0)
    else:
        assert False, (
            f"Unknown advantage_gating_mode: {gating_mode}. Must be 'goal_myopic', "
            "'goal_future', 'grandparent_myopic', or 'grandparent_future'."
        )

    return gate_weights


def compute_qipm_td_target(
    subtask_status: jnp.ndarray,
    next_q: jnp.ndarray,
    masks: jnp.ndarray,
    discount,
    horizon_length: int,
    no_of_subtasks: int,
    alpha: float = 0.1,
    td_loss_mode: str = 'gated',
) -> jnp.ndarray:
    """Compute per-subtask TD targets for Q-IPM.

    Supports two modes controlled by ``td_loss_mode``:

    **gated** (default) – per-subtask done gating:
        For each subtask:
        - If done: reward = +5, next value = V_subgoal (bootstrap)
        - If not done: reward = -1, next value = Q_next
        TD target is capped at V_subgoal.

    **simple** – standard TD backup (used with goal conditioning):
        target_q = rewards + discount^horizon * masks * next_q
        No done-gating or V_subgoal capping.

    Args:
        subtask_status: (B, horizon_length, S) done flags per subtask per timestep.
        next_q: (S, B) aggregated next Q-values per subtask.
        masks: (B,) or (B, horizon_length) termination masks.
        discount: Discount factor — scalar float (same for all subtasks) or
                  array of shape (S,) for per-subtask discounts.
        horizon_length: Horizon length for action chunking.
        no_of_subtasks: Number of subtasks.
        alpha: Absorbing state value coefficient.
        td_loss_mode: "gated" or "simple".

    Returns:
        TD targets of shape (S, B).
    """
    if masks.ndim > 1:
        masks = masks[..., -1]

    # Per-subtask discount factor: scalar broadcasts naturally,
    # array (S,) is reshaped to (S, 1) for (S, B) broadcasting.
    disc_factor = jnp.asarray(discount) ** horizon_length
    if disc_factor.ndim > 0:
        disc_factor = disc_factor[:, None]  # (S, 1)

    # (S, B) done mask from last timestep
    subgoal_done_mask = subtask_status[..., -1, :].T  # (S, B)

    if td_loss_mode == 'simple':
        rewards = jnp.where(subgoal_done_mask, 5.0, -1.0)
        target_q = rewards + (discount ** horizon_length) * masks * next_q
        return target_q

    # --- gated (default) ---
    assert td_loss_mode == f"gated", "TD Loss Mode is an invalid option: {td_loss_mode}"
    
    V_subgoal = 15.0
    done_float = subgoal_done_mask.astype(jnp.float32)
    subgoal_not_done = 1.0 - done_float

    rewards = jnp.where(subgoal_done_mask, 5.0, -1.0)

    target_q = jnp.minimum(
        rewards + disc_factor * masks * (
            done_float * V_subgoal +
            subgoal_not_done * next_q
        ),
        V_subgoal
    )

    # Terminal fix: when episode ends and subtask is done, target = V_subgoal
    # (mask=0 zeroes out bootstrap, but done subtasks should be at V_subgoal)
    terminal = (1.0 - masks.astype(jnp.float32))
    target_q = jnp.where(
        (terminal > 0) & (done_float > 0),
        V_subgoal,
        target_q
    )

    return target_q


def compute_done_predictor_loss(
    done_logits: jnp.ndarray,
    done_probs: jnp.ndarray,
    subtask_status: jnp.ndarray,
    no_of_subtasks: int,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute BCE loss for done prediction.

    Args:
        done_logits: (B, S) raw logits from GateNet.
        done_probs: (B, S) sigmoid probabilities from GateNet.
        subtask_status: (B, horizon_length, S) ground truth done flags.
        no_of_subtasks: Number of subtasks.

    Returns:
        Tuple of (loss, info_dict).
    """
    true_done = subtask_status[:, -1, :]  # (B, S)

    done_loss = optax.sigmoid_binary_cross_entropy(
        done_logits, true_done.astype(jnp.float32)
    )
    done_loss = done_loss.mean()

    done_preds = (done_probs > 0.5).astype(jnp.float32)
    accuracy = (done_preds == true_done).mean()

    per_subtask_acc = {
        f'done_acc_subtask_{i}': (done_preds[:, i] == true_done[:, i]).mean()
        for i in range(no_of_subtasks)
    }

    info = {
        'done_loss': done_loss,
        'done_accuracy': accuracy,
        'done_prob_mean': done_probs.mean(),
        **per_subtask_acc,
    }

    return done_loss, info


def compute_per_subtask_mc_returns(
    subtask_status_history: list,
    decision_indices: list,
    no_of_subtasks: int,
    discount = 0.99,
) -> dict:
    """Compute MC returns for each subtask Q-function.

    For subtask i:
      reward = 0 if subtask_status[i] == 1 (completed)
      reward = -1/no_of_subtasks if subtask_status[i] == 0 (pending)

    Uses only the last timestep of each action chunk for discounting.

    Args:
        subtask_status_history: List of lists, each [done0, done1, ..., doneS].
        decision_indices: Timesteps where decisions were made.
        no_of_subtasks: Number of subtasks.
        discount: Scalar float (same for all subtasks) or list/array of
                  per-subtask discount factors of length no_of_subtasks.

    Returns:
        Dict mapping subtask index -> list of MC returns at decision points.
    """
    T = len(subtask_status_history)
    if T == 0:
        return {i: [] for i in range(no_of_subtasks)}

    # Resolve per-subtask discounts
    if isinstance(discount, (int, float)):
        discounts = [float(discount)] * no_of_subtasks
    else:
        discounts = [float(d) for d in discount]

    status_array = np.array(subtask_status_history)
    sorted_decision_indices = sorted(decision_indices)

    # Extract last timestep of each action chunk
    chunk_last_timesteps = []
    for i in range(len(sorted_decision_indices)):
        chunk_start = sorted_decision_indices[i]
        if i + 1 < len(sorted_decision_indices):
            chunk_end = sorted_decision_indices[i + 1]
        else:
            chunk_end = T
        if chunk_end > chunk_start:
            chunk_last_timesteps.append(chunk_end - 1)

    chunk_last_status = (
        status_array[chunk_last_timesteps, :]
        if chunk_last_timesteps
        else np.zeros((0, no_of_subtasks))
    )

    subtask_mc = {}
    for subtask_i in range(no_of_subtasks):
        chunk_rewards = np.where(
            chunk_last_status[:, subtask_i] == 1,
            0.0,
            -1.0 / no_of_subtasks,
        )
        num_chunks = len(chunk_rewards)
        if num_chunks == 0:
            subtask_mc[subtask_i] = []
            continue

        subtask_discount = discounts[subtask_i]
        chunk_returns = []
        for chunk_idx in range(num_chunks):
            future_rewards = chunk_rewards[chunk_idx:]
            disc_pows = subtask_discount ** np.arange(len(future_rewards))
            return_val = np.sum(future_rewards * disc_pows)
            chunk_returns.append(float(return_val))

        decision_to_chunk = {
            dec_idx: chunk_idx
            for chunk_idx, dec_idx in enumerate(sorted_decision_indices)
        }
        subtask_mc[subtask_i] = [
            chunk_returns[decision_to_chunk[t]] for t in decision_indices
        ]

    return subtask_mc


def _aggregate_safety_min_max(q_safety: jnp.ndarray) -> jnp.ndarray:
    """``safety_aggregation_strategy='min_max'``: reduce the safety-head slice
    to one scalar per element, replacing ``sum_k agg_m Q_km``.

    Lives here rather than in agents/safe_qipm.py (its 2026-08-13 home) since
    2026-08-14 (docs/changes/2026-08-14-fql-minmax-aggregation/): the FQL
    one-step sites need it, and agents/modules/fql_helper.py cannot import
    from agents/safe_qipm.py — safe_qipm imports fql_helper at module scope,
    so the reverse import is a hard cycle. Five call sites: safe_qipm's
    ``_compute_advantages`` (aggregated path), ``sample_actions_BON`` and
    ``get_q_values`` — which import the name from here and re-export it — plus
    fql_helper's ``one_step_distillation_loss`` and ``sample_one_step_BON``,
    which import it lazily at the use site.

    Two reductions, in this order (docs/changes/2026-08-13-safety-aggregation-
    strategy/ — Decisions 3-5):

      1. Over the ENSEMBLE (axis 1), a MAX restricted to members with
         ``Q <= 0``. The true cost-Q is <= 0 identically (rewards are
         ``-1 * unsafe steps``, see ``_compute_safety_td_target``), so a
         positive member is a mathematically invalid estimate and is dropped
         rather than averaged in. Among the valid members the max is the
         LEAST pessimistic one — the tightest bound the ensemble still agrees
         is admissible. A head with NO valid member contributes exactly 0.0,
         the cost-Q upper bound (Decision 4).
      2. Over the HEADS (axis 0), a MIN: the worst-violated constraint alone
         sets the safety pressure, so ``safety_coeff`` no longer scales with
         K (the summed form ran ~2.4x the pressure at K=4 vs K=1). At K=1
         (``safety_head_mode: single``) the cross-head min is trivial and the
         strategy degenerates to the masked ensemble max — still a real
         change, since it replaces the ``q_agg`` reduction (``mean`` in the
         shipped configs).

    Member-set contract (Decision 9): the caller passes the SAME member set
    the task-side reduction used — the subsampled slice under
    ``q_agg='subsample'`` / ``subsample_bon``, the full ensemble otherwise —
    so the two terms of the scalarization never disagree about which critics
    they trust. This function never subsamples.

    Gradient safety: the ``-inf`` sentinel is a CONSTANT, and ``jnp.where``'s
    VJP routes cotangents by selection (``select(cond, g, 0)``) rather than by
    multiplication, so no cotangent ever touches it and no ``0 * inf`` arises.
    An all-invalid head takes the ``0.0`` branch and therefore propagates a
    ZERO cotangent to every one of its members (pinned by a ``jax.grad`` test
    in tests/test_safe_qipm.py). Valid heads get the usual max/min one-hot.

    WARNING: NaN fails ``<= 0``, so a NaN member is masked out as invalid —
    where ``simple_summation`` would propagate the NaN into the loss and kill
    the run visibly, ``min_max`` silently ignores that member (and an
    all-NaN head reads as 0.0, i.e. "safe"). A genuine ``-inf`` member
    collides with the masking sentinel the same way: a head whose only valid
    members are ``-inf`` reads as 0.0 ("safe") instead of propagating the
    divergence. Nothing checks the critic for finiteness, and no runtime
    value assert is possible under jit/vmap.

    Args:
        q_safety: safety-head Q-values with K heads on axis 0 and the ensemble
            members the task term used on axis 1 — (K, M, B) or (K, 2, B) at
            the advantage, telemetry and FQL distillation sites, (K, M, B, N)
            or (K, 2, B, N) at the two best-of-N sites (2 = the subsampled
            member pair).

    Returns:
        (B,) or (B, N) — the scalar safety term, before ``safety_coeff``.
    """
    chex.assert_rank(q_safety, {3, 4})
    valid = q_safety <= 0.0
    per_head = jnp.where(valid, q_safety, -jnp.inf).max(axis=1)
    # -inf survives only where a head had no valid member -> 0.0 (Decision 4);
    # it is a masking sentinel and must never reach the returned value.
    per_head = jnp.where(jnp.isneginf(per_head), 0.0, per_head)
    return per_head.min(axis=0)
