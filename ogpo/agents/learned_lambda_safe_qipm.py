"""OGPO Q-IPM with always-active safety heads priced by a LEARNED Lambda matrix.

A copy-fork of ogpo/agents/safe_qipm.py for the ``learned_lambda_safe_qipm``
algorithm, following the same fork policy as safe_qipm_hjb.py: the two agents
are compared head-to-head on the same task and seeds, so they must not drift
through a shared base class, and safe_qipm.py stays the untouched default.

THE DELTAS vs safe_qipm (everything else is deliberately identical):

  1. ``safety_aggregation_strategy`` defaults to ``'learned_lambda'`` here.
     The fixed scalar ``safety_coeff`` is replaced by a learned, non-negative
     (S_task x K) matrix of Lagrange multipliers, and the safety-head
     aggregation is removed entirely -- the K heads are consumed raw:

         Q_total  = sum_i w_i(s) Q^task_i  +  sum_k lambda_k(s) Q^safety_k
         lambda(s) = Lambda^T phi(s),   phi(s) = w(s) / sum(w(s))
         Lambda    = softplus(Theta)

     Rows index task subgoals, columns index safety constraints, so the price
     of each constraint depends on which task phase the policy is in.

  2. Lambda is trained by PROJECTED dual ascent on the CMDP Lagrangian, not
     tuned. After each dual step Theta is clipped into
     [softplus^-1(lambda_floor), softplus^-1(lambda_max)] -- see
     ``_maybe_update_lambda`` for why this must be a parameter-space
     projection and NOT ``SoftplusMatrixParam(max_value=...)``.

  3. New config keys: ``lambda_lr``, ``lambda_init``, ``lambda_max``,
     ``lambda_floor``, ``lambda_warmup_steps``, ``safety_budget`` /
     ``safety_budget_rate``.

Everything below this line is inherited from safe_qipm's docstring and still
describes the critic, the gating and the safety heads accurately.

---

A near-fork of ogpo/agents/qipm.py for the ``safe_qipm`` algorithm. The
per-subtask critic is widened from S_task heads to
S_task + no_of_safety_subtasks heads on the same shared backbone:

  - Heads [0, S_task): task subgoals — standard QIPM TD (+5 done / -1 pending,
    capped at V_subgoal), gate-weighted in the actor objective.
  - Heads [S_task, S): safety subgoals — CMDP cost critics trained with
    ``_compute_safety_td_target`` (reward = -1 * unsafe-steps-in-chunk, plain
    Bellman backup, zero upper cap, no lower cap, no absorbing state,
    terminals SELECTED rather than multiplied by a 0/1 mask), ALWAYS
    ACTIVE in the actor objective: never gated by task progress.

Scalarization everywhere (advantages, BoN, Q logging). The task term is
always ``sum_task gate_s * Q_s``. Two orthogonal config keys control the
safety term, and they compose:

(1) ``safety_aggregation_strategy`` — how the ENSEMBLE members of the safety
    heads are reduced (docs/changes/2026-08-13-safety-aggregation-strategy/):

  - "simple_summation" (default; bit-identical to every run before
    2026-08-13) — reduce the ensemble with ``q_agg`` exactly like the task
    heads, then SUM the K per-head values:

        Q_total = sum_task gate_s * Q_s + safety_coeff * sum_safety Q_k

  - "min_max" — per safety head, the MAX over the ensemble members whose
    value is <= 0 (a positive cost-Q is an invalid estimate and is ignored; a
    head with no valid member contributes exactly 0.0, the cost-Q upper
    bound), then the MIN across heads, so the worst-violated constraint alone
    sets the pressure and ``safety_coeff`` stops scaling with K:

        Q_total = sum_task gate_s * Q_s
                  + safety_coeff * min_k ( max_{m : Q_km <= 0} Q_km )

    The member set is the one the TASK-side reduction used (subsampled under
    ``q_agg='subsample'`` / ``subsample_bon``). Implemented on the five
    REACHABLE scalarization sites — ``_compute_advantages``'s plain-ensemble
    aggregated path, ``sample_actions_BON`` and ``get_q_values`` here, plus
    the FQL one-step distillation Q-term and one-step BoN in
    agents/modules/fql_helper.py (2026-08-14,
    docs/changes/2026-08-14-fql-minmax-aggregation/); the MIP-Q and
    full-ensemble (conservative/penultimate) copies inside
    ``_compute_advantages`` stay summation-shaped because ``create()`` asserts
    those modes off. ``get_per_subtask_q_values`` is per-head telemetry and is
    strategy-independent by design.

(2) ``safety_agg`` — how the K per-head values (already ensemble-reduced) fuse
    into one scalar. Applies whenever ``safety_aggregation_strategy`` is
    "simple_summation"; "min_max" does its own head reduction and ignores it,
    which ``create()`` asserts.

    Q_total = sum_task gate_s * Q_s + safety_coeff * agg_safety(Q_k)

where agg_safety is chosen by ``safety_agg`` (see ``_aggregate_safety_heads``):
  - "sum" (default): plain sum_safety Q_k, as before.
  - "relu": sum_safety ReLU(-Q_k), negated — each head's contribution is
    clamped to a non-negative cost so a head that mispredicts a positive Q
    (the live prediction is unclipped, unlike the <=0 TD target) cannot
    offset another head's real negative cost.
  - "biased_relu": same as "relu" but with a per-call margin subtracted
    inside the ReLU before clipping: sum_safety ReLU(-Q_k - margin),
    negated, where margin = mean(Q_k) over all heads in the call (no
    filtering by unsafe_count or sign — a first-pass simplification, see
    docs/changes or session notes for the filtered/EMA refinements
    considered but not yet implemented). This widens the zeroed-out band
    from Q_k >= 0 to Q_k >= -margin, forgiving small negative predictions
    as noise while still fully zeroing mispredicted-positive heads exactly
    as "relu" does.
  - "min_relu"/"max_relu": reduce the K heads to a SINGLE Q via
    ``_select_safety_q`` and apply one biased negative ReLU:
    -ReLU(-z - margin) == min(0, z + margin). "min_relu" takes the plain
    min over heads (worst constraint; fires if ANY head is past the
    margin). "max_relu" takes the max over the non-positive heads, falling
    back to the min over the positives when every head has mispredicted
    (fires only if ALL heads are past the margin). Because the result is a
    single head's value its magnitude does not scale with K, and it is
    exactly 0 whenever z >= -margin. `margin` is agent state
    (``safety_margin``), estimated per critic batch as the (1-q)-quantile
    of z over the transitions that are NOT deteriorating
    (z(s',a') >= z(s,a)), EMA-smoothed and clipped to [0, cap]. See
    ``safety_margin_mode`` for the alternatives.

The env wrapper appends the per-step safety costs as trailing columns of
``subtask_status`` (see RobomimicLowdimWrapper.configure_safety_status in
envs/robomimic_utils.py), so ``batch['subtask_status']`` is
(B, H, S_task + no_of_safety_subtasks). Two sources, chosen by the config key
``safety_head_mode`` — the math below is identical either way, only K changes:

  - "single": one column from ``info['unsafe_step']``, the aggregate unsafe bit.
  - "multi": K columns from ``info['safety_violations']``, one per safety
    CONSTRAINT (joint_limits / velocity / collision / workspace / keepout),
    giving one always-active cost head per constraint. See
    configs/algos/safe_qipm_multihead.yaml.

NOTE: in "multi" an all-zero safety row DOES mean the step was safe.
``info['unsafe_step']`` is exactly ``any(info['safety_violations'])`` — both are
decided from the REQUEST ``q_des`` before the safety filter runs, so no cost mass
can escape the heads. This was one-way before 2026-08-04, when violations were
latched from the FILTERED command behind a magnitude threshold; the
``env/episode_unattributed_unsafe_steps`` metric that existed for that case has
been removed.

Because detection is demand-side, the rows describe what the policy ASKED for and
are identical whether or not the shield prevented it — the property that makes the
four experimental conditions comparable.

``no_of_subtasks`` keeps meaning the
TASK count everywhere; the total head count is derived at the seams, never
stored. v1 restrictions (no MIP-Q / separate critics / goal conditioning /
CalQL-with-safety) are asserted fail-fast in ``create()``. FQL one-step mode
is supported since 2026-08-11 (docs/changes/2026-08-10-safe-qipm-fql-mode/) —
under BOTH aggregation strategies since 2026-08-14
(docs/changes/2026-08-14-fql-minmax-aggregation/) — and the one-step net is
built at the sliced actor width. Its two scalarizations live in fql_helper.py
and do not both match this file's shapes:

  - the distillation Q-term is the gate-weighted sum of the ``q_agg``-reduced
    task heads + ``safety_coeff`` * the strategy's safety term over that same
    member set, i.e. the advantage-site formula;
  - one-step BoN sums the heads BEFORE the member reduction, so
    "simple_summation" scores ``min_m ( sum_task Q_s + coeff * sum_K Q_k )``
    and "min_max" scores ``min_m ( sum_task Q_s )
    + coeff * min_k max_{m : Q_km <= 0} Q_km``, with the member pair drawn per
    (b, n) candidate cell rather than once globally as in
    ``sample_actions_BON`` here.

This file is intentionally self-contained: no inheritance from OGPOQIPMAgent
to avoid Flax PyTreeNode subclassing pitfalls.
"""
import copy
from typing import Any, Optional, Tuple, List
from functools import partial
import numpy as np

import chex
import flax

import jax
import jax.numpy as jnp
import optax
import pickle

from ogpo.networks.encoders import encoder_modules, vit_encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import (
    ActorVectorField, Value, ActorVectorFieldTF, ValueTF,
    ActorVectorFieldSimBa, ValueSimBa, NoiseInjectionNetwork,
    SoftplusMatrixParam,
)
from ogpo.networks.goal_net import GoalConditionedNetwork
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets, get_shortcut_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    get_mip_targets,
    apply_chunking_mask,
    compute_flow_bc_loss,
    compute_mip_bc_loss,
    compute_flow_bc_loss_with_denoiser,
)
from ogpo.agents.modules.maintenance import (
    create_lr_schedule,
    compute_param_norm,
    compute_param_diff_norm,
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
    create_optimizer,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    compute_td_target,
    compute_td_loss,
    compute_calql_regularizer,
    get_q_values as _get_q_values,
    get_mip_q_targets,
    sample_mip_q_values,
    compute_mip_q_predictions,
    sample_mip_q_ensemble_values,
    compute_mip_q_ensemble_predictions,
)
from ogpo.agents.modules.qipm_helper import (
    compute_qipm_td_target,
    compute_gate_weights,
    # Module-private by name but imported deliberately (and re-exported by
    # this module: tests import it from ogpo.agents.safe_qipm). It lived here
    # until 2026-08-14, when fql_helper needed it too and could not import
    # from this file — this module imports fql_helper below, so the reverse
    # direction is a cycle (docs/changes/2026-08-14-fql-minmax-aggregation/).
    _aggregate_safety_min_max,
)
from ogpo.agents.modules.fql_helper import (
    create_one_step_network,
    one_step_distillation_loss,
    sample_one_step,
    sample_one_step_BON,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
    sample_mip_actions_ode,
    sample_flow_actions_ode_with_correction,
    sample_flow_actions_sde,
    sample_mip_actions_sde,
    compute_flow_log_prob,
    compute_mip_log_prob,
    compute_ppo_loss,
    compute_ogpo_advantages,
    compute_fpo_cfm_ratios,
    compute_fpo_ppo_loss,
)


def _split_keys(rng: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Split a PRNGKey into one key and a list of n subkeys."""
    keys = jax.random.split(rng, n + 1)
    return keys[0], list(keys[1:])


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


def _masked_quantile(values, mask, q, fallback, min_count=32):
    """jit-safe quantile of `values` over the entries where `mask` is True.

    Sorts with masked-out entries pushed to +inf so they land past every real
    value, then indexes the q-quantile position within the first `n` slots.
    Returns `fallback` when fewer than `min_count` entries survive the mask,
    so a batch with an empty (or near-empty) reference population holds the
    previous estimate instead of spiking.

    Args:
        values: (B,) float array.
        mask: (B,) boolean array selecting the reference population.
        q: quantile in [0, 1].
        fallback: scalar returned when the population is too small.
        min_count: minimum population size to trust the batch.
    """
    mask = mask.astype(bool)
    n = jnp.sum(mask)
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf))
    idx = jnp.clip(jnp.floor(q * jnp.maximum(n - 1, 0)).astype(jnp.int32),
                   0, values.shape[0] - 1)
    picked = ordered[idx]
    return jnp.where(n >= min_count, picked, jnp.asarray(fallback, values.dtype))


def _select_safety_q(q_safety_per_head, safety_agg):
    """Reduce the head axis (0) to ONE Q per (state, action) — the min/max modes.

      "min_relu": plain min over heads — the most-negative (worst) constraint.

      "max_relu": plain max over heads — the least-negative (most permissive)
             constraint, so the term fires only when EVERY head agrees the
             state is unsafe (unanimity).

    Both modes commute with the downstream zero-clip in
    _aggregate_safety_heads, which is why no positivity guard is needed here:

        max_k min(0, Q_k)  ==  min(0, max_k Q_k)  ==  -relu(-max_k Q_k)

    i.e. clipping each head at zero first and then selecting is identical to
    selecting first and then clipping. A head that mispredicts positive
    (q_subtask_*_max has been logged at +141; see the zero-upper-cap note on
    _compute_safety_td_target) therefore drives max_relu to exactly 0 — the
    safety term switches off — which is the intended semantics: under
    max_relu a single head reporting "safe" is enough. Pair this mode with
    q_agg="min" so a positive Q_k requires ALL ensemble members to
    mispredict rather than just one.

    Args:
        q_safety_per_head: (K, ...) — K safety heads on axis 0.
        safety_agg: "min_relu" or "max_relu".
    Returns:
        (...) — the trailing shape of the input, head axis removed.
    """
    if safety_agg == 'min_relu':
        return jnp.min(q_safety_per_head, axis=0)
    return jnp.max(q_safety_per_head, axis=0)


def _aggregate_safety_heads(q_safety_per_head, safety_agg, margin=0.0):
    """Fuse per-head safety Q-values (axis 0 = heads) into one signed contribution.

    Returns a value X such that `q_task + safety_coeff * X` is the correct
    total for both modes, so every call site keeps the same fusion line
    regardless of `safety_agg`.

      "sum"  (default, legacy behavior): plain sum of the K heads. A head that
             mispredicts a positive Q (see the zero-upper-cap note on
             _compute_safety_td_target — the TARGET is <= 0 but nothing caps
             the live PREDICTION) can silently cancel another head's real
             negative cost.

      "relu": each head is converted to a non-negative cost via
             ReLU(-Q_k) = max(-Q_k, 0) before summing, then negated back to a
             signed contribution. A head can never contribute a bonus — only
             ever >= 0 cost — so a mispredicted-positive head contributes
             exactly 0 instead of offsetting a genuinely dangerous head.

      "min_relu"/"max_relu": reduce the K heads to a SINGLE Q via
             _select_safety_q, then apply the biased negative ReLU
             -ReLU(-z - margin) == min(0, z + margin). The result is one
             head's value, so its magnitude does not scale with K and it is
             exactly 0 whenever z >= -margin. `margin` is supplied by the
             caller (agent state, see safety_margin) rather than derived from
             this call's slice.

      "biased_relu": same shape as "relu", but a margin is subtracted from
             each Q_k BEFORE the ReLU: ReLU(-Q_k - margin). margin =
             mean(Q_k) over all heads in this call, computed fresh each
             call (no persistent state, no filtering by unsafe_count or
             sign — a first-pass simplification). Putting the margin inside
             the ReLU (not subtracted after) is what actually moves the
             zeroed-out band from Q_k >= 0 to Q_k >= -margin, rather than
             just shifting the total by a constant that would cancel out in
             every action-comparison use of this aggregation.

    Args:
        q_safety_per_head: (K, ...) — K safety heads on axis 0, any trailing
            shape (batch, ensemble, samples, ...).
        safety_agg: "sum", "relu", "biased_relu", "min_relu", or "max_relu".
        margin: dead-zone width for the min/max modes; >= 0, ignored by the
            others. Comes from agent state (`safety_margin`), estimated in
            get_td_loss. Larger margin = MORE forgiving.
    """
    if safety_agg in ('min_relu', 'max_relu'):
        q_sel = _select_safety_q(q_safety_per_head, safety_agg)
        return -jax.nn.relu(-q_sel - margin)
    if safety_agg == 'biased_relu':
        m = jnp.mean(q_safety_per_head)
        return -jnp.sum(jax.nn.relu(-q_safety_per_head - m), axis=0)
    if safety_agg == 'relu':
        return -jnp.sum(jax.nn.relu(-q_safety_per_head), axis=0)
    return q_safety_per_head.sum(axis=0)


def _clip_cost_q_members(q_safety_members):
    """Clip per-member cost-Q estimates to their valid range, Q <= 0.

    The safety heads are CMDP cost critics: the reward is -(unsafe steps in the
    chunk), so the true Q is a discounted sum of non-positive terms and is <= 0
    IDENTICALLY. A positive prediction is not a value, it is an estimation error
    -- and it is not rare. Measured over every logged step of 19giclgc and
    6u1v068o, 100% of logs had at least one member predicting a positive cost-Q,
    reaching +12.4.

    Averaging those in biases the estimate TOWARD ZERO, i.e. makes the state look
    SAFER than the critic actually believes -- the dangerous direction. Clipping
    each member first removes each invalid estimate on its own instead of letting
    it cancel a legitimately negative one.

    ORDER MATTERS, and this is the conservative order. min(.,0) is nonlinear, so
    by Jensen  mean_m[min(Q_m, 0)] <= min(mean_m[Q_m], 0)  -- clip-then-mean is
    always the more pessimistic of the two. (Under a MIN reduction the two
    coincide, since min is monotone; only the mean is affected.)

    This is the same premise two other parts of the codebase already act on: the
    safety TD target is capped with jnp.minimum(target_q, 0.0), and
    _aggregate_safety_min_max masks positive members as invalid. The
    learned_lambda path was the one place that consumed them raw.

    Args:
        q_safety_members: (K, M, B) or (K, M, B, N) -- the safety slice, BEFORE
            the ensemble reduction.
    Returns:
        Same shape, with positive entries set to 0.
    """
    return jnp.minimum(q_safety_members, 0.0)


def _normalized_gate_weights(gate_weights):
    """Gate weights renormalized to a simplex over the ACTIVE task subgoals.

    ``compute_gate_weights`` returns unnormalized 0/1 indicators whose column
    sum varies between 1 and S_task with task progress. The task term wants
    that (it is a SUM of the active heads' Q). The learned-lambda safety term
    does not: if phi did not sum to 1, the effective multiplier would scale
    with how many subtasks happen to be open, which (a) makes lambda's
    magnitude uninterpretable and (b) breaks the reduction to the fixed-coeff
    baseline at initialization.

    Every gating mode force-sets the last row to 1.0, so the column sum is >= 1
    and the division is safe; the maximum() is a belt-and-braces guard, not a
    fallback that can silently engage under normal operation.

    Args:
        gate_weights: (S_task, B).
    Returns:
        (S_task, B), each column summing to 1.
    """
    return gate_weights / jnp.maximum(gate_weights.sum(axis=0, keepdims=True), 1e-8)


def _effective_lambda(lam, gate_weights):
    """The state-dependent multiplier vector lambda(s) = Lambda^T phi(s).

    Args:
        lam: (S_task, K) non-negative learned matrix. Rows index task subgoals,
            columns index safety constraints.
        gate_weights: (S_task, B) raw gate weights, or None at the call sites
            that carry no task-phase information.
    Returns:
        (K, B) when gate_weights is given; (K,) when it is None.

    The None case covers ``sample_actions_BON`` and ``get_q_values``, which sum
    the TASK heads unweighted because no subtask_status is available there.
    The matching convention on the safety side is the UNIFORM simplex
    phi = 1/S_task, i.e. the column MEAN of Lambda. The column sum would be the
    wrong choice: it would make the safety term S_task times larger at those
    sites than at the gated ones, so the action ranking used at rollout time
    would not match the objective the actor is trained on.
    """
    if gate_weights is None:
        return lam.mean(axis=0)
    return jnp.einsum('ib,ik->kb', _normalized_gate_weights(gate_weights), lam)


def _zero_dual(s_task, s_safety):
    """Placeholder dual signal for the scalarization branches that cannot be
    reached under 'learned_lambda'.

    ``compute_q_for_group`` must return the same pytree structure from every
    branch for ``jax.vmap`` to stack it, so the non-learned-lambda branches
    return a zero (S_task, K). This is NOT a fallback that could silently
    engage: ``create()`` asserts that 'learned_lambda' is incompatible with
    use_mip_q and with adv_strategy != 'vanilla', which are exactly the
    branches that return this, so a zero dual can never reach the dual step.
    """
    return jnp.zeros((s_task, s_safety), dtype=jnp.float32)


def _dual_signal(q_safety_per_head, gate_weights, budget):
    """Per-entry constraint slack: the dual objective's gradient wrt Lambda.

        g_ik = mean_b[ phi_i(s_b) * ( Q_k(s_b, a_b) + d_k ) ]

    The safety heads are CMDP cost critics, so Q_k <= 0 and the constraint
    "constraint k costs at most d_k" reads Q_k >= -d_k. Hence:

        g_ik < 0  <=>  Q_k < -d_k  <=>  constraint VIOLATED in task phase i
                   =>  the dual step raises Lambda_ik
        g_ik > 0  <=>  slack in the budget
                   =>  Lambda_ik decays back toward 0

    Q is stop-gradient'd: the dual MEASURES the critic and must not train it,
    or the critic would learn to report safety rather than predict it.

    Args:
        q_safety_per_head: (K, B) ensemble-reduced safety Q at the current
            policy's actions.
        gate_weights: (S_task, B) raw gate weights.
        budget: (K,) non-negative cost budgets d_k.
    Returns:
        (S_task, K)
    """
    phi = _normalized_gate_weights(gate_weights)                        # (S_task, B)
    slack = jax.lax.stop_gradient(q_safety_per_head) + budget[:, None]  # (K, B)
    return jnp.einsum('ib,kb->ik', phi, slack) / phi.shape[1]


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
    away — asserted off in safe_qipm v1.
    """
    if actor_obs_dim is None:
        return observations
    return observations[..., :actor_obs_dim]


class OGPOLearnedLambdaSafeQIPMAgentSharedBase(flax.struct.PyTreeNode):
    """OGPO Q-IPM with always-active safety Q-heads on a shared-backbone critic."""
    rng: Any
    actor_network: TrainState
    critic_network: TrainState  # Shared backbone with num_subtasks -- output shape (S, M, B)
    config: Any = nonpytree_field()
    one_step_network: Optional[TrainState] = None
    # Margin for safety_agg in {min_relu, max_relu}: -mean(z) from the PREVIOUS
    # critic update, so contrib = min(0, z - z_prev_mean) fires on deterioration
    # (moving into worse-than-recent safety values) and is 0 when improving.
    # 0.0 cold start => the modes reduce exactly to plain -ReLU(-z).
    safety_margin: Any = 0.0
    # Learned Lagrange multipliers Lambda, (S_task, K). Present only under
    # safety_aggregation_strategy='learned_lambda'; None for every other
    # strategy, which is what keeps this arm bit-identical to the pre-existing
    # ones when it is off. It gets its OWN TrainState rather than an entry in
    # the actor's ModuleDict (the sac.py `temp` pattern) because the actor
    # optimizer runs a cosine schedule decaying to actor_end_value=2e-5 by
    # actor_decay_steps=50000, which would freeze the dual one tenth of the way
    # into a 500k-step run. A dual variable needs a flat step size.
    lambda_network: Optional[TrainState] = None

    # --- Vision encoding helpers (same as OGPOAgent) ---

    def _encode_for_actor(self, observations, images=None, params=None, use_target=False):
        """Encode observations for actor. Returns encoded obs (flat features).

        This is the single funnel for every actor input (BC loss incl. the
        success-buffer branch, PPO, sampling, BoN, one-step, target actor) —
        the actor-obs slice lives here so all paths get it. Critics never
        call this and keep the full observation.
        """
        observations = _slice_actor_obs(observations, self.config['actor_obs_dim'])
        if not self.config.encoder:
            return observations
        module = 'target_actor' if use_target else 'actor'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        return self.actor_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)

    def _encode_for_critic(self, observations, images=None, params=None, use_target=False):
        """Encode observations for critic. Returns encoded obs (flat features)."""
        if not self.config.encoder:
            return observations
        module = 'target_critic' if use_target else 'critic'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        return self.critic_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)

    # --- Log prob / sampling (identical to OGPOAgent) ---

    def compute_log_probs_and_entropy(
        self,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        params: flax.core.FrozenDict,
        get_entropy: bool = False,
        use_target: bool = False,
        is_encoded: bool = False,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute log-probabilities and optionally entropy using pg_helper functions.
        """
        actor_module_name = 'target_actor' if use_target else 'actor'
        noise_module_name = 'target_noise_net' if use_target else 'noise_net'

        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, params=params, use_target=use_target)

        # Create wrapper functions for the network calls
        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'mip':
            logprob, entropy_rate_est, info = compute_mip_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                mip_t_star=self.config.mip_t_star,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
            )
        else:
            logprob, entropy_rate_est, info = compute_flow_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                flow_steps=self.config.flow_steps,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
            )

        if get_entropy:
            return logprob, entropy_rate_est, info
        else:
            return logprob, None

    def compute_log_probs(self, observations: jnp.ndarray, chains: jnp.ndarray, params: flax.core.FrozenDict, use_target: bool = False,
                          is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """ Compute sum of log-probabilities over the denoising chain. """
        logprob, _ = self.compute_log_probs_and_entropy(observations, chains, params, get_entropy=False, use_target=use_target, is_encoded=is_encoded, images=images)
        return logprob

    def _sample_actions(self, obs: jnp.ndarray, rng: jnp.ndarray, use_target: bool, num_samples: int, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Sample actions from TARGET policy """
        sample_rngs = jax.random.split(rng, num_samples)
        vmapped_sample = jax.vmap(
            lambda obs_g, rng_g: self.sample_actions_with_noise(obs_g, rng=rng_g, use_target=use_target, is_encoded=True),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, use_target=use_target)
        actions, chains, old_lp, sigmas = vmapped_sample(obs_encoded, sample_rngs)

        return actions, chains, old_lp, sigmas

    def _sample_actions_ode(self, obs_encoded: jnp.ndarray, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample actions from target policy via ODE (no logprob, no chains). For FPO."""
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('target_actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        def sample_ode_single(rng_g):
            if self.config.error_correct_sde_to_ode and self.config.use_denoiser:
                return sample_flow_actions_ode_with_correction(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    constant_noise_std=self.config['constant_noise_std'],
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )
            else:
                return sample_flow_actions_ode(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )

        sample_rngs = jax.random.split(rng, num_samples)
        return jax.vmap(sample_ode_single)(sample_rngs)  # [G, batch, act_dim]

    def _compute_current_log_probs(self, obs: jnp.ndarray, chains: jnp.ndarray, grad_params: flax.core.FrozenDict, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """ Compute log probabilities under CURRENT policy for actions sampled from target policy """
        vmapped_logprob = jax.vmap(
            lambda obs_g, chains_g: self.compute_log_probs_and_entropy(
                obs_g, chains_g, grad_params, get_entropy=True, is_encoded=True
            ),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)
        lp, entropy_rate_est, info_vmapped = vmapped_logprob(obs_encoded, chains)

        return lp, entropy_rate_est, info_vmapped

    # --- QIPM-specific advantage computation ---

    def _compute_advantages(self, obs: jnp.ndarray, actions: jnp.ndarray, rng: jnp.ndarray, subtask_status: jnp.ndarray = None, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GRPO-style advantages with per-subtask Q-values and gold done gating."""
        critic_obs_enc = self._encode_for_critic(obs, images=images, use_target=True)

        # Gold dones: subtask_status (B, S) with 1=done, 0=pending
        assert subtask_status is not None, (
            "_compute_advantages requires gold subtask_status from batch. "
            "Ensure batch contains 'subtask_status' key."
        )
        S_task = self.config['no_of_subtasks']
        S_safety = self.config['no_of_safety_subtasks']
        safety_coeff = self.config['safety_coeff']
        safety_agg = self.config.get('safety_agg', 'sum')
        safety_strategy = self.config['safety_aggregation_strategy']
        learned_lambda = safety_strategy == 'learned_lambda'
        clip_cost_q = bool(self.config.get('safety_clip_cost_q', True))
        # Lambda enters Q_total DETACHED -- read from the committed params, not
        # from any grad_params. If the actor could differentiate through it, it
        # would drive Lambda to 0: Q_safety <= 0, so a smaller multiplier is
        # unconditionally a larger objective, and the constraint would never
        # bind. Only `lambda_loss` sees Lambda through a gradient. Same split as
        # sac.py's entropy temperature (agents/sac.py:200-221).
        lam = self._lambda_matrix() if learned_lambda else None
        lam_budget = self._safety_budget() if learned_lambda else None
        # Gates come from TASK columns only: compute_gate_weights force-sets its
        # last row to 1 and derives parent_done from adjacent rows, so safety
        # columns must never enter it. Safety heads are added un-gated (always
        # active) in the scalarizations below.
        gate_weights = compute_gate_weights(
            subtask_status_last=subtask_status[:, :S_task],
            gating_mode=self.config.get('advantage_gating_mode', 'grandparent_future'),
            dag=self.config.get('subtask_dag'),
        )
        done_weights = gate_weights

        adv_strategy = self.config.get('adv_strategy', 'vanilla')
        need_full_ensemble = adv_strategy in ('conservative', 'penultimate')

        if self.config.get('use_mip_q', False):
            # MIP-Q path: two-step sampling, output is (S, M, B) or (S, N, B)
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = critic_obs_enc.shape[0]
            S = S_task + self.config['no_of_safety_subtasks']

            if self.config.get('mip_q_ensemble', False):
                # Explicit ensemble: single shared noise for fair comparison
                rng, noise_rng = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_rng, (batch_size, S),
                    minval=-noise_scale, maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_ensemble_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_sample=noise_sample,
                        mip_t_star=mip_t_star,
                        is_encoded=bool(self.config.encoder)
                    )  # (S, M, B)

                    if need_full_ensemble:
                        q_task = (q_vals[:S_task] * done_weights[:, None, :]).sum(axis=0)  # (M, B)
                        q_safety = _aggregate_safety_heads(q_vals[S_task:], safety_agg, self.safety_margin)  # (M, B) — always active, never gated
                        return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)
                    else:
                        if self.config['q_agg'] == 'min':
                            q_ensemble_agg = q_vals.min(axis=1)
                        elif self.config['q_agg'] == 'subsample':
                            subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                            q_ensemble_agg = q_vals[:, subsample_idxs, :].min(axis=1)
                        else:
                            q_ensemble_agg = q_vals.mean(axis=1)
                        q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)
                        q_safety = _aggregate_safety_heads(q_ensemble_agg[S_task:], safety_agg, self.safety_margin)
                        return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)
            else:
                # Implicit ensemble: multiple noises
                rng, noise_rng = jax.random.split(rng)
                noise_samples = jax.random.uniform(
                    noise_rng, (num_ensemble_members, batch_size, S),
                    minval=-noise_scale, maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_samples=noise_samples,
                        mip_t_star=mip_t_star,
                        is_encoded=bool(self.config.encoder)
                    )  # (S, N, B)

                    if need_full_ensemble:
                        q_task = (q_vals[:S_task] * done_weights[:, None, :]).sum(axis=0)  # (N, B)
                        q_safety = _aggregate_safety_heads(q_vals[S_task:], safety_agg, self.safety_margin)  # (N, B) — always active, never gated
                        return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)
                    else:
                        if self.config['q_agg'] == 'min':
                            q_ensemble_agg = q_vals.min(axis=1)
                        elif self.config['q_agg'] == 'subsample':
                            subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                            q_ensemble_agg = q_vals[:, subsample_idxs, :].min(axis=1)
                        else:
                            q_ensemble_agg = q_vals.mean(axis=1)
                        q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)
                        q_safety = _aggregate_safety_heads(q_ensemble_agg[S_task:], safety_agg, self.safety_margin)
                        return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)
        else:
            # Original ensemble-Q path
            def compute_q_for_group(actions_g, rng_g):
                q_vals = self.critic_network.select('target_critic')(
                    critic_obs_enc, actions_g, is_encoded=bool(self.config.encoder))
                # q_vals shape: (S, M, B) for batched critic

                if need_full_ensemble:
                    q_task = (q_vals[:S_task] * done_weights[:, None, :]).sum(axis=0)  # (M, B)
                    q_safety = _aggregate_safety_heads(q_vals[S_task:], safety_agg, self.safety_margin)  # (M, B) — always active, never gated
                    return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)
                else:
                    # q_members is the member set the TASK reduction consumes;
                    # min_max reduces the safety heads over exactly that set
                    # (Decision 9), so 'subsample' subsamples both terms.
                    if self.config['q_agg'] == 'min':
                        q_members = q_vals  # (S, M, B)
                        q_ensemble_agg = q_members.min(axis=1)  # (S, B)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, self.config.num_qs)
                        q_members = q_vals[:, subsample_idxs, :]  # (S, 2, B)
                        q_ensemble_agg = q_members.min(axis=1)  # (S, B)
                    else:
                        q_members = q_vals  # (S, M, B)
                        q_ensemble_agg = q_members.mean(axis=1)  # (S, B)

                    q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)  # (B,)
                    if learned_lambda:
                        # No head aggregation at all: the K safety Q-values are
                        # consumed raw and mixed by the learned matrix.
                        #
                        # But they ARE clipped to the valid cost-Q range first,
                        # per ensemble member, BEFORE the reduction -- see
                        # _clip_cost_q_members for why the order matters. This
                        # feeds both the actor (through q_total) and the dual
                        # (through dual_g), so primal and dual see one
                        # consistent constraint function.
                        q_saf_members = q_members[S_task:]
                        if clip_cost_q:
                            q_saf_members = _clip_cost_q_members(q_saf_members)
                        if self.config['q_agg'] in ('min', 'subsample'):
                            q_saf_heads = q_saf_members.min(axis=1)              # (K, B)
                        else:
                            q_saf_heads = q_saf_members.mean(axis=1)             # (K, B)
                        lam_eff = _effective_lambda(lam, done_weights)           # (K, B)
                        saf = jnp.einsum('kb,kb->b', lam_eff, q_saf_heads)   # (B,)
                        q_total = q_task + saf
                        dual_g = _dual_signal(q_saf_heads, done_weights, lam_budget)
                        return q_total, dual_g, saf
                    if safety_strategy == 'min_max':
                        q_safety = _aggregate_safety_min_max(q_members[S_task:])  # (B,)
                    else:
                        q_safety = _aggregate_safety_heads(q_ensemble_agg[S_task:], safety_agg, self.safety_margin)  # (B,)
                    return (q_task + safety_coeff * q_safety, _zero_dual(S_task, S_safety),
                            safety_coeff * q_safety)

        subsample_rngs = jax.random.split(rng, actions.shape[0])
        vmapped_critic = jax.vmap(compute_q_for_group, in_axes=(0, 0))
        q_out, dual_g_per_group, saf_per_group = vmapped_critic(actions, subsample_rngs)
        # (G, S_task, K) -> (S_task, K). The G action groups are i.i.d. samples
        # from the same target policy at the same states, so the dual signal is
        # a plain mean over them -- the same population the advantage baseline
        # is taken over.
        dual_g = dual_g_per_group.mean(axis=0)

        if need_full_ensemble:
            # q_out: (G, M, B) — apply conservative/penultimate aggregation
            adv_per_q = q_out - q_out.mean(axis=0, keepdims=True)  # (G, M, B)
            if adv_strategy == 'conservative':
                # safe_max: min if all positive, max if all negative, 0 if disagreement
                q_min = adv_per_q.min(axis=1)
                q_max = adv_per_q.max(axis=1)
                adv = q_min * (q_min > 0) + q_max * (q_max < 0)  # (G, B)
            else:  # penultimate
                q_sorted = jnp.sort(adv_per_q, axis=1)
                q_min = q_sorted[:, 0, :]
                q_max = q_sorted[:, -1, :]
                q_min2 = q_sorted[:, 1, :]
                q_max2 = q_sorted[:, -2, :]
                adv = q_min2 * (q_min > 0) + q_max2 * (q_max < 0)  # (G, B)
            adv_pre = q_out.mean(axis=1)  # (G, B) for logging
        else:
            adv_pre = q_out  # (G, B)
            adv = adv_pre - adv_pre.mean(axis=0, keepdims=True)  # (G, B)

        if self.config.adv_clip_min is not None:
            adv = jnp.clip(adv, self.config.adv_clip_min)

        # --- Does the safety term actually influence action selection? --------
        # The advantage is q_total_g minus its mean over the G candidate action
        # groups, so ONLY variation ACROSS g can change which action the actor is
        # pushed toward. A safety term with a large mean but no across-group
        # spread shifts every candidate equally and cancels in the baseline
        # subtraction -- it is invisible to the policy no matter how big lambda
        # is. That is the failure this measures, and it is not visible in
        # lambda/violation_* or in the Q means.
        #
        # saf_frac  = std_g(safety) / (std_g(task) + std_g(safety))
        #             ~0   -> safety cannot move the ranking; lambda is inert
        #             ~0.5 -> the two terms have comparable say
        # rank_flip = fraction of states where including the safety term changes
        #             which candidate group has the highest Q. This is the
        #             behavioural version of the same question and needs no
        #             scale calibration.
        if saf_per_group.ndim == q_out.ndim:                 # (G, B) reachable path
            task_per_group = q_out - saf_per_group
            sd_saf = saf_per_group.std(axis=0)               # (B,)
            sd_task = task_per_group.std(axis=0)             # (B,)
            saf_frac = (sd_saf / jnp.maximum(sd_task + sd_saf, 1e-8)).mean()
            rank_flip = (q_out.argmax(axis=0)
                         != task_per_group.argmax(axis=0)).mean()
            adv_diag = {
                'adv_safety_std_frac': saf_frac,
                'adv_rank_flip_rate': rank_flip,
                'adv_safety_std': sd_saf.mean(),
                'adv_task_std': sd_task.mean(),
            }
        else:
            adv_diag = {}

        return adv, adv_pre, dual_g, adv_diag

    def _compute_bc_loss(self, rng, grad_params, batch, batch_success, success_flag, obs_encoded=None, batch_success_encoded=None, images=None, images_success=None):
        rng, key = jax.random.split(rng)

        def true_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch_success, grad_params, key, obs_encoded=batch_success_encoded, images=images_success)
            return bc_loss

        def false_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch, grad_params, key, obs_encoded=obs_encoded, images=images)
            return bc_loss

        bc_loss = jax.lax.cond(success_flag, true_branch, false_branch,
            (batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success),
        )
        return bc_loss

    def actor_loss(self, batch: dict, batch_success: dict, grad_params: flax.core.FrozenDict, success_flag: bool, rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """ Compute PPO surrogate loss + entropy + BC regularization with QIPM advantage calculation. """
        # FQL mode: skip PG loss when one-step policy handles RL (flow actor only gets BC)
        skip_pg_loss = not self.config.get('fql_train_main_policy', True)

        # Default values (used when PG is skipped in FQL mode)
        pg_loss = jnp.float32(0.0)
        ent_loss = jnp.float32(0.0)
        ent = jnp.float32(0.0)
        lp = jnp.float32(0.0)
        old_lp = jnp.float32(0.0)
        adv = jnp.zeros(1)
        adv_pre = jnp.zeros(1)
        # Shape must match the real dual signal so the FQL skip_pg_loss path
        # stays traceable; create() rejects learned_lambda with FQL mode, so
        # this zero can never reach the dual step.
        dual_g = _zero_dual(self.config['no_of_subtasks'], self.config['no_of_safety_subtasks'])
        adv_diag = {}
        sigmas = jnp.float32(0.0)
        actions = jnp.zeros(1)
        pg_stats = {}
        info_extra = {}
        G = self.config.group_num_samples

        if not skip_pg_loss:
            # Subsample observations if needed
            obs = batch['observations']
            images = batch.get('images')
            subtask_status = batch.get('subtask_status')  # (B, T, S) from sequence sampling
            if subtask_status is not None and subtask_status.ndim == 3:
                subtask_status = subtask_status[:, -1, :]  # last timestep -> (B, S)
            if self.config.ppo_batch_size < self.config.batch_size:
                rng, key = jax.random.split(rng)
                idxs = jax.random.choice(key, obs.shape[0], (self.config.ppo_batch_size,), replace=False)
                obs = obs[idxs]
                if images is not None:
                    images = images[idxs]
                if subtask_status is not None:
                    subtask_status = subtask_status[idxs]

            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)

            rng, sample_rng = jax.random.split(rng)

            if self.config.get('use_fpo', False):
                # ===== FPO PATH =====
                # Step 1: Sample actions via ODE from target policy (no chains, no logprob)
                actions = self._sample_actions_ode(obs_encoded, sample_rng, G)  # [G, batch, act_dim]

                # Step 2: Compute advantages (QIPM-specific, gold dones)
                rng, adv_rng = jax.random.split(rng)
                adv, adv_pre, dual_g, adv_diag = self._compute_advantages(obs, actions, adv_rng, subtask_status=subtask_status, images=images)

                # Step 3: Compute FPO CFM ratios (vmapped over G groups)
                def old_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('target_actor')(obs_a, x_tau, t, is_encoded=True)
                def new_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('actor')(obs_a, x_tau, t, params=grad_params, is_encoded=True)

                rng, ratio_rng = jax.random.split(rng)
                def compute_ratio_single(actions_g, rng_g):
                    return compute_fpo_cfm_ratios(
                        old_actor_fn=old_actor_fn, new_actor_fn=new_actor_fn,
                        observations=obs_encoded, actions=actions_g,
                        n_mc=self.config.fpo_n_mc, rng=rng_g,
                        per_sample=self.config.fpo_per_sample,
                        cfm_loss_clamp=self.config.fpo_cfm_loss_clamp,
                        cfm_diff_clamp=self.config.fpo_cfm_diff_clamp,
                    )
                ratio_rngs = jax.random.split(ratio_rng, G)
                ratios, fpo_stats = jax.vmap(compute_ratio_single)(actions, ratio_rngs)

                # Step 4: PPO loss with FPO ratios
                adv_flat = adv.reshape(-1)  # [G * batch]
                if self.config.fpo_per_sample:
                    n_mc = self.config.fpo_n_mc
                    ratios_reshaped = ratios.transpose(1, 0, 2).reshape(n_mc, -1)
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_reshaped, adv_flat, self.config.clip_epsilon,
                        per_sample=True, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )
                else:
                    ratios_flat = ratios.reshape(-1)  # [G*batch]
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_flat, adv_flat, self.config.clip_epsilon,
                        per_sample=False, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )

                # FPO has no tractable entropy
                ent = jnp.float32(0.0)
                ent_loss = jnp.float32(0.0)
                sigmas = jnp.float32(0.0)
                lp = jnp.float32(0.0)
                old_lp = jnp.float32(0.0)

                # Merge FPO-specific stats into pg_stats
                pg_stats.update({f'fpo_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in fpo_stats.items()})
                info_extra = {}

            else:
                # ===== ORIGINAL OGPO PATH (unchanged) =====
                actions, chains, old_lp, sigmas = self._sample_actions(obs, sample_rng, use_target=True, num_samples=G, obs_encoded=obs_encoded, images=images)

                rng, adv_rng = jax.random.split(rng)
                adv, adv_pre, dual_g, adv_diag = self._compute_advantages(obs, actions, adv_rng, subtask_status=subtask_status, images=images)
                lp, entropy_rate_est, info_vmapped = self._compute_current_log_probs(obs, chains, grad_params, obs_encoded=obs_encoded)

                lp = lp.reshape(-1)  # [G * batch]
                old_lp = old_lp.reshape(-1)  # [G * batch]
                adv = adv.reshape(-1)  # [G * batch]
                pg_loss, pg_stats = compute_ppo_loss(lp, old_lp, adv, self.config)

                ent = jnp.mean(entropy_rate_est)
                ent_loss = -ent
                info_extra = {f'info_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in info_vmapped.items()}

        # BC regularization (always runs, including FQL mode)
        bc_loss = 0.0
        if self.config.use_bc_regularization:
            rng, key = jax.random.split(rng)

            batch_images = batch.get('images')
            batch_success_images = batch_success.get('images')
            batch_obs_encoded = self._encode_for_actor(batch['observations'], images=batch_images, params=grad_params)
            batch_success_encoded = self._encode_for_actor(batch_success['observations'], images=batch_success_images, params=grad_params)

            bc_loss = self._compute_bc_loss(rng, grad_params, batch, batch_success, success_flag, obs_encoded=batch_obs_encoded, batch_success_encoded=batch_success_encoded, images=batch_images, images_success=batch_success_images)

        total_loss = pg_loss + self.config.entropy_coeff * ent_loss + self.config.bc_coeff * bc_loss
        info = {
            'pg_loss': pg_loss,
            'ent_loss': ent_loss,
            'bc_loss': bc_loss,
            'lp': lp.mean() if hasattr(lp, 'mean') else lp,
            'lp_min': lp.min() if hasattr(lp, 'min') else lp,
            'lp_max': lp.max() if hasattr(lp, 'max') else lp,
            'old_lp': old_lp.mean() if hasattr(old_lp, 'mean') else old_lp,
            'entropy': ent,
            'adv': adv.mean(),
            'adv_min': adv.min(),
            'adv_max': adv.max(),
            'adv_std': adv.std(),
            'adv_pre_norm': adv_pre.mean(),
            'adv_pre_min': adv_pre.min(),
            'adv_pre_max': adv_pre.max(),
            'q_values': adv_pre.mean(),
            'sigmas': sigmas.mean() if hasattr(sigmas, 'mean') else sigmas,
            'actions': actions.mean(),
            'actions_min': actions.min(),
            'actions_max': actions.max(),
            'actions_std': actions.std(),
            'group_num_samples': G,
            # (S_task, K) array, NOT a scalar metric. `_update` pops it before
            # the info dict is returned, so it never reaches the tree_map mean
            # in batch_update. Carried on the info dict rather than recomputed
            # so the dual reads the SAME critic evaluation the actor is
            # optimizing against, at no extra forward pass.
            'dual_g': dual_g,
            **adv_diag,
            'grpo_adv_group_std': jnp.std(adv_pre, axis=0).mean(),
            'grpo_adv_between_group_std': jnp.std(adv_pre.mean(axis=0)),
            **pg_stats,
            **info_extra,
        }

        return total_loss, info

    # --- QIPM-specific TD loss ---

    def get_td_loss(self, batch, batch_actions, next_actions, grad_params, rng, obs_encoded=None, next_obs_encoded=None, images=None, next_images=None):
        """Compute TD loss with per-subtask QIPM targets — supports both ensemble-Q and MIP-Q."""
        # Get next Q-values from target network
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        S_task = self.config['no_of_subtasks']
        n_safety = self.config['no_of_safety_subtasks']
        S = S_task + n_safety

        if self.config.get('use_mip_q', False):
            # MIP-Q path: two-step sampling for next Q-values
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = next_obs_encoded.shape[0]

            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                rng, noise_rng = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_rng, (batch_size, S),
                    minval=-noise_scale, maxval=noise_scale
                )
                next_qs = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=next_obs_encoded,
                    actions=next_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # (S, M, B)
            else:
                rng, noise_rng = jax.random.split(rng)
                noise_samples = jax.random.uniform(
                    noise_rng, (num_ensemble_members, batch_size, S),
                    minval=-noise_scale, maxval=noise_scale
                )
                next_qs = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=next_obs_encoded,
                    actions=next_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # (S, N, B)

            # Aggregate over ensemble/noise axis (axis=1) to get (S, B)
            rng, agg_rng = jax.random.split(rng)
            num_ens = num_ensemble_members
            if self.config['q_agg'] == 'min':
                next_q = next_qs.min(axis=1)
            elif self.config['q_agg'] == 'subsample':
                subsample_idxs = jax.random.randint(agg_rng, (2,), 0, num_ens)
                next_q = next_qs[:, subsample_idxs, :].min(axis=1)
            else:
                next_q = next_qs.mean(axis=1)
        else:
            # Original ensemble-Q path
            next_qs = self.critic_network.select('target_critic')(next_obs_encoded, actions=next_actions, is_encoded=bool(self.config.encoder))
            # next_qs: (S, M, B)

            rng, agg_rng = jax.random.split(rng)
            if self.config['q_agg'] == 'min':
                next_q = next_qs.min(axis=1)
            elif self.config['q_agg'] == 'subsample':
                subsample_idxs = jax.random.randint(agg_rng, (2,), 0, self.config.num_qs)
                next_q = next_qs[:, subsample_idxs, :].min(axis=1)
            else:
                next_q = next_qs.mean(axis=1)

        # Per-subtask TD targets: task heads use QIPM synthesis (+5/-1, capped
        # at V_subgoal); safety heads use the plain CMDP cost backup via
        # _compute_safety_td_target (zero upper cap, no lower cap, no
        # absorbing state).
        if self.config.get('subtask_based_discount', False) and self.config.get('subtask_discounts') is not None:
            subtask_discounts = self.config['subtask_discounts']
            assert len(subtask_discounts) == S_task, (
                f"subtask_discounts length {len(subtask_discounts)} != no_of_subtasks {S_task} "
                "(task heads only; safety heads use safety_discount)"
            )
            task_discount = jnp.array(subtask_discounts)
        else:
            task_discount = self.config['discount']
        safety_discount = jnp.full((n_safety,), self.config['safety_discount'])
        masks = batch['masks']
        if masks.ndim > 1:
            masks = masks[..., -1]

        target_q_task = compute_qipm_td_target(
            subtask_status=batch['subtask_status'][..., :S_task],
            next_q=next_q[:S_task],
            masks=masks,
            discount=task_discount,
            horizon_length=self.config['horizon_length'],
            no_of_subtasks=S_task,
            alpha=0.1,
            td_loss_mode=self.config.get('td_loss_mode', 'gated'),
        )
        target_q_safety = _compute_safety_td_target(
            subtask_status=batch['subtask_status'][..., S_task:],
            next_q=next_q[S_task:],
            masks=masks,
            discount=safety_discount,
            horizon_length=self.config['horizon_length'],
        )
        target_q = jnp.concatenate([target_q_task, target_q_safety], axis=0)
        # target_q: (S, B)

        # Current Q-values
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        is_enc = bool(self.config.encoder)

        valid = batch.get('valid')
        if valid is not None and valid.ndim > 1:
            valid = valid[..., -1]
        else:
            valid = jnp.ones(target_q.shape[1])

        if self.config.get('use_mip_q', False):
            # MIP-Q: two-term loss
            critic_base_fn = self.critic_network.select('critic')
            def critic_fn_with_params(obs, actions, scalar_input, time, is_encoded, return_logits):
                return critic_base_fn(
                    obs, actions=actions, scalar_input=scalar_input, time=time,
                    params=grad_params, is_encoded=is_encoded, return_logits=return_logits
                )

            if self.config.get('mip_q_ensemble', False):
                q_0, q = compute_mip_q_ensemble_predictions(
                    critic_fn=critic_fn_with_params,
                    observations=obs_encoded,
                    actions=batch_actions,
                    target_q=target_q,
                    rng=rng,
                    noise_scale=noise_scale,
                    mip_t_star=mip_t_star,
                    is_encoded=is_enc,
                    return_logits=False
                )
                # q_0, q: (S, M, B)
            else:
                q_0, q = compute_mip_q_predictions(
                    critic_fn=critic_fn_with_params,
                    observations=obs_encoded,
                    actions=batch_actions,
                    target_q=target_q,
                    rng=rng,
                    noise_scale=noise_scale,
                    mip_t_star=mip_t_star,
                    num_ensemble_members=num_ensemble_members,
                    is_encoded=is_enc,
                    return_logits=False
                )
                # q_0, q: (S, N, B)

            # Two-term MSE loss
            td_err_0 = jnp.square(q_0 - target_q[:, None, :])  # (S, M/N, B)
            td_err_t = jnp.square(q - target_q[:, None, :])     # (S, M/N, B)
            td_loss_0 = (td_err_0 * valid).mean()
            td_loss_t = (td_err_t * valid).mean()
            td_loss = td_loss_0 + td_loss_t
        else:
            # Original ensemble-Q path
            q = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=is_enc)
            # q: (S, M, B)
            td_err = jnp.square(q - target_q[:, None, :])  # (S, M, B)
            td_loss = (td_err * valid).mean()

        # Per-subtask Q-value statistics
        q_per_subtask_mean = q.mean(axis=(1, 2))  # (S,)
        q_per_subtask_min = q.min(axis=(1, 2))    # (S,)
        q_per_subtask_max = q.max(axis=(1, 2))    # (S,)

        stats = {
            **{f'q_subtask_{i}_mean': q_per_subtask_mean[i] for i in range(S)},
            **{f'q_subtask_{i}_min': q_per_subtask_min[i] for i in range(S)},
            **{f'q_subtask_{i}_max': q_per_subtask_max[i] for i in range(S)},
        }
        if self.config.get('use_mip_q', False):
            stats['td_loss_regression_t0'] = td_loss_0
            stats['td_loss_denoising_tstar'] = td_loss_t

        # --- min/max single-Q safety: directional margin estimator + telemetry ---
        # The actor-facing penalty is A = min(0, z + margin) (see
        # _aggregate_safety_heads). `margin` is one scalar of agent state so that
        # every call site -- including sample_actions_BON, which runs at ROLLOUT
        # time with no batch in scope -- can read it.
        #
        # It is estimated here because this is the only place where both the live
        # prediction q at (s, a) and the target-critic value next_q at (s', a')
        # are in scope, which makes the PER-TRANSITION change
        #
        #     dz_i = z(s'_i, a'_i) - z(s_i, a_i)
        #
        # free: next_q is already computed for the TD target. dz < 0 means the
        # predicted future safety cost GREW across this transition, i.e. the
        # trajectory is moving into worse safety. (Note the sign: because Q is a
        # cost-TO-GO, z rises again once a violation has actually been paid, so
        # dz < 0 marks the APPROACH to danger, not the violation itself.)
        #
        # The margin is then the (1-q)-quantile of z over the NON-deteriorating
        # transitions only, so ~q of that reference population contributes
        # exactly 0. Selecting the reference population by dz is what makes the
        # statistic directional; taking a quantile rather than a mean is what
        # stops ~half the batch firing by construction.
        safety_agg = self.config.get('safety_agg', 'sum')
        if safety_agg in ('min_relu', 'max_relu') and n_safety > 0:
            q_saf = jax.lax.stop_gradient(q[S_task:])          # (K, M, B)
            if self.config['q_agg'] in ('min', 'subsample'):
                q_ens = q_saf.min(axis=1)                       # (K, B)
            else:
                q_ens = q_saf.mean(axis=1)
            z = _select_safety_q(q_ens, safety_agg)             # (B,)
            # next_q is already ensemble-reduced with the same q_agg: (S, B).
            z_next = _select_safety_q(
                jax.lax.stop_gradient(next_q[S_task:]), safety_agg)   # (B,)
            dz = z_next - z                                     # (B,)

            margin_mode = self.config.get('safety_margin_mode', 'directional_quantile')
            valid_f = jnp.asarray(valid, dtype=z.dtype).reshape(-1)
            if margin_mode == 'directional_quantile':
                # Reference population: valid AND not deteriorating.
                ref = (dz >= 0.0) & (valid_f > 0)
                qtl = float(self.config.get('safety_margin_quantile', 0.9))
                m_batch = -_masked_quantile(
                    z, ref, 1.0 - qtl, fallback=-self.safety_margin,
                    min_count=int(self.config.get('safety_margin_min_count', 32)))
            elif margin_mode == 'batch_mean':
                # Mean predicted safety cost over the VALID transitions only.
                # z <= 0, so -mean(z) >= 0 and the margin is forgiving; the
                # dead zone becomes z >= mean(z), i.e. every transition whose
                # predicted cost is below the batch average contributes 0.
                #
                # The valid mask matters: `valid` zeroes padded / past-terminal
                # rows of the chunk, and their z is whatever the critic happens
                # to emit there. Averaging them in biases the margin by an
                # amount that depends on how much padding the batch drew.
                # (directional_quantile above already filters on valid_f; this
                # branch did not until 2026-08-22.)
                n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)
                m_batch = -jnp.sum(z * valid_f) / n_valid
            else:  # 'none' -- pin the margin at 0, giving plain -ReLU(-z)
                m_batch = jnp.zeros((), dtype=z.dtype)

            margin_prev = self.safety_margin
            contrib = -jax.nn.relu(-z - margin_prev)            # what the actor saw
            stats['safety_margin_batch'] = m_batch
            stats['safety_margin'] = jnp.asarray(margin_prev, dtype=z.dtype)
            stats['safety_z_mean'] = jnp.mean(z)
            stats['safety_z_min'] = jnp.min(z)
            stats['safety_z_max'] = jnp.max(z)
            stats['safety_dz_mean'] = jnp.mean(dz)
            # Fraction of transitions moving INTO worse predicted safety. This is
            # the population the margin is calibrated against; if it approaches
            # 1.0 the reference set is empty and the margin freezes.
            stats['safety_deteriorating_rate'] = jnp.mean((dz < 0).astype(jnp.float32))
            # Fraction of (s,a) where the safety term actually fires. If this
            # sits near 0 the arm is a no-op -- kill it early rather than
            # waiting out a full run.
            stats['safety_activation_rate'] = jnp.mean((contrib < 0).astype(jnp.float32))
            stats['safety_contrib_mean'] = jnp.mean(contrib)
            stats['safety_frac_heads_positive'] = jnp.mean((q_ens > 0).astype(jnp.float32))
            for k in range(n_safety):
                stats[f'safety_sel_frac_head_{k}'] = jnp.mean(
                    jnp.isclose(q_ens[k], z).astype(jnp.float32))

        return td_loss, q, next_q, stats

    def get_calql_diff(self, batch, batch_actions, grad_params, rng, obs_encoded=None, images=None):
        """CalQL regularizer for QIPM: sum over subtask axis before CQL computation."""
        B = batch['observations'].shape[0]
        is_enc = bool(self.config.encoder)
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        S_task = self.config['no_of_subtasks']
        q_pred = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=is_enc)
        # q_pred: (S, M, B) -> sum TASK heads only for CQL: the mc_returns lower
        # bound is task-reward based, so safety cost heads are excluded.
        q_pred = q_pred[:S_task].sum(axis=0)  # (M, B)

        # CQL Part
        rng, action_rng = jax.random.split(rng)
        N = self.config["cql_n_actions"]
        cql_random_actions = jax.random.uniform(action_rng, shape=(B, N, self.config["full_act_dim"]),
                                                minval=-1.0, maxval=1.0,)
        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        sample_rngs = jax.random.split(current_a_rng, N)
        vmapped_sample = jax.vmap(
            lambda obs, rng_i: self.sample_actions(obs, rng=rng_i),
            in_axes=(None, 0)
        )

        vmapped_q_fn = jax.vmap(
            lambda a: self.critic_network.select('critic')(obs_encoded, actions=a, params=grad_params, is_encoded=is_enc),
            in_axes=1, out_axes=-1
        )

        cql_current_actions = vmapped_sample(batch['observations'], sample_rngs)
        cql_current_actions = jnp.transpose(cql_current_actions, (1, 0, 2))
        cql_next_actions = vmapped_sample(batch['next_observations'], sample_rngs)
        cql_next_actions = jnp.transpose(cql_next_actions, (1, 0, 2))
        all_actions = jnp.concatenate([cql_random_actions, cql_current_actions, cql_next_actions], axis=1)

        cql_qs = vmapped_q_fn(all_actions)
        # cql_qs: (S, M, B, N*3)
        chex.assert_shape(cql_qs, (S_task + self.config['no_of_safety_subtasks'], self.config.num_qs, B, N * 3))

        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(subsample_key, (self.config.calql_q_subsample,), 0, self.config.num_qs,)
        # After subsampling ensembles: sum over subtask axis for cql_qs, subsample q_pred
        cql_qs = cql_qs[:, subsample_idcs, ...]  # (S, subsample, B, N*3)
        # Sum TASK heads only to match standard CalQL flow (safety excluded)
        cql_qs = cql_qs[:S_task].sum(axis=0)  # (subsample, B, N*3)
        q_pred = q_pred[subsample_idcs]  # (subsample, B)

        # CalQL Part
        n_actions_for_calql = N * 3
        mc_lower_bound = jnp.repeat(batch["mc_returns"].reshape(-1, 1), n_actions_for_calql, axis=1,)
        cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
        cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)], axis=-1)
        cql_qs -= jnp.log(cql_qs.shape[-1]) * self.config.cql_temp

        cql_ood_values = jax.scipy.special.logsumexp(cql_qs / self.config.cql_temp, axis=-1) * self.config.cql_temp
        calql_regularizer = (cql_ood_values - q_pred).mean()
        return calql_regularizer

    def bc_critic_loss(self, batch, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """Standard TD loss with CalQL for BC phase. Q stats use .sum(axis=0) over subtasks."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            next_actions = jnp.reshape(batch["next_actions"], (batch["next_actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
            next_actions = batch["next_actions"][..., 0, :]
        rng, sample_rng = jax.random.split(rng)

        images = batch.get('images')
        next_images = batch.get('next_images')
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=images, next_images=next_images)
        calql_regularizer = self.get_calql_diff(batch, batch_actions, grad_params, rng, obs_encoded=obs_encoded, images=images)
        critic_loss = td_loss + self.config.cql_alpha * calql_regularizer

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.sum(axis=0).mean(),
            'q_max': q.sum(axis=0).max(),
            'q_min': q.sum(axis=0).min(),
            'next_q_mean': next_q.sum(axis=0).mean(),
            'calql_regularizer': calql_regularizer.mean(),
            **stats
        }

    def critic_loss(self, batch: dict, grad_params: flax.core.FrozenDict,
                    rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Standard TD loss. Q stats use .sum(axis=0) over subtasks. Includes per-subtask Q stats."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)

        images = batch.get('images')
        next_images = batch.get('next_images')

        # Pre-encode observations
        obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        # Sample next actions from target actor (or one-step policy for FQL)
        next_actor_obs_encoded = self._encode_for_actor(batch['next_observations'], images=next_images, use_target=True)
        if self.config.get('use_one_step_for_targets', False) and self.one_step_network is not None:
            next_actions = self.sample_actions_one_step(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)
        else:
            next_actions = self.sample_actions(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)

        td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=images, next_images=next_images)
        return td_loss, {
            'td_loss': td_loss,
            'q_mean': q.sum(axis=0).mean(),
            'q_max': q.sum(axis=0).max(),
            'q_min': q.sum(axis=0).min(),
            'next_actions': next_actions.mean(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_actions_std': next_actions.std(),
            'next_q_mean': next_q.sum(axis=0).mean(),
            # How often the cost-Q clip actually bites, and how much it moves the
            # estimate. `q` is (S, M, B), so the safety slice still carries the
            # per-member axis the clip is applied on. frac_positive is the share
            # of member predictions that are invalid (true cost-Q is <= 0);
            # clip_shift is how much more negative clip-then-mean is than the raw
            # mean, i.e. the bias that was being averaged in before.
            'safety_q_frac_positive': (q[self.config['no_of_subtasks']:] > 0).mean(),
            'safety_q_clip_shift': (
                q[self.config['no_of_subtasks']:].mean(axis=1)
                - jnp.minimum(q[self.config['no_of_subtasks']:], 0.0).mean(axis=1)
            ).mean(),
            **stats
        }

    # --- QIPM-specific: done predictor loss ---

    def actor_total_loss(
        self,
        batch: dict,
        batch_success: dict,
        grad_params: flax.core.FrozenDict,
        success_flag,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Actor loss for separate optimization."""
        a_loss, a_info = self.actor_loss(batch, batch_success, grad_params, success_flag, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info

    def critic_total_loss(self, batch: dict, grad_params: flax.core.FrozenDict, rng: jnp.ndarray):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    # --- Learned Lagrange multipliers (safety_aggregation_strategy='learned_lambda') ---

    def _lambda_matrix(self, grad_params=None):
        """The (S_task, K) multiplier matrix Lambda = softplus(Theta), clipped
        at ``lambda_max``.

        ``grad_params=None`` reads the COMMITTED parameters, which is what
        every consumer inside the actor/critic losses must use: Lambda enters
        Q_total detached, so the actor cannot differentiate through it. Passing
        ``grad_params`` is reserved for ``lambda_loss``, the one place that
        trains it. Exactly the split sac.py uses for the entropy temperature.
        """
        if grad_params is None:
            return self.lambda_network.select('safety_lambda')()
        return self.lambda_network.select('safety_lambda')(params=grad_params)

    def _safety_budget(self):
        """(K,) per-constraint cost budgets d_k >= 0, resolved once in create()."""
        return jnp.asarray(self.config['safety_budget_resolved'], dtype=jnp.float32)

    def lambda_loss(self, dual_g, grad_params: flax.core.FrozenDict):
        """Dual ascent on the CMDP Lagrangian, written as a minimisation.

            L(Theta) = sum_ik softplus(Theta_ik) * g_ik
            dL/dTheta_ik = sigmoid(Theta_ik) * g_ik

        with g_ik = mean_b[ phi_i(s_b) * (sg[Q_k(s_b, a_b)] + d_k) ] supplied by
        ``_dual_signal`` (see its docstring for the sign convention). g is the
        SAME critic evaluation the actor was optimised against on this step --
        it is carried on the actor info dict rather than recomputed, so the
        primal and dual see one consistent pair (Q, pi) and no extra forward
        pass is spent.

        Descending L raises Lambda_ik exactly where the constraint is violated
        (g_ik < 0) and decays it toward 0 where there is budget slack. Note
        that gradient flow stops at ``lambda_max`` because ``_lambda_matrix``
        clips the OUTPUT -- an intentional stop, not a silent saturation: the
        logged ``lambda/frac_at_max`` reports it.
        """
        lam = self._lambda_matrix(grad_params)              # (S_task, K)
        loss = (lam * dual_g).sum()

        s_task, k = lam.shape
        info = {
            'dual_loss': loss,
            'matrix_mean': lam.mean(),
            'matrix_max': lam.max(),
            'matrix_min': lam.min(),
            # Fraction of entries pinned at the cap. Non-zero means the dual
            # has run away: the budget is unreachable at the current policy.
            # Under Theta-projection an entry sitting at the cap is
            # RECOVERABLE: it still has a live gradient and descends as soon as
            # its constraint is met. A frac_at_max that stays high AFTER the
            # violations go negative means the projection is not working, and
            # is a bug rather than a tuning signal.
            'frac_at_max': (lam >= float(self.config['lambda_max']) - 1e-6).mean(),
            # The mirror image, and the failure mode the floor introduces:
            # entries parked at lambda_floor are effectively switched off. A
            # healthy dual keeps most entries strictly interior, with
            # frac_at_max and frac_at_min both small and both moving.
            'frac_at_min': (lam <= float(self.config['lambda_floor']) + 1e-6).mean(),
            # Mean violation per constraint, sign-flipped so POSITIVE = over
            # budget, matching the usual reading of a constraint violation.
            # Summed over i because phi is a simplex (sum_i phi_i = 1), so this
            # is exactly mean_b[-(Q_k + d_k)].
            'violation_mean': (-dual_g).sum(axis=0).mean(),
        }
        for kk in range(k):
            info[f'violation_{kk}'] = -dual_g[:, kk].sum()
        # Every entry individually: 20 scalars at 4x5, and the whole point of
        # giving Lambda a task-phase axis is being able to read them.
        for i in range(s_task):
            for kk in range(k):
                info[f'L_{i}_{kk}'] = lam[i, kk]
        return loss, info

    def lambda_total_loss(self, dual_g, grad_params: flax.core.FrozenDict):
        """Namespaced wrapper, mirroring actor_total_loss/critic_total_loss.

        The prefix is load-bearing: `_update` merges the three info dicts with
        a plain {**a, **b}, so an unprefixed 'dual_loss' would collide.
        """
        l_loss, l_info = self.lambda_loss(dual_g, grad_params)
        return l_loss, {f'lambda/{k}': v for k, v in l_info.items()}

    @staticmethod
    def _maybe_update_lambda(agent, actor_info):
        """One projected-dual-ascent step on Lambda, if this arm has one.

        Returns (new_lambda_state, lambda_info). Both are inert passthroughs
        when ``lambda_network is None`` (every non-learned_lambda arm), which
        is what keeps those arms bit-identical to their pre-change behaviour.

        Warmup is handled by the optimizer's LEARNING-RATE SCHEDULE (built in
        `create()` as join_schedules([0, lambda_lr]) at lambda_warmup_steps),
        not by a Python branch here: `batch_update` runs this inside a
        `lax.scan` whose carry must keep a static pytree structure, so an `if`
        on the step count would not trace. Lambda therefore sits at its
        initialisation (== safety_coeff, i.e. exactly the sum baseline) until
        `lambda_warmup_steps`, by which point the critic's safety heads carry
        enough signal for the violation estimate to mean something. Ascending
        the dual against a randomly-initialised critic just injects noise.
        Adam's moment estimates do accumulate through the zero-lr phase, so the
        first live step is warm rather than a cold start -- deliberate, and the
        reason for a schedule rather than a masked gradient.
        """
        if agent.lambda_network is None:
            return None, {}

        dual_g = actor_info['actor/dual_g']

        def lambda_loss_fn(params):
            return agent.lambda_total_loss(dual_g, params)

        new_lambda_state, lambda_info = agent.lambda_network.apply_loss_fn(lambda_loss_fn)
        # apply_loss_fn injects grad/max, grad/min and grad/norm OUTSIDE the
        # loss's own info dict (flax_utils.py:129-131), so `lambda_total_loss`'s
        # prefixing does not reach them. `_update` merges the info dicts with
        # {**actor, **critic, **lambda}, so leaving them bare would silently
        # overwrite the ACTOR's gradient metrics with the dual's -- caught by
        # the baseline-equivalence test, which saw grad/norm drop 75 -> 0.3.
        lambda_info = {
            (f'lambda/{k}' if k.startswith('grad/') else k): v
            for k, v in lambda_info.items()
        }

        # Bound Lambda by PROJECTING Theta, not by clipping the output.
        #
        # SoftplusMatrixParam's max_value clips softplus(Theta), and jnp.minimum
        # is flat above the cap, so dLambda/dTheta becomes exactly 0 -- in BOTH
        # directions. A capped entry then cannot come back down when its
        # constraint is later satisfied. That is not a guard, it is an absorbing
        # state: it froze 12 of 16 entries for 500k+ steps in every learned_lambda
        # fleet (10327119-21, 10334353, 10353365-67) while 3 of 4 constraints
        # were in fact being met. The single entry that escaped -- L_1_2,
        # collision, run 19giclgc -- did so only because collision became
        # satisfied BEFORE that entry reached the cap, and it then decayed
        # correctly to 0. That contrast is the evidence for this fix.
        #
        # Clipping Theta keeps the boundary gradient at sigmoid(Theta) != 0, so
        # an entry starts moving on the first step after its slack changes sign.
        #
        # The floor is the same argument at the other end: a satisfied constraint
        # drives Theta down without limit (L_1_2 reached Lambda=0.0000, i.e.
        # Theta < -9.9), and that distance must be unwound before the multiplier
        # can re-engage if the constraint is violated again. Simulated on the real
        # update rule: re-engagement takes ~190 steps with the floor, ~1650
        # without.
        theta_max = float(np.log(np.expm1(float(agent.config['lambda_max']))))
        theta_min = float(np.log(np.expm1(float(agent.config['lambda_floor']))))
        new_lambda_state = new_lambda_state.replace(
            params=jax.tree_util.tree_map(
                lambda t: jnp.clip(t, theta_min, theta_max),
                new_lambda_state.params,
            )
        )
        return new_lambda_state, lambda_info

    # --- Update methods ---

    @staticmethod
    def _next_safety_margin(agent, critic_info):
        """margin <- clip( EMA(margin, m_batch), 0, cap ).

        `m_batch` is the directional statistic from this critic batch (see the
        estimator in get_td_loss). Two guards:

          * clipped at 0 from below, so the modes can never be HARSHER than a
            plain -ReLU(-z). Without it a batch where every head mispredicted
            positive would drive the margin negative and start penalising
            genuinely safe (z = 0) states.
          * capped from above, so a globally unsafe policy cannot widen its own
            dead-zone without bound and switch the safety term off.

        Returns the unchanged margin for every safety_agg that does not use one
        (the estimator only writes its key for min_relu/max_relu).
        """
        key = 'critic/safety_margin_batch'
        if key not in critic_info:
            return agent.safety_margin
        tau = float(agent.config.get('safety_margin_tau', 0.005))
        cap = float(agent.config.get('safety_margin_cap', 5.0))
        m = (1.0 - tau) * agent.safety_margin + tau * critic_info[key]
        return jnp.clip(m, 0.0, cap)

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.bc_loss(batch, p, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        agent.target_update(new_actor_state, 'actor')

        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # FQL one-step distillation (BC-only, no Q loss)
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_offline', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng2, use_q_loss=0.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        bc_info = {**actor_info, **one_step_info}
        new_rng, _ = jax.random.split(agent.rng)  # match upstream: re-split from agent.rng
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), bc_info

    @staticmethod
    def _update_offline_calql(agent, batch: dict) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        """Apply gradient update to both critic and done networks."""
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.critic_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_critic_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.bc_critic_loss(batch, p, rng1)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        info = {**critic_info}
        new_rng, _ = jax.random.split(new_rng)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @jax.jit
    def calql_update(self, batch):
        return self._update_offline_calql(self, batch)

    @staticmethod
    def _update(agent, batch_tuple, success_flag) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        """Apply gradient update to actor and critic networks."""
        batch, batch_success = batch_tuple
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        # Update actor network
        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=agent.actor_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.actor_total_loss(batch, batch_success, p, success_flag, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Update actor target networks
        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # Update critic network
        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.critic_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.critic_total_loss(batch, p, rng2)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        # FQL one-step distillation (online, with Q loss)
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng3, _ = jax.random.split(rng2)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng3, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        # Dual ascent on Lambda. Runs AFTER the actor step but reads the dual
        # signal the actor's own forward pass produced, so primal and dual see
        # the same (Q, pi) pair. No-op (None, {}) for every arm without a
        # learned Lambda.
        new_lambda_state, lambda_info = agent._maybe_update_lambda(agent, actor_info)

        # Combine info. dual_g is an (S_task, K) array, not a metric: pop it
        # before the dicts merge, or batch_update's tree_map mean would emit it
        # as a meaningless scalar and the per-entry lambda/L_i_k logs would be
        # redundant with it.
        actor_info = {k: v for k, v in actor_info.items() if k != 'actor/dual_g'}
        info = {**actor_info, **critic_info, **one_step_info, **lambda_info}
        new_margin = agent._next_safety_margin(agent, critic_info)
        return agent.replace(actor_network=new_actor_state, critic_network=new_critic_state, one_step_network=new_one_step_state, rng=new_rng, safety_margin=new_margin, lambda_network=new_lambda_state), info

    @jax.jit
    def batch_update(self, batch, batch_success, success_flag):
        policy_delay = self.config.get('policy_delay', 1)

        if policy_delay <= 1:
            # Original: joint actor+critic updates every step
            batch_tuple = (batch, batch_success)
            def scan_update(agent, batch_tuple):
                return self._update(agent, batch_tuple, success_flag)
            agent, infos = jax.lax.scan(scan_update, self, batch_tuple)
            return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

        # Policy delay > 1: two-phase scan
        # Phase 1: critic updates on ALL utd slices
        agent, critic_infos = jax.lax.scan(self._update_critic_only, self, batch)

        # Phase 2: actor updates on every policy_delay-th slice
        actor_batch = jax.tree.map(
            lambda x: x[policy_delay - 1::policy_delay], batch)
        actor_success = jax.tree.map(
            lambda x: x[policy_delay - 1::policy_delay], batch_success)

        def scan_actor(agent, inputs):
            return self._update_actor_only(agent, inputs, success_flag)
        agent, actor_infos = jax.lax.scan(
            scan_actor, agent, (actor_batch, actor_success))

        return agent, {
            **jax.tree_util.tree_map(lambda x: x.mean(), critic_infos),
            **jax.tree_util.tree_map(lambda x: x.mean(), actor_infos),
        }

    @staticmethod
    def _update_critic_only(agent, batch) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        """Apply gradient update to critic network only."""
        new_rng, sample_rng = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        new_margin = agent._next_safety_margin(agent, critic_info)
        return agent.replace(critic_network=new_critic_state, rng=new_rng,
                             safety_margin=new_margin), critic_info

    @staticmethod
    def _update_actor_only(agent, batch_tuple, success_flag) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        """Apply gradient update to actor network only (used with policy delay)."""
        batch, batch_success = batch_tuple
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables, batch, batch_success,
                    rngs={'dropout': rng1},
                    method=agent.actor_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.actor_total_loss(batch, batch_success, p, success_flag, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # FQL one-step distillation
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng2, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        # Same dual step as in `_update`. Present here too so that
        # policy_delay > 1 does not silently freeze the dual: this is the only
        # path that runs the actor in that mode.
        new_lambda_state, lambda_info = agent._maybe_update_lambda(agent, actor_info)

        actor_info = {k: v for k, v in actor_info.items() if k != 'actor/dual_g'}
        info = {**actor_info, **one_step_info, **lambda_info}
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng, lambda_network=new_lambda_state), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update_critic_only, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    batch_critic_only_update = batch_q_warmup_update

    @staticmethod
    def _bc_refine_update(agent, batch_success) -> Tuple['OGPOLearnedLambdaSafeQIPMAgentSharedBase', dict]:
        """Apply BC loss update to actor network only using success buffer."""
        new_rng, rng1 = jax.random.split(agent.rng)

        # Update actor network with BC loss only
        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=lambda batch, params: agent.bc_loss(batch, params, rng1),
                    mutable=['batch_stats']
                )
                bc_info = {f'actor/bc_{k}': v for k, v in info.items()}
                return (loss, bc_info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                bc_loss_val, bc_info = agent.bc_loss(batch_success, p, rng1)
                info = {f'actor/bc_{k}': v for k, v in bc_info.items()}
                return bc_loss_val, info
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Update actor target networks
        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        return agent.replace(actor_network=new_actor_state, rng=new_rng), actor_info

    @jax.jit
    def batch_bc_refine_update(self, batch_success):
        """BC refinement update using only the success buffer."""
        agent, infos = jax.lax.scan(self._bc_refine_update, self, batch_success)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # --- FQL one-step sampling ---

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions_one_step(self, observations, rng=None, is_encoded=False, images=None):
        return sample_one_step(self, observations, rng, is_encoded, images)

    @jax.jit
    def sample_actions_one_step_BON(self, observations, rng=None, images=None):
        return sample_one_step_BON(self, observations, rng, images)

    def one_step_distillation_loss(self, batch, grad_params, rng, use_q_loss=1.0):
        return one_step_distillation_loss(self, batch, grad_params, rng, use_q_loss)

    def target_update(self, network, module_name):
        """Update the target network."""
        if module_name in ['actor', 'noise_net']:
            tau = self.config['actor_tau']
        else:
            tau = self.config['tau']

        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    # --- Inference methods ---

    @jax.jit
    def sample_actions_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None, images: jnp.ndarray = None) -> jnp.ndarray:
        """Fully vectorized Best-of-N action selection with per-subtask Q-values.
        Best action -> max ( sum_task ( min_M Q ) + safety_coeff * <safety term> )
        (task heads unweighted as in qipm; safety heads always active, weighted
        by safety_coeff — safety-aware action selection at inference.)

        The safety term follows safety_aggregation_strategy:
        "simple_summation" -> sum_safety ( min_M Q ), "min_max" ->
        min_k ( max over the members with Q <= 0 ) via
        _aggregate_safety_min_max. The TASK heads keep their hard-coded
        (subsampled) ensemble MIN under both strategies — this site has always
        ignored q_agg — and both terms read the same member set, so
        subsample_bon subsamples the safety reduction too.
        """
        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[None]
            if images is not None:
                images = images[None]

        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n

        # 1. Generate all B*N candidate actions in one forward pass
        rng, action_key = jax.random.split(rng)
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
        images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None
        sampled_actions, _, _, _ = self.sample_actions_with_noise(
            obs_expanded, rng=action_key, use_target=True, is_encoded=False, images=images_expanded
        )

        # 2. Evaluate Q-ensemble for all candidates
        critic_obs = self._encode_for_critic(obs_expanded, images=images_expanded, use_target=True)
        S_task = self.config['no_of_subtasks']
        safety_coeff = self.config['safety_coeff']
        safety_agg = self.config.get('safety_agg', 'sum')
        safety_strategy = self.config['safety_aggregation_strategy']
        S = S_task + self.config['no_of_safety_subtasks']

        if self.config.get('use_mip_q', False):
            # MIP-Q path
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            total_candidates = batch_size * num_samples

            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                rng, noise_key = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_key, (total_candidates, S),
                    minval=-noise_scale, maxval=noise_scale
                )
                q_values_flat = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs,
                    actions=sampled_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # (S, M, B*N)
            else:
                rng, noise_key = jax.random.split(rng)
                noise_samples_base = jax.random.uniform(
                    noise_key, (num_ensemble_members, 1, S),
                    minval=-noise_scale, maxval=noise_scale
                )
                noise_samples = jnp.broadcast_to(
                    noise_samples_base,
                    (num_ensemble_members, total_candidates, S)
                )
                q_values_flat = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs,
                    actions=sampled_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # (S, N, B*N)

            # Reshape (S, M/N, B*N) -> (S, M/N, B, N)
            q_ensemble = q_values_flat.reshape(S, num_ensemble_members, batch_size, num_samples)
            num_ens = num_ensemble_members
        else:
            # Original ensemble-Q path
            q_ensemble = self.critic_network.select('target_critic')(
                critic_obs, actions=sampled_actions, is_encoded=bool(self.config.encoder)
            )
            # q_ensemble: (S, M, B*N) -> reshape to (S, M, B, N)
            q_ensemble = q_ensemble.reshape(S, self.config.num_qs, batch_size, num_samples)
            num_ens = self.config.num_qs

        # 3. min over ensemble axis=1 -> (S, B, N), reduce over subtasks axis=0
        #    -> (B, N). q_members is the member set both terms read (Decision 9).
        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2,), 0, num_ens)
            q_members = q_ensemble[:, idxs, :, :]  # (S, 2, B, N)
        else:
            q_members = q_ensemble  # (S, M, B, N)
        min_qs = q_members.min(axis=1)  # (S, B, N)

        if safety_strategy == 'learned_lambda':
            # No head aggregation: the K safety values are mixed by the learned
            # multipliers. No subtask_status reaches this site, so the task term
            # is an unweighted sum; the matching safety convention is the
            # uniform simplex phi = 1/S_task (see _effective_lambda). Lambda is
            # read detached -- action selection never differentiates.
            lam_eff = _effective_lambda(self._lambda_matrix(), None)          # (K,)
            # Clipped to the valid cost-Q range, as on the advantage path. This
            # site reduces with min, and min is monotone, so clipping before or
            # after the reduction is identical here -- but NOT clipping at all
            # is not: an all-positive head would otherwise contribute a positive
            # "cost" and make an unsafe candidate score better.
            saf_m = q_members[S_task:]
            if self.config.get('safety_clip_cost_q', True):
                saf_m = _clip_cost_q_members(saf_m)
            saf_qs = saf_m.min(axis=1)                                        # (K, B, N)
            q_safety = jnp.tensordot(lam_eff, saf_qs, axes=([0], [0]))        # (B, N)
            q_total = min_qs[:S_task].sum(axis=0) + q_safety                  # (B, N)
        else:
            if safety_strategy == 'min_max':
                q_safety = _aggregate_safety_min_max(q_members[S_task:])  # (B, N)
            else:
                q_safety = _aggregate_safety_heads(min_qs[S_task:], safety_agg, self.safety_margin)  # (B, N)
            q_total = min_qs[:S_task].sum(axis=0) + safety_coeff * q_safety  # (B, N)

        # 4. Select the best candidate per observation
        best_idx = jnp.argmax(q_total, axis=1)  # (B,)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)  # (B, act_dim)

        return best_actions

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None, is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """Sample final actions via the flow policy with noise."""
        add_batch_dim = False
        if not is_encoded:
            if observations.ndim == 1:
                add_batch_dim = True
                observations = observations[None]
                if images is not None:
                    images = images[None]

        actions, chains, _, _ = self.sample_actions_with_noise(observations, use_target=True, rng=rng, is_encoded=is_encoded, images=images)

        if add_batch_dim:
            actions = actions[0]
        return actions

    def _check_add_batch_dim(self, observations, is_encoded):
        """Check if batch dimension needs to be added (based on static ndim)."""
        if is_encoded:
            return False
        return observations.ndim == 1

    def _encode_observations_for_flow(self, observations, is_encoded, images=None):
        """Encode observations for flow sampling. Called from within jitted functions."""
        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images)
        return observations

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions() instead."""
        # Add batch dim if needed (static argument)
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper function for actor network
        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'mip':
            # Use pg_helper for MIP ODE sampling
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            # Shortcut logic
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            # Use pg_helper for standard ODE flow (no correction)
            actions = sample_flow_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,
                clip_value=self.config['denoised_clip_value'],
                noises=ode_noises,
                is_encoded=True,
            )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions(self, observations, rng=None, noises=None, is_encoded=False, images=None):
        """Compute actions using pure ODE flow (no SDE-to-ODE correction)."""
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_corrected_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions_corrected() instead."""
        # Add batch dim if needed (static argument)
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper function for actor network (with denoiser support)
        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'mip':
            # MIP doesn't use correction (different formulation)
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            # Shortcut doesn't use correction
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            # Use pg_helper for ODE with SDE-to-ODE correction
            actions = sample_flow_actions_ode_with_correction(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                constant_noise_std=self.config['constant_noise_std'],
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,
                clip_value=self.config['denoised_clip_value'],
                noises=ode_noises,
                is_encoded=True,
            )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions_corrected(self, observations, rng=None, noises=None, is_encoded=False, images=None):
        """Compute actions using ODE flow with SDE-to-ODE correction."""
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_corrected_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    def sample_actions_with_noise(
            self,
            observations: jnp.ndarray,
            grad_params: flax.core.FrozenDict = None,
            rng: jnp.ndarray = None,
            use_target: bool = True,
            is_encoded: bool = False,
            images: jnp.ndarray = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate actions, full chains, and log_probs using pg_helper SDE sampling."""
        params = grad_params or self.actor_network.params
        actor_module_name = 'target_actor'
        noise_module_name = 'target_noise_net'

        # Encode observations
        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, use_target=use_target)

        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper functions for the network calls
        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'mip':
            actions, chains, logprob, sigmas = sample_mip_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                is_encoded=True,
                params=params,
            )
        else:
            actions, chains, logprob, sigmas = sample_flow_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
                is_encoded=True,
                params=params,
            )

        # Apply normalization options (matching compute_log_probs_and_entropy)
        logprob_steps = 3 if self.config.policy_type == 'mip' else self.config.flow_steps + 1
        if self.config['normalize_denoising_horizon']:
            logprob = logprob / logprob_steps

        if self.config['normalize_act_space_dimension']:
            logprob = logprob / act_dim

        return actions, chains, logprob, sigmas

    @jax.jit
    def bc_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        obs_encoded: jnp.ndarray = None,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute flow matching BC loss.

        Supports three modes:
        1. MIP mode: Two-term loss (regression at t=0 + denoising at t=t*)
        2. Flow + Denoiser mode: Velocity + noise prediction (OGPO-specific SDE-ODE correction)
        3. Standard flow matching: Velocity prediction only

        Uses bc_helper functions for common operations.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        # Encoder handling
        if images is None:
            images = batch.get('images')
        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(batch['observations'], images=images, params=grad_params)
        # obs_encoded is always the _encode_for_actor output (passed in or
        # computed above): with an encoder it is the encoded features, without
        # one it is the raw obs AFTER the actor_obs_dim slice — using raw
        # batch['observations'] here would bypass the slice (the pre-slice
        # code did exactly that; equivalent then, wrong now).
        actor_obs = obs_encoded

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))

        if self.config.policy_type == 'mip':
            # ========== MIP Mode: Two-term loss (uses bc_helper) ==========
            x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(
                batch_actions, rng, self.config.mip_t_star
            )

            # Term 1: Regression at t=0
            pred_0 = self.actor_network.select('actor')(
                actor_obs, x_0, t_0_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_regression_raw = jnp.square(pred_0 - batch_actions)

            # Term 2: Denoising at t=t*
            pred_t_star = self.actor_network.select('actor')(
                actor_obs, x_t_star, t_star_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_denoising_raw = jnp.square(pred_t_star - batch_actions)

            combined_loss_raw = loss_regression_raw + loss_denoising_raw

            bc_loss = apply_chunking_mask(
                combined_loss_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_regression = apply_chunking_mask(
                loss_regression_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_denoising = apply_chunking_mask(
                loss_denoising_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            info = {
                'bc_loss': bc_loss,
                'bc_mip_loss_regression': loss_regression,
                'bc_mip_loss_denoising': loss_denoising,
            }
        else:
            # ========== Standard Flow Matching Mode ==========
            rng, targets_rng = jax.random.split(rng)

            if self.config.use_shortcut:
                # OGPO-specific: Shortcut targets with bootstrapping
                def model_call_fn(obs, x_t, t, dt_base):
                    return self.actor_network.select('target_actor')(obs, x_t, t, dt_base)

                # actor_obs, not batch['observations']: raw obs here would
                # bypass the actor_obs_dim slice (branch currently unreachable
                # in safe_qipm — no shortcut_denoising_steps key — but keep it
                # funnel-correct).
                x_t, vel, t, dt_base, _ = get_shortcut_targets(
                    actor_obs,
                    batch_actions,
                    model_call_fn,
                    targets_rng,
                    denoise_timesteps=self.config.shortcut_denoising_steps,
                    bootstrap_ratio=self.config.bootstrap_ratio,
                )
                pred = self.actor_network.select('actor')(
                    actor_obs, x_t, t, dt_base, params=grad_params
                )
                loss_denoising = 0.0
            else:
                x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)
                loss_denoising = 0.0

                if self.config.use_denoiser:
                    # OGPO-specific: Denoiser for SDE-to-ODE correction
                    rng, z_rng = jax.random.split(rng)
                    z = jax.random.normal(z_rng, x_t.shape)

                    sigma = self.config.constant_noise_std
                    x_t_noisy = x_t + sigma * z

                    pred, pred_z = self.actor_network.select('actor')(
                        actor_obs, x_t_noisy, t,
                        params=grad_params,
                        is_encoded=self.config.encoder,
                        return_denoiser=True
                    )

                    loss_denoising = apply_chunking_mask(
                        jnp.square(pred_z - z), valid_mask, batch_size,
                        self.config["horizon_length"], self.config["action_dim"],
                        self.config["action_chunking"]
                    )
                else:
                    pred = self.actor_network.select('actor')(
                        actor_obs, x_t, t,
                        params=grad_params,
                        is_encoded=self.config.encoder
                    )

            # Velocity loss
            loss_vel = apply_chunking_mask(
                jnp.square(pred - vel), valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            bc_loss = loss_vel + loss_denoising
            info = {
                'bc_loss': bc_loss,
                'bc_flow_loss_vel': loss_vel,
                'bc_flow_loss_denoising': loss_denoising
            }

        return bc_loss, info

    # --- Q-value query methods ---

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False, images=None):
        """Computes the Q-value for a given state-action pair for logging.
        Returns the true scalarized objective: task-head sum + safety_coeff *
        safety term (min or mean over the ensemble first).

        The safety term follows safety_aggregation_strategy: "simple_summation"
        sums the q_agg-reduced heads; "min_max" runs _aggregate_safety_min_max
        over the FULL ensemble — mirroring the task reduction here, which uses
        the full ensemble even under q_agg='subsample' (pre-existing: this site
        maps 'subsample' onto a plain min).
        """
        if not is_encoded:
            observations = self._encode_for_critic(observations, images=images, use_target=True)
        actions = jnp.reshape(actions, (actions.shape[0], -1))
        qs = self._get_subtask_q_ensemble(observations, actions)
        # qs: (S, M, B)
        S_task = self.config['no_of_subtasks']
        safety_coeff = self.config['safety_coeff']
        safety_agg = self.config.get('safety_agg', 'sum')
        if self.config['q_agg'] == 'min' or self.config['q_agg'] == 'subsample':
            q_agg = qs.min(axis=1)  # (S, B)
        else:
            q_agg = qs.mean(axis=1)  # (S, B)
        if self.config['safety_aggregation_strategy'] == 'learned_lambda':
            # Same ungated convention as sample_actions_BON: uniform phi, so the
            # logged Q matches the objective actions are actually selected by.
            lam_eff = _effective_lambda(self._lambda_matrix(), None)  # (K,)
            # Clipped per member before reducing, matching the advantage path --
            # otherwise the logged Q would not be the quantity the policy is
            # actually optimising.
            saf_m = qs[S_task:]
            if self.config.get('safety_clip_cost_q', True):
                saf_m = _clip_cost_q_members(saf_m)
            if self.config['q_agg'] in ('min', 'subsample'):
                saf_q = saf_m.min(axis=1)  # (K, B)
            else:
                saf_q = saf_m.mean(axis=1)  # (K, B)
            q_safety = jnp.tensordot(lam_eff, saf_q, axes=([0], [0]))  # (B,)
            return q_agg[:S_task].sum(axis=0) + q_safety
        if self.config['safety_aggregation_strategy'] == 'min_max':
            q_safety = _aggregate_safety_min_max(qs[S_task:])  # (B,)
        else:
            q_safety = _aggregate_safety_heads(q_agg[S_task:], safety_agg, self.safety_margin)  # (B,)
        return q_agg[:S_task].sum(axis=0) + safety_coeff * q_safety

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_per_subtask_q_values(self, observations, actions, is_encoded=False):
        """Returns Q-value for each subtask separately. Shape: (S, B)"""
        actions = jnp.reshape(actions, (actions.shape[0], -1))
        qs = self._get_subtask_q_ensemble(observations, actions, is_encoded=is_encoded)
        # qs: (S, M, B)
        if self.config['q_agg'] == 'min' or self.config['q_agg'] == 'subsample':
            return qs.min(axis=1)  # (S, B)
        else:
            return qs.mean(axis=1)  # (S, B)

    def _get_subtask_q_ensemble(self, observations, actions, is_encoded=None):
        """Get (S, M/N, B) Q-values from target critic, handling both ensemble-Q and MIP-Q."""
        if is_encoded is None:
            is_encoded = bool(self.config.encoder)
        if self.config.get('use_mip_q', False):
            S = self.config['no_of_subtasks'] + self.config['no_of_safety_subtasks']
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = observations.shape[0]
            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                # Deterministic eval: zero noise
                noise_sample = jnp.zeros((batch_size, S))
                return sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=observations,
                    actions=actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                    is_encoded=is_encoded
                )  # (S, M, B)
            else:
                # Deterministic eval: use fixed seed noise spread
                noise_samples = jnp.zeros((num_ensemble_members, batch_size, S))
                return sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=observations,
                    actions=actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                    is_encoded=is_encoded
                )  # (S, N, B)
        else:
            return self.critic_network.select('target_critic')(
                observations, actions=actions, is_encoded=is_encoded
            )  # (S, M, B)

    # --- LR info and optimizer reset ---

    def _get_lr_info(self, actor_state=None, critic_state=None):
        """Get learning rate information and parameter difference norms for logging."""
        actor_state = actor_state or self.actor_network
        critic_state = critic_state or self.critic_network

        # Parameter norms
        actor_norm = compute_param_norm(actor_state.params['modules_actor'])
        target_actor_norm = compute_param_norm(actor_state.params['modules_target_actor'])
        critic_norm = compute_param_norm(critic_state.params['modules_critic'])
        target_critic_norm = compute_param_norm(critic_state.params['modules_target_critic'])

        # Parameter difference norms
        actor_diff_norm = compute_param_diff_norm(
            actor_state.params['modules_actor'],
            actor_state.params['modules_target_actor']
        )
        critic_diff_norm = compute_param_diff_norm(
            critic_state.params['modules_critic'],
            critic_state.params['modules_target_critic']
        )

        # OGPO-specific: noise network norms
        noise_info = {}
        if not self.config.use_constant_noise:
            noise_norm = compute_param_norm(actor_state.params['modules_noise_net'])
            target_noise_norm = compute_param_norm(actor_state.params['modules_target_noise_net'])
            noise_diff_norm = compute_param_diff_norm(
                actor_state.params['modules_noise_net'],
                actor_state.params['modules_target_noise_net']
            )
            noise_info = {
                'noise_param_norm': noise_norm,
                'target_noise_param_norm': target_noise_norm,
                'noise_target_diff_norm': noise_diff_norm,
            }

        return {
            'actor_lr': actor_state.opt_state.hyperparams['learning_rate'],
            'critic_lr': critic_state.opt_state.hyperparams['learning_rate'],
            'actor_step': actor_state.step,
            'critic_step': critic_state.step,
            'actor_param_norm': actor_norm,
            'target_actor_param_norm': target_actor_norm,
            'critic_param_norm': critic_norm,
            'target_critic_param_norm': target_critic_norm,
            'actor_target_diff_norm': actor_diff_norm,
            'critic_target_diff_norm': critic_diff_norm,
            **noise_info,
        }

    def reset_optimizers_with_lr(self) -> 'OGPOLearnedLambdaSafeQIPMAgentSharedBase':
        """Reset optimizers with new LR."""
        actor_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.actor_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.actor_warmup_steps,
            decay_steps=self.config.actor_decay_steps,
            end_value=self.config.actor_end_value
        )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.critic_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.critic_warmup_steps,
            decay_steps=self.config.critic_decay_steps,
            end_value=self.config.critic_end_value
        )

        new_actor_tx = create_optimizer(
            lr_schedule=actor_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.actor_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )
        new_critic_tx = create_optimizer(
            lr_schedule=critic_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.critic_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )

        opt_type = "Muon" if self.config.use_muon else "AdamW"
        print(f"Done: reset actor optimizer with {opt_type} (LR: {self.config.ppo_lr}, scheduler: {self.config.actor_scheduler})")
        print(f"Done: reset critic optimizer with {opt_type} (LR: ppo_lr={self.config.ppo_lr}, scheduler: {self.config.critic_scheduler})")

        new_actor_opt_state = new_actor_tx.init(self.actor_network.params)
        new_actor_network = self.actor_network.replace(tx=new_actor_tx, opt_state=new_actor_opt_state, step=1)

        new_critic_opt_state = new_critic_tx.init(self.critic_network.params)
        new_critic_network = self.critic_network.replace(tx=new_critic_tx, opt_state=new_critic_opt_state, step=1)

        return self.replace(actor_network=new_actor_network, critic_network=new_critic_network)

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=bool(self.config.encoder))

    @classmethod
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config: Any, adv_clip_min=None, ex_images=None) -> 'OGPOLearnedLambdaSafeQIPMAgentSharedBase':
        """Initialize an OGPOLearnedLambdaSafeQIPMAgentSharedBase with network states."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        # derive dims
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        # full action dim for chunking
        if config.action_chunking:
            full_actions = jnp.concatenate([ex_actions] * config.horizon_length, axis=-1)
        else:
            full_actions = ex_actions
        full_act_dim = full_actions.shape[-1]
        config['full_act_dim'] = full_act_dim
        config.adv_clip_min = adv_clip_min
        config['proprio_dim'] = ex_observations.shape[-1]

        # --- v1 restrictions: combinations unverified with safety heads are
        # rejected fail-fast (docs/changes/2026-07-22-safe-qipm-safety-heads) ---
        assert config['no_of_safety_subtasks'] >= 1, (
            "safe_qipm requires no_of_safety_subtasks >= 1; use algo=qipm for a "
            "pure task-head agent"
        )
        assert config.get('safety_agg', 'sum') in ('sum', 'relu', 'biased_relu', 'min_relu', 'max_relu'), (
            f"safety_agg must be one of sum/relu/biased_relu/min_relu/max_relu, got {config.get('safety_agg')!r}"
        )
        assert not config.get('use_mip_q', False), \
            "safe_qipm v1 does not support use_mip_q (its four MIP-Q " \
            "scalarization branches in _compute_advantages are dead code and " \
            "were left summation-shaped: safety_aggregation_strategy='min_max' " \
            "is implemented on the reachable ensemble-Q paths only)"
        assert not config.get('separate_critic_per_subtask', False), \
            "safe_qipm v1 does not support separate_critic_per_subtask"
        assert not config.get('scalar_goal_conditioned_critic', False), \
            "safe_qipm v1 does not support scalar_goal_conditioned_critic"
        assert not config.get('goal_conditioning_mode'), \
            "safe_qipm v1 does not support goal conditioning"
        assert config.get('adv_strategy', 'vanilla') == 'vanilla', \
            "safe_qipm v1 supports only adv_strategy='vanilla' (the " \
            "conservative/penultimate full-ensemble branch in " \
            "_compute_advantages is dead code and was left summation-shaped: " \
            "safety_aggregation_strategy='min_max' is implemented on the " \
            "reachable aggregated path only)"
        assert config.get('td_loss_mode', 'gated') == 'gated', \
            "safe_qipm v1 supports only td_loss_mode='gated'"

        # Safety-scalarization strategy (2026-08-13). There is NO config-schema
        # or enum-validation layer (config_loader.py merges YAML + CLI
        # overrides verbatim), so this assert is the only guard against a typo:
        # an unrecognized value would otherwise fall through the `else` at each
        # scalarization site and silently run simple_summation.
        assert config['safety_aggregation_strategy'] in (
            'simple_summation', 'min_max', 'learned_lambda'), (
            f"unknown safety_aggregation_strategy="
            f"{config['safety_aggregation_strategy']!r}; expected "
            f"'simple_summation', 'min_max' or 'learned_lambda' "
            f"(docs/changes/2026-08-13-safety-aggregation-strategy/)"
        )

        # --- learned_lambda (2026-09-05) -------------------------------------
        # Replaces the fixed scalar safety_coeff with a learned, non-negative
        # (S_task, K) matrix of Lagrange multipliers, and drops the head
        # aggregation entirely. The asserts below are what makes the
        # `_zero_dual` placeholder in _compute_advantages unreachable rather
        # than a silent fallback: they reject exactly the branches that emit it.
        learned_lambda = config['safety_aggregation_strategy'] == 'learned_lambda'
        if learned_lambda:
            assert config.get('safety_agg', 'sum') == 'sum', (
                f"safety_aggregation_strategy='learned_lambda' consumes the K "
                f"safety heads RAW (that is the point of the design: the "
                f"multiplier matrix does the mixing) and never calls "
                f"_aggregate_safety_heads, so safety_agg="
                f"{config.get('safety_agg')!r} would be silently ignored. "
                f"Leave safety_agg='sum'."
            )
            assert not config.get('use_one_step_policy', False), (
                "safety_aggregation_strategy='learned_lambda' does not support "
                "use_one_step_policy: the FQL distillation Q-term in "
                "fql_helper.py carries its own scalarization, which has not "
                "been ported to the learned multipliers"
            )
            assert float(config['lambda_lr']) >= 0.0, (
                f"lambda_lr must be >= 0 (0 = frozen dual, the "
                f"baseline-equivalence test), got {config['lambda_lr']}"
            )
            assert float(config['lambda_max']) > 0.0, (
                f"lambda_max must be > 0, got {config['lambda_max']}"
            )
            # Lambda is bounded by projecting Theta into
            # [softplus^-1(lambda_floor), softplus^-1(lambda_max)] after each
            # dual step (see _maybe_update_lambda). Both ends must be strictly
            # positive: softplus^-1(0) is undefined, and a floor of exactly 0
            # would let a satisfied constraint's Theta run to -inf, which is the
            # windup that made the old output clip unrecoverable.
            assert 0.0 < float(config['lambda_floor']) < float(config['lambda_max']), (
                f"lambda_floor must satisfy 0 < floor < lambda_max, got "
                f"lambda_floor={config['lambda_floor']}, "
                f"lambda_max={config['lambda_max']}"
            )

            # Budget d_k. Expressed as an allowed per-step unsafe RATE rho so
            # it is directly comparable to the logged env/episode_unsafe_rate,
            # then converted to the units the cost critic actually predicts:
            # a discounted sum of (-unsafe steps per chunk) over horizon H.
            #     d = rho * H / (1 - gamma_safety^H)
            # d_k = 0 is a trap and is rejected below: at the TD fixed point
            # Q_k <= 0, so the slack (Q_k + d_k) would be <= 0 almost
            # everywhere, g_ik <= 0, and Lambda would ramp monotonically until
            # the policy froze. An explicit per-head `safety_budget` list
            # overrides the rate.
            n_safety = int(config['no_of_safety_subtasks'])
            explicit = config.get('safety_budget', None)
            if explicit is not None:
                budget = [float(b) for b in (
                    explicit if isinstance(explicit, (list, tuple)) else [explicit] * n_safety)]
                assert len(budget) == n_safety, (
                    f"safety_budget has {len(budget)} entries but "
                    f"no_of_safety_subtasks={n_safety}"
                )
            else:
                # safety_budget_rate may be a scalar (one rate for every
                # constraint) or a per-constraint list, in the SAME order as
                # the env's safety_constraint_names roster
                # (main.py:343; toolhang = joint_limits, velocity, collision,
                # workspace). A list of the wrong length is rejected rather
                # than broadcast -- a silently mis-aligned budget column would
                # price the wrong constraint and be nearly undetectable.
                rate = config['safety_budget_rate']
                rates = ([float(r) for r in rate]
                         if isinstance(rate, (list, tuple))
                         else [float(rate)] * n_safety)
                assert len(rates) == n_safety, (
                    f"safety_budget_rate has {len(rates)} entries but "
                    f"no_of_safety_subtasks={n_safety}; the list must be in "
                    f"safety_constraint_names order"
                )
                H = int(config['horizon_length'])
                gamma = float(config['safety_discount'])
                scale = H / (1.0 - gamma ** H)
                budget = [r * scale for r in rates]
            assert all(b > 0.0 for b in budget), (
                f"every safety budget must be > 0 (d_k=0 makes the constraint "
                f"unsatisfiable at the TD fixed point, where Q_k <= 0, so the "
                f"dual ramps without bound); got {budget}"
            )
            config['safety_budget_resolved'] = budget

            # null -> safety_coeff, so that at step 0 lambda(s) = safety_coeff
            # for every state (phi is a simplex) and Q_total is EXACTLY the
            # safety_agg='sum' baseline. That exact reduction is what makes the
            # 3-seed comparison against that baseline interpretable.
            if config.get('lambda_init', None) is None:
                config['lambda_init'] = float(config['safety_coeff'])
            assert float(config['lambda_init']) > 0.0, (
                f"lambda_init must be > 0 (softplus^-1 is undefined at 0), got "
                f"{config['lambda_init']}"
            )
            # Checked here, after lambda_init has been resolved from
            # safety_coeff. If init were below the projection floor the very
            # first dual step would clamp Lambda off its initialisation, which
            # would break the exact reduction to the fixed-coefficient baseline
            # that the equivalence test asserts.
            assert float(config['lambda_init']) >= float(config['lambda_floor']), (
                f"lambda_init={config['lambda_init']} is below "
                f"lambda_floor={config['lambda_floor']}: the projection would "
                f"move Lambda off its initialisation on the first dual step"
            )
        # ---------------------------------------------------------------------

        # The two safety knobs are orthogonal but not composable: 'min_max'
        # does its own head reduction from the raw member axis, so it never
        # calls _aggregate_safety_heads and would silently ignore safety_agg.
        assert not (
            config['safety_aggregation_strategy'] == 'min_max'
            and config.get('safety_agg', 'sum') != 'sum'
        ), (
            f"safety_aggregation_strategy='min_max' reduces the heads itself "
            f"and ignores safety_agg={config.get('safety_agg')!r}; leave "
            f"safety_agg='sum' with min_max, or use "
            f"safety_aggregation_strategy='simple_summation' to get the "
            f"relu/biased_relu/min_relu/max_relu head fusion."
        )

        # Categorical-critic support-grid guard (2026-08-02). Deliberately NOT
        # safe_qipm_hjb's blanket `critic_loss_type == 'mse'`: that assert
        # exists because positive HJ margins are unrepresentable when
        # q_max = 0, and q_max = 0 is exactly RIGHT for cost heads (the true
        # cost-Q is <= 0 identically). The real exposure here is the other
        # edge — a hlgauss support grid is linspace(q_min, q_max, num_bins)
        # SHARED across every head, so if q_min stops above the safety head's
        # floor the head saturates in the bottom bin and simply stops
        # reporting how much cost it has accumulated, with no error. At
        # robomimic's q_min = -1/(1-discount) that is fine for
        # discount = 0.999 (-1000 vs a -103.5 floor) and NOT fine for
        # discount = 0.99 (-100 vs -103.5).
        if config['critic_loss_type'] != 'mse':
            _safety_floor = -float(config['horizon_length']) / (
                1.0 - float(config['safety_discount']) ** config['horizon_length']
            )
            _q_min = config['q_min']
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

        # MIP config validation
        if config['policy_type'] == 'mip':
            assert 0.0 < config.mip_t_star < 1.0, "mip_t_star must be in (0, 1)"
            print(f"[MIP Mode] Using Minimal Iterative Policy with t*={config.mip_t_star}")
        else:
            print(f"[Flow Mode] Using Flow Matching with {config.flow_steps} steps")

        # encoders
        encoders = dict()
        # Vision path: VisionProprioEncoder for actor_obs/critic_obs == 'image'
        if config.actor_obs == 'image' or config.critic_obs == 'image':
            from ogpo.networks.encoders import VisionProprioEncoder
            assert ex_images is not None, "ex_images required when actor_obs or critic_obs is 'image'"
            vit_img_h, vit_img_w, vit_num_ch = ex_images.shape
            print(f"[Vision] ViT input: {vit_img_h}x{vit_img_w}, {vit_num_ch} channels")
        if config.actor_obs == 'image':
            actor_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['actor'] = VisionProprioEncoder(
                vit=actor_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
            )
            config['encoder'] = 'vision'
        if config.critic_obs == 'image':
            critic_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['critic'] = VisionProprioEncoder(
                vit=critic_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
            )
            config['encoder'] = 'vision'
        # Legacy encoder path (existing CNN/ViT encoders without Approach B)
        if config.encoder and config.actor_obs == 'state' and config.critic_obs == 'state':
            if 'vit' in config.encoder:
                encoder_module = vit_encoder_modules[config.encoder]
            else:
                encoder_module = encoder_modules[config.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Resolve backbone per-network (useTF is a convenience shortcut)
        a_backbone = config['actor_backbone']
        c_backbone = config['critic_backbone']
        if config.useTF:
            a_backbone = 'tf'
            c_backbone = 'tf'
        config['actor_backbone'] = a_backbone
        config['critic_backbone'] = c_backbone

        # --- Time embedding (sinusoidal or None for raw scalar) ---
        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(
                embed_dim=config.get('time_embedding_dim', 32),
            )
            print(f"[time_embedding] sinusoidal (dim={config.get('time_embedding_dim', 32)})")

        # --- Actor definition ---
        if a_backbone == 'tf':
            actor_def = ActorVectorFieldTF(
                hidden_dim=config.tf_pi_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=False,
                num_layers=config.tf_pi_layers,
                num_heads=config.tf_pi_heads,
                dropout_rate=config.tf_pi_dropout,
                use_denoiser=config.use_denoiser,
                encoder=encoders.get('actor'),
            )
        elif config.useSimBa:
            actor_def = ActorVectorFieldSimBa(
                hidden_dims=config['simba_actor_hidden_dims'],
                action_dim=full_act_dim,
                encoder=encoders.get('actor'),
                rs_norm_momentum=0.99,
                rs_norm_epsilon=1e-8,
                time_embedding=time_emb_module,
            )
        else:
            actor_def = ActorVectorField(
                hidden_dims=config.actor_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.actor_layer_norm,
                encoder=encoders.get('actor'),
                use_film=config.useFiLM,
                use_denoiser=config.use_denoiser,
                time_embedding=time_emb_module,
            )

        # --- Critic definition: single Value with shared backbone, multi-subtask output ---
        if config.get('use_mip_q', False):
            S = config['no_of_subtasks']
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
                    num_subtasks=S,
                )
                print(f"[QIPM MIP-Q Ensemble] EXPLICIT ensemble with {config.get('num_ensemble_members', 10)} networks, "
                      f"num_subtasks={S}, noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
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
                    num_subtasks=S,
                )
                print(f"[QIPM MIP-Q] IMPLICIT ensemble with {config.get('num_ensemble_members', 10)} noise samples, "
                      f"num_subtasks={S}, noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
        else:
            if config.get('scalar_goal_conditioned_critic', False):
                from ogpo.networks import ScalarGoalConditionedCritic
                print("[QIPM] Creating SCALAR goal-conditioned critic (single shared Q, S one-hot calls)")
                critic_def = ScalarGoalConditionedCritic(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    num_ensembles=config['num_qs'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config.get('num_bins', 256),
                    q_min=config.get('q_min', None),
                    q_max=config.get('q_max', None),
                    action_repeat=config.get('action_repeat', False),
                    use_film=config.useFiLM,
                    num_subtasks=config['no_of_subtasks'],
                    goal_embed_dim=config.get('scalar_critic_goal_embed_dim', 32),
                )
            elif config.get('separate_critic_per_subtask', False):
                from ogpo.networks import ValuePerSubtask
                print("[QIPM] Creating SEPARATE critic per subtask (vmapped Value, shared encoder)")
                critic_def = ValuePerSubtask(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    num_ensembles=config['num_qs'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config.get('num_bins', 256),
                    q_min=config.get('q_min', None),
                    q_max=config.get('q_max', None),
                    action_repeat=config.get('action_repeat', False),
                    use_film=config.useFiLM,
                    num_subtasks=config['no_of_subtasks'],
                )
            else:
                print("[safe_qipm] Creating shared-backbone critic: "
                      f"{config['no_of_subtasks']} task + "
                      f"{config['no_of_safety_subtasks']} safety heads")
                critic_def = Value(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    num_ensembles=config['num_qs'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config.get('num_bins', 256),
                    q_min=config.get('q_min', None),
                    q_max=config.get('q_max', None),
                    action_repeat=config.get('action_repeat', False),
                    use_film=config.useFiLM,
                    num_subtasks=config['no_of_subtasks'] + config['no_of_safety_subtasks'],
                )

        # --- Goal conditioning wrappers ---
        # When goal_conditioning_mode is set, wrap actor and critic so the goal
        # suffix (length = no_of_subtasks) is either projected or stripped.
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

        # Add batch dimension if missing
        ex_observations = ex_observations[None, ...]
        # Actor-side nets are BUILT at the sliced width so that any obs path
        # bypassing the _encode_for_actor funnel fails loudly with a shape
        # error instead of silently consuming the full observation.
        _aod = config['actor_obs_dim']
        if _aod is not None:
            assert 0 < _aod <= ex_observations.shape[-1], (
                f"actor_obs_dim={_aod} must be in (0, obs_dim="
                f"{ex_observations.shape[-1]}]"
            )
        ex_actor_observations = _slice_actor_obs(ex_observations, _aod)
        full_actions = full_actions[None, ...]
        ex_times = ex_times[None, ...]
        if ex_images is not None:
            ex_images = ex_images[None, ...]

        print(ex_observations.shape, full_actions.shape, ex_times.shape)

        # --- Build init args using dicts (ModuleDict Mapping dispatch) ---
        actor_nets = {'actor': actor_def, 'target_actor': copy.deepcopy(actor_def)}

        actor_init_kw = dict(
            observations=ex_actor_observations,
            actions=full_actions,
            times=ex_times,
            dt=jnp.ones_like(ex_times) if config.use_shortcut else None,
        )
        if config.use_denoiser:
            actor_init_kw['is_encoded'] = False
            actor_init_kw['return_denoiser'] = True
            if config['actor_backbone'] == 'tf':
                actor_init_kw['deterministic'] = True
        if config.actor_obs == 'image' and ex_images is not None:
            actor_init_kw['images'] = ex_images

        actor_args = {
            'actor': actor_init_kw,
            'target_actor': actor_init_kw,
        }

        critic_nets = {'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)}
        critic_init_kw = dict(observations=ex_observations, actions=full_actions)
        if config.critic_obs == 'image' and ex_images is not None:
            critic_init_kw['images'] = ex_images
        critic_args = {'critic': critic_init_kw, 'target_critic': critic_init_kw}

        if config.use_constant_noise:
            noise_def = None
        else:
            noise_def = NoiseInjectionNetwork(
                hidden_dims=config.noise_net_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.noise_net_layer_norm,
                min_noise_std=config.min_noise_std,
                max_noise_std=config.max_noise_std,
            )
            actor_nets['noise_net'] = noise_def
            actor_nets['target_noise_net'] = copy.deepcopy(noise_def)
            if config.encoder:
                if config.actor_obs == 'image':
                    vit = encoders['actor'].vit
                    img_dim = config.vit_img_proj_dim if config.vit_img_proj_dim > 0 else (vit.num_patches * vit.embed_dim)
                    state_dim = config.vit_state_proj_dim if config.vit_state_proj_dim > 0 else config.proprio_dim
                    ex_noise_obs_shape = (img_dim + state_dim,)
                elif 'vit' in config.encoder:
                    ex_noise_obs_shape = (encoders['actor'].num_patches * encoders['actor'].embed_dim,)
                else:
                    ex_noise_obs_shape = (encoders['actor'].mlp_hidden_dims[-1],)
                ex_noise_obs = jnp.zeros((1, *ex_noise_obs_shape))
            else:
                ex_noise_obs = ex_actor_observations
            actor_args['noise_net'] = (ex_noise_obs, ex_times)
            actor_args['target_noise_net'] = (ex_noise_obs, ex_times)

        # Create separate network definitions
        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)

        # --- Optimizers ---
        # Ensure all numeric values are Python scalars (not arrays)
        actor_lr = float(config['actor_lr'])
        critic_lr = float(config['critic_lr'])
        clip_grad_norm = float(config.clip_grad_norm)
        actor_weight_decay = float(config.actor_weight_decay)
        critic_weight_decay = float(config.critic_weight_decay)

        if config.use_constant_scheduler_for_bc:
            actor_lr_schedule = actor_lr  # constant for BC
        else:
            actor_lr_schedule = create_lr_schedule(
                scheduler_type=config.actor_scheduler,
                base_lr=actor_lr,
                warmup_steps=int(config.actor_warmup_steps),
                decay_steps=int(config.actor_decay_steps),
                end_value=float(config.actor_end_value)
            )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=config.critic_scheduler,
            base_lr=critic_lr,
            warmup_steps=int(config.critic_warmup_steps),
            decay_steps=int(config.critic_decay_steps),
            end_value=float(config.critic_end_value)
        )

        # Create separate optimizers with injected hyperparameters
        if config.use_muon:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=actor_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=actor_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=critic_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=critic_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
        else:
            @optax.inject_hyperparams
            def adam_optimizer(learning_rate, weight_decay):
                return optax.adamw(learning_rate, weight_decay=weight_decay, eps=1e-8)
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=actor_lr_schedule, weight_decay=actor_weight_decay)
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=critic_lr_schedule, weight_decay=critic_weight_decay)
            )

        # Initialize parameters
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # MIP-Q needs rng for initialization (ValueMIP/ValueMIPEnsemble can sample noise internally)
        if config.get('use_mip_q', False):
            critic_init_rng, target_critic_init_rng = jax.random.split(critic_rng)
            critic_args = {
                'critic': {**critic_args['critic'], 'rng': critic_init_rng},
                'target_critic': {**critic_args['target_critic'], 'rng': target_critic_init_rng},
            }

        if config.useSimBa:
            actor_variables = actor_net_def.init(actor_rng, **actor_args)
            critic_variables = critic_net_def.init(critic_rng, **critic_args)

            # Separate trainable params from non-trainable batch_stats
            actor_params = actor_variables.pop('params')
            actor_batch_stats = actor_variables.pop('batch_stats', None)
            critic_params = critic_variables.pop('params')
            critic_batch_stats = critic_variables.pop('batch_stats', None)

            # Create train states, now including the batch_stats
            actor_state = TrainState.create(
                apply_fn=actor_net_def.apply,
                params=actor_params,
                tx=actor_tx,
                batch_stats=actor_batch_stats
            )
            critic_state = TrainState.create(
                apply_fn=critic_net_def.apply,
                params=critic_params,
                tx=critic_tx,
                batch_stats=critic_batch_stats
            )
        else:
            actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
            critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
            # Create separate train states
            actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
            critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)

        # Initialize target networks
        actor_params = actor_state.params
        actor_params['modules_target_actor'] = actor_params['modules_actor']
        if not config.use_constant_noise:
            actor_params['modules_target_noise_net'] = actor_params['modules_noise_net']

        params = critic_state.params
        params['modules_target_critic'] = params['modules_critic']

        # FQL one-step policy. Built at the SLICED actor width
        # (ex_actor_observations, not ex_observations): every call site feeds
        # it _encode_for_actor output, which applies the actor_obs_dim slice —
        # a full-width build would shape-crash on first use under the
        # family-default proprio-widened obs (the trap qipm used to assert
        # off).
        one_step_state = create_one_step_network(
            config, encoders, full_act_dim, ex_actor_observations, ex_images,
            init_rng, clip_grad_norm, actor_weight_decay,
        )

        # --- Learned Lagrange multipliers -----------------------------------
        # Its OWN TrainState, not an entry in the actor's ModuleDict (the
        # sac.py `temp` pattern), because the actor optimizer runs a cosine
        # schedule decaying to actor_end_value by actor_decay_steps and shares
        # a clip_by_global_norm with the policy; a dual variable needs a flat,
        # unclipped step size over the whole run. `None` for every other
        # strategy, which is what keeps those arms structurally unchanged.
        lambda_state = None
        if config['safety_aggregation_strategy'] == 'learned_lambda':
            # Reuse init_rng rather than splitting `rng`. SoftplusMatrixParam's
            # init is deterministic (jnp.full), so the key is unused -- but
            # consuming the agent's rng stream here would shift every downstream
            # sample and destroy the bit-exact reduction to the sum baseline at
            # lambda_lr=0, which is this arm's correctness test.
            lambda_rng = init_rng
            lambda_def = SoftplusMatrixParam(
                shape=(int(config['no_of_subtasks']),
                       int(config['no_of_safety_subtasks'])),
                init_value=float(config['lambda_init']),
                # max_value stays None ON PURPOSE. An output clip zeroes
                # dLambda/dTheta in both directions, so a capped entry can never
                # relax again -- see SoftplusMatrixParam's docstring. The cap is
                # enforced by projecting Theta in _maybe_update_lambda instead,
                # which keeps the boundary gradient alive.
                max_value=None,
            )
            # ModuleDict dispatches an empty tuple as a no-arg call, the same
            # registration sac.py uses for its scalar temperature.
            lambda_net_def = ModuleDict({'safety_lambda': lambda_def})
            lambda_params = lambda_net_def.init(
                lambda_rng, safety_lambda=())['params']
            # Zero lr until lambda_warmup_steps, then flat lambda_lr. A
            # schedule rather than a branch because the update runs inside a
            # lax.scan; see _maybe_update_lambda.
            lambda_lr = float(config['lambda_lr'])
            lambda_warmup = int(config['lambda_warmup_steps'])
            if lambda_warmup > 0:
                lambda_schedule = optax.join_schedules(
                    schedules=[optax.constant_schedule(0.0),
                               optax.constant_schedule(lambda_lr)],
                    boundaries=[lambda_warmup],
                )
            else:
                lambda_schedule = optax.constant_schedule(lambda_lr)
            lambda_tx = optax.adam(learning_rate=lambda_schedule)
            lambda_state = TrainState.create(
                lambda_net_def, lambda_params, lambda_tx)

        # store dims
        config.ob_dims = ob_dims
        config.action_dim = action_dim

        print(config)
        return cls(rng, actor_state, critic_state, config,
                   one_step_network=one_step_state, lambda_network=lambda_state)
