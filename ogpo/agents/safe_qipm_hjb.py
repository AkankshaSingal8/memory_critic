"""OGPO Q-IPM with always-active HJB (reachability) safety Q-heads.

A copy-fork of ogpo/agents/safe_qipm.py for the ``safe_qipm_hjb`` algorithm
(fork policy: docs/code/best_practices.md Decisions Q2 — the two agents are
compared head-to-head, so they must not drift through a shared base class).
Change record: docs/changes/2026-07-26-hjb-safety-heads/.

THE ONLY DELTAS vs safe_qipm (everything else is byte-identical, deliberately):

  1. ``_compute_safety_hjb_target`` (renamed from ``_compute_safety_td_target``
     because the math is incompatible) replaces the CMDP cost backup with the
     discounted Hamilton-Jacobi-Bellman reachability backup, composed stepwise
     backward over the H-step chunk.
  2. The safety columns of ``subtask_status`` carry signed MARGINS, not 0/1
     cost flags (env side: ``safety_signal='margin'``).
  3. ``safety_margin_scales`` — per-head divisors applied inside the backup
     only, to make heterogeneous physical units commensurable.
  4. Extra ``create()`` guards (multi mode, mse critic, dense rewards, scales).

Head semantics FLIP relative to safe_qipm: Q_safety is a discounted worst-case
future margin, so **HIGHER = SAFER** (safe_qipm's cost heads are lower = safer,
and are <= 0 identically). Consequences worth stating once:

  - The bound on the values is ASYMMETRIC — read the backup's docstring before
    relying on it. UPWARD it is per-backup and unconditional (`V_t <= m_t`
    whatever `next_q` is), which is why this agent needs no analogue of the
    zero upper cap safe_qipm had bolted on in 2026-07-27. DOWNWARD there is no
    per-backup bound; only the `gamma^H`-contraction pulls values back into the
    margin hull, and at the launcher's `safety_discount=0.999` with H=8 that is
    `gamma^H = 0.992` — about 0.8% per backup. Bounded, but slowly.
  - ``adv_clip_min`` (null by default) clips advantages one-sidedly FROM BELOW.
    Under cost semantics that trims the pessimistic tail; under margin
    semantics the pessimistic tail is the same side but now means "less safe",
    so the clip is more aggressive than it reads. Left unchanged; do not set it
    without re-deriving what it does here.
  - ``q_agg='min'`` over the critic ensemble WOULD be the conservative choice
    for these heads — the minimum of an ensemble of "worst-case future margin"
    estimates is the least-safe estimate. Note the toolhang launcher runs
    ``q_agg=mean``, so the TD bootstrap is NOT conservative in the shipped
    config; only Best-of-N is, and only because ``sample_actions_BON`` takes a
    hard ``.min()`` over the ensemble regardless of ``q_agg``.

Scalarization is UNCHANGED from safe_qipm (advantages, BoN, Q logging):

    Q_total = sum_task gate_s * Q_s + safety_coeff * sum_safety Q_hj_k

No sign surgery is needed anywhere: advantages are group-centered differences
and BoN takes an argmax, so "higher Q_total = better" already means "higher
margin = better". Task heads are untouched — same QIPM TD (+5 done / -1
pending, capped at V_subgoal), same gate weights from task columns only.

Margin source and column contract (MUST match across three files — a drift
here silently mis-attributes a constraint to the wrong critic head):

  - ``envs/robomimic_utils._safety_margin_columns`` reads
    ``info['safety_margins']`` in ``info['safety_constraints']`` order and
    appends it after the task columns, so ``batch['subtask_status']`` is
    (B, H, S_task + no_of_safety_subtasks).
  - ``safety_margin_scales`` in the YAML is a length-K vector in **that same
    roster order** (toolhang: joint_limits, velocity, collision, workspace).
    Reordering the controller's safety_config without reordering the scales
    would rescale the wrong heads — the env-side roster assert catches a
    changed roster, but nothing can catch a mis-ordered scales vector.
  - Units are raw and heterogeneous upstream: rad, rad/s, m, m. Positive =
    safe; ZERO IS THE CONFIGURED ENVELOPE BOUNDARY, not the physical failure
    surface. Margins are min-latched over the inner sim steps of one policy
    step. Contract of record:
    ../safe-robosuite/docs/changes/2026-07-26-safety-margin-logging/.

v1 restrictions (multi mode only, mse critic loss, dense rewards on, plus the
eight carried over from safe_qipm) are asserted fail-fast in ``create()``.
``no_of_subtasks`` keeps meaning the TASK count everywhere; the total head
count is derived at the seams, never stored.

This file is intentionally self-contained: no inheritance from OGPOQIPMAgent
or OGPOSafeQIPMAgentSharedBase, to avoid Flax PyTreeNode subclassing pitfalls.
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


def _compute_safety_hjb_target(
    subtask_status: jnp.ndarray,
    next_q: jnp.ndarray,
    masks: jnp.ndarray,
    discount,
    horizon_length: int,
    margin_scales=None,
) -> jnp.ndarray:
    """Discounted HJB (reachability) TD target for the always-active safety heads.

    Renamed from ``_compute_safety_td_target`` (safe_qipm) on purpose: same
    call site, incompatible math. The env wrapper appends per-step signed
    MARGINS as the trailing columns of ``subtask_status`` — positive = safe,
    zero = the configured envelope boundary, negative = violated; instantaneous,
    NOT ratcheting.

    Q_hj is the discounted worst-case future margin. The single-step discounted
    safety Bellman operator (Fisac et al. 2019) is

        V_t = (1 - gamma) * m_t + gamma * min(m_t, V_{t+1})

    and this composes it BACKWARD over the whole H-step chunk, so the chunk
    update as a whole is a gamma-contraction rather than a sum:

        V     <- masks * next_q + (1 - masks) * m_{H-1}   # bootstrap
        for t = H-1 .. 0:  V = (1 - gamma) * m_t + gamma * min(m_t, V)
        target = V

    (margins are 0-indexed over the chunk: m_0 is the step the target attaches
    to, m_{H-1} the last.)

    TWO PINNED PROPERTIES — the reason this agent exists:

    1. **Bounded above by the chunk's own margins, UNCONDITIONALLY.** Each
       iteration gives ``V_t <= (1-g)*m_t + g*m_t = m_t``, because the ``min``
       caps its second term at ``m_t`` — regardless of what ``V_{t+1}`` (and
       therefore ``next_q``) is. So the returned target is ``<= m_0`` even when
       the bootstrap is +1e9: positive bootstrap overestimation cannot
       propagate, and no zero upper cap has to be bolted on the way safe_qipm's
       was (2026-07-27). Downward the per-backup statement is weaker —
       ``V_t >= min(m_t, V_{t+1})``, so ONE backup does not clip a wildly
       negative bootstrap — and what rules divergence out on that side is the
       contraction: the composed chunk operator is a gamma^H-contraction in V,
       so iterating drives every fixed point into the convex hull of the
       observed margins.

       That downward guarantee is only as fast as `1 - gamma^H`, and it is NOT
       automatic: the `min` biases every backup downward, so if the pull-back is
       smaller than that bias the value ratchets away. MEASURED 2026-07-28 on
       GPU: at `safety_discount=0.999` (H=8 -> 0.80%/backup) the heads DIVERGE
       to -20..-60 against margin hulls of +-1 over four seeds; at 0.99
       (7.7%/backup) they converge and stay in the hull. The launcher therefore
       ships 0.99. Do not raise it without re-running that check —
       docs/changes/2026-07-26-hjb-safety-heads/VERIFICATION.md section D.4.
    2. **Absorbing at the running minimum.** If ``m_t <= V_{t+1}`` then
       ``min(m_t, V_{t+1}) = m_t`` and ``V_t = (1-g)*m_t + g*m_t = m_t``
       EXACTLY — a violation deep in the chunk is not diluted by the safe steps
       backed up before it. This is what makes the head learn "how close did we
       come", not "how often did we trip".

    Order matters: the recursion is not commutative in t (the ``min`` is
    absorbing), so the loop must run from the LAST chunk step to the first,
    matching ``sample_sequence``'s convention that index 0 is the transition
    the target is attached to.

    The terminal bootstrap is ``m_{H-1}``, NOT zero: under margin semantics zero
    means "sitting exactly on the envelope boundary", so zeroing terminals
    would drag every episode-ending head toward the boundary value. Chunks that
    straddle an episode boundary need no masking here — the critic loss
    multiplies by valid[..., -1], which is 0 for exactly those chunks
    (ogpo/utils/datasets.py sample_sequence); that factor is load-bearing.

    ``margin_scales`` divides the margins (and ONLY the margins) per head
    before the recursion, making rad / rad/s / m commensurable under a single
    scalar ``safety_coeff``. It is applied here, not in the env wrapper, so the
    replay buffer and every logged metric stay in raw physical units and the
    scales can be re-tuned without invalidating a rolling checkpoint. Note the
    recursion is positively homogeneous in the margins, so scaling all K
    entries by c scales the target by c — but ``next_q`` is NOT rescaled (it
    comes from a critic already trained in scaled units), which is why changing
    scales mid-run makes the bootstrap inconsistent.

    Args:
        subtask_status: (B, horizon_length, K) per-step signed margins.
        next_q: (K, B) aggregated next Q-values for the safety heads.
        masks: (B,) or (B, horizon_length) termination masks.
        discount: scalar or (K,) per-safety-head discount factors (gamma; note
            this is gamma itself, NOT gamma**H — the horizon enters through the
            composition, not through an exponent).
        horizon_length: action chunking horizon.
        margin_scales: optional length-K positive divisors, in roster order.
            None = raw margins.

    Returns:
        TD targets of shape (K, B).
    """
    chex.assert_shape(subtask_status, (None, horizon_length, None))
    if masks.ndim > 1:
        masks = masks[..., -1]

    # (B, H, K) -> (H, K, B): the recursion walks axis 0 in Python, and the
    # remaining (K, B) matches next_q / the returned target.
    margins = jnp.transpose(subtask_status, (1, 2, 0)).astype(jnp.float32)
    if margin_scales is not None:
        margins = margins / jnp.asarray(margin_scales, jnp.float32)[None, :, None]

    gamma = jnp.asarray(discount, jnp.float32)
    if gamma.ndim > 0:
        gamma = gamma[:, None]  # (K, 1)

    # Bootstrap: terminal chunks fall back to the last margin, not zero.
    # SELECT, do not blend: the multiplicative form `masks*next_q +
    # (1-masks)*m` leaks a non-finite next_q into the terminals it is supposed
    # to mask out, because IEEE gives 0.0*inf = 0.0*nan = nan. Nothing checks
    # next_q for finiteness (the env-side assert covers the MARGIN columns
    # only), so one poisoned critic element would nan the whole batch's
    # td_loss through the mean. masks are binary here (1 - float(terminated)).
    v = jnp.where(masks[None, :] > 0, next_q, margins[-1])  # (K, B)
    for t in range(horizon_length - 1, -1, -1):  # backward over the chunk
        m_t = margins[t]
        v = (1.0 - gamma) * m_t + gamma * jnp.minimum(m_t, v)
    return v


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


class OGPOSafeQIPMHJBAgentSharedBase(flax.struct.PyTreeNode):
    """OGPO Q-IPM with always-active safety Q-heads on a shared-backbone critic."""
    rng: Any
    actor_network: TrainState
    critic_network: TrainState  # Shared backbone with num_subtasks -- output shape (S, M, B)
    config: Any = nonpytree_field()
    one_step_network: Optional[TrainState] = None

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
        safety_coeff = self.config['safety_coeff']
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
                        q_safety = q_vals[S_task:].sum(axis=0)  # (M, B) — always active, never gated
                        return q_task + safety_coeff * q_safety
                    else:
                        if self.config['q_agg'] == 'min':
                            q_ensemble_agg = q_vals.min(axis=1)
                        elif self.config['q_agg'] == 'subsample':
                            subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                            q_ensemble_agg = q_vals[:, subsample_idxs, :].min(axis=1)
                        else:
                            q_ensemble_agg = q_vals.mean(axis=1)
                        q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)
                        q_safety = q_ensemble_agg[S_task:].sum(axis=0)
                        return q_task + safety_coeff * q_safety
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
                        q_safety = q_vals[S_task:].sum(axis=0)  # (N, B) — always active, never gated
                        return q_task + safety_coeff * q_safety
                    else:
                        if self.config['q_agg'] == 'min':
                            q_ensemble_agg = q_vals.min(axis=1)
                        elif self.config['q_agg'] == 'subsample':
                            subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                            q_ensemble_agg = q_vals[:, subsample_idxs, :].min(axis=1)
                        else:
                            q_ensemble_agg = q_vals.mean(axis=1)
                        q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)
                        q_safety = q_ensemble_agg[S_task:].sum(axis=0)
                        return q_task + safety_coeff * q_safety
        else:
            # Original ensemble-Q path
            def compute_q_for_group(actions_g, rng_g):
                q_vals = self.critic_network.select('target_critic')(
                    critic_obs_enc, actions_g, is_encoded=bool(self.config.encoder))
                # q_vals shape: (S, M, B) for batched critic

                if need_full_ensemble:
                    q_task = (q_vals[:S_task] * done_weights[:, None, :]).sum(axis=0)  # (M, B)
                    q_safety = q_vals[S_task:].sum(axis=0)  # (M, B) — always active, never gated
                    return q_task + safety_coeff * q_safety
                else:
                    if self.config['q_agg'] == 'min':
                        q_ensemble_agg = q_vals.min(axis=1)  # (S, B)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, self.config.num_qs)
                        q_ensemble_agg = q_vals[:, subsample_idxs, :].min(axis=1)  # (S, B)
                    else:
                        q_ensemble_agg = q_vals.mean(axis=1)  # (S, B)

                    q_task = (q_ensemble_agg[:S_task] * done_weights).sum(axis=0)  # (B,)
                    q_safety = q_ensemble_agg[S_task:].sum(axis=0)  # (B,)
                    return q_task + safety_coeff * q_safety

        subsample_rngs = jax.random.split(rng, actions.shape[0])
        vmapped_critic = jax.vmap(compute_q_for_group, in_axes=(0, 0))
        q_out = vmapped_critic(actions, subsample_rngs)

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

        return adv, adv_pre

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
                adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, subtask_status=subtask_status, images=images)

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
                adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, subtask_status=subtask_status, images=images)
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
        # at V_subgoal); safety heads use the discounted HJB reachability
        # backup via _compute_safety_hjb_target (per-backup bounded ABOVE by
        # the chunk's own margins, so no upper cap is needed; the downward
        # bound is the contraction, not per-backup — see its docstring).
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
        target_q_safety = _compute_safety_hjb_target(
            subtask_status=batch['subtask_status'][..., S_task:],
            next_q=next_q[S_task:],
            masks=masks,
            discount=safety_discount,
            horizon_length=self.config['horizon_length'],
            # Bracket access, not .get(): a config without the key is a foreign
            # YAML (safe_qipm's), and silently running on raw margins would put
            # the safety term ~100x below the task heads without a word.
            margin_scales=self.config['safety_margin_scales'],
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

    # --- Update methods ---

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
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
    def _update_offline_calql(agent, batch: dict) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
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
    def _update(agent, batch_tuple, success_flag) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
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

        # Combine info
        info = {**actor_info, **critic_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, critic_network=new_critic_state, one_step_network=new_one_step_state, rng=new_rng), info

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
    def _update_critic_only(agent, batch) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
        """Apply gradient update to critic network only."""
        new_rng, sample_rng = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @staticmethod
    def _update_actor_only(agent, batch_tuple, success_flag) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
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

        info = {**actor_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update_critic_only, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    batch_critic_only_update = batch_q_warmup_update

    @staticmethod
    def _bc_refine_update(agent, batch_success) -> Tuple['OGPOSafeQIPMHJBAgentSharedBase', dict]:
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
        Best action -> max ( sum_task ( min_M Q ) + safety_coeff * sum_safety ( min_M Q ) )
        (task heads unweighted as in qipm; safety heads always active, weighted
        by safety_coeff — safety-aware action selection at inference. Under HJB
        semantics the safety terms are worst-case future MARGINS, so this argmax
        already prefers the safer candidate with no sign change; the ensemble
        min over safety heads is the conservative aggregate.)
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

        # 3. min over ensemble axis=1 -> (S, B, N), sum over subtasks axis=0 -> (B, N)
        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2,), 0, num_ens)
            min_qs = q_ensemble[:, idxs, :, :].min(axis=1)  # (S, B, N)
        else:
            min_qs = q_ensemble.min(axis=1)  # (S, B, N)

        q_total = min_qs[:S_task].sum(axis=0) + safety_coeff * min_qs[S_task:].sum(axis=0)  # (B, N)

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
        safety-head sum (min or mean over the ensemble first).
        """
        if not is_encoded:
            observations = self._encode_for_critic(observations, images=images, use_target=True)
        actions = jnp.reshape(actions, (actions.shape[0], -1))
        qs = self._get_subtask_q_ensemble(observations, actions)
        # qs: (S, M, B)
        S_task = self.config['no_of_subtasks']
        safety_coeff = self.config['safety_coeff']
        if self.config['q_agg'] == 'min' or self.config['q_agg'] == 'subsample':
            q_agg = qs.min(axis=1)  # (S, B)
        else:
            q_agg = qs.mean(axis=1)  # (S, B)
        return q_agg[:S_task].sum(axis=0) + safety_coeff * q_agg[S_task:].sum(axis=0)

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

    def reset_optimizers_with_lr(self) -> 'OGPOSafeQIPMHJBAgentSharedBase':
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
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config: Any, adv_clip_min=None, ex_images=None) -> 'OGPOSafeQIPMHJBAgentSharedBase':
        """Initialize an OGPOSafeQIPMHJBAgentSharedBase with network states."""
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
        assert not config.get('use_mip_q', False), \
            "safe_qipm v1 does not support use_mip_q"
        assert not config.get('separate_critic_per_subtask', False), \
            "safe_qipm v1 does not support separate_critic_per_subtask"
        assert not config.get('scalar_goal_conditioned_critic', False), \
            "safe_qipm v1 does not support scalar_goal_conditioned_critic"
        assert not config.get('goal_conditioning_mode'), \
            "safe_qipm v1 does not support goal conditioning"
        assert config.get('adv_strategy', 'vanilla') == 'vanilla', \
            "safe_qipm v1 supports only adv_strategy='vanilla'"
        assert config.get('td_loss_mode', 'gated') == 'gated', \
            "safe_qipm v1 supports only td_loss_mode='gated'"
        assert not config.get('use_one_step_policy', False), \
            "safe_qipm v1 does not support the FQL one-step policy"

        # --- HJB-specific v1 restrictions (docs/changes/2026-07-26-hjb-safety-heads) ---
        assert config['safety_head_mode'] == 'multi', (
            "safe_qipm_hjb v1 supports only safety_head_mode='multi': the "
            "single-mode aggregate has no margin analogue (see "
            "envs/robomimic_utils.configure_safety_status)"
        )
        assert config['critic_loss_type'] == 'mse', (
            "safe_qipm_hjb v1 requires critic_loss_type='mse': the categorical "
            "(hlgauss) support grid is linspace(q_min, q_max) with q_max=0 for "
            "robomimic and is SHARED across all heads, so positive HJ margins "
            "are unrepresentable and the safety heads would silently saturate "
            "at 0 (see ogpo/utils/main_helper.get_q_bounds)"
        )
        assert config['use_dense_rewards'], (
            "safe_qipm_hjb requires use_dense_rewards=true: the safety columns "
            "are appended by the same wrapper path that dense rewards enable, "
            "so with it off subtask_status stays all-zero — which under MARGIN "
            "semantics reads as 'permanently on the envelope boundary' (a cost "
            "no-op for safe_qipm, a silent poisoning here)"
        )
        _scales = config['safety_margin_scales']
        if _scales is not None:
            # Roster order, one divisor per head — see the module docstring's
            # cross-file contract. Validated eagerly so a bad vector fails at
            # agent construction, not hours later inside a jitted backup.
            _scales = [float(s) for s in _scales]
            assert len(_scales) == config['no_of_safety_subtasks'], (
                f"safety_margin_scales has {len(_scales)} entries but "
                f"no_of_safety_subtasks={config['no_of_safety_subtasks']}"
            )
            assert all(np.isfinite(s) and s > 0.0 for s in _scales), (
                f"safety_margin_scales must be finite and strictly positive, "
                f"got {_scales} (they are DIVISORS; a zero or negative entry "
                f"would blow up or flip the margin sign)"
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
                print("[safe_qipm_hjb] Creating shared-backbone critic: "
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

        # FQL one-step policy
        one_step_state = create_one_step_network(
            config, encoders, full_act_dim, ex_observations, ex_images,
            init_rng, clip_grad_norm, actor_weight_decay,
        )

        # store dims
        config.ob_dims = ob_dims
        config.action_dim = action_dim

        print(config)
        return cls(rng, actor_state, critic_state, config, one_step_network=one_step_state)
