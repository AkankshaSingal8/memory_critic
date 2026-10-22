"""FQL (Flow Q-Learning) helper functions.

Shared between OGPOAgent and OGPOQIPMAgent. All functions are pure
(take agent as argument, return new state) for JAX/Flax compatibility.

The safe_qipm family's config keys (``safety_coeff``,
``safety_aggregation_strategy``) are read with brackets and ONLY inside the
``no_of_safety_subtasks > 0`` arms of the two scalarization sites, so ogpo and
qipm — which never declare them — keep the identical K=0 control flow.
"""
import copy
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from ogpo.networks.actors_fql import OneStepPolicy
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.maintenance import create_lr_schedule, create_optimizer
from ogpo.agents.modules.bc_helper import preprocess_actions
from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.agents.modules.q_helper import aggregate_q_values


def create_one_step_network(
    config: Any,
    encoders: Dict,
    full_act_dim: int,
    ex_observations: jnp.ndarray,
    ex_images: Optional[jnp.ndarray],
    init_rng: jnp.ndarray,
    clip_grad_norm: float,
    actor_weight_decay: float,
) -> Optional[TrainState]:
    """Create one-step policy network for FQL, or None if disabled."""
    if not config.get('use_one_step_policy', False):
        return None

    print("[One-Step Policy] Initializing for FQL distillation")

    one_step_def = OneStepPolicy(
        hidden_dims=config.get('one_step_hidden_dims', (512, 512, 512, 512)),
        action_dim=full_act_dim,
        layer_norm=config.get('one_step_layer_norm', False),
        encoder=encoders.get('actor'),
    )
    one_step_nets = {
        'one_step': one_step_def,
        'target_one_step': copy.deepcopy(one_step_def),
    }
    one_step_net_def = ModuleDict(one_step_nets)

    # Initialize with (observations, noise)
    ex_noise = jax.random.normal(init_rng, (1, full_act_dim))
    one_step_init_kw = dict(observations=ex_observations, noise=ex_noise, is_encoded=False)
    if config.get('actor_obs') == 'image' and ex_images is not None:
        one_step_init_kw['images'] = ex_images
    one_step_args = {
        'one_step': one_step_init_kw,
        'target_one_step': one_step_init_kw,
    }

    # Create optimizer (use actor config as defaults)
    one_step_lr = float(config.get('one_step_lr') or config['actor_lr'])
    one_step_scheduler = config.get('one_step_scheduler') or config.get('actor_scheduler', 'cosine')
    one_step_warmup = int(config.get('one_step_warmup_steps') or config.get('actor_warmup_steps', 2000))
    one_step_decay = int(config.get('one_step_decay_steps') or config.get('actor_decay_steps', 50000))
    one_step_end = float(config.get('one_step_end_value') or config.get('actor_end_value', 2e-5))

    one_step_lr_schedule = create_lr_schedule(
        scheduler_type=one_step_scheduler,
        base_lr=one_step_lr,
        warmup_steps=one_step_warmup,
        decay_steps=one_step_decay,
        end_value=one_step_end,
    )
    one_step_tx = create_optimizer(
        lr_schedule=one_step_lr_schedule,
        clip_grad_norm=clip_grad_norm,
        weight_decay=actor_weight_decay,
    )

    one_step_rng, _ = jax.random.split(init_rng)
    one_step_params = one_step_net_def.init(one_step_rng, **one_step_args)['params']
    one_step_state = TrainState.create(one_step_net_def, one_step_params, one_step_tx)

    # Initialize target = online
    one_step_state.params['modules_target_one_step'] = one_step_state.params['modules_one_step']

    return one_step_state


def one_step_distillation_loss(
    agent: Any,
    batch: dict,
    grad_params: Any,
    rng: jnp.ndarray,
    use_q_loss: float = 1.0,
) -> Tuple[jnp.ndarray, dict]:
    """FQL loss: -Q(s, μ_w(s,z)) + α * ||μ_w(s,z) - ODE(θ, z)||².

    Args:
        agent: OGPOAgent or OGPOQIPMAgent with one_step_network.
        batch: Training batch.
        grad_params: Gradient parameters for one_step_network.
        rng: PRNG key.
        use_q_loss: 1.0 for online RL, 0.0 for BC-only training.
    """
    batch_size = batch['observations'].shape[0]
    rng, noise_rng, ode_rng, q_rng = jax.random.split(rng, 4)

    batch_actions = preprocess_actions(batch, agent.config.get('action_chunking', True))
    act_dim = batch_actions.shape[-1]

    images = batch.get('images')
    obs_encoded_actor = agent._encode_for_actor(batch['observations'], images=images, params=grad_params)

    # Sample noise
    noise = jax.random.normal(noise_rng, (batch_size, act_dim))

    # One-step policy prediction
    actions_one_step = agent.one_step_network.select('one_step')(
        obs_encoded_actor, noise, is_encoded=True, params=grad_params
    )

    # === Q-Maximization Loss ===
    obs_encoded_critic = agent._encode_for_critic(batch['observations'], images=images, use_target=True)
    is_enc = bool(agent.config.get('encoder'))

    q_values = agent.critic_network.select('target_critic')(
        obs_encoded_critic, actions=actions_one_step, is_encoded=is_enc
    )

    if q_values.ndim == 3:
        # QIPM critic: (S, M, B). Aggregate ensemble first, then apply gate
        # weights per subtask, then sum. Mirrors sac_qipm.actor_loss so
        # advantage_gating_mode propagates to the one-step policy.
        from ogpo.agents.modules.qipm_helper import compute_gate_weights
        # (S, M, B) -> (M, S, B) so aggregate_q_values treats M as ensemble axis
        q_values_t = jnp.transpose(q_values, (1, 0, 2))
        # return_indices exposes the member draw (None unless
        # q_agg='subsample') so the min_max safety term below can reduce over
        # exactly the members the task term used — value- and rng-identical to
        # the flagless call for every consumer, K=0 included.
        q_per_subtask, subsample_idxs = aggregate_q_values(
            q_values_t,
            method=agent.config.get('q_agg', 'mean'),
            rng=q_rng,
            num_qs=agent.config.get('num_qs', 10),
            return_indices=True,
        )  # (S, B), (2,) | None

        subtask_status = batch.get('subtask_status')
        if subtask_status is None:
            raise ValueError(
                "FQL Q-max on QIPM critic requires batch['subtask_status']."
            )
        if subtask_status.ndim == 3:
            subtask_status = subtask_status[:, -1, :]  # (B, S)
        # Safe-family critics carry K trailing safety cost heads
        # (no_of_safety_subtasks; absent => 0 => the original all-task
        # behavior, bit-identical). Safety heads must never enter
        # compute_gate_weights — it force-sets rows by position — and are
        # added un-gated (always active), weighted by safety_coeff,
        # mirroring safe_qipm's advantage scalarization. Gates read TASK
        # status columns only: offline Datasets carry task-width status
        # (no safety columns), which is also what lets the offline
        # distillation pass run at all.
        n_safety = agent.config.get('no_of_safety_subtasks', 0)
        n_task = q_per_subtask.shape[0] - n_safety
        gate_weights = compute_gate_weights(
            subtask_status[:, :n_task],
            gating_mode=agent.config.get('advantage_gating_mode', 'grandparent_future'),
            dag=agent.config.get('subtask_dag'),
        )  # (S_task, B)
        q_aggregated = (q_per_subtask[:n_task] * gate_weights).sum(axis=0)  # (B,)
        if n_safety > 0:
            # How the K heads collapse is safety_aggregation_strategy's call
            # (2026-08-14, docs/changes/2026-08-14-fql-minmax-aggregation/;
            # semantics from the 08-13 record). Bracket read, and only here
            # under K > 0: ogpo/qipm never declare the key, and a safe-family
            # config missing it must fail loudly instead of silently running
            # summation — the rule safety_coeff already follows.
            # WARNING: safe_qipm_hjb INHERITS the key from safe_qipm.yaml and
            # its create() one-step guard is the only thing keeping this
            # branch dead for it. min_max must never be applied to HJB heads:
            # their margins go positive by design, so every member fails
            # Q <= 0 and the term collapses to a constant 0.0 ("always safe").
            if agent.config['safety_aggregation_strategy'] == 'min_max':
                from ogpo.agents.modules.qipm_helper import _aggregate_safety_min_max
                # RAW (S, M, B) critic output, NOT the (M, S, B) transpose:
                # the helper wants the members on axis 1. subsample_idxs
                # addresses axis 0 of the transpose == axis 1 here, i.e. the
                # member pair the TASK reduction consumed (Decision 9); None
                # under q_agg mean/min, where both terms see the full ensemble.
                q_safety = q_values[n_task:]                    # (K, M, B)
                if subsample_idxs is not None:
                    q_safety = q_safety[:, subsample_idxs, :]   # (K, 2, B)
                q_aggregated = q_aggregated + agent.config['safety_coeff'] * \
                    _aggregate_safety_min_max(q_safety)
            else:
                q_aggregated = q_aggregated + agent.config['safety_coeff'] * q_per_subtask[n_task:].sum(axis=0)
    else:
        q_aggregated = aggregate_q_values(
            q_values,
            method=agent.config.get('q_agg', 'mean'),
            rng=q_rng,
            num_qs=agent.config.get('num_qs', 10),
        )

    q_loss = -jnp.mean(q_aggregated) * use_q_loss

    # === Distillation Loss ===
    def actor_fn(obs, actions, t, is_encoded=True):
        return agent.actor_network.select('target_actor')(obs, actions, t, is_encoded=is_encoded)

    actions_flow_ode = sample_flow_actions_ode(
        actor_fn=actor_fn,
        observations=obs_encoded_actor,
        rng=ode_rng,
        flow_steps=agent.config.get('flow_steps', 10),
        act_dim=act_dim,
        act_min=agent.config.get('act_min', -1),
        act_max=agent.config.get('act_max', 1),
        clip_intermediate=agent.config.get('clip_intermediate_actions', True),
        clip_value=agent.config.get('denoised_clip_value', 1.0),
        noises=noise,  # Same noise for fair comparison
        is_encoded=True,
    )
    actions_flow_ode = jax.lax.stop_gradient(actions_flow_ode)

    distill_loss = jnp.mean(jnp.square(actions_one_step - actions_flow_ode))

    # === Total Loss ===
    fql_distillation_coeff = agent.config.get('fql_distillation_coeff', 10.0)
    total_loss = q_loss + fql_distillation_coeff * distill_loss

    info = {
        'one_step_total_loss': total_loss,
        'one_step_q_loss': q_loss,
        'one_step_distill_loss': distill_loss,
        'one_step_q_mean': q_aggregated.mean(),
        'one_step_actions_mean': actions_one_step.mean(),
        'one_step_actions_std': actions_one_step.std(),
        'flow_ode_actions_mean': actions_flow_ode.mean(),
        'fql_distillation_coeff': fql_distillation_coeff,
    }

    return total_loss, info


def sample_one_step(
    agent: Any,
    observations: jnp.ndarray,
    rng: jnp.ndarray,
    is_encoded: bool = False,
    images: jnp.ndarray = None,
) -> jnp.ndarray:
    """Sample actions via one-step policy (direct noise → action mapping)."""
    if agent.one_step_network is None:
        return agent.sample_actions(observations, rng, is_encoded, images)

    add_batch_dim = False
    if not is_encoded:
        if observations.ndim == 1:
            add_batch_dim = True
            observations = observations[None]
            if images is not None:
                images = images[None]

    obs_encoded = agent._encode_for_actor(observations, images=images, use_target=True) if not is_encoded else observations

    act_dim = agent.config.action_dim * (agent.config.horizon_length if agent.config.get('action_chunking', True) else 1)
    noise = jax.random.normal(rng, (obs_encoded.shape[0], act_dim))

    actions = agent.one_step_network.select('target_one_step')(
        obs_encoded, noise, is_encoded=True
    )
    actions = jnp.clip(actions, agent.config.get('act_min', -1), agent.config.get('act_max', 1))

    if add_batch_dim:
        actions = actions[0]
    return actions


def sample_one_step_BON(
    agent: Any,
    observations: jnp.ndarray,
    rng: jnp.ndarray,
    images: jnp.ndarray = None,
) -> jnp.ndarray:
    """Best-of-N action selection using one-step policy.

    On a QIPM critic the candidate score depends on
    ``safety_aggregation_strategy`` (safe family, K > 0 only), and the two
    strategies differ in the TASK term as well — a consequence of this
    helper's established sum-heads-before-ensemble-min order, unlike
    safe_qipm's own BoN which mins first under both strategies:

      - "simple_summation": ``min_m ( sum_task Q_s + coeff * sum_K Q_k )``
        — the pre-2026-08-14 formula, kept verbatim.
      - "min_max": ``min_m ( sum_task Q_s ) + coeff * min_k max_{m : Q<=0} Q_k``
        — the member reduction still runs over the task sum alone
        (docs/changes/2026-08-14-fql-minmax-aggregation/, D4).

    Under ``subsample_bon`` the member pair is drawn PER (b, n) CELL here
    (``idxs`` is (2, B, N)), not once globally as in safe_qipm's BoN; min_max
    reuses that one draw for the safety heads so both terms read the same
    members (Decision 9).
    """
    if agent.one_step_network is None:
        return agent.sample_actions_BON(observations, rng, images)

    is_single_obs = observations.ndim == 1
    if is_single_obs:
        observations = observations[None]
        if images is not None:
            images = images[None]

    batch_size = observations.shape[0]
    num_samples = agent.config.get('best_of_n', 8)
    act_dim = agent.config.action_dim * (agent.config.horizon_length if agent.config.get('action_chunking', True) else 1)

    # Expand observations to B*N
    obs_expanded = jnp.repeat(observations, num_samples, axis=0)
    images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None

    obs_encoded_actor = agent._encode_for_actor(obs_expanded, images=images_expanded, use_target=True)

    # Generate B*N noise samples and forward pass
    rng, noise_rng = jax.random.split(rng)
    noise_samples = jax.random.normal(noise_rng, (batch_size * num_samples, act_dim))

    sampled_actions = agent.one_step_network.select('target_one_step')(
        obs_encoded_actor, noise_samples, is_encoded=True
    )
    sampled_actions = jnp.clip(sampled_actions, agent.config.get('act_min', -1), agent.config.get('act_max', 1))

    # Evaluate Q-ensemble
    critic_obs = agent._encode_for_critic(obs_expanded, images=images_expanded, use_target=True)
    q_ensemble = agent.critic_network.select('target_critic')(
        critic_obs, actions=sampled_actions, is_encoded=bool(agent.config.get('encoder'))
    )

    # Handle QIPM multi-subtask: critic returns (S, M, B*N) — sum over
    # subtasks (axis=0). Safe-family safety heads (trailing K rows) join the
    # sum weighted by safety_coeff, un-gated — the
    # safe_qipm.sample_actions_BON inference convention. The helper's
    # sum-before-ensemble-min order is deliberately kept (its established
    # convention; safe_qipm's own BoN mins first). n_safety absent => 0 =>
    # plain all-head sum, bit-identical for ogpo/qipm.
    # Under safety_aggregation_strategy='min_max' (2026-08-14,
    # docs/changes/2026-08-14-fql-minmax-aggregation/) the safety heads leave
    # that sum: they are held aside as raw members and reduced by
    # _aggregate_safety_min_max after the task term's ensemble min, so the
    # member reduction covers the TASK sum alone (D4). Bracket read of the
    # strategy, short-circuited behind K > 0 so ogpo/qipm never touch the key.
    num_qs = agent.config.get('num_qs', 10)   # one read: both reshapes below
    safety_members = None                     # min_max only: raw (K, M, B, N)
    if q_ensemble.ndim == 3:
        n_safety = agent.config.get('no_of_safety_subtasks', 0)
        n_task = q_ensemble.shape[0] - n_safety
        if n_safety > 0 and agent.config['safety_aggregation_strategy'] == 'min_max':
            # (K, M, B*N) -> (K, M, B, N): the candidate axis is flat and
            # b-major (jnp.repeat above), so this is a reshape, never a
            # transpose — the same convention the task reshape below uses.
            safety_members = q_ensemble[n_task:].reshape(
                n_safety, num_qs, batch_size, num_samples)
            q_ensemble = q_ensemble[:n_task].sum(axis=0)   # (M, B*N)
        else:
            q_total = q_ensemble[:n_task].sum(axis=0)
            if n_safety > 0:
                q_total = q_total + agent.config['safety_coeff'] * q_ensemble[n_task:].sum(axis=0)
            q_ensemble = q_total

    # Reshape to (num_qs, B, N)
    q_values = q_ensemble.reshape(num_qs, batch_size, num_samples)

    # Aggregate (min across ensemble, optionally subsampled)
    if agent.config.get('subsample_bon', False):
        rng, sub_key = jax.random.split(rng)
        idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, num_qs)
        qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)
        min_qs = jnp.min(qs_sub, axis=0)
        if safety_members is not None:
            # ONE draw for both terms (Decision 9): idxs[None] is (1, 2, B, N)
            # and broadcasts over the K axis -> (K, 2, B, N). min_max never
            # draws its own members, so the rng stream is untouched.
            safety_members = jnp.take_along_axis(safety_members, idxs[None], axis=1)
    else:
        min_qs = jnp.min(q_values, axis=0)

    if safety_members is not None:
        from ogpo.agents.modules.qipm_helper import _aggregate_safety_min_max
        q_scores = min_qs + agent.config['safety_coeff'] * _aggregate_safety_min_max(safety_members)
    else:
        q_scores = min_qs   # summation: no arithmetic here, bit-identical

    # Select best action per observation
    best_idx = jnp.argmax(q_scores, axis=1)
    actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
    best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)

    if is_single_obs:
        best_actions = best_actions[0]
    return best_actions
