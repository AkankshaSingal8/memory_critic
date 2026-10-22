"""Online RL training runner for Q-IPM algorithm."""

import copy
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.evaluation_qipm import evaluate_qipm, evaluate_qipm_parallel, DenseRewardEnvWrapper
from ogpo.utils.evaluation import _split_obs, capture_and_log_policy_videos
from ogpo.utils.main_helper import (
    LoggingHelper, save_rolling_checkpoint, create_success_buffer_batch,
    _collect_rng_states, _restore_rng_states, is_preemption_requested,
    PreemptionDuringEval, raise_if_preempted,
)
from ogpo.utils.mem_monitor import log_memory
from ogpo.agents.modules.maintenance import rebind_online_optimizers


def _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name,
                      safety_views=None):
    if n_eval_envs <= 1:
        return None
    flags = SimpleNamespace(
        n_eval_envs=n_eval_envs, eval_episodes=eval_episodes,
        discount=discount, run_name=run_name, env_name=env_name,
    )
    # Only set when given, so a FLAGS built without views is unchanged; the
    # eval reads it with getattr(..., None) (ogpo/utils/eval_safety.py).
    if safety_views is not None:
        flags.safety_views = safety_views
    return flags


def _run_qipm_evaluation(agent, eval_env, eval_envs, eval_flags, action_dim,
                          actor_fn, log_step, eval_episodes, run_name,
                          no_of_subtasks, discount, plot, subtask_discounts=None,
                          eval_rng=None, safety_views=None):
    if eval_flags is None:
        eval_info, _, _ = evaluate_qipm(
            agent=agent, env=eval_env, global_step=log_step,
            num_eval_episodes=eval_episodes, action_dim=action_dim,
            actor_fn=actor_fn, env_name=run_name,
            no_of_subtasks=no_of_subtasks, discount=discount, plot=plot,
            subtask_discounts=subtask_discounts,
            eval_rng=eval_rng,
            safety_views=safety_views,
        )
    else:
        if safety_views is not None:
            # The explicit kwarg wins on both branches; copy so the caller's FLAGS
            # is not mutated.
            eval_flags = copy.copy(eval_flags)
            eval_flags.safety_views = safety_views
        eval_info, _, _ = evaluate_qipm_parallel(
            envs=eval_envs, agent=agent, FLAGS=eval_flags,
            action_dim=action_dim, global_step=log_step,
            actor_fn=actor_fn, no_of_subtasks=no_of_subtasks, plot=plot,
            eval_rng=eval_rng,
        )
    return eval_info


def _reshape_for_utd(batch, utd_ratio, batch_size):
    return jax.tree.map(lambda x: x.reshape((utd_ratio, batch_size) + x.shape[1:]), batch)


def _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                        horizon_length, discount, offline_ratio, history_length=0):
    offline_batch_size = int(batch_size * offline_ratio)
    online_batch_size = batch_size - offline_batch_size

    offline_batch = None
    if offline_batch_size > 0:
        offline_batch = train_dataset.sample_sequence(
            offline_batch_size * utd_ratio,
            sequence_length=horizon_length, discount=discount,
            history_length=history_length)
        offline_batch = _reshape_for_utd(offline_batch, utd_ratio, offline_batch_size)

    online_batch = None
    if online_batch_size > 0:
        min_online_size = online_batch_size * utd_ratio * horizon_length
        if replay_buffer.size >= min_online_size:
            online_batch = replay_buffer.sample_sequence(
                online_batch_size * utd_ratio,
                sequence_length=horizon_length, discount=discount,
                history_length=history_length)
            online_batch = _reshape_for_utd(online_batch, utd_ratio, online_batch_size)
        else:
            online_batch_size = 0

    if offline_batch is not None and online_batch is not None:
        common_keys = set(offline_batch.keys()) & set(online_batch.keys())
        return jax.tree.map(
            lambda off, on: np.concatenate([off, on], axis=1),
            {k: offline_batch[k] for k in common_keys},
            {k: online_batch[k] for k in common_keys},
        )
    return offline_batch if offline_batch is not None else online_batch


def _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount,
                         history_length=0):
    batch = replay_buffer.sample_sequence(
        batch_size * utd_ratio, sequence_length=horizon_length, discount=discount,
        history_length=history_length)
    return _reshape_for_utd(batch, utd_ratio, batch_size)


def _sample_success_batch(replay_buffer, success_buffer_batch_size, utd_ratio,
                          horizon_length, discount):
    num_successful = replay_buffer._traj_success_mask[:replay_buffer.size].sum()
    min_required = success_buffer_batch_size * utd_ratio * horizon_length
    if num_successful < min_required:
        return None, False
    batch = create_success_buffer_batch(
        replay_buffer, success_buffer_batch_size * utd_ratio,
        horizon_length, discount)
    batch = _reshape_for_utd(batch, utd_ratio, success_buffer_batch_size)
    return batch, True


def _eval_start(start_training, bc_refine_steps, q_warmup_steps):
    """First online step that evaluates. Independent of the critic warmup, so a
    run with q_warmup_steps > 0 evaluates on the same schedule as one without
    and their curves line up; evaluations during the warmup show the policy the
    warmup leaves in place."""
    del q_warmup_steps
    return start_training + bc_refine_steps


def _resolve_rollout_methods(agent, config, best_of_n):
    """(main, warmup) names of the agent method that picks rollout actions.

    `main` acts from q_warmup_end on. `warmup` acts before it (data collection
    and the critic-only warmup): the same method unless
    config.warmup_behavior_policy == 'actor', which uses the frozen actor
    alone so the critic being trained cannot rank the candidates.
    """
    if config.get('use_one_step_for_rollouts', False) and hasattr(agent, 'sample_actions_one_step'):
        if best_of_n > 1 and hasattr(agent, 'sample_actions_one_step_BON'):
            main = 'sample_actions_one_step_BON'
        else:
            main = 'sample_actions_one_step'
    elif best_of_n > 1 and hasattr(agent, 'sample_actions_BON'):
        main = 'sample_actions_BON'
    else:
        main = 'sample_actions'
    policy = config.get('warmup_behavior_policy', 'bon')
    if policy not in ('bon', 'actor'):
        raise ValueError(f"warmup_behavior_policy must be 'bon' or 'actor', got {policy!r}")
    if policy != 'actor':
        return main, main
    # The actor-only warmup drops best-of-n, not the one-step policy: switching
    # families at q_warmup_end would silently change which network acts.
    one_step = main.startswith('sample_actions_one_step')
    return main, ('sample_actions_one_step' if one_step else 'sample_actions')


def run_qipm_online_rl(
    agent: Any,
    env: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    train_dataset: Dataset,
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    algo_name: str,
    online_steps: int,
    start_training: int,
    batch_size: int,
    horizon_length: int,
    discount: float,
    # Required, not defaulted: both are written into every rolling checkpoint
    # (dataset_rng into the rng payload, config_hash into the resume guard), and
    # a runner that could omit either would write a checkpoint the loader has to
    # refuse. dataset_rng is the Generator shared with the offline Dataset.
    dataset_rng: Any,
    config_hash: str,
    utd_warmup: int = 1,
    utd_online: int = 1,
    utd_q: Optional[int] = None,
    utd_pi: Optional[int] = None,
    offline_ratio: float = 0.0,
    warmup_offline_ratio: float = 0.0,
    bc_refine_steps: int = 0,
    q_warmup_steps: int = 0,
    use_success_buffer: bool = False,
    success_buffer_batch_size: int = 256,
    best_of_n: int = 1,
    eval_interval: int = 100000,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    video_env: Optional[Any] = None,
    video_interval: int = 0,
    video_max_attempts: int = 8,
    plot_q_vs_mc: bool = False,
    run_name: str = 'qipm',
    env_name: str = '',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    save_interval: int = 500000,
    preemption_backup_interval: int = -1,
    log_enabled: bool = True,
    sparse: bool = False,
    p_aug: Optional[float] = None,
    buffer_size: int = 2000000,
    seed: int = 0,
    is_resumed: bool = False,
    resumed_online_step: int = 0,
    resumed_episode_counter: int = 0,
    resumed_replay_buffer: Optional[ReplayBuffer] = None,
    start_step: int = 0,
    wandb_run_id: Optional[str] = None,
    resumed_rng_states: Optional[dict] = None,
    # QIPM-specific
    use_dense_rewards: bool = False,
    normalize_rewards: bool = False,
    no_of_subtasks: int = 3,
    no_of_safety_subtasks: int = 0,
    safety_constraint_names: Optional[list] = None,
    safety_signal: str = 'cost',
    viz_callback: Optional[Any] = None,
    **kwargs,
) -> Tuple[Any, int, Dict, ReplayBuffer]:

    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]
    has_images = 'images' in example_batch
    # subtask_status buffer width: task columns + trailing safety columns
    # (safe_qipm; 0 safety columns for every other algo)
    total_status_width = no_of_subtasks + no_of_safety_subtasks

    needs_offline_data = offline_ratio > 0.0
    needs_warmup_offline = warmup_offline_ratio > 0.0
    if not needs_offline_data and not needs_warmup_offline:
        train_dataset = None
        import gc; gc.collect()

    # Replay buffer with QIPM extra fields
    if is_resumed and resumed_replay_buffer is not None:
        replay_buffer = resumed_replay_buffer
    else:
        rb_template = dict(
            observations=example_batch['observations'],
            actions=example_batch['actions'],
            rewards=np.array(0.0), terminals=np.array(0.0),
            masks=np.array(1.0), next_observations=example_batch['observations'],
            subtask_status=np.zeros(total_status_width, dtype=np.float32),
        )
        if has_images:
            rb_template['images'] = example_batch['images']
            rb_template['next_images'] = example_batch['images']
        replay_buffer = ReplayBuffer.create(
            rb_template, buffer_size, numpy_rng=np.random.default_rng(seed + 30000))

    if p_aug is not None:
        replay_buffer.p_aug = p_aug

    episode_counter = resumed_episode_counter
    if resumed_episode_counter > 0:
        print(f"Restored episode counter: {resumed_episode_counter}")
    # What the trailing subtask_status columns carry (set by agent identity in
    # main.py). "cost" = 0/1 violation flags (safe_qipm); "margin" = signed
    # distances, higher = safer (safe_qipm_hjb). The two metric sets are
    # disjoint apart from episode_unsafe_any_*, which is defined identically in
    # both and is therefore THE cross-arm comparable.
    assert safety_signal in ('cost', 'margin'), (
        f"safety_signal must be 'cost' or 'margin', got {safety_signal!r}"
    )
    margin_mode = no_of_safety_subtasks > 0 and safety_signal == 'margin'
    # Per-constraint unsafe-step counters (safe_qipm), one entry per safety
    # column. In "single" mode there is one entry and it is the aggregate; in
    # "multi" the entries are ordered like safety_constraint_names.
    episode_unsafe_steps = np.zeros(max(no_of_safety_subtasks, 1), dtype=np.float64)
    # Steps flagged unsafe by the env aggregate. Comparable across both modes,
    # unlike the column sum (which counts a step twice if two constraints fire).
    episode_unsafe_any = 0.0
    # NOTE: `episode_unattributed_unsafe` was removed on 2026-08-05. It counted unsafe steps that
    # matched no configured constraint family — cost mass the per-constraint heads could not see.
    # safe-robosuite now decides `unsafe_step` as exactly `any(safety_violations)` (both from the
    # REQUEST, before filtering), so that state is unrepresentable and the counter was permanently
    # zero. The invariant is guarded upstream by
    # test_safety_constraint_logging.py::test_unsafe_step_is_exactly_the_or_of_the_latch, which is
    # mutation-verified. Do not re-add without first checking that contract still holds.
    # Margin-mode accumulators (all inert when margin_mode is False). Counting
    # is per-head where a per-head number is meaningful and aggregate where the
    # cross-arm comparison needs one number.
    episode_margin_min = np.full(max(no_of_safety_subtasks, 1), np.inf, dtype=np.float64)
    episode_margin_sum = np.zeros(max(no_of_safety_subtasks, 1), dtype=np.float64)
    episode_neg_margin_steps = np.zeros(max(no_of_safety_subtasks, 1), dtype=np.float64)
    # Steps where ANY head's margin went negative — the margin-side analogue of
    # the cost arm's "was this step unsafe", but derived from margins, not flags.
    episode_any_neg_margin = 0.0
    # Steps the env flagged unsafe while NO margin had gone negative (a margin
    # of exactly 0.0 — sitting on the envelope boundary — counts as not-negative
    # here, matching the neg-count definition below). Sizes the documented
    # lookahead lead: upstream's violation detectors predict a few steps ahead
    # while margins are state-only, so this is expected to be nonzero and is a
    # diagnostic, not an error.
    episode_unsafe_with_positive_margins = 0.0
    # Logger key suffixes. Names are used ONLY to build metric keys; nothing
    # downstream of the wrapper's append lets a name reach the TD math.
    safety_names = list(safety_constraint_names or [])
    if safety_names:
        assert len(safety_names) == no_of_safety_subtasks, (
            f"safety_constraint_names has {len(safety_names)} entries but "
            f"no_of_safety_subtasks={no_of_safety_subtasks}"
        )
    if margin_mode:
        # Margin mode is multi-only, so the roster is always resolved and every
        # per-head metric key is nameable — unconditional, unlike the cost path
        # (whose "single" arm legitimately has an empty roster).
        assert len(safety_names) == no_of_safety_subtasks, (
            f"safety_signal='margin' needs one constraint name per head, got "
            f"{len(safety_names)} names for {no_of_safety_subtasks} heads"
        )
    # + episode_counter: on a resume the interrupted episode's partial trajectory
    # is discarded, so re-running episode #episode_counter with the seed it was
    # first given continues the initial-state sequence instead of re-walking it
    # from 0. Identical to the old `seed + 10000` for a fresh run (counter == 0).
    raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
    ob, ob_images = _split_obs(raw_ob)
    trajectory_buffer = []
    action_queue = []

    # Reset whenever the ONLINE phase starts fresh. `not is_resumed` alone left a
    # BC-phase resume (resumed_online_step == 0) running the whole online phase on
    # the BC learning rate, silently, since BC preemption sets is_resumed too.
    if (not is_resumed or resumed_online_step == 0) and hasattr(agent, 'reset_optimizers_with_lr'):
        agent = agent.reset_optimizers_with_lr()
        print("Reset optimizer with online learning rate")
    elif hasattr(agent, 'reset_optimizers_with_lr'):
        # Mid-online resume: the checkpoint restored opt_state but not tx, so
        # the agent is on the BC schedule. Rebind tx; keep the Adam state.
        agent = rebind_online_optimizers(agent)

    # Resolve per-subtask discounts for evaluation MC returns
    subtask_discounts = None
    if config.get('subtask_based_discount', False) and config.get('subtask_discounts') is not None:
        subtask_discounts = list(config['subtask_discounts'])

    # Resolve separate Q/Pi UTD ratios (default to utd_online)
    eff_utd_q = utd_q if utd_q is not None else utd_online
    eff_utd_pi = utd_pi if utd_pi is not None else utd_online
    split_utd = eff_utd_q != eff_utd_pi
    if split_utd:
        has_critic_only = hasattr(agent, 'batch_critic_only_update')
        if eff_utd_pi > eff_utd_q:
            print(f"[UTD] WARNING: utd_pi={eff_utd_pi} > utd_q={eff_utd_q} not supported, "
                  f"using max={eff_utd_pi} for both")
            split_utd = False
        elif not has_critic_only:
            print(f"[UTD] WARNING: {algo_name} has no batch_critic_only_update, "
                  f"using max(utd_q={eff_utd_q}, utd_pi={eff_utd_pi}) for both")
            split_utd = False
        else:
            print(f"[UTD] Split updates: critic={eff_utd_q}, actor={eff_utd_pi} per env step")
    elif eff_utd_q != utd_online:
        print(f"[UTD] critic={eff_utd_q}, actor={eff_utd_pi} per env step")

    # SAC agents are Gaussian (not flow-based), so SDE eval is identical to ODE eval
    needs_dual_eval = not algo_name.startswith('sac')
    should_log_step_data = step_data_log_interval > 0
    q_warmup_end = start_training + bc_refine_steps + q_warmup_steps
    eval_start = _eval_start(start_training, bc_refine_steps, q_warmup_steps)

    log_step = start_step
    eval_step = 0
    update_info = {}

    # Value check, not key presence: a BC-phase checkpoint carries the same
    # payload with online_rng=None (bc_runner has no online key).
    if is_resumed and resumed_rng_states is not None and resumed_rng_states.get('online_rng') is not None:
        online_rng = resumed_rng_states['online_rng']
        eval_rng = resumed_rng_states['eval_rng']
        print("Restored online_rng and eval_rng from checkpoint")
    else:
        online_rng = jax.random.PRNGKey(seed + 40000)
        eval_rng = jax.random.PRNGKey(seed + 50000)

    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)

    _actor_method_name, _warmup_actor_method_name = _resolve_rollout_methods(
        agent, config, best_of_n)
    print(f"[rollout] steps < {q_warmup_end} (collection + critic-only warmup) act with "
          f"{_warmup_actor_method_name}; from {q_warmup_end} with {_actor_method_name}")

    start_online_step = 1
    if is_resumed and resumed_online_step > 0:
        start_online_step = resumed_online_step + 1
        print(f"Resuming online loop at step {start_online_step}")

    eval_info = {}
    mem_log_interval = max(eval_interval, 20000)
    # Success/failure video capture: fire at the first eval at/after each
    # video_interval boundary of the global log_step (threshold, not modulo, so
    # the video cadence need not divide the eval cadence).
    video_enabled = video_env is not None and video_interval > 0
    next_video_at = ((log_step // video_interval) + 1) * video_interval if video_enabled else None
    log_memory("online_start", step=log_step, logger=logger)

    for i in tqdm.tqdm(
        range(start_online_step, online_steps + 1),
        dynamic_ncols=True, desc="QIPM Online RL",
        initial=start_online_step - 1, total=online_steps,
    ):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Action chunking
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            act_method = (_warmup_actor_method_name if i < q_warmup_end
                          else _actor_method_name)
            action = getattr(agent, act_method)(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        # Environment step (dense reward configured persistently via configure_dense_reward)
        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        next_ob, next_ob_images = _split_obs(raw_next_ob)
        done = terminated or truncated

        # Get subtask status from info
        reward_info = info.get('reward_info', {})
        subtask_status = np.array(
            reward_info.get('subtask_status', [0] * total_status_width),
            dtype=np.float32)
        if no_of_safety_subtasks > 0:
            # Trailing entries are the per-step safety signal (safe_qipm
            # family), one column per always-active safety head.
            safety_cols = subtask_status[no_of_subtasks:]
            assert safety_cols.shape[0] == no_of_safety_subtasks, (
                f"env returned {safety_cols.shape[0]} safety columns but "
                f"{no_of_safety_subtasks} safety heads are configured"
            )
            # .get: only robosuite-family envs publish unsafe_step; the
            # safe_qipm family is robomimic-only today, but the runner is
            # shared with other envs.
            env_unsafe = bool(info.get('unsafe_step', False))
            if margin_mode:
                # Margins are signed distances: minimum, not sum, and "unsafe"
                # by margin means negative. Kept separate from the cost
                # accumulators so neither set is ever read in the wrong units.
                margin_min_this_step = float(safety_cols.min())
                np.minimum(episode_margin_min, safety_cols, out=episode_margin_min)
                episode_margin_sum += safety_cols
                neg = safety_cols < 0.0
                episode_neg_margin_steps += neg
                if neg.any():
                    episode_any_neg_margin += 1.0
                if env_unsafe:
                    episode_unsafe_any += 1.0
                    if not neg.any():
                        episode_unsafe_with_positive_margins += 1.0
            else:
                unsafe_this_step = float(safety_cols.sum())
                episode_unsafe_steps += safety_cols
                if env_unsafe:
                    episode_unsafe_any += 1.0

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
            subtask_status=subtask_status,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        trajectory_buffer.append(transition)

        if should_log_step_data and i % step_data_log_interval == 0:
            # Built OUTSIDE the try so a width/roster bug raises instead of being
            # swallowed; only the log call itself is best-effort.
            step_log = {
                'phase': 'online', 'reward': float(int_reward), 'done': float(done),
                'batch_obs_mean': float(np.mean(ob)), 'batch_obs_std': float(np.std(ob)),
                'batch_action_mean': float(np.mean(action)), 'batch_action_std': float(np.std(action)),
                'subtasks_completed': float(reward_info.get('subtasks_completed', 0)),
            }
            if no_of_safety_subtasks > 0:
                if margin_mode:
                    # NOTE: 'unsafe' here is the env's aggregate FLAG (0/1), not
                    # the cost arm's column sum — same key, different definition
                    # across arms. Compare arms on env/episode_unsafe_any_* (the
                    # one identically-defined pair), not on this.
                    step_log['unsafe'] = float(env_unsafe)
                    step_log['margin_min'] = margin_min_this_step
                    for k, name in enumerate(safety_names):
                        step_log[f'margin_{name}'] = float(safety_cols[k])
                else:
                    step_log['unsafe'] = unsafe_this_step
                    for k, name in enumerate(safety_names):
                        step_log[f'unsafe_{name}'] = float(safety_cols[k])
            try:
                logger.log(step_log, "step_data", step=log_step)
            except Exception:  # best-effort: step telemetry must not kill training
                pass

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9
            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
            if no_of_safety_subtasks > 0:
                ep_len = len(trajectory_buffer)
                # Scalar keys only — CsvLogger joins with ',' and does not quote,
                # so a list value would corrupt column alignment (and wandb
                # cannot chart it). Every key is emitted every episode, built
                # from the fixed roster, because CSV headers freeze on row 1.
                if margin_mode:
                    total_neg = float(episode_neg_margin_steps.sum())
                    safety_log = {
                        # Defined identically to the cost arm — THE cross-arm
                        # comparable, and the only key both arms emit.
                        'episode_unsafe_any_steps': episode_unsafe_any,
                        'episode_unsafe_any_rate': episode_unsafe_any / ep_len,
                        # Worst margin reached anywhere this episode, over all
                        # heads: the headline safety number under HJB semantics.
                        'episode_min_margin': float(episode_margin_min.min()),
                        # Summed over heads, so this "rate" EXCEEDS 1.0 when two
                        # heads go negative on the same step — same convention
                        # (and same caveat) as the cost arm's
                        # episode_unsafe_rate. episode_any_neg_margin_rate
                        # below is the one that lives in [0, 1].
                        'episode_neg_margin_steps': total_neg,
                        'episode_neg_margin_rate': total_neg / ep_len,
                        'episode_any_neg_margin_steps': episode_any_neg_margin,
                        'episode_any_neg_margin_rate': episode_any_neg_margin / ep_len,
                        # Flags lead margins by design (lookahead detectors vs
                        # state-only margins) — this sizes that lead rather than
                        # signalling a fault.
                        'episode_unsafe_with_positive_margins':
                            episode_unsafe_with_positive_margins,
                    }
                    for k, name in enumerate(safety_names):
                        safety_log[f'episode_min_margin_{name}'] = float(episode_margin_min[k])
                        safety_log[f'episode_mean_margin_{name}'] = float(
                            episode_margin_sum[k] / ep_len)
                        neg_k = float(episode_neg_margin_steps[k])
                        safety_log[f'episode_neg_margin_steps_{name}'] = neg_k
                        # Same denominator as the aggregate, so the per-constraint
                        # rates stay directly comparable to it.
                        safety_log[f'episode_neg_margin_rate_{name}'] = neg_k / ep_len
                else:
                    total_cost = float(episode_unsafe_steps.sum())
                    safety_log = {
                        'episode_unsafe_steps': total_cost,
                        'episode_unsafe_rate': total_cost / ep_len,
                        'episode_unsafe_any_steps': episode_unsafe_any,
                        'episode_unsafe_any_rate': episode_unsafe_any / ep_len,
                    }
                    for k, name in enumerate(safety_names):
                        steps_k = float(episode_unsafe_steps[k])
                        safety_log[f'episode_unsafe_steps_{name}'] = steps_k
                        # Same denominator as the aggregate, so the per-constraint
                        # rates stay directly comparable to it.
                        safety_log[f'episode_unsafe_rate_{name}'] = steps_k / ep_len
                logger.log(safety_log, "env", step=log_step)
                # Both accumulator sets reset unconditionally: one of them is
                # inert in either mode, and resetting only the active set would
                # leave a landmine if the other is ever read.
                episode_unsafe_steps[:] = 0.0
                episode_unsafe_any = 0.0
                episode_margin_min[:] = np.inf
                episode_margin_sum[:] = 0.0
                episode_neg_margin_steps[:] = 0.0
                episode_any_neg_margin = 0.0
                episode_unsafe_with_positive_margins = 0.0
            trajectory_buffer = []
            episode_counter += 1
            raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
            ob, ob_images = _split_obs(raw_ob)
            action_queue = []
        else:
            ob = next_ob
            ob_images = next_ob_images

        # Rolling checkpoint + preemption poll, ABOVE the two `continue`s below.
        # Behind them, a SIGTERM arriving during the buffer-fill phase
        # (start_training=10000 env steps) was silently dropped and the job died
        # uncheckpointed. Cost of the move: the checkpoint records the state
        # BEFORE this iteration's update, so a resume skips iteration i's
        # gradient step along with its already-discarded partial-episode
        # transition — one update, versus losing the whole run.
        if preemption_backup_interval > 0 and i % preemption_backup_interval == 0:
            print(f"Saving rolling checkpoint at step {log_step}")
            save_rolling_checkpoint(
                agent, replay_buffer, log_step, i, seed, env_name,
                wandb_run_id=wandb_run_id, algo_name=algo_name,
                rng_states=_collect_rng_states(online_rng, eval_rng, dataset_rng),
                episode_counter=episode_counter,
                config_hash=config_hash,
            )

        if is_preemption_requested():
            print(f"[PREEMPTION] Saving emergency checkpoint at step {log_step}")
            save_rolling_checkpoint(
                agent, replay_buffer, log_step, i, seed, env_name,
                wandb_run_id=wandb_run_id, algo_name=algo_name,
                rng_states=_collect_rng_states(online_rng, eval_rng, dataset_rng),
                episode_counter=episode_counter,
                config_hash=config_hash,
            )
            print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
            import sys
            sys.exit(0)

        # Training update
        if i < start_training:
            continue

        is_warmup = (i < bc_refine_steps or i < q_warmup_end)
        if is_warmup:
            utd_ratio = utd_warmup
        elif split_utd:
            utd_ratio = max(eff_utd_q, eff_utd_pi)
        else:
            utd_ratio = utd_online

        if replay_buffer.size < batch_size * utd_ratio * horizon_length:
            continue

        # Sample batch. history_length is non-zero only for agents that read
        # the replay history (temporal_bias_safe_qipm); 0 leaves every other
        # agent's batch unchanged.
        history_length = int(agent.config.get('safety_bias_window', 0) or 0)
        if is_warmup and needs_warmup_offline:
            batch = _sample_mixed_batch(
                replay_buffer, train_dataset, batch_size, utd_ratio,
                horizon_length, discount, warmup_offline_ratio, history_length)
        elif needs_offline_data:
            batch = _sample_mixed_batch(
                replay_buffer, train_dataset, batch_size, utd_ratio,
                horizon_length, discount, offline_ratio, history_length)
        else:
            batch = _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount,
                                         history_length)

        # Free dataset after warmup ends if only needed for warmup
        if not is_warmup and needs_warmup_offline and not needs_offline_data and train_dataset is not None:
            train_dataset = None
            needs_warmup_offline = False
            import gc; gc.collect()

        # Success buffer
        batch_success = None
        success_flag = False
        if use_success_buffer:
            batch_success, success_flag = _sample_success_batch(
                replay_buffer, success_buffer_batch_size, utd_ratio,
                horizon_length, discount)
        if batch_success is None:
            batch_success = batch
            success_flag = False

        # Phase dispatch
        if i < bc_refine_steps:
            agent, update_info["online_agent"] = agent.batch_bc_refine_update(batch_success)
        elif i < q_warmup_end:
            agent, update_info["online_agent"] = agent.batch_q_warmup_update(batch)
        elif split_utd:
            # Split: joint actor+critic on first eff_utd_pi slices, critic-only on the rest
            import jax as _jax
            batch_joint = _jax.tree.map(lambda x: x[:eff_utd_pi], batch)
            bs_joint = _jax.tree.map(lambda x: x[:eff_utd_pi], batch_success)
            agent, update_info["online_agent"] = agent.batch_update(batch_joint, bs_joint, success_flag)
            batch_extra = _jax.tree.map(lambda x: x[eff_utd_pi:eff_utd_q], batch)
            agent, critic_info = agent.batch_critic_only_update(batch_extra)
            update_info["online_agent"].update(critic_info)
        else:
            agent, update_info["online_agent"] = agent.batch_update(batch, batch_success, success_flag)

        jax.block_until_ready(agent)

        # Logging
        if i % log_interval == 0 and update_info:
            for key, info_dict in update_info.items():
                logger.log(info_dict, key, step=log_step)
            update_info = {}

        # Evaluation
        if i >= eval_start and (i == online_steps or (eval_interval > 0 and i % eval_interval == 0)):
            eval_step += 1

            # ODE evaluation
            ode_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            eval_rng, eval_key = jax.random.split(eval_rng)
            try:
                eval_info_ode = _run_qipm_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    ode_actor_fn, log_step, eval_episodes, run_name,
                    no_of_subtasks, discount,
                    plot=(eval_step % 2 == 0) and plot_q_vs_mc,
                    subtask_discounts=subtask_discounts,
                    eval_rng=eval_key,
                )
                subtask_str = " | ".join(
                    f"sub{s}={eval_info_ode.get(f'subtask_{s}_success_rate', 'N/A')}"
                    for s in range(no_of_subtasks) if f'subtask_{s}_success_rate' in eval_info_ode
                )
                subtask_msg = f" | {subtask_str}" if subtask_str else ""
                print(f"[EVAL ODE] step={i} | success={float(eval_info_ode.get('success', 0.0)):.4f}{subtask_msg}")
                logger.log(eval_info_ode, "eval", step=log_step)

                if video_enabled and log_step >= next_video_at:
                    capture_and_log_policy_videos(
                        video_env, ode_actor_fn, action_dim, log_step, logger,
                        video_frame_skip=video_frame_skip, max_attempts=video_max_attempts)
                    next_video_at = ((log_step // video_interval) + 1) * video_interval

                if needs_dual_eval:
                    sde_actor_fn = getattr(agent, _actor_method_name)
                    eval_rng, eval_key = jax.random.split(eval_rng)
                    eval_info_sde = _run_qipm_evaluation(
                        agent, eval_env, eval_envs, eval_flags, action_dim,
                        sde_actor_fn, log_step, eval_episodes, run_name,
                        no_of_subtasks, discount, plot=False,
                        subtask_discounts=subtask_discounts,
                        eval_rng=eval_key,
                    )
                    subtask_str_sde = " | ".join(
                        f"sub{s}={eval_info_sde.get(f'subtask_{s}_success_rate', 'N/A')}"
                        for s in range(no_of_subtasks) if f'subtask_{s}_success_rate' in eval_info_sde
                    )
                    subtask_msg_sde = f" | {subtask_str_sde}" if subtask_str_sde else ""
                    print(f"[EVAL SDE] step={i} | success={float(eval_info_sde.get('success', 0.0)):.4f}{subtask_msg_sde}")
                    try:
                        logger.log(eval_info_sde, "eval_sde", step=log_step)
                    except Exception:
                        pass

                if viz_callback is not None:
                    # Checked here because the callback's own best-effort
                    # except would swallow a preemption raised inside it.
                    raise_if_preempted()
                    try:
                        viz_callback(agent, eval_env, log_step, save_dir)
                    except PreemptionDuringEval:
                        raise
                    except Exception as _viz_err:
                        print(f"[viz_callback] error at step {log_step}: {_viz_err}")

                eval_info = eval_info_ode
            except PreemptionDuringEval as exc:
                # Eval is minutes long; raise_if_preempted aborts it so this
                # emergency write can still finish inside KillWait=120s.
                print(f"[PREEMPTION] {exc}")
                print(f"[PREEMPTION] Saving emergency checkpoint at step {log_step}")
                save_rolling_checkpoint(
                    agent, replay_buffer, log_step, i, seed, env_name,
                    wandb_run_id=wandb_run_id, algo_name=algo_name,
                    rng_states=_collect_rng_states(online_rng, eval_rng, dataset_rng),
                    episode_counter=episode_counter,
                    config_hash=config_hash,
                )
                print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
                import sys
                sys.exit(0)

        if i % mem_log_interval == 0:
            log_memory("online", step=log_step, logger=logger)

        if log_enabled and save_interval > 0 and i % save_interval == 0:
            save_agent(agent, save_dir, log_step)

    save_agent(agent, save_dir, log_step)
    print(f"QIPM Online RL training complete. Final step: {log_step}")
    return agent, log_step, eval_info, replay_buffer
