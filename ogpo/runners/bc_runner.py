"""BC and CalQL training runners."""

import glob
import threading
import queue
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple
import jax
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent, save_actor_checkpoint
from ogpo.utils.evaluation import evaluate, evaluate_parallel, capture_and_log_policy_videos
from ogpo.utils.main_helper import (
    LoggingHelper, is_preemption_requested, save_rolling_checkpoint,
    _collect_rng_states, PreemptionDuringEval,
)
from ogpo.utils.mem_monitor import log_memory


class _BatchPrefetcher:
    """Prefetch batches on a background thread to overlap CPU data loading with GPU compute.

    With state_fn, each queued item is (batch, state_fn() taken right after that
    batch was drawn), so the consumer knows the sampler position at the last
    batch it actually used, not the thread's (which runs up to maxsize+1 ahead).
    """

    def __init__(self, sample_fn, maxsize=2, state_fn=None):
        self._queue = queue.Queue(maxsize=maxsize)
        self._sample_fn = sample_fn
        self._state_fn = state_fn
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        item = None
        while not self._stop.is_set():
            try:
                if item is None:
                    batch = self._sample_fn()
                    item = (batch, self._state_fn() if self._state_fn is not None else None)
                self._queue.put(item, timeout=0.5)
                item = None  # successfully enqueued; sample a new one next iteration
            except queue.Full:
                continue  # retry putting the SAME batch, don't re-sample
            except Exception as e:
                self._error = e
                break

    def get(self):
        return self.get_with_state()[0]

    def get_with_state(self):
        if self._error is not None:
            raise self._error
        return self._queue.get()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)


def _generator_at(rng, state):
    """A new Generator of rng's bit-generator type, positioned at `state`."""
    g = np.random.Generator(type(rng.bit_generator)())
    g.bit_generator.state = state
    return g


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


def _run_evaluation(agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot, eval_rng=None):
    if eval_flags is None:
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, global_step=log_step,
            action_dim=action_dim, num_eval_episodes=eval_episodes,
            num_video_episodes=video_episodes, video_frame_skip=video_frame_skip,
            actor_fn=actor_fn, env_name=run_name, plot=False,
            eval_rng=eval_rng,
        )
    else:
        eval_info = evaluate_parallel(
            eval_envs, agent, eval_flags, action_dim, log_step,
            actor_fn=actor_fn, plot=False,
            eval_rng=eval_rng,
        )
    return eval_info


def _log_step_data(logger, batch, phase, log_step):
    try:
        logger.log({
            'phase': phase,
            'batch_obs_mean': float(batch['observations'].mean()),
            'batch_obs_std': float(batch['observations'].std()),
            'batch_action_mean': float(batch['actions'].mean()),
            'batch_action_std': float(batch['actions'].std()),
        }, "step_data", step=log_step)
    except Exception:
        pass


# Fixed key for the periodic BC-phase SDE eval: independent of eval_rng (which
# is checkpointed and consumed by the ODE eval) and of every training stream.
_SDE_EVAL_KEY = 1


def _safety_msg(info):
    """' | safety: v_episode_rate=..' suffix for the eval print, '' without safety keys."""
    parts = [f"{k[len('safety/'):]}={v:.3f}" for k, v in info.items()
             if k.startswith('safety/') and k.endswith('_episode_rate')]
    return f" | safety: {' '.join(parts)}" if parts else ""


_LOSS_KEYS = ('actor/bc_flow_loss', 'bc_flow_loss', 'actor_loss', 'bc_loss', 'loss')


def _find_loss_value(info):
    for k in _LOSS_KEYS:
        if k in info:
            return float(info[k])
    return None


def run_bc_training(
    agent: Any,
    train_dataset: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    offline_steps: int,
    horizon_length: int,
    discount: float,
    batch_size: int,
    # Required, not defaulted: both are written into every rolling checkpoint
    # (dataset_rng into the rng payload, config_hash into the resume guard), and
    # a caller that could omit either would write a checkpoint the loader has to
    # refuse. dataset_rng is the Generator this phase samples its batches with.
    dataset_rng: Any,
    config_hash: str,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    eval_interval_bc: int = 10000,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    video_env: Optional[Any] = None,
    video_interval: int = 0,
    video_max_attempts: int = 8,
    plot_q_vs_mc: bool = False,
    run_name: str = 'bc',
    log_interval: int = 5000,
    save_interval_bc: int = -1,
    preemption_backup_interval: int = -1,
    step_data_log_interval: int = 10,
    clip_bc: bool = False,
    clip_bc_threshold: float = 0.45,
    env_name: str = '',
    start_step: int = 0,
    log_to_file: Optional[str] = None,
    custom_eval_fn: Optional[Any] = None,
    seed: int = 0,
    algo_name: str = 'ogpo',
    wandb_run_id: Optional[str] = None,
    # Restored on a BC resume so the eval episode sequence continues instead of
    # restarting (the emergency saves already snapshot this key).
    resumed_eval_rng: Any = None,
    # Periodic SDE eval (agent.sample_actions) right after every ODE eval, logged
    # under "eval_sde" at the same step. It draws from its own fixed key, never
    # from eval_rng / the training streams, so the ODE eval and training are
    # bit-identical with it on or off.
    bc_sde_eval: bool = False,
    # {view: [constraint names]} (or a JSON string of one) for the per-floor
    # safety/ eval keys; see ogpo/utils/eval_safety.py. None = one "any" view.
    safety_views: Any = None,
    # Periodic save_interval_bc checkpoints hold only the actor network's params
    # (actor_<step>.pkl, flax_utils.save_actor_checkpoint) -- what the ODE and SDE
    # evals read. The final checkpoint stays a full params_<step>.pkl.
    save_actor_only: bool = False,
) -> Tuple[Any, int, Dict]:
    log_step = start_step
    eval_step = 0
    # Read, not sampled: Dataset.sample draws from dataset_rng, which a resume has
    # just restored, so a draw here would shift every post-resume batch by one.
    action_dim = train_dataset['actions'].shape[-1]
    dataset_idx = 0

    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name,
                                   safety_views=safety_views)
    # Restored on a BC resume: the emergency saves snapshot this key, and
    # rebuilding it from the seed would restart the eval episode sequence.
    eval_rng = (resumed_eval_rng if resumed_eval_rng is not None
                else jax.random.PRNGKey(seed + 50000))
    should_log_step_data = step_data_log_interval > 0
    # Success/failure video capture: fire at the first eval at/after each
    # video_interval boundary of the global log_step (threshold, not modulo, so
    # the video cadence need not divide the eval cadence).
    video_enabled = video_env is not None and video_interval > 0
    next_video_at = ((start_step // video_interval) + 1) * video_interval if video_enabled else None
    test_log_data = {'losses': [], 'eval_success': [], 'steps': []}
    eval_info = {}
    bc_mem_interval = max(eval_interval_bc, 50000) if eval_interval_bc > 0 else 50000

    def _emergency_exit_on_eval_preemption(exc: BaseException) -> None:
        print(f"[PREEMPTION] {exc}")
        print(f"[PREEMPTION] Saving emergency checkpoint at BC step {log_step}")
        _save_bc_rolling_checkpoint(log_step)
        print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
        import sys
        sys.exit(0)

    def _save_bc_rolling_checkpoint(step: int) -> None:
        """Rolling checkpoint for the BC phase.

        replay_buffer=None and online_loop_step=0 mark it as BC-phase for the
        resume path (main.py's bc_resumed branch, and the online runners'
        optimizer reset). online_rng=None for the same reason: this phase never
        steps a train env. Reads `agent`/`eval_rng` from the enclosing scope at
        CALL time, so it always checkpoints the current ones.
        """
        save_rolling_checkpoint(
            agent, None, step, 0, seed, env_name,
            wandb_run_id=wandb_run_id, algo_name=algo_name,
            # The dataset_rng position right after the last batch a train step
            # consumed, not the live Generator: the prefetcher has drawn ahead of
            # it, and a resume from its position would skip those batches.
            rng_states=_collect_rng_states(
                None, eval_rng, _generator_at(dataset_rng, consumed_dataset_state)),
            episode_counter=0,
            config_hash=config_hash,
        )

    log_memory("bc_start", step=log_step, logger=logger)

    # Sampler position after the last consumed batch; before any step, the
    # position the prefetcher starts from (which a resume has restored).
    consumed_dataset_state = dataset_rng.bit_generator.state
    prefetcher = _BatchPrefetcher(
        lambda: train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount),
        maxsize=2,
        state_fn=lambda: dataset_rng.bit_generator.state,
    )

    try:
        for i in tqdm.tqdm(range(1, offline_steps + 1), dynamic_ncols=True, desc="BC Training"):
            log_step += 1

            batch, consumed_dataset_state = prefetcher.get_with_state()
            agent, offline_info = agent.bc_update(batch)

            if should_log_step_data and log_step % step_data_log_interval == 0:
                _log_step_data(logger, batch, 'bc', log_step)

            if log_step % log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)
                if log_to_file is not None:
                    loss_val = _find_loss_value(offline_info)
                    if loss_val is not None:
                        test_log_data['losses'].append(loss_val)
                        test_log_data['steps'].append(log_step)

            # eval_interval_bc < 0: never evaluate, not even at the final step
            # (evals run afterwards on CPU from the saved checkpoints,
            # scripts/safety_pretrain/eval_bc_run.py); eval_info stays {}.
            # 0: final step only; > 0: every eval_interval_bc steps and the final.
            if eval_interval_bc >= 0 and (
                    i == offline_steps
                    or (eval_interval_bc != 0 and log_step % eval_interval_bc == 0)):
                jax.block_until_ready(agent)
                eval_step += 1

                # ODE eval
                eval_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
                try:
                    if custom_eval_fn is not None:
                        eval_info = custom_eval_fn(
                            agent=agent, eval_env=eval_env, eval_envs=eval_envs,
                            eval_flags=eval_flags, action_dim=action_dim,
                            actor_fn=eval_actor_fn, log_step=log_step,
                            eval_episodes=eval_episodes, run_name=run_name,
                        )
                    else:
                        eval_rng, eval_key = jax.random.split(eval_rng)
                        eval_info = _run_evaluation(
                            agent, eval_env, eval_envs, eval_flags, action_dim,
                            eval_actor_fn, log_step, eval_episodes, video_episodes,
                            video_frame_skip, run_name, plot=False,
                            eval_rng=eval_key,
                        )
                except PreemptionDuringEval as exc:
                    print(f"[PREEMPTION] {exc}")
                    print(f"[PREEMPTION] Saving emergency checkpoint at BC step {log_step}")
                    _save_bc_rolling_checkpoint(log_step)
                    print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
                    import sys
                    sys.exit(0)
                success_rate = eval_info.get('success', 0.0)
                subtask_str = " | ".join(
                    f"sub{s}={eval_info.get(f'subtask_{s}_success_rate', 'N/A')}"
                    for s in range(10) if f'subtask_{s}_success_rate' in eval_info
                )
                subtask_msg = f" | {subtask_str}" if subtask_str else ""
                print(f"[EVAL ODE] step={i:6d} | success={success_rate:.4f} | return={eval_info.get('return', 0.0):.4f}{subtask_msg}{_safety_msg(eval_info)}", flush=True)
                logger.log(eval_info, "eval", step=log_step)

                # SDE eval: same eval path and start states (the env seeds come
                # from log_step), stochastic actor. Kept in its own variable:
                # eval_info / success_rate stay ODE for clip_bc, the test log
                # and the return value.
                # "ODE eval unchanged by the SDE eval" holds for the PARALLEL
                # eval (n_eval_envs > 1, resets seeded by global_step). The serial
                # path (n_eval_envs <= 1) resets unseeded, so with bc_sde_eval on
                # later ODE evals there see different start states.
                if bc_sde_eval:
                    try:
                        if custom_eval_fn is not None:
                            eval_info_sde = custom_eval_fn(
                                agent=agent, eval_env=eval_env, eval_envs=eval_envs,
                                eval_flags=eval_flags, action_dim=action_dim,
                                actor_fn=agent.sample_actions, log_step=log_step,
                                eval_episodes=eval_episodes, run_name=run_name,
                                eval_rng=jax.random.PRNGKey(_SDE_EVAL_KEY),
                            )
                        else:
                            eval_info_sde = _run_evaluation(
                                agent, eval_env, eval_envs, eval_flags, action_dim,
                                agent.sample_actions, log_step, eval_episodes,
                                0, video_frame_skip, run_name, plot=False,  # no SDE videos
                                eval_rng=jax.random.PRNGKey(_SDE_EVAL_KEY),
                            )
                    except PreemptionDuringEval as exc:
                        # The ODE point for this step is already logged; the
                        # checkpoint is at this step, so the resumed run starts
                        # after it and this step's SDE point is never re-run.
                        _emergency_exit_on_eval_preemption(exc)
                    sde_subtask_str = " | ".join(
                        f"sub{s}={eval_info_sde.get(f'subtask_{s}_success_rate', 'N/A')}"
                        for s in range(10) if f'subtask_{s}_success_rate' in eval_info_sde
                    )
                    sde_subtask_msg = f" | {sde_subtask_str}" if sde_subtask_str else ""
                    print(f"[EVAL SDE] step={i:6d} | success={eval_info_sde.get('success', 0.0):.4f} | return={eval_info_sde.get('return', 0.0):.4f}{sde_subtask_msg}{_safety_msg(eval_info_sde)}", flush=True)
                    logger.log(eval_info_sde, "eval_sde", step=log_step)

                # Before the clip_bc break below, so a clipping run still emits a
                # video at the step it stops on. Capture aborts on SIGTERM
                # (it is minutes of rendered sim), and that abort has to reach
                # the emergency checkpoint rather than the process boundary.
                if video_enabled and log_step >= next_video_at:
                    try:
                        capture_and_log_policy_videos(
                            video_env, eval_actor_fn, action_dim, log_step, logger,
                            video_frame_skip=video_frame_skip, max_attempts=video_max_attempts)
                    except PreemptionDuringEval as exc:
                        print(f"[PREEMPTION] {exc}")
                        print(f"[PREEMPTION] Saving emergency checkpoint at BC step {log_step}")
                        _save_bc_rolling_checkpoint(log_step)
                        print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
                        import sys
                        sys.exit(0)
                    next_video_at = ((log_step // video_interval) + 1) * video_interval

                if log_to_file is not None:
                    test_log_data['eval_success'].append(float(success_rate))

                if clip_bc and success_rate >= clip_bc_threshold:
                    # Poll BEFORE the break: this is the one loop exit that skips
                    # the poll at the bottom of the iteration, so a SIGTERM
                    # landing on a clipping eval used to be dropped entirely.
                    if is_preemption_requested():
                        print(f"[PREEMPTION] Saving emergency checkpoint at BC step {log_step}")
                        _save_bc_rolling_checkpoint(log_step)
                        print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
                        import sys
                        sys.exit(0)
                    print(f"Breaking due to BC clipping (ODE success {success_rate:.4f} >= {clip_bc_threshold})")
                    save_agent(agent, save_dir, log_step)
                    break

            if save_interval_bc > 0 and log_step % save_interval_bc == 0:
                if save_actor_only:
                    save_actor_checkpoint(agent, save_dir, log_step)
                else:
                    save_agent(agent, save_dir, log_step)

            if log_step % bc_mem_interval == 0:
                log_memory("bc", step=log_step, logger=logger)

            # Periodic BC checkpoint. Until 2026-08-24 the BC phase (500k steps
            # in the active launchers) was SIGTERM-only, so any non-preemption
            # death restarted it from step 0. Cheap next to the online one —
            # replay_buffer is None here, so this writes the agent, not GBs.
            if preemption_backup_interval > 0 and log_step % preemption_backup_interval == 0:
                print(f"Saving rolling checkpoint at BC step {log_step}")
                _save_bc_rolling_checkpoint(log_step)

            if is_preemption_requested():
                print(f"[PREEMPTION] Saving emergency checkpoint at BC step {log_step}")
                _save_bc_rolling_checkpoint(log_step)
                print("[PREEMPTION] Checkpoint saved. Exiting gracefully.")
                import sys
                sys.exit(0)
    finally:
        prefetcher.stop()

    log_memory("bc_end", step=log_step, logger=logger)
    save_agent(agent, save_dir, log_step)

    if log_to_file is not None:
        import json
        with open(log_to_file, 'w') as f:
            json.dump(test_log_data, f, indent=2)
        print(f"Test log written to {log_to_file}")

    return agent, log_step, eval_info


def run_calql_training(
    agent: Any,
    train_dataset: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    calql_steps: int,
    horizon_length: int,
    discount: float,
    batch_size: int,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    eval_interval_bc: int = 10000,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    video_env: Optional[Any] = None,
    video_interval: int = 0,
    video_max_attempts: int = 8,
    plot_q_vs_mc: bool = False,
    run_name: str = 'calql',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    env_name: str = '',
    start_step: int = 0,
    seed: int = 0,
) -> Tuple[Any, int, Dict]:
    if calql_steps <= 0:
        return agent, start_step, {}

    log_step = start_step
    eval_step = 0
    # Read, not sampled: Dataset.sample draws from dataset_rng, which a resume has
    # just restored, so a draw here would shift every post-resume batch by one.
    action_dim = train_dataset['actions'].shape[-1]
    dataset_idx = 0
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    eval_rng = jax.random.PRNGKey(seed + 50000)
    # Success/failure video capture: threshold, not modulo (see run_bc_training).
    video_enabled = video_env is not None and video_interval > 0
    next_video_at = ((start_step // video_interval) + 1) * video_interval if video_enabled else None
    should_log_step_data = step_data_log_interval > 0
    calql_log_interval = max(1, log_interval // 5)
    eval_info = {}

    for i in tqdm.tqdm(range(1, calql_steps + 1), dynamic_ncols=True, desc="CalQL Training"):
        log_step += 1

        batch = train_dataset.sample_sequence(
            batch_size, sequence_length=horizon_length, discount=discount,
            ret_next_act=True, ret_mc=True)
        agent, offline_info = agent.calql_update(batch)

        if should_log_step_data and i % step_data_log_interval == 0:
            _log_step_data(logger, batch, 'calql', log_step)

        if i % calql_log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        if eval_interval_bc >= 0 and (i == calql_steps or (eval_interval_bc != 0 and i % eval_interval_bc == 0)):
            eval_step += 1
            eval_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            eval_rng, eval_key = jax.random.split(eval_rng)
            try:
                eval_info = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    eval_actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot=False,
                    eval_rng=eval_key,
                )
            except PreemptionDuringEval as exc:
                # evaluate() calls raise_if_preempted; uncaught it would kill
                # main() with no checkpoint, while the video hook five lines
                # below already breaks cleanly.
                print(f"[PREEMPTION] {exc}; ending the CalQL phase")
                break
            print(f"CalQL Step {i}: Success Rate = {eval_info.get('success', 'N/A')}")
            logger.log(eval_info, "eval", step=log_step)

            if video_enabled and log_step >= next_video_at:
                try:
                    capture_and_log_policy_videos(
                        video_env, eval_actor_fn, action_dim, log_step, logger,
                        video_frame_skip=video_frame_skip, max_attempts=video_max_attempts)
                except PreemptionDuringEval as exc:
                    # CalQL has no rolling checkpoint of its own; hand control
                    # back so the caller's next preemption poll can save.
                    print(f"[PREEMPTION] {exc}; ending the CalQL phase")
                    break
                next_video_at = ((log_step // video_interval) + 1) * video_interval

    return agent, log_step, eval_info
