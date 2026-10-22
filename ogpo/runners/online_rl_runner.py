"""Online RL training runner for all algorithms."""

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.evaluation import evaluate, evaluate_parallel, _split_obs
from ogpo.utils.main_helper import (
    LoggingHelper, save_rolling_checkpoint, create_success_buffer_batch,
    _collect_rng_states, _restore_rng_states, is_preemption_requested,
)
from ogpo.agents.modules.maintenance import rebind_online_optimizers
from ogpo.utils.mem_monitor import log_memory
from ogpo.utils.robometer_client import RobometerRewardClient

ALGORITHMS_WITH_BATCH_BC = {'ogpo', 'bptt', 'expo'}
ALGORITHMS_PURE_ONLINE = {'dsrl', 'dsrl_bon', 'dsrl_plus_expo'}
ALGORITHMS_WITH_PHASES = {'ogpo', 'bptt', 'sac', 'sac_qipm'}
ALGORITHMS_DUAL_EVAL = {'ogpo', 'expo'}


def _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name):
    if n_eval_envs <= 1:
        return None
    return SimpleNamespace(
        n_eval_envs=n_eval_envs, eval_episodes=eval_episodes,
        discount=discount, run_name=run_name, env_name=env_name,
    )


def _run_evaluation(agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot, eval_rng=None):
    if eval_flags is None:
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, global_step=log_step,
            action_dim=action_dim, num_eval_episodes=eval_episodes,
            num_video_episodes=video_episodes, video_frame_skip=video_frame_skip,
            actor_fn=actor_fn, env_name=run_name, plot=plot,
            eval_rng=eval_rng,
        )
    else:
        eval_info = evaluate_parallel(
            eval_envs, agent, eval_flags, action_dim, log_step,
            actor_fn=actor_fn, plot=plot,
            eval_rng=eval_rng,
        )
    return eval_info

def _reshape_for_utd(batch, utd_ratio, batch_size):
    return jax.tree.map(lambda x: x.reshape((utd_ratio, batch_size) + x.shape[1:]), batch)

def _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                        horizon_length, discount, offline_ratio):
    """Sample a mixed online/offline batch and reshape for UTD."""
    offline_batch_size = int(batch_size * offline_ratio)
    online_batch_size = batch_size - offline_batch_size

    offline_batch = None
    if offline_batch_size > 0:
        offline_batch = train_dataset.sample_sequence(
            offline_batch_size * utd_ratio,
            sequence_length=horizon_length, discount=discount)
        offline_batch = _reshape_for_utd(offline_batch, utd_ratio, offline_batch_size)

    online_batch = None
    if online_batch_size > 0:
        min_online_size = online_batch_size * utd_ratio * horizon_length
        if replay_buffer.size >= min_online_size:
            online_batch = replay_buffer.sample_sequence(
                online_batch_size * utd_ratio,
                sequence_length=horizon_length, discount=discount)
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


def _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount):
    """Sample a pure-online batch and reshape for UTD."""
    batch = replay_buffer.sample_sequence(
        batch_size * utd_ratio, sequence_length=horizon_length, discount=discount)
    return _reshape_for_utd(batch, utd_ratio, batch_size)


def _sample_success_batch(replay_buffer, success_buffer_batch_size, utd_ratio,
                          horizon_length, discount):
    """Sample from success buffer. Returns (batch, True) or (None, False)."""
    num_successful = replay_buffer._traj_success_mask[:replay_buffer.size].sum()
    min_required = success_buffer_batch_size * utd_ratio * horizon_length
    if num_successful < min_required:
        return None, False
        
    batch = create_success_buffer_batch(
        replay_buffer, success_buffer_batch_size * utd_ratio,
        horizon_length, discount)
    batch = _reshape_for_utd(batch, utd_ratio, success_buffer_batch_size)
    return batch, True

def run_online_rl(
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
    plot_q_vs_mc: bool = False,
    run_name: str = 'online_rl',
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
    progress_rewards: bool = False,
    robometer_server_url: Optional[str] = None,
    robometer_url_file: Optional[str] = None,
    robometer_wait_s: float = 1800.0,
    robometer_max_frames: int = 8,
    robometer_use_pbrs: bool = False,
) -> Tuple[Any, int, Dict, ReplayBuffer]:
    if algo_name in ALGORITHMS_PURE_ONLINE and offline_ratio != 0.0:
        raise ValueError(f"{algo_name} requires offline_ratio=0.0, got {offline_ratio}")

    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]
    has_images = 'images' in example_batch

    # If the env produces dict obs with an 'image' key but the offline dataset
    # does not (e.g. LIBERO with live_render_images=true), inject a zero-image
    # template so the online replay buffer schema includes images. The actual
    # frames are filled in from rollouts; offline data is never sent to robometer.
    if progress_rewards and not has_images:
        env_obs_space = getattr(env, 'observation_space', None)
        img_space = getattr(env_obs_space, 'spaces', {}).get('image') if env_obs_space is not None else None
        if img_space is not None:
            zero_img = np.zeros(img_space.shape, dtype=img_space.dtype)
            example_batch = {**example_batch, 'images': zero_img, 'next_images': zero_img}
            has_images = True

    robometer_client: Optional[RobometerRewardClient] = None
    if progress_rewards:
        assert has_images, (
            "progress_rewards=True requires image observations, but neither the "
            "offline dataset nor the env exposes them. Set env.live_render_images=true "
            "for LIBERO, or use an image-based env."
        )
        robometer_client = RobometerRewardClient.from_env(
            env_name=env_name,
            server_url=robometer_server_url,
            url_file=robometer_url_file,
            wait_s=float(robometer_wait_s),
            max_frames=int(robometer_max_frames),
        )

    needs_offline_data = offline_ratio > 0.0
    needs_warmup_offline = warmup_offline_ratio > 0.0
    if not needs_offline_data and not needs_warmup_offline:
        train_dataset = None
        import gc; gc.collect()

    # Replay buffer
    if is_resumed and resumed_replay_buffer is not None:
        replay_buffer = resumed_replay_buffer
    else:
        rb_template = dict(
            observations=example_batch['observations'],
            actions=example_batch['actions'],
            rewards=np.array(0.0), terminals=np.array(0.0),
            masks=np.array(1.0), next_observations=example_batch['observations'],
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

    # Validate policy delay vs UTD ratio
    policy_delay = getattr(agent.config, 'policy_delay', 1) if hasattr(agent, 'config') else 1
    if policy_delay > 1:
        for _utd, _name in [(utd_warmup, 'utd_warmup'), (utd_online, 'utd_online')]:
            if _utd % policy_delay != 0:
                raise ValueError(
                    f"policy_delay={policy_delay} must evenly divide {_name}={_utd}")
        print(f"[Policy Delay] {policy_delay}x — {utd_online} critic updates, "
              f"{utd_online // policy_delay} actor update(s) per env step")

    # Pre-compute invariants
    needs_batch_bc = algo_name in ALGORITHMS_WITH_BATCH_BC
    has_phases = algo_name in ALGORITHMS_WITH_PHASES
    needs_dual_eval = algo_name in ALGORITHMS_DUAL_EVAL
    should_log_step_data = step_data_log_interval > 0
    q_warmup_end = start_training + bc_refine_steps + q_warmup_steps
    eval_start = q_warmup_end if has_phases else start_training

    log_step = start_step
    eval_step = 0
    update_info = {}

    # Restore JAX RNG states from checkpoint if available, otherwise init fresh
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

    if config.get('use_one_step_for_rollouts', False) and hasattr(agent, 'sample_actions_one_step'):
        if best_of_n > 1 and hasattr(agent, 'sample_actions_one_step_BON'):
            _actor_method_name = 'sample_actions_one_step_BON'
        else:
            _actor_method_name = 'sample_actions_one_step'
    elif best_of_n > 1 and hasattr(agent, 'sample_actions_BON'):
        _actor_method_name = 'sample_actions_BON'
    else:
        _actor_method_name = 'sample_actions'

    start_online_step = 1
    if is_resumed and resumed_online_step > 0:
        start_online_step = resumed_online_step + 1
        print(f"Resuming online loop at step {start_online_step}")

    eval_info = {}
    mem_log_interval = max(eval_interval, 20000)  # Log memory at least every eval

    log_memory("online_start", step=log_step, logger=logger)

    for i in tqdm.tqdm(
        range(start_online_step, online_steps + 1),
        dynamic_ncols=True, desc="Online RL",
        initial=start_online_step - 1, total=online_steps,
    ):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Action chunking
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            action = getattr(agent, _actor_method_name)(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        # Environment step
        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        next_ob, next_ob_images = _split_obs(raw_next_ob)
        done = terminated or truncated

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        if env_info:
            try:
                logger.log(env_info, "env", step=log_step)
            except Exception:
                pass

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        trajectory_buffer.append(transition)

        if should_log_step_data and i % step_data_log_interval == 0:
            try:
                logger.log({
                    'phase': 'online', 'reward': float(int_reward), 'done': float(done),
                    'batch_obs_mean': float(np.mean(ob)), 'batch_obs_std': float(np.std(ob)),
                    'batch_action_mean': float(np.mean(action)), 'batch_action_std': float(np.std(action)),
                }, "step_data", step=log_step)
            except Exception:
                pass

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9

            if robometer_client is not None and trajectory_buffer:
                try:
                    progress_arr, success_arr = robometer_client.compute_from_trajectory(trajectory_buffer)
                    print(f"[progress_rewards] traj_len={len(trajectory_buffer)} "
                          f"progress.shape={progress_arr.shape} success.shape={success_arr.shape}")
                    assert progress_arr.shape[0] == len(trajectory_buffer), (
                        f"progress length {progress_arr.shape[0]} != traj len {len(trajectory_buffer)}"
                    )
                    if robometer_use_pbrs:
                        # PBRS: r_t = γ·Φ(s_{t+1}) − Φ(s_t) with Φ = robometer progress, Φ(s_0)=0.
                        # Sum telescopes to γ^T·Φ(s_T) − Φ(s_0), so plateau-stalling earns ~0
                        # cumulative shaping reward — kills the "look done forever" exploit.
                        # +50 sparse bonus on true success terminal (masks<0.5) keeps the
                        # task-completion signal task-aligned with the env predicate.
                        phi = progress_arr.astype(np.float32)
                        phi_prev = np.concatenate(
                            [np.array([0.0], dtype=np.float32), phi[:-1]]
                        )
                        for idx, t in enumerate(trajectory_buffer):
                            is_success_terminal = float(t.get('masks', 1.0)) < 0.5
                            shaping = float(discount) * float(phi[idx]) - float(phi_prev[idx])
                            t['rewards'] = shaping + (50.0 if is_success_terminal else 0.0)
                    else:
                        # Dense robometer progress (shifted to [-1, 0]) + sparse success bonus.
                        # Gate the bonus on masks (not terminals) — see prior commit comment.
                        for idx, t in enumerate(trajectory_buffer):
                            is_success_terminal = float(t.get('masks', 1.0)) < 0.5
                            t['rewards'] = (float(progress_arr[idx]) - 1.0) + (
                                50.0 if is_success_terminal else 0.0
                            )
                except Exception as e:
                    print(f"[progress_rewards] query failed: {e}")
            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
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

        # Handle SLURM preemption: save checkpoint and exit cleanly
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

        is_warmup = has_phases and (i < bc_refine_steps or i < q_warmup_end)
        if is_warmup:
            utd_ratio = utd_warmup
        elif split_utd:
            utd_ratio = max(eff_utd_q, eff_utd_pi)
        else:
            utd_ratio = utd_online

        if replay_buffer.size < batch_size * utd_ratio * horizon_length:
            continue

        # Sample batch
        if is_warmup and needs_warmup_offline:
            batch = _sample_mixed_batch(
                replay_buffer, train_dataset, batch_size, utd_ratio,
                horizon_length, discount, warmup_offline_ratio)
        elif needs_offline_data:
            batch = _sample_mixed_batch(
                replay_buffer, train_dataset, batch_size, utd_ratio,
                horizon_length, discount, offline_ratio)
        else:
            batch = _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount)

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
        if has_phases and i < bc_refine_steps:
            agent, update_info["online_agent"] = agent.batch_bc_refine_update(batch_success)
        elif has_phases and i < q_warmup_end:
            agent, update_info["online_agent"] = agent.batch_q_warmup_update(batch)
        elif split_utd:
            # Split: joint actor+critic on first eff_utd_pi slices, critic-only on the rest
            batch_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch)
            bs_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch_success)
            agent, update_info["online_agent"] = agent.batch_update(batch_joint, bs_joint, success_flag)
            batch_extra = jax.tree.map(lambda x: x[eff_utd_pi:eff_utd_q], batch)
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

            ode_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            eval_rng, eval_key = jax.random.split(eval_rng)
            eval_info = _run_evaluation(
                agent, eval_env, eval_envs, eval_flags, action_dim,
                ode_actor_fn, log_step, eval_episodes, video_episodes,
                video_frame_skip, run_name, plot=(eval_step % 2 == 0) and plot_q_vs_mc,
                eval_rng=eval_key,
            )
            print(f"[EVAL ODE] step={i} | success={float(eval_info.get('success', 0.0)):.4f}")
            logger.log(eval_info, "eval", step=log_step)

            if needs_dual_eval:
                sde_actor_fn = getattr(agent, _actor_method_name)
                eval_rng, eval_key = jax.random.split(eval_rng)
                eval_info_sde = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    sde_actor_fn, log_step, eval_episodes, 0,
                    video_frame_skip, run_name, plot=False,
                    eval_rng=eval_key,
                )
                print(f"[EVAL SDE] step={i} | success={float(eval_info_sde.get('success', 0.0)):.4f}")
                try:
                    logger.log(eval_info_sde, "eval_sde", step=log_step)
                except Exception:
                    pass

        # Memory monitoring (every eval or mem_log_interval steps)
        if i % mem_log_interval == 0:
            log_memory("online", step=log_step, logger=logger)

        # Saving
        if log_enabled and save_interval > 0 and i % save_interval == 0:
            save_agent(agent, save_dir, log_step)

    save_agent(agent, save_dir, log_step)
    print(f"Online RL training complete. Final step: {log_step}")
    return agent, log_step, eval_info, replay_buffer
