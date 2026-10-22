from collections import defaultdict
from ogpo.utils.main_helper import PreemptionDuringEval, raise_if_preempted
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tqdm
import umap


def supply_rng(f, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)
    return wrapped


def _split_obs(obs):
    """Split dict observation into (state, images) or return (obs, None) for flat arrays."""
    if isinstance(obs, dict):
        return obs['state'], obs['image']
    return obs, None


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_async(d, idx, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v[idx]))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def discounted_returns_from_prefix(rewards, indices, discount=0.99):
    T = len(rewards)
    disc_pows = discount ** np.arange(T)
    # Rewards are pre-transformed to [-1, 0] by env wrappers (e.g. robomimic step())
    prefix = np.cumsum(np.array(rewards) * disc_pows)

    def G(t):
        total = prefix[-1]
        prev = 0.0 if t == 0 else prefix[t - 1]
        return float((total - prev) / disc_pows[t])

    return [G(t) for t in indices]


def _aggregate_stats(stats, env_name):
    """Aggregate evaluation stats, handling Kitchen envs specially."""
    if "kitchen" in env_name.lower():
        final_stats = {}
        for k, v in stats.items():
            if not v:
                continue
            # np.bool_ is named explicitly: it subclasses NEITHER int (as Python's
            # bool does) NOR np.number, so without it a flag key falls through to
            # the last-value branch and latches the final episode instead of
            # averaging into a rate. The vectorized eval path produces exactly
            # that dtype -- gymnasium's VectorEnv._add_info batches a Python bool
            # info value into an np.zeros(n, dtype=bool) array.
            if isinstance(v[0], (bool, int, float, np.bool_, np.number)):
                final_stats[k] = np.mean(v)
            else:
                final_stats[k] = v[-1]
    else:
        final_stats = {}
        for k, v in stats.items():
            # Same guard as the kitchen branch above. Envs may publish
            # non-numeric info keys and flatten_async forwards ALL of them:
            # safe-robosuite writes info['safety_constraints'] (constraint
            # NAMES) and ['safety_violations'] on every step whenever a safety
            # filter exists, and np.mean on strings raises TypeError — which
            # killed every n_eval_envs>1 eval run under a safe controller.
            if not v:
                # Preserve the previous np.mean([]) -> nan so the key set is the
                # same from one eval to the next; a key that vanishes on some
                # evals makes the logged series ragged.
                final_stats[k] = np.nan
            elif isinstance(v[0], (bool, int, float, np.bool_, np.number)):
                # np.bool_ is load-bearing here -- see the kitchen branch above.
                final_stats[k] = np.mean(v)
            elif isinstance(v[0], (list, tuple, np.ndarray)):
                # Sequence-valued keys have no scalar aggregate, and CsvLogger
                # joins with ',' without quoting — emitting one would corrupt
                # every subsequent row. Drop rather than corrupt.
                continue
            else:
                final_stats[k] = v[-1]
    return final_stats


def visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix, save_dir='./plots'):
    if not scatter_data:
        return

    save_dir = os.path.join(save_dir, 'q_accuracy_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)
    reward_desc = "MC return-to-go (episode end)"
    fig.suptitle(f'Q-Function Accuracy Analysis ({reward_desc})\nStep {global_step} ({suffix})', fontsize=12)

    # Subplot 1: per-episode time series
    ax1 = axes[0]
    for i, episode in enumerate(time_series_data):
        timesteps = np.arange(len(episode['q_preds']))
        color = plt.cm.plasma(i / max(1, len(time_series_data) - 1))
        if len(timesteps) == 0:
            continue
        ax1.plot(timesteps, episode['q_preds'], marker='s', linestyle='--',
                 markersize=4, color=color, alpha=0.5, label='Q-Pred' if i == 0 else None)
        ax1.plot(timesteps, episode['mc_returns'], marker='^', linestyle='-',
                 markersize=4, color=color, alpha=0.5, label='MC Return' if i == 0 else None)

    ax1.set_title('Q-Value vs MC Return at Decision Points')
    ax1.set_xlabel('Decision Point Index (per episode)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: scatter correlation
    ax2 = axes[1]
    q_preds = [d[0] for d in scatter_data]
    mc_returns = [d[1] for d in scatter_data]

    ax2.scatter(q_preds, mc_returns, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    if q_preds and mc_returns:
        min_val = min(min(q_preds), min(mc_returns))
        max_val = max(max(q_preds), max(mc_returns))
        ax2.plot([min_val, max_val], [min_val, max_val], '--', alpha=0.75, linewidth=2, label='y = x')

    ax2.set_title('Predicted Q vs MC Return (All Episodes)')
    ax2.set_xlabel('Predicted Q-Value')
    ax2.set_ylabel('MC Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    if len(q_preds) > 1:
        correlation = np.corrcoef(q_preds, mc_returns)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    save_path = os.path.join(save_dir, f'q_accuracy_step_{global_step}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-function accuracy plot saved to: {save_path}")

def evaluate(
    agent, env, global_step,
    num_eval_episodes=50, num_video_episodes=0, video_frame_skip=3,
    eval_temperature=0, eval_gaussian=None,
    action_shape=None, observation_shape=None, action_dim=None,
    actor_fn=None, env_name='', plot=False,
    eval_rng=None,
):
    if actor_fn is None:
        suffix = "SDE"
        actor_fn = agent.sample_actions
    else:
        suffix = "ODE"

    if eval_rng is None:
        eval_rng = jax.random.PRNGKey(0)
    eval_rng, actor_rng = jax.random.split(eval_rng)
    actor_fn = supply_rng(actor_fn, rng=actor_rng)
    trajs = []
    stats = defaultdict(list)
    renders = []

    # Cache critic_fn once to avoid re-jitting (same issue as evaluate_parallel)
    _cached_get_q_fn = None
    if plot:
        from ogpo.agents.modules.q_helper import get_q_values as _get_q_values_jit
        _cached_critic_fn = agent.critic_network.select('target_critic')
        _cfg = agent.config if isinstance(agent.config, dict) else vars(agent.config)
        _cached_q_agg = _cfg.get('q_agg', 'mean')
        _critic_needs_rng = _cfg.get('use_mip_q', False)

        def _cached_get_q_fn(observations, actions, rng):
            return _get_q_values_jit(
                critic_fn=_cached_critic_fn,
                observations=observations,
                actions=actions,
                q_agg=_cached_q_agg,
                rng=rng if _critic_needs_rng else None,
            )

    scatter_data = []   # list of (q_pred, mc_return) across all episodes
    time_series_data = []  # per-episode lists
    for i in tqdm.tqdm(range(num_eval_episodes + num_video_episodes), dynamic_ncols=True):
        raise_if_preempted()
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        # Seed each episode distinctly for diverse, reproducible resets
        raw_observation, info = env.reset(seed=global_step + i)
        observation, obs_images = _split_obs(raw_observation)

        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        rewards = []                     # per-step rewards
        decision_indices = []            # step indices where a new chunk was decided
        q_preds_at_decisions = []        # Q(s_t, a_t) at each decision point
        action_chunk_lens = defaultdict(int)

        action_queue = []
        while not done:
            if len(action_queue) == 0:
                actor_kwargs = dict(observations=observation)
                if obs_images is not None:
                    actor_kwargs['images'] = obs_images
                action = actor_fn(**actor_kwargs)
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)

                if plot and obs_images is None:
                    eval_rng, q_rng = jax.random.split(eval_rng)
                    q_pred = _cached_get_q_fn(
                        jax.device_put(observation[np.newaxis, ...]),
                        jax.device_put(action[np.newaxis, ...]),
                        q_rng)[0]
                    q_preds_at_decisions.append(float(q_pred))

                decision_indices.append(step)
                action_chunk_lens[f"action_chunk_length_{action_chunk_len}"] += 1
                info['action_chunk_length'] = action_chunk_lens

            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            raw_next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            next_observation, next_obs_images = _split_obs(raw_next_observation)
            rewards.append(float(reward))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
            obs_images = next_obs_images

        mc_returns = discounted_returns_from_prefix(rewards, decision_indices, discount=0.99)
        time_series_data.append({
            'q_preds': q_preds_at_decisions,
            'mc_returns': mc_returns,
        })
        scatter_data.extend(list(zip(q_preds_at_decisions, mc_returns)))

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    final_stats = _aggregate_stats(stats, env_name)

    if plot:
        visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix=env_name)

    return final_stats, trajs, renders


def capture_success_failure_videos(
    actor_fn, env, action_dim, global_step,
    video_frame_skip=3, max_attempts=8,
):
    """Roll out up to max_attempts episodes on a renderable env, keeping the
    frames of the first successful and first failed episode.

    Returns {'success': (t, h, w, c) uint8 array or None, 'failure': ... or None}.
    Stops early once both outcomes are captured.

    `env` must have been built with make_env(..., renderable=True); a plain eval
    env has no offscreen context and env.render() will raise.
    """
    actor_fn = supply_rng(actor_fn, rng=jax.random.PRNGKey(global_step + 987))

    out = {'success': None, 'failure': None}
    # env.reset(seed=...) calls np.random.seed() globally in the robomimic
    # wrapper; save/restore so video capture doesn't perturb training RNG.
    _np_state = np.random.get_state()
    try:
        for attempt in range(max_attempts):
            raw_observation, info = env.reset(seed=global_step + attempt)
            observation, obs_images = _split_obs(raw_observation)

            done = False
            step = 0
            frames = []
            action_queue = []
            while not done:
                # Up to max_attempts rendered episodes is minutes of sim; a
                # SIGTERM must reach the emergency checkpoint inside KillWait.
                raise_if_preempted()
                if len(action_queue) == 0:
                    actor_kwargs = dict(observations=observation)
                    if obs_images is not None:
                        actor_kwargs['images'] = obs_images
                    action = actor_fn(**actor_kwargs)
                    action = np.array(action).reshape(-1, action_dim)
                    for a in action:
                        action_queue.append(a)

                action = action_queue.pop(0)
                raw_next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                observation, obs_images = _split_obs(raw_next_observation)
                done = terminated or truncated
                step += 1

                if step % video_frame_skip == 0 or done:
                    frames.append(env.render().copy())

            succeeded = bool(info.get('success', 0))
            key = 'success' if succeeded else 'failure'
            if out[key] is None:
                out[key] = np.array(frames)
            if out['success'] is not None and out['failure'] is not None:
                break
    finally:
        np.random.set_state(_np_state)

    return out


def capture_and_log_policy_videos(
    video_env, actor_fn, action_dim, log_step, logger,
    video_frame_skip=3, max_attempts=8, fps=15,
):
    """Capture one success + one failure episode video and log to wandb as
    eval/video_success and eval/video_failure.

    Only the outcomes actually found within max_attempts are logged; a missing
    outcome is simply omitted rather than logged blank. Keys use the "eval" prefix
    so a video shares the x-axis with the success-rate point it came from.

    Goes to wandb directly rather than through LoggingHelper.log: videos are
    wandb-only media, and LoggingHelper.log unconditionally writes a CSV row too,
    which would append a second row to eval.csv at this same step with every
    scalar column blank (CsvLogger filters wandb.Video out via disallowed_types,
    so the row would carry nothing but its step).
    """
    if video_env is None:
        return
    # Checked before capturing, not after: capture is minutes of blocking sim, and
    # LoggingHelper pairs wandb_logger=None with log_enabled=False on every rank
    # that does not log (main_helper.py:682-686). Doing the work then discarding
    # it would cost those ranks a full capture per eval boundary.
    if not getattr(logger, 'log_enabled', True) or getattr(logger, 'wandb_logger', None) is None:
        return
    from ogpo.utils.log_utils import get_wandb_video_file
    # best-effort: video is telemetry; a GL failure must not kill a multi-hour run
    try:
        result = capture_success_failure_videos(
            actor_fn, video_env, action_dim, log_step,
            video_frame_skip=video_frame_skip, max_attempts=max_attempts)
        data, tmp_paths = {}, []
        try:
            for key in ('success', 'failure'):
                if result[key] is not None:
                    video, path = get_wandb_video_file(result[key], fps=fps)
                    data[f'video_{key}'] = video
                    tmp_paths.append(path)
            if data:
                logger.wandb_logger.log(
                    {f'eval/{k}': v for k, v in data.items()}, step=log_step)
        finally:
            # wandb copies media files into the run dir during log, so cleanup is
            # safe once log() returns -- and must also happen if it raised.
            for path in tmp_paths:
                # best-effort: temp-file cleanup must not fail the training step
                try:
                    os.remove(path)
                except OSError:
                    pass
        print(f"[video] step={log_step}: success={'yes' if 'video_success' in data else 'no'}, "
              f"failure={'yes' if 'video_failure' in data else 'no'}", flush=True)
    except PreemptionDuringEval:
        # Not a capture failure: the caller's handler owns the emergency save.
        raise
    except Exception as e:
        print(f"[video] capture failed at step {log_step}: {e}", flush=True)


def visualize_q_diagnostics(scatter_data, episode_q_mc_sequences, actions_data, global_step, suffix, dir_suffix, save_dir='./plots'):
    """
    Visualizes Q-function diagnostics with four subplots:
    1. Q vs. MC scatter plot, colored by value.
    2. Q-value drop vs. MC-return drop scatter plot.
    3. UMAP of actions colored by Q-values.
    4. UMAP of (actions + Q-values) colored by Q-values.
    """
    if not scatter_data:
        return

    save_dir = os.path.join(save_dir, 'q_diagnostics_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(32, 7), constrained_layout=True)
    fig.suptitle(f'Q-Function Diagnostics\nStep {global_step} ({suffix})', fontsize=16)

    q_preds = np.array([d[0] for d in scatter_data])
    mc_returns = np.array([d[1] for d in scatter_data])
    ax1 = axes[0]
    sc = ax1.scatter(q_preds, mc_returns, c=mc_returns, cmap='plasma', alpha=0.5, s=15, edgecolors='none')
    fig.colorbar(sc, ax=ax1, label='MC Return Value')
    min_val, max_val = min(q_preds.min(), mc_returns.min()), max(q_preds.max(), mc_returns.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.8, linewidth=2, label='y = x')
    ax1.set_title('Predicted Q vs. MC Return')
    ax1.set_xlabel('Predicted Q-Value')
    ax1.set_ylabel('Actual MC Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', 'box')

    # --- Subplot 2: Value Drop Comparison ---
    ax2 = axes[1]
    q_diffs, mc_diffs = [], []
    for q_seq, mc_seq in episode_q_mc_sequences:
        if len(q_seq) > 1:
            q_diffs.extend(q_seq[:-1] - q_seq[1:])
            mc_diffs.extend(mc_seq[:-1] - mc_seq[1:])
    
    ax2.scatter(q_diffs, mc_diffs, alpha=0.2, s=15, edgecolors='none', color='royalblue')
    min_val, max_val = min(min(q_diffs), min(mc_diffs)), max(max(q_diffs), max(mc_diffs))
    ax2.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.8, linewidth=2, label='y = x')
    ax2.set_title('Value Drop Consistency')
    ax2.set_xlabel("Predicted Drop: $Q_t - Q_{t+1}$")
    ax2.set_ylabel("Actual Drop: $MC_t - MC_{t+1}$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    # --- Subplot 3: UMAP of Actions colored by Q-values ---
    ax3 = axes[2]
    if actions_data is not None and len(actions_data) > 0:
        actions = np.array(actions_data)
        # Standardize actions before UMAP
        actions_scaled = StandardScaler().fit_transform(actions)

        # Run UMAP on actions
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        action_embeddings = reducer.fit_transform(actions_scaled)

        # Plot UMAP embeddings colored by Q-values
        sc_actions = ax3.scatter(action_embeddings[:, 0], action_embeddings[:, 1],
                                 c=q_preds, cmap='plasma', alpha=0.6, s=15)
        fig.colorbar(sc_actions, ax=ax3, label='Q-Value')
        ax3.set_title('UMAP of Actions (colored by Q-values)')
        ax3.set_xlabel('UMAP Dimension 1')
        ax3.set_ylabel('UMAP Dimension 2')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No action data available', transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('UMAP of Actions (colored by Q-values)')

    # --- Subplot 4: UMAP of (Actions + Q-values) colored by Q-values ---
    ax4 = axes[3]
    if actions_data is not None and len(actions_data) > 0:
        # Combine actions and Q-values
        actions = np.array(actions_data)
        combined_features = np.column_stack([actions, q_preds.reshape(-1, 1)])
        combined_scaled = StandardScaler().fit_transform(combined_features)

        # Run UMAP on combined features
        reducer_combined = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        combined_embeddings = reducer_combined.fit_transform(combined_scaled)

        # Plot UMAP embeddings colored by Q-values
        sc_combined = ax4.scatter(combined_embeddings[:, 0], combined_embeddings[:, 1],
                                  c=q_preds, cmap='plasma', alpha=0.6, s=15)
        fig.colorbar(sc_combined, ax=ax4, label='Q-Value')
        ax4.set_title('UMAP of (Actions + Q-values) (colored by Q-values)')
        ax4.set_xlabel('UMAP Dimension 1')
        ax4.set_ylabel('UMAP Dimension 2')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No action data available', transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('UMAP of (Actions + Q-values) (colored by Q-values)')

    save_path = os.path.join(save_dir, f'step_{global_step}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-function diagnostics plot saved to: {save_path}")
    
def evaluate_parallel(envs, agent, FLAGS, action_dim, global_steps, actor_fn=None, plot=False, eval_rng=None):
    total_episodes_to_run = FLAGS.n_eval_envs * FLAGS.eval_episodes
    if actor_fn is None:
        suffix = "SDE"
        actor_fn = agent.sample_actions
    else:
        suffix = "ODE"

    if eval_rng is None:
        eval_rng = jax.random.PRNGKey(0)
    eval_rng, actor_rng = jax.random.split(eval_rng)
    actor_fn = supply_rng(actor_fn, rng=actor_rng)

    # When plotting Q-diagnostics, we need to call get_q_values at each decision
    # point. The standalone _get_q_values() in q_helper uses critic_fn as a
    # static_argname, but TrainState.select() creates a new functools.partial
    # each call with a different id() — causing JAX to re-jit on every call.
    # Fix: cache the select() result ONCE and reuse the same object.
    _cached_get_q_fn = None
    if plot:
        from ogpo.agents.modules.q_helper import get_q_values as _get_q_values_jit
        _cached_critic_fn = agent.critic_network.select('target_critic')
        _cfg = agent.config if isinstance(agent.config, dict) else vars(agent.config)
        _cached_q_agg = _cfg.get('q_agg', 'mean')
        _critic_needs_rng = _cfg.get('use_mip_q', False)

        def _cached_get_q_fn(observations, actions, rng):
            return _get_q_values_jit(
                critic_fn=_cached_critic_fn,
                observations=observations,
                actions=actions,
                q_agg=_cached_q_agg,
                rng=rng if _critic_needs_rng else None,
            )

    stats = defaultdict(list)
    scatter_data = []
    episode_q_mc_sequences = []
    actions_data = []

    pbar = tqdm.tqdm(total=total_episodes_to_run, desc="Parallel Evaluation", dynamic_ncols=True)

    for rollout_idx in range(FLAGS.eval_episodes):
        # Seed each env distinctly per rollout for diverse, reproducible resets
        rollout_seed = global_steps + rollout_idx * FLAGS.n_eval_envs
        raw_observations, _ = envs.reset(seed=rollout_seed)
        observations, obs_images = _split_obs(raw_observations)

        # Buffers for the current batch
        action_queues = [[] for _ in range(FLAGS.n_eval_envs)]
        rewards_buffers = [[] for _ in range(FLAGS.n_eval_envs)]
        q_preds_buffers = [[] for _ in range(FLAGS.n_eval_envs)]
        decision_indices = [[] for _ in range(FLAGS.n_eval_envs)]
        actions_buffers = [[] for _ in range(FLAGS.n_eval_envs)]
        active_envs = np.ones(FLAGS.n_eval_envs, dtype=bool)
        steps = np.zeros(FLAGS.n_eval_envs, dtype=int)

        while np.any(active_envs):
            raise_if_preempted()
            envs_needing_actions_indices = [i for i, q in enumerate(action_queues) if not q and active_envs[i]]
            if envs_needing_actions_indices:
                obs_to_act_on = observations[envs_needing_actions_indices]
                imgs_to_act_on = obs_images[envs_needing_actions_indices] if obs_images is not None else None
                n_need = len(envs_needing_actions_indices)

                # Pad to fixed batch size to avoid JIT recompilation of actor_fn
                # when envs finish at different times and the subset size changes.
                actor_kwargs = {}
                if n_need < FLAGS.n_eval_envs:
                    pad_size = FLAGS.n_eval_envs - n_need
                    obs_padding = np.zeros((pad_size,) + obs_to_act_on.shape[1:], dtype=obs_to_act_on.dtype)
                    obs_padded = np.concatenate([obs_to_act_on, obs_padding], axis=0)
                    actor_kwargs['observations'] = obs_padded
                    if imgs_to_act_on is not None:
                        img_padding = np.zeros((pad_size,) + imgs_to_act_on.shape[1:], dtype=imgs_to_act_on.dtype)
                        actor_kwargs['images'] = np.concatenate([imgs_to_act_on, img_padding], axis=0)
                    action_chunks = np.array(actor_fn(**actor_kwargs))
                    action_chunks = action_chunks.reshape(FLAGS.n_eval_envs, -1, action_dim)[:n_need]
                else:
                    actor_kwargs['observations'] = obs_to_act_on
                    if imgs_to_act_on is not None:
                        actor_kwargs['images'] = imgs_to_act_on
                    action_chunks = np.array(actor_fn(**actor_kwargs))
                    action_chunks = action_chunks.reshape(n_need, -1, action_dim)

                # Only compute Q-values when plotting diagnostics (skip for image envs).
                if plot and obs_images is None:
                    eval_rng, q_rng = jax.random.split(eval_rng)
                    q_preds = _cached_get_q_fn(jax.device_put(obs_to_act_on), jax.device_put(action_chunks), q_rng)
                    for i, env_idx in enumerate(envs_needing_actions_indices):
                        q_preds_buffers[env_idx].append(float(q_preds[i]))
                        decision_indices[env_idx].append(steps[env_idx])
                        actions_buffers[env_idx].append(action_chunks[i].flatten())

                for i, env_idx in enumerate(envs_needing_actions_indices):
                    for action in action_chunks[i]:
                        action_queues[env_idx].append(action)

            step_actions = []
            for i, q in enumerate(action_queues):
                if active_envs[i] and q:
                    step_actions.append(q.pop(0))
                else:
                    dummy_action = np.zeros(action_dim)
                    step_actions.append(dummy_action)
            raw_observations, rewards, terminations, truncations, infos = envs.step(np.array(step_actions))
            observations, obs_images = _split_obs(raw_observations)
            dones = np.logical_or(terminations, truncations)

            for i in range(FLAGS.n_eval_envs):
                if active_envs[i]:
                    rewards_buffers[i].append(float(rewards[i]))
                    steps[i] += 1
                    if dones[i]:
                        if plot:
                            q_sequence = np.array(q_preds_buffers[i])
                            mc_sequence = np.array(discounted_returns_from_prefix(
                                rewards_buffers[i],
                                decision_indices[i],
                                discount=FLAGS.discount
                            ))
                            scatter_data.extend(list(zip(q_sequence, mc_sequence)))
                            episode_q_mc_sequences.append((q_sequence, mc_sequence))
                            actions_data.extend(actions_buffers[i])

                        add_to(stats, flatten_async(infos, i))
                        active_envs[i] = False
                        pbar.update(1)

    pbar.close()

    final_stats = _aggregate_stats(stats, FLAGS.env_name)

    if plot:
        visualize_q_diagnostics(scatter_data, episode_q_mc_sequences, actions_data, global_steps, suffix, dir_suffix=FLAGS.run_name)

    return final_stats
