"""QIPM evaluation utilities: per-subtask Q vs MC tracking."""

import os
from collections import defaultdict
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.qipm_helper import compute_per_subtask_mc_returns
from ogpo.utils.evaluation import (
    _split_obs, flatten_async, add_to, discounted_returns_from_prefix,
    _aggregate_stats,
)
from ogpo.utils.main_helper import raise_if_preempted
from ogpo.utils.eval_safety import (
    EpisodeSafetyRecorder, parse_safety_views, record_shield_safely, record_step_safely,
    step_safety_from_info, step_safety_from_vector_info, step_shield_from_info,
    step_shield_from_vector_info, summarize_safety_safely,
)


def supply_rng(f, rng):
    """Wrap a function to automatically supply and advance an RNG key."""
    _rng = [rng]
    @wraps(f)
    def wrapper(*args, **kwargs):
        nonlocal _rng
        _rng[0], key = jax.random.split(_rng[0])
        return f(*args, rng=key, **kwargs)
    return wrapper


def visualize_per_subtask_q_vs_mc(subtask_scatter_data, global_step, dir_suffix='', save_dir='./plots'):
    """Create per-subtask Q vs MC scatter plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping Q vs MC plots")
        return

    no_of_subtasks = len(subtask_scatter_data)
    ncols = min(3, no_of_subtasks)
    nrows = (no_of_subtasks + ncols - 1) // ncols

    save_dir = os.path.join(save_dir, 'subtask_q_mc_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f'Per-Subtask Q vs MC Returns (Step {global_step})', fontsize=14)

    for idx in range(no_of_subtasks):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        data = subtask_scatter_data[idx]
        if data:
            q_vals = [d[0] for d in data]
            mc_vals = [d[1] for d in data]
            ax.scatter(q_vals, mc_vals, alpha=0.4, s=10)
            if q_vals and mc_vals:
                vmin = min(min(q_vals), min(mc_vals))
                vmax = max(max(q_vals), max(mc_vals))
                ax.plot([vmin, vmax], [vmin, vmax], '--r', alpha=0.5)
                if len(q_vals) > 1:
                    corr = np.corrcoef(q_vals, mc_vals)[0, 1]
                    ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'Subtask {idx}')
        ax.set_xlabel('Q-Value')
        ax.set_ylabel('MC Return')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)

    for idx in range(no_of_subtasks, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    save_path = os.path.join(save_dir, f'subtask_q_mc_step_{global_step}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-subtask Q vs MC plot saved to: {save_path}")


class DenseRewardEnvWrapper:
    """Wrapper for individual env to support dense reward parameters."""
    def __init__(self, env, use_dense_reward=True, normalize_rewards=False, no_of_subtasks=3):
        self.env = env
        self.use_dense_reward = use_dense_reward
        self.normalize_rewards = normalize_rewards
        self.no_of_subtasks = no_of_subtasks

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(
            action,
            use_dense_reward=self.use_dense_reward,
            normalize_rewards=self.normalize_rewards,
            no_of_subtasks=self.no_of_subtasks,
        )

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


def evaluate_qipm(
    agent,
    env,
    global_step,
    num_eval_episodes=50,
    action_dim=None,
    actor_fn=None,
    env_name='',
    no_of_subtasks=3,
    discount=0.99,
    plot=True,
    subtask_discounts=None,
    eval_rng=None,
    safety_views=None,
):
    """Evaluation with per-subtask Q vs MC tracking for Q-IPM.

    ``safety_views`` ({view: [constraint names]}, or a JSON string of one) turns
    on the ``safety/`` eval keys (ogpo/utils/eval_safety.py) when the env info
    carries a safe-robosuite roster. None (default) = off: no safety recording
    at all, so the returned keys are exactly the pre-existing ones.
    """
    if actor_fn is None:
        actor_fn = agent.sample_actions
    if eval_rng is None:
        eval_rng = jax.random.PRNGKey(0)
    eval_rng, actor_rng = jax.random.split(eval_rng)
    actor_fn = supply_rng(actor_fn, rng=actor_rng)

    stats = defaultdict(list)
    subtask_scatter_data = {i: [] for i in range(no_of_subtasks)}
    total_scatter_data = []
    safety_episodes = []
    safety_views = parse_safety_views(safety_views)
    safety_on = safety_views is not None

    for ep_idx in tqdm.tqdm(range(num_eval_episodes), desc="Q-IPM Eval", dynamic_ncols=True):
        # Same reasoning as the parallel path: episode granularity is fine
        # enough to return control well inside the 120s kill window.
        raise_if_preempted()
        observation, info = env.reset()
        done = False
        step = 0
        action_queue = []
        subtask_status_history = []
        rewards = []
        decision_indices = []
        per_subtask_q_preds = {i: [] for i in range(no_of_subtasks)}
        total_q_preds = []
        safety_rec = EpisodeSafetyRecorder()

        while not done:
            if len(action_queue) == 0:
                action = actor_fn(observations=observation)
                action = np.array(action).reshape(-1, action_dim)
                for a in action:
                    action_queue.append(a)

                obs_batch = jax.device_put(observation[np.newaxis, ...])
                act_batch = jax.device_put(action[np.newaxis, ...])
                subtask_qs = agent.get_per_subtask_q_values(obs_batch, act_batch)
                subtask_qs = np.array(subtask_qs)[:, 0]

                for i in range(no_of_subtasks):
                    per_subtask_q_preds[i].append(float(subtask_qs[i]))
                # Task heads only: the safe_qipm family's critic carries trailing
                # safety heads whose MC counterpart below is task-only, so summing
                # all rows would compare different quantities. No-op elsewhere.
                total_q_preds.append(float(subtask_qs[:no_of_subtasks].sum()))
                decision_indices.append(step)

            action = action_queue.pop(0)
            next_observation, reward, terminated, truncated, info = env.step(
                np.clip(action, -1, 1),
            )
            reward_info = info.get('reward_info', {})
            subtask_status = reward_info.get('subtask_status', [0] * no_of_subtasks)
            subtask_status_history.append(subtask_status)
            rewards.append(float(reward))
            if safety_on:
                record_step_safely(safety_rec, step_safety_from_info, info)
                record_shield_safely(safety_rec, step_shield_from_info, info)
            done = terminated or truncated
            step += 1
            observation = next_observation

        mc_discount = subtask_discounts if subtask_discounts is not None else discount
        subtask_mc = compute_per_subtask_mc_returns(
            subtask_status_history, decision_indices, no_of_subtasks, mc_discount)
        for i in range(no_of_subtasks):
            q_list = per_subtask_q_preds[i]
            mc_list = subtask_mc[i]
            for q_val, mc_val in zip(q_list, mc_list):
                subtask_scatter_data[i].append((q_val, mc_val))

        total_mc = [
            sum(subtask_mc[i][j] for i in range(no_of_subtasks))
            for j in range(len(decision_indices))
        ]
        total_scatter_data.extend(list(zip(total_q_preds, total_mc)))

        final_reward_info = info.get('reward_info', {})
        final_subtask_status = final_reward_info.get('subtask_status', [0] * no_of_subtasks)

        stats['success'].append(float(info.get('success', 0)))
        stats['episode_reward'].append(sum(rewards))
        stats['episode_length'].append(step)
        stats['subtasks_completed'].append(float(final_reward_info.get('subtasks_completed', 0)))

        for i in range(no_of_subtasks):
            stats[f'subtask_{i}_success'].append(float(final_subtask_status[i]))

        if safety_on and safety_rec.has_data:
            safety_episodes.append(safety_rec.finish(float(info.get('success', 0))))

    final_stats = {k: np.mean(v) for k, v in stats.items()}
    final_stats['avg_subtasks_completed'] = final_stats.get('subtasks_completed', 0)

    for i in range(no_of_subtasks):
        final_stats[f'subtask_{i}_success_rate'] = final_stats.pop(f'subtask_{i}_success', 0)

    # Merged after the generic aggregation, which would otherwise see the
    # per-episode records as list-valued stats.
    if safety_on:
        final_stats.update(summarize_safety_safely(safety_episodes, safety_views))

    if plot:
        visualize_per_subtask_q_vs_mc(subtask_scatter_data, global_step, dir_suffix=env_name)

    return final_stats, subtask_scatter_data, total_scatter_data


def evaluate_qipm_parallel(
    envs,
    agent,
    FLAGS,
    action_dim,
    global_step,
    actor_fn=None,
    no_of_subtasks=3,
    plot=True,
    eval_rng=None,
):
    """Parallel evaluation with per-subtask Q and total Q vs MC tracking.

    Uses vectorized envs (AsyncVectorEnv) to run multiple episodes in parallel.
    Per-subtask scatter data is skipped in parallel mode because vectorized envs
    don't expose per-step subtask_status reliably. Use the single-env
    ``evaluate_qipm`` for accurate per-subtask plots.

    Args:
        envs: Vectorized env (e.g. gymnasium AsyncVectorEnv).
        agent: OGPOQIPMAgent with ``get_per_subtask_q_values`` method.
        FLAGS: SimpleNamespace with n_eval_envs, eval_episodes, discount,
               run_name, env_name, and optionally safety_views
               ({view: [constraint names]}; see ogpo/utils/eval_safety.py).
        action_dim: Scalar action dimension.
        global_step: Current training step (for plot naming).
        actor_fn: Optional policy function override.
        no_of_subtasks: Number of Q-IPM subtasks.
        plot: Whether to produce Q vs MC scatter plots.

    Returns:
        (final_stats, subtask_scatter_data, total_scatter_data)
    """
    total_episodes_to_run = FLAGS.n_eval_envs * FLAGS.eval_episodes

    if actor_fn is None:
        actor_fn = agent.sample_actions
    if eval_rng is None:
        eval_rng = jax.random.PRNGKey(0)
    eval_rng, actor_rng = jax.random.split(eval_rng)
    actor_fn = supply_rng(actor_fn, rng=actor_rng)

    stats = defaultdict(list)
    subtask_scatter_data = {i: [] for i in range(no_of_subtasks)}
    total_scatter_data = []
    safety_episodes = []
    # Opt-in: no FLAGS.safety_views (or None) = no safety recording at all.
    safety_views = parse_safety_views(getattr(FLAGS, 'safety_views', None))
    safety_on = safety_views is not None

    pbar = tqdm.tqdm(total=total_episodes_to_run, desc="Q-IPM Parallel Eval",
                     dynamic_ncols=True)

    for batch_idx in range(FLAGS.eval_episodes):
        rollout_seed = global_step + batch_idx * FLAGS.n_eval_envs
        raw_observations, _ = envs.reset(seed=rollout_seed)
        observations, obs_images = _split_obs(raw_observations)

        action_queues = [[] for _ in range(FLAGS.n_eval_envs)]
        rewards_buffers = [[] for _ in range(FLAGS.n_eval_envs)]
        per_subtask_q_preds = [
            {s: [] for s in range(no_of_subtasks)} for _ in range(FLAGS.n_eval_envs)
        ]
        total_q_preds_buffers = [[] for _ in range(FLAGS.n_eval_envs)]
        decision_indices = [[] for _ in range(FLAGS.n_eval_envs)]
        active_envs = np.ones(FLAGS.n_eval_envs, dtype=bool)
        steps = np.zeros(FLAGS.n_eval_envs, dtype=int)
        safety_recs = [EpisodeSafetyRecorder() for _ in range(FLAGS.n_eval_envs)]

        while np.any(active_envs):
            # A full eval is minutes of simulation with no other preemption
            # check in it, and SIGKILL follows SIGTERM by 120s. Bail out here so
            # the training loop can reach its emergency checkpoint.
            raise_if_preempted()
            envs_needing_actions = [
                i for i, q in enumerate(action_queues) if not q and active_envs[i]
            ]

            if envs_needing_actions:
                obs_to_act = observations[envs_needing_actions]
                imgs_to_act = (obs_images[envs_needing_actions]
                               if obs_images is not None else None)
                n_need = len(envs_needing_actions)

                # Pad to fixed batch size to avoid JIT recompilation
                actor_kwargs = {}
                if n_need < FLAGS.n_eval_envs:
                    pad = FLAGS.n_eval_envs - n_need
                    obs_padded = np.concatenate([
                        obs_to_act,
                        np.zeros((pad,) + obs_to_act.shape[1:], dtype=obs_to_act.dtype),
                    ], axis=0)
                    actor_kwargs['observations'] = obs_padded
                    if imgs_to_act is not None:
                        img_padded = np.concatenate([
                            imgs_to_act,
                            np.zeros((pad,) + imgs_to_act.shape[1:], dtype=imgs_to_act.dtype),
                        ], axis=0)
                        actor_kwargs['images'] = img_padded
                    action_chunks = np.array(actor_fn(**actor_kwargs))
                    action_chunks = action_chunks.reshape(
                        FLAGS.n_eval_envs, -1, action_dim)[:n_need]
                else:
                    actor_kwargs['observations'] = obs_to_act
                    if imgs_to_act is not None:
                        actor_kwargs['images'] = imgs_to_act
                    action_chunks = np.array(actor_fn(**actor_kwargs))
                    action_chunks = action_chunks.reshape(n_need, -1, action_dim)

                # Per-subtask Q predictions at decision points
                if obs_images is None:
                    subtask_qs = agent.get_per_subtask_q_values(
                        jax.device_put(obs_to_act),
                        jax.device_put(action_chunks),
                    )
                    subtask_qs = np.array(subtask_qs)  # (S, n_need)

                    for i, env_idx in enumerate(envs_needing_actions):
                        for s in range(no_of_subtasks):
                            per_subtask_q_preds[env_idx][s].append(
                                float(subtask_qs[s, i]))
                        # Task heads only — see the single-env path above.
                        total_q_preds_buffers[env_idx].append(
                            float(subtask_qs[:no_of_subtasks, i].sum()))
                        decision_indices[env_idx].append(int(steps[env_idx]))

                for i, env_idx in enumerate(envs_needing_actions):
                    for action in action_chunks[i]:
                        action_queues[env_idx].append(action)

            # Step all environments
            step_actions = []
            for i, q in enumerate(action_queues):
                if active_envs[i] and q:
                    step_actions.append(q.pop(0))
                else:
                    step_actions.append(np.zeros(action_dim))

            raw_observations, rewards, terminations, truncations, infos = envs.step(
                np.clip(np.array(step_actions), -1, 1))
            observations, obs_images = _split_obs(raw_observations)
            dones = np.logical_or(terminations, truncations)

            for i in range(FLAGS.n_eval_envs):
                if active_envs[i]:
                    rewards_buffers[i].append(float(rewards[i]))
                    steps[i] += 1
                    # Autoreset is DISABLED for the eval envs, so this step's
                    # infos (the terminal step's included) belong to the episode
                    # being counted; envs that are already done are skipped above.
                    if safety_on:
                        record_step_safely(safety_recs[i], step_safety_from_vector_info,
                                           infos, i)
                        record_shield_safely(safety_recs[i], step_shield_from_vector_info,
                                             infos, i)

                    if dones[i]:
                        # Total Q vs MC scatter (reward-based MC returns)
                        if decision_indices[i]:
                            mc_returns = discounted_returns_from_prefix(
                                rewards_buffers[i],
                                decision_indices[i],
                                discount=FLAGS.discount,
                            )
                            total_scatter_data.extend(
                                list(zip(total_q_preds_buffers[i], mc_returns)))

                        # Aggregate stats from vectorized env info
                        flat_info = flatten_async(infos, i)
                        add_to(stats, flat_info)
                        if safety_on and safety_recs[i].has_data:
                            safety_episodes.append(safety_recs[i].finish(
                                float(flat_info.get('success', 0))))

                        # Per-subtask success from info
                        subtask_status = None
                        # AutoresetMode.NEXT_STEP wraps terminal info in final_info
                        if 'final_info' in infos and infos['final_info'][i] is not None:
                            fi = infos['final_info'][i]
                            if isinstance(fi, dict) and 'reward_info' in fi:
                                subtask_status = fi['reward_info'].get(
                                    'subtask_status', None)
                        # AutoresetMode.DISABLED: info is directly available
                        if subtask_status is None and 'reward_info' in infos:
                            ri = infos['reward_info']
                            if isinstance(ri, dict) and 'subtask_status' in ri:
                                subtask_status = ri['subtask_status'][i]
                        if subtask_status is not None:
                            for s in range(min(no_of_subtasks, len(subtask_status))):
                                stats[f'subtask_{s}_success'].append(
                                    float(subtask_status[s]))

                        active_envs[i] = False
                        pbar.update(1)

    pbar.close()

    final_stats = _aggregate_stats(stats, FLAGS.env_name)
    # Merged after _aggregate_stats so the per-episode records never go through
    # the generic list path (which drops sequence-valued keys).
    if safety_on:
        final_stats.update(summarize_safety_safely(safety_episodes, safety_views))

    for i in range(no_of_subtasks):
        final_stats[f'subtask_{i}_success_rate'] = final_stats.pop(
            f'subtask_{i}_success', 0)

    if plot:
        visualize_per_subtask_q_vs_mc(
            subtask_scatter_data, global_step, dir_suffix=FLAGS.run_name)

    return final_stats, subtask_scatter_data, total_scatter_data
