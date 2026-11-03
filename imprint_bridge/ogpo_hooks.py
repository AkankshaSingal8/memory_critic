"""Experiment B: give OGPO's critic (B1) or actor and critic (B2) an IMPRINT memory of the episode so far.

Called once from OGPO's main.py, right after the envs and the demo dataset are built (ogpo_memory/
ogpo_public.patch). Everything else in OGPO stays as it is, because its existing plumbing already carries
the memory once the envs and the dataset provide it:

  mode "critic" (B1)        observations stay the 23-d state for the actor; the critic reads
                            full_states = [obs, m_t] (OGPO's privileged-critic input). The env returns
                            {"state": obs, "full_state": [obs, m_t]}; OGPO's _split_obs, replay buffer
                            (full_states / next_full_states), BoN (full_state kwarg), TD target, actor
                            advantage and eval padding already handle that.
  mode "actor_critic" (B2)  observations become [obs, m_t] everywhere: demos, online env, eval envs.
  mode "none"               no-op (the no-memory baseline on the same code).

Memory comes from imprint_bridge.tracker.MemoryTracker (frozen Stage-1 encoder, updated every env step):
one tracker for the online env, one batched tracker for the vectorized eval envs, and a pass over every
demo episode for the dataset. m_t for a demo step is the memory of that demo's frames 0..t; next_* get
the memory of frames 0..t+1.
"""

import numpy as np

from imprint_bridge.tracker import MemoryTracker, episode_memory

MODES = ("none", "critic", "actor_critic")


def _make_tracker(cfg, n_envs):
    return MemoryTracker(cfg["stage1_dir"], n_envs=n_envs, feature=cfg.get("feature", "mean"),
                         ckpt=cfg.get("ckpt", "stage1_stepbest.npz"),
                         projection=cfg.get("projection") or None)


def _augment(obs, mem, mode):
    full = np.concatenate([np.asarray(obs, np.float32), mem.astype(np.float32)], axis=-1)
    return {"state": obs, "full_state": full} if mode == "critic" else full


class MemoryEnv:
    """Single (online) env: tracks the episode's memory and returns augmented observations."""

    def __init__(self, env, tracker, mode):
        self.env, self.tracker, self.mode = env, tracker, mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.tracker.reset(np.array([0]), np.asarray(obs)[None])
        return _augment(obs, self.tracker.read()[0], self.mode), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.tracker.step(np.asarray(action)[None], np.asarray(obs)[None])
        return _augment(obs, self.tracker.read()[0], self.mode), rew, term, trunc, info


class MemoryVectorEnv:
    """Vectorized eval envs (autoreset disabled): one batched tracker, reset with the envs."""

    def __init__(self, envs, tracker, mode):
        self.envs, self.tracker, self.mode = envs, tracker, mode

    def __getattr__(self, name):
        return getattr(self.envs, name)

    def reset(self, *args, **kwargs):
        obs, info = self.envs.reset(*args, **kwargs)
        self.tracker.reset(np.arange(self.tracker.n), np.asarray(obs))
        return _augment(obs, self.tracker.read(), self.mode), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.envs.step(actions)
        # finished envs keep receiving dummy actions until the rollout ends; their memory is never read
        self.tracker.step(np.asarray(actions), np.asarray(obs))
        return _augment(obs, self.tracker.read(), self.mode), rew, term, trunc, info


def demo_episode_bounds(observations, next_observations):
    """(start, end) of each demo: robomimic writes next_obs[t] == obs[t+1] inside an episode."""
    cut = np.flatnonzero(np.any(next_observations[:-1] != observations[1:], axis=1)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(observations)]
    return list(zip(starts.tolist(), ends.tolist()))


def demo_memory(dataset, tracker):
    """(mem, next_mem), each (N, dim): memory of frames 0..t and 0..t+1 of each demo step."""
    obs, nxt, act = (np.asarray(dataset[k], np.float32) for k in ("observations", "next_observations", "actions"))
    bounds = demo_episode_bounds(obs, nxt)
    mem = np.zeros((len(obs), tracker.dim), np.float32)
    next_mem = np.zeros_like(mem)
    for g in range(0, len(bounds), tracker.n):
        group = bounds[g:g + tracker.n]
        # each episode plus its final next_obs, so row t+1 is the memory after the step from t
        eps = [(np.concatenate([obs[s:e], nxt[e - 1:e]]), act[s:e]) for s, e in group]
        for (s, e), m in zip(group, episode_memory(tracker, eps)):
            mem[s:e], next_mem[s:e] = m[:-1], m[1:]
    return mem, next_mem, bounds


def install(env, eval_env, eval_envs, train_dataset, cfg, n_eval_envs):
    """Wrap the envs and add memory to the demo dataset. Returns (env, eval_env, eval_envs, train_dataset)."""
    mode = cfg.get("mode") or "none"   # OGPO's CLI parser turns "none" into None
    assert mode in MODES, f"memory.mode must be one of {MODES}, got {mode!r}"
    if mode == "none":
        return env, eval_env, eval_envs, train_dataset
    from ogpo.utils.datasets import Dataset

    demo_tracker = _make_tracker(cfg, n_envs=128)
    mem, next_mem, bounds = demo_memory(train_dataset, demo_tracker)
    fields = {k: np.asarray(v) for k, v in train_dataset.items()}
    if mode == "critic":
        fields["full_states"] = np.concatenate([fields["observations"], mem], axis=-1)
        fields["next_full_states"] = np.concatenate([fields["next_observations"], next_mem], axis=-1)
    else:
        fields["observations"] = np.concatenate([fields["observations"], mem], axis=-1)
        fields["next_observations"] = np.concatenate([fields["next_observations"], next_mem], axis=-1)
    train_dataset = Dataset.create(**fields)
    print(f"[memory] mode={mode}: {len(bounds)} demo episodes, {len(mem)} steps, memory dim {mem.shape[1]} "
          f"(stage1 {cfg['stage1_dir']}, feature {cfg.get('feature', 'mean')}, "
          f"projection {cfg.get('projection') or 'none'})", flush=True)

    env = MemoryEnv(env, _make_tracker(cfg, 1), mode)
    if eval_env is not None:
        eval_env = MemoryEnv(eval_env, _make_tracker(cfg, 1), mode)
    if eval_envs is not None:
        eval_envs = MemoryVectorEnv(eval_envs, _make_tracker(cfg, n_eval_envs), mode)
    return env, eval_env, eval_envs, train_dataset
