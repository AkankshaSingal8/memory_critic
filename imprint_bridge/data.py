"""Square low-dim rollouts -> IMPRINT memory-encoder inputs (pure numpy, shared by Stage 1 and the cache).

Low-dim shortcut of docs/imprint.md §4.2: the 14-d `object` state stands in for the visual streams.

  obs (23) = [eef_pos 3, eef_quat 4, gripper_qpos 2 | object 14]; proprio = obs[:, :9], object = obs[:, 9:].

History contract at decision step t (IMPRINT docs/TRAINING.md "Data Contract", frames 0..t, no downsample):
  long-term visual: one token per finalized chunk of TF frames, = concat of the TF object vectors (TF*14)
  recent visual:    one 14-d object token per frame after the last finalized chunk (< TF tokens)
  action stream:    causal executed action, a_causal[f] = a[f-1], zero at f = 0     (7)
  proprio stream:   proprio[f]                                                   (9)
All streams are z-scored with train-split statistics (`Norm`), saved next to the Stage-1 checkpoint.

Stage-1 targets at Q random past frames q <= t: object[q] (align), a_causal[q], proprio[q], and one
current-step progress scalar = mean of the cumulative subtask flags at t (grasped / lifted-near-peg /
success). The flags are sticky, so progress is a function of history, not of s_t alone.
"""

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

PROPRIO_DIM = 9
OBJECT_DIM = 14
ACTION_DIM = 7
POLICY_RE = re.compile(r"square_(ph|mh)_(\d+)demos_seed(\d+)_(sde|ode)")


@dataclass
class Episode:
    policy: str
    episode_id: int
    obj: np.ndarray        # (T, 14)
    proprio: np.ndarray    # (T, 9)
    action_causal: np.ndarray  # (T, 7)
    progress: np.ndarray   # (T,) in [0, 1], or None when the file has no subtask flags


@dataclass
class Norm:
    obj_mean: np.ndarray
    obj_std: np.ndarray
    pro_mean: np.ndarray
    pro_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def to_dict(self):
        return {k: np.asarray(v, np.float32) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: np.asarray(d[k], np.float32) for k in cls.__dataclass_fields__})

    @classmethod
    def fit(cls, episodes):
        o = np.concatenate([e.obj for e in episodes])
        p = np.concatenate([e.proprio for e in episodes])
        a = np.concatenate([e.action_causal for e in episodes])
        s = lambda x: np.maximum(x.std(0), 1e-3)  # noqa: E731
        return cls(o.mean(0), s(o), p.mean(0), s(p), a.mean(0), s(a))

    def apply(self, ep):
        return Episode(ep.policy, ep.episode_id,
                       ((ep.obj - self.obj_mean) / self.obj_std).astype(np.float32),
                       ((ep.proprio - self.pro_mean) / self.pro_std).astype(np.float32),
                       ((ep.action_causal - self.act_mean) / self.act_std).astype(np.float32),
                       ep.progress)


def rollout_files(data_dir):
    files = [p for p in sorted(glob.glob(os.path.join(data_dir, "*.npz")))
             if POLICY_RE.fullmatch(os.path.basename(p)[:-4])]
    assert files, f"no square_* rollout files in {data_dir}"
    return files


def causal_actions(action):
    out = np.zeros_like(action, dtype=np.float32)
    out[1:] = action[:-1]
    return out


def split_episodes(z, policy):
    """Cut one rollout npz into Episode objects (rows are grouped by episode_id, in time order)."""
    ep_ids = z["episode_id"]
    obs, act = z["obs"].astype(np.float32), z["action"].astype(np.float32)
    sub = z["subtask_status"] if "subtask_status" in z.files else None
    # collect_flow_bc_rollouts.py writes -1 when the env reports no subtask flags
    has_sub = sub is not None and sub.ndim == 2 and sub.shape[1] > 0 and (sub >= 0).all()
    timestep = z["timestep"]   # NpzFile re-reads a member on every access
    bounds = np.flatnonzero(np.diff(ep_ids)) + 1
    out = []
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(ep_ids)]):
        assert np.all(timestep[s:e] == np.arange(e - s)), f"{policy}: episode {ep_ids[s]} not contiguous"
        out.append(Episode(policy, int(ep_ids[s]), obs[s:e, PROPRIO_DIM:PROPRIO_DIM + OBJECT_DIM],
                           obs[s:e, :PROPRIO_DIM], causal_actions(act[s:e]),
                           sub[s:e].astype(np.float32).mean(1) if has_sub else None))
    return out


def load_episodes(data_dir, episode_filter=None):
    eps = []
    for p in rollout_files(data_dir):
        z = np.load(p)
        eps += [e for e in split_episodes(z, os.path.basename(p)[:-4])
                if episode_filter is None or episode_filter(e.episode_id)]
    return eps


@dataclass(frozen=True)
class Shapes:
    """Fixed padded shapes, so the jitted encoder compiles once."""
    temporal_factor: int = 4
    max_steps: int = 400          # longest episode (square time limit)
    num_query: int = 16

    @property
    def max_chunks(self):
        return self.max_steps // self.temporal_factor

    @property
    def max_recent(self):
        return self.temporal_factor


def history_inputs(ep, t, sh):
    """Encoder inputs for the history 0..t of one (normalized) episode, padded to `sh`."""
    TF = sh.temporal_factor
    n = t + 1
    assert n <= sh.max_steps, f"history of {n} steps > max_steps {sh.max_steps}"
    nc = n // TF
    lat = np.zeros((sh.max_chunks, TF * OBJECT_DIM), np.float32)
    lat_mask = np.zeros(sh.max_chunks, bool)
    lat[:nc] = ep.obj[:nc * TF].reshape(nc, TF * OBJECT_DIM)
    lat_mask[:nc] = True
    rec = np.zeros((sh.max_recent, OBJECT_DIM), np.float32)
    rec_mask = np.zeros(sh.max_recent, bool)
    r = n - nc * TF
    rec[:r] = ep.obj[nc * TF:n]
    rec_mask[:r] = True
    act = np.zeros((sh.max_steps, ACTION_DIM), np.float32)
    pro = np.zeros((sh.max_steps, PROPRIO_DIM), np.float32)
    step_mask = np.zeros(sh.max_steps, bool)
    act[:n] = ep.action_causal[:n]
    pro[:n] = ep.proprio[:n]
    step_mask[:n] = True
    return {"cosmo_latent_emb": lat, "cosmo_mem_mask": lat_mask,
            "cosmo_recent_emb": rec, "cosmo_recent_mask": rec_mask,
            "cosmo_action_emb": act, "cosmo_action_mask": step_mask,
            "cosmo_proprio_emb": pro, "cosmo_proprio_mask": step_mask}


def stage1_sample(ep, t, sh, rng):
    out = history_inputs(ep, t, sh)
    n = t + 1
    K = min(sh.num_query, n)
    q = np.zeros(sh.num_query, np.int64)
    qm = np.zeros(sh.num_query, bool)
    q[:K] = np.sort(rng.choice(n, K, replace=False))
    qm[:K] = True
    out.update(query_steps=q, query_mask=qm,
               align_target=ep.obj[q][:, None, :],   # (Q, P=1, 14)
               action_target=ep.action_causal[q], proprio_target=ep.proprio[q])
    if ep.progress is not None:
        out["reward_target"] = np.float32(ep.progress[t])
    return out


def collate(samples):
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


def stage1_batches(episodes, sh, batch_size, rng):
    """Infinite iterator: uniform over (episode, t) pairs, like IMPRINT's samples.json."""
    lengths = np.array([len(e.obj) for e in episodes])
    cum = np.cumsum(lengths)
    while True:
        flat = rng.integers(0, cum[-1], batch_size)
        ei = np.searchsorted(cum, flat, side="right")
        ts = flat - np.r_[0, cum[:-1]][ei]
        yield collate([stage1_sample(episodes[i], int(t), sh, rng) for i, t in zip(ei, ts)])
