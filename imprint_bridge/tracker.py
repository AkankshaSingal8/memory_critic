"""Online IMPRINT memory for live rollouts: one recurrent memory state per env, updated every env step.

Experiment B (docs/ogpo_memory_plan.md) needs m_t while OGPO acts and trains, so it can't come from the
offline cache. `MemoryTracker` keeps IMPRINT's incremental gated-delta state per env and follows the same
history contract and normalization as imprint_bridge/data.py:

  reset(env, obs_0)          frame 0 = obs_0, causal action = 0
  step(actions, next_obs)    frame t+1 = next_obs, causal action = the action just executed
  read()                     m_t from frames 0..t: finalized 4-frame chunks live in the recurrent state
                             (update_incremental); the < 4 frames after them are fused exactly
                             (read_incremental), as the full forward pass in cache_memory.py does.

So `read()` after t steps equals cache_memory's features for history 0..t (test_tracker.py checks this).

Self-contained on purpose: it imports only IMPRINT's model files (jax, flax.nnx), not its trainer, and
reads the history YAML with PyYAML. That keeps it importable in OGPO's env (jax 0.10 / flax 0.12),
where IMPRINT's modules need `pytree=False` (they hold plain Python lists of submodules).
"""

import json
import os

import numpy as np

from imprint_bridge import data as D


class AttrDict(dict):
    """Minimal stand-in for an OmegaConf node: IMPRINT reads its config with getattr(cfg, key, default)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    @classmethod
    def wrap(cls, x):
        if isinstance(x, dict):
            return cls({k: cls.wrap(v) for k, v in x.items()})
        if isinstance(x, list):
            return [cls.wrap(v) for v in x]
        return x


def load_history_config(path, compute_dtype, overrides=()):
    """The history YAML as train_stage1.load_config builds it, without OmegaConf."""
    import yaml
    cfg = AttrDict.wrap(yaml.safe_load(open(path)))
    cfg.cosmo_bottleneck_memory.memory_compute_dtype = compute_dtype
    cfg.cosmo_bottleneck_memory.delta_wy_matmul_dtype = compute_dtype
    for item in overrides:   # OmegaConf dotlist entries, e.g. "memory_token_dim=64"
        key, val = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
    return cfg


def _compat(cls):
    """flax >= 0.12 refuses plain lists of submodules in a pytree Module; opt this class out of pytree mode."""
    try:
        return type(cls.__name__, (cls,), {}, pytree=False)
    except TypeError:   # flax 0.10 (IMPRINT's own env): no pytree flag, lists already fine
        return cls


def _load_params(module, npz, prefix):
    """Fill `module` from a Stage-1 npz ('<prefix>/<path>' keys, as IMPRINT's _save_checkpoint writes)."""
    import flax.traverse_util as trav
    from flax import nnx
    graphdef, st = nnx.split(module)
    flat = trav.flatten_dict(st.to_pure_dict())
    missing = []
    for key, cur in flat.items():
        name = f"{prefix}/" + "/".join(str(x) for x in key)
        if name not in npz.files:
            missing.append(name)
            continue
        arr = np.asarray(npz[name])
        assert arr.shape == np.shape(cur), f"{name}: checkpoint {arr.shape} vs model {np.shape(cur)}"
        flat[key] = arr.astype(np.asarray(cur).dtype)
    if missing:
        raise KeyError(f"Stage-1 checkpoint lacks {len(missing)} {prefix} params, first {missing[:3]}")
    st.replace_by_pure_dict(trav.unflatten_dict(flat))
    return nnx.merge(graphdef, st)


class Projection:
    """Fixed map from the pooled 1024-d memory to the network input: x -> ((x - mu) @ W) / scale."""

    def __init__(self, mu, components=None, scale=None):
        self.mu = np.asarray(mu, np.float32)
        self.W = None if components is None else np.asarray(components, np.float32).T
        self.scale = np.asarray(scale, np.float32) if scale is not None else None

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mu"], z["components"] if "components" in z.files else None,
                   z["scale"] if "scale" in z.files else None)

    @property
    def out_dim(self):
        return len(self.mu) if self.W is None else self.W.shape[1]

    def __call__(self, x):
        x = x - self.mu
        if self.W is not None:
            x = x @ self.W
        return x / self.scale if self.scale is not None else x


class MemoryTracker:
    """Batched online IMPRINT memory for `n_envs` envs. Arrays are numpy in, numpy out."""

    def __init__(self, stage1_dir, n_envs, feature="mean", ckpt="stage1_stepbest.npz", projection=None,
                 compute_dtype=None):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from mme_vla_suite.models.representation.cosmo_decoder import CosmoMemoryDecoder
        from mme_vla_suite.models.representation.gated_delta_mem import GatedDeltaBottleneckMemory

        assert feature in ("mean", "progress"), feature
        s1 = json.load(open(os.path.join(stage1_dir, "stage1_summary.json")))["config"]
        dtype_name = compute_dtype or s1["compute_dtype"]
        cfg = load_history_config(s1["config"], dtype_name, s1.get("config_override", []))
        dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
        rngs = nnx.Rngs(0)
        enc = _compat(GatedDeltaBottleneckMemory)(cfg, rngs=rngs, dtype=dtype)
        dec = _compat(CosmoMemoryDecoder)(cfg, rngs=rngs, dtype=dtype)
        npz = np.load(os.path.join(stage1_dir, ckpt), allow_pickle=True)
        enc, dec = _load_params(enc, npz, "bottleneck"), _load_params(dec, npz, "decoder")
        self.norm = D.Norm.from_dict(np.load(os.path.join(stage1_dir, "norm.npz")))
        self.projection = Projection.load(projection) if isinstance(projection, str) else projection
        self.n, self.TF, self.feature = n_envs, enc.chunk_frames, feature
        self.dim = self.projection.out_dim if self.projection is not None else int(cfg.memory_token_dim)
        graphdef, params = nnx.split((enc, dec))
        self._graphdef, self._params = graphdef, params

        def init(n):
            return nnx.merge(graphdef, params)[0].init_incremental_state(n)

        @jax.jit
        def update(p, state, chunk, act, pro, which):
            e, _ = nnx.merge(graphdef, p)
            new = e.update_incremental(state, chunk, act, pro)
            keep = lambda a, b: jnp.where(which.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)  # noqa: E731
            return jax.tree.map(keep, new, state)

        @jax.jit
        def read(p, state, rec, act, pro, mask):
            e, d = nnx.merge(graphdef, p)
            mem, _, _ = e.read_incremental(state, rec, act, pro, recent_mask=mask, action_mask=mask,
                                           proprio_mask=mask)
            if feature == "mean":
                return mem.mean(1).astype(jnp.float32)
            # the decoder's progress query (as in cache_memory.py), stopped before its scalar head
            rq = jnp.broadcast_to(d.reward_query.value.astype(d.dtype)[None, None], (mem.shape[0], 1, d.dim))
            rq = rq + d.reward_cross(d.reward_norm_q(rq), d.reward_norm_kv(mem))
            return rq[:, 0].astype(jnp.float32)

        @jax.jit
        def reset_state(state, fresh, which):
            keep = lambda a, b: jnp.where(which.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)  # noqa: E731
            return jax.tree.map(keep, fresh, state)

        self._update, self._read, self._reset_state = update, read, reset_state
        self._fresh = init(n_envs)
        self.state = self._fresh
        TF = self.TF
        self.rec_obj = np.zeros((n_envs, TF, D.OBJECT_DIM), np.float32)
        self.rec_act = np.zeros((n_envs, TF, D.ACTION_DIM), np.float32)
        self.rec_pro = np.zeros((n_envs, TF, D.PROPRIO_DIM), np.float32)
        self.r = np.zeros(n_envs, np.int64)          # frames since the last finalized chunk
        self.frames = np.zeros(n_envs, np.int64)     # frames seen this episode
        self.zero_act = ((np.zeros(D.ACTION_DIM, np.float32) - self.norm.act_mean) / self.norm.act_std)

    # ------------------------------------------------------------------ frames

    def _push(self, envs, obs, act_norm):
        """Append one normalized frame to each env in `envs` (index array)."""
        obs = np.asarray(obs, np.float32)
        obj = (obs[:, D.PROPRIO_DIM:D.PROPRIO_DIM + D.OBJECT_DIM] - self.norm.obj_mean) / self.norm.obj_std
        pro = (obs[:, :D.PROPRIO_DIM] - self.norm.pro_mean) / self.norm.pro_std
        slot = self.r[envs]
        self.rec_obj[envs, slot], self.rec_act[envs, slot], self.rec_pro[envs, slot] = obj, act_norm, pro
        self.r[envs] += 1
        self.frames[envs] += 1
        full = envs[self.r[envs] == self.TF]
        if len(full):   # finalize the chunk: it moves from the exact recent window into the recurrent state
            which = np.zeros(self.n, bool)
            which[full] = True
            # copies: on CPU, jax may alias numpy buffers and runs asynchronously, so zeroing the
            # buffers below could otherwise race with the update that reads them
            chunk = self.rec_obj.reshape(self.n, 1, self.TF * D.OBJECT_DIM).copy()
            self.state = self._update(self._params, self.state, chunk, self.rec_act.copy(), self.rec_pro.copy(),
                                      which)
            self.r[full] = 0
            self.rec_obj[full] = self.rec_act[full] = self.rec_pro[full] = 0.0

    def reset(self, envs, obs0):
        """Start a new episode in each env of `envs` (indices or bool mask) at observation `obs0` (len(envs), 23)."""
        envs = np.flatnonzero(envs) if np.asarray(envs).dtype == bool else np.asarray(envs, np.int64)
        if not len(envs):
            return
        which = np.zeros(self.n, bool)
        which[envs] = True
        self.state = self._reset_state(self.state, self._fresh, which)
        self.r[envs] = 0
        self.frames[envs] = 0
        self.rec_obj[envs] = self.rec_act[envs] = self.rec_pro[envs] = 0.0
        self._push(envs, obs0, np.broadcast_to(self.zero_act, (len(envs), D.ACTION_DIM)))

    def step(self, actions, next_obs, active=None):
        """After env.step: actions (n, 7) just executed, next_obs (n, 23). `active` (bool, n) skips finished envs."""
        envs = np.arange(self.n) if active is None else np.flatnonzero(active)
        if not len(envs):
            return
        a = (np.asarray(actions, np.float32)[envs] - self.norm.act_mean) / self.norm.act_std
        self._push(envs, np.asarray(next_obs)[envs], a)

    def read(self):
        """(n_envs, dim) memory of each env's history 0..t (projected if a projection is set)."""
        mask = np.arange(self.TF)[None, :] < self.r[:, None]
        m = np.asarray(self._read(self._params, self.state, self.rec_obj.copy(), self.rec_act.copy(),
                                  self.rec_pro.copy(), mask))
        return self.projection(m) if self.projection is not None else m


def episode_memory(tracker, episodes):
    """Per-step memory for whole recorded episodes, e.g. the demos for B2's BC phase.

    episodes: list of (obs (T, 23), actions (T, 7)) with len(episodes) <= tracker.n.
    Returns a list of (T, dim) arrays: row t = memory of frames 0..t, for observations[t].
    """
    k = len(episodes)
    assert 0 < k <= tracker.n
    lengths = np.array([len(o) for o, _ in episodes])
    out = [np.zeros((L, tracker.dim), np.float32) for L in lengths]
    tracker.reset(np.arange(k), np.stack([o[0] for o, _ in episodes]))
    for t in range(lengths.max()):
        m = tracker.read()
        for i in np.flatnonzero(lengths > t):
            out[i][t] = m[i]
        live = np.zeros(tracker.n, bool)
        live[:k] = lengths > t + 1
        if not live.any():
            break
        acts = np.zeros((tracker.n, D.ACTION_DIM), np.float32)
        nxt = np.zeros((tracker.n, D.PROPRIO_DIM + D.OBJECT_DIM), np.float32)
        for i in np.flatnonzero(live[:k]):
            acts[i], nxt[i] = episodes[i][1][t], episodes[i][0][t + 1]
        tracker.step(acts, nxt, live)
    return out
