"""Tracker vs full forward: the online memory must equal the offline (cache_memory.py) memory.

    python -m imprint_bridge.test_tracker <stage1_dir> <rollout_dir> [--feature mean|progress]

Runs in either env (.venv-imprint, or OGPO's with flax 0.12). For a few episodes, steps a MemoryTracker
through every frame and compares read() at each step with the encoder's full forward pass over the
padded history 0..t (imprint_bridge.data.history_inputs), the computation cache_memory.py does.
"""

import argparse
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--feature", default="mean")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from imprint_bridge import data as D
    from imprint_bridge.tracker import MemoryTracker, episode_memory

    tr = MemoryTracker(args.stage1_dir, n_envs=args.episodes, feature=args.feature, compute_dtype="float32")
    s1 = json.load(open(os.path.join(args.stage1_dir, "stage1_summary.json")))["config"]
    sh = D.Shapes(max_steps=s1["max_steps"])
    z = np.load(D.rollout_files(args.data_dir)[0])
    raw = D.split_episodes(z, "p")[:args.episodes]
    obs, act = z["obs"].astype(np.float32), z["action"].astype(np.float32)
    bounds = np.r_[0, np.flatnonzero(np.diff(z["episode_id"])) + 1]
    eps = [(obs[b:b + len(e.obj)], act[b:b + len(e.obj)]) for b, e in zip(bounds, raw)]

    @jax.jit
    def full(p, b):
        e, d = nnx.merge(tr._graphdef, p)
        mem, _, _ = e(b["cosmo_latent_emb"], b["cosmo_recent_emb"], b["cosmo_action_emb"], b["cosmo_proprio_emb"],
                      b["cosmo_mem_mask"], recent_mask=b["cosmo_recent_mask"], action_mask=b["cosmo_action_mask"],
                      proprio_mask=b["cosmo_proprio_mask"])
        if args.feature == "mean":
            return mem.mean(1)
        rq = jnp.broadcast_to(d.reward_query.value[None, None], (mem.shape[0], 1, d.dim))
        return (rq + d.reward_cross(d.reward_norm_q(rq), d.reward_norm_kv(mem)))[:, 0]

    online = episode_memory(tr, eps)
    worst = 0.0
    for ep_raw, m_on in zip(raw, online):
        ep = tr.norm.apply(ep_raw)
        ref = np.concatenate([np.asarray(full(tr._params, D.collate([D.history_inputs(ep, t, sh)
                                                                  for t in range(i, min(i + 64, len(ep.obj)))])))
                              for i in range(0, len(ep.obj), 64)])
        err = float(np.abs(ref - m_on).max())
        scale = float(np.abs(ref).max())
        print(f"episode {ep_raw.episode_id}: {len(ep.obj)} steps, max |online - full| = {err:.2e} (|full| max {scale:.2f})")
        worst = max(worst, err)
    ok = worst <= args.atol
    print("PASS" if ok else "FAIL", f"worst {worst:.2e} vs atol {args.atol}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
