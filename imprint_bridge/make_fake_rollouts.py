"""Write synthetic square rollout files in the scripts/collect_flow_bc_rollouts.py format (smoke tests only).

    python -m imprint_bridge.make_fake_rollouts <out_dir> [--episodes 400] [--policies 2]
"""

import argparse
import os

import numpy as np


def fake_policy(rng, n_episodes, max_len):
    rows = {k: [] for k in ("obs", "action", "reward", "episode_id", "timestep", "chunk_pos", "subtask_status")}
    ep_success, ep_length = [], []
    for e in range(n_episodes):
        T = int(rng.integers(max_len // 4, max_len + 1))
        succ = T < max_len
        obs = np.cumsum(rng.normal(0, 0.05, (T, 23)), 0).astype(np.float32)
        sub = np.zeros((T, 3), np.int8)
        for j, frac in enumerate((0.3, 0.6, 1.0)):
            if succ or j < 2:
                sub[int(frac * (T - 1)):, j] = 1
        rows["obs"].append(obs)
        rows["action"].append(rng.uniform(-1, 1, (T, 7)).astype(np.float32))
        r = -np.ones(T, np.float32)
        if succ:
            r[-1] = 0.0
        rows["reward"].append(r)
        rows["episode_id"].append(np.full(T, e, np.int32))
        rows["timestep"].append(np.arange(T, dtype=np.int16))
        rows["chunk_pos"].append((np.arange(T) % 8).astype(np.int8))
        rows["subtask_status"].append(sub)
        ep_success.append(succ)
        ep_length.append(T)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out.update(ep_success=np.array(ep_success), ep_length=np.array(ep_length))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--policies", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    names = ["square_ph_10demos_seed0_sde", "square_mh_20demos_seed7_sde", "square_ph_50demos_seed42_sde"]
    for name in names[:args.policies]:
        np.savez(os.path.join(args.out_dir, f"{name}.npz"), **fake_policy(rng, args.episodes, args.max_len))
    print(f"wrote {args.policies} fake policies to {args.out_dir}")


if __name__ == "__main__":
    main()
