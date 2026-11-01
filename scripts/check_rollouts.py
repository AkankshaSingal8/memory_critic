"""Sanity checks on collected rollout files: python scripts/check_rollouts.py <dir> [<dir> ...]."""
import glob, json, os, sys
import numpy as np

for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p)
        T = len(z["reward"]); E = len(z["ep_length"])
        errs = []
        if T != z["ep_length"].astype(int).sum(): errs.append("T != sum(ep_length)")
        ends = np.flatnonzero(z["done"])
        if len(ends) != E: errs.append(f"{len(ends)} done flags for {E} episodes")
        same_ep = z["episode_id"][1:] == z["episode_id"][:-1]
        gap = np.abs(z["next_obs"][:-1][same_ep] - z["obs"][1:][same_ep]).max() if same_ep.any() else 0
        if gap > 1e-5: errs.append(f"next_obs[t] != obs[t+1] (max gap {gap})")
        if not np.array_equal(z["chunk_pos"], z["timestep"] % 8): errs.append("chunk_pos != timestep % 8")
        if not np.array_equal(z["terminated"] | z["truncated"], z["done"]): errs.append("done != term|trunc")
        last = np.flatnonzero(np.r_[~same_ep, True])
        if not np.array_equal(z["ep_success"], z["is_success"][last]): errs.append("ep_success != last is_success")
        if (z["subtask_status"] < 0).all(): errs.append("subtask_status missing (all -1)")
        print(f"{os.path.basename(p)}: {E} eps, {T} transitions, obs {z['obs'].shape[1]}, "
              f"success {z['ep_success'].mean():.3f}, len {z['ep_length'].mean():.0f}  "
              + ("OK" if not errs else "FAIL: " + "; ".join(errs)))
