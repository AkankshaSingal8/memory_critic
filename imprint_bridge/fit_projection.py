"""Fit the fixed map from pooled 1024-d IMPRINT memory to OGPO's network input (tracker.Projection).

Fit on experiment A's memory cache (imprint_bridge/cache_memory.py), train-split episodes only, the
same rows and the same transform train_mc_critic.py's memory_inputs uses:

  --n_components K > 0   whitened PCA: (x - mu) @ V_K / sigma_K       -> K dims
  --n_components 0       z-score only: (x - mu) / std                 -> 1024 dims

    python -m imprint_bridge.fit_projection --cache_dir <memory/<init>> --feature mean --n_components 64 --out proj.npz
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def fit(x, n_components):
    mu = x.mean(0)
    if n_components <= 0:
        return {"mu": mu, "scale": np.maximum(x.std(0), 1e-6)}, None
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    std = s[:n_components] / np.sqrt(len(x) - 1)
    explained = float((s[:n_components] ** 2).sum() / (s ** 2).sum())
    return {"mu": mu, "components": vt[:n_components], "scale": np.maximum(std, 1e-6)}, explained


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--feature", choices=("mean", "progress"), default="mean")
    ap.add_argument("--n_components", type=int, default=64)
    ap.add_argument("--val_start", type=int, default=288, help="train split: episode_id < val_start")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.cache_dir, "square_*.npz")))
    assert files, f"no cache files in {args.cache_dir}"
    rows = []
    for p in files:
        z = np.load(p)
        rows.append(z[f"feat_{args.feature}"][z["episode_id"] < args.val_start].astype(np.float32))
    x = np.concatenate(rows)
    params, explained = fit(x, args.n_components)
    np.savez(args.out, **{k: v.astype(np.float32) for k, v in params.items()})
    meta = {"cache_dir": os.path.abspath(args.cache_dir), "feature": args.feature,
            "n_components": args.n_components, "n_rows": int(len(x)), "n_files": len(files),
            "explained_variance": explained}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[projection] {len(x)} train rows from {len(files)} files -> {args.out}: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
