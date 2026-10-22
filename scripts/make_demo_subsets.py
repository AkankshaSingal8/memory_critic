"""Nested demo subsets of a robomimic low_dim dataset (e.g. square ph: 10 < 20 < 50 < 100 < 200).

Same construction as SafeVaDaR scripts/safety_pretrain/make_relabel_subsets.py: demo groups are
copied verbatim from the source file, data.attrs carried over and "total" recomputed, mask/ keeps
only listed demos, and the chosen demo keys are recorded in data.attrs["subset_provenance"].
Demos are drawn from ONE numpy default_rng(SEED) permutation of all demos, and subset N is its
first N entries, so every smaller subset is contained in every larger one.

Output: <out_root>/<task>_<type>_<N>/<task>/<type>/low_dim_v15.hdf5, i.e. a ROBOMIMIC_DATASET_ROOT
per subset (the layout envs/robomimic_utils._check_dataset_exists resolves). Existing outputs are
left alone after checking they hold the expected demos.

    .venv/bin/python scripts/make_demo_subsets.py --task square --dataset_type ph --sizes 10 20 50 100 200
"""
import argparse
import json
import os

import h5py
import numpy as np


def demo_key(k):
    return int(k.split("_")[1])


def main():
    user = os.environ["USER"]
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--dataset_type", default="ph")
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src_root", default=f"/data/user_data/{user}/robomimic")
    p.add_argument("--out_root", default=f"/data/user_data/{user}/robomimic_subsets")
    args = p.parse_args()

    rel = os.path.join(args.task, args.dataset_type, "low_dim_v15.hdf5")
    src_path = os.path.join(args.src_root, rel)
    with h5py.File(src_path, "r") as src:
        all_demos = sorted(src["data"].keys(), key=demo_key)
        assert max(args.sizes) <= len(all_demos), f"{src_path} has only {len(all_demos)} demos"
        order = [str(k) for k in np.random.default_rng(args.seed).permutation(all_demos)]
        prev = set()
        for n in sorted(args.sizes):
            keys = sorted(order[:n], key=demo_key)
            assert len(set(keys)) == n and prev <= set(keys)
            prev = set(keys)
            out = os.path.join(args.out_root, f"{args.task}_{args.dataset_type}_{n}", rel)
            if os.path.exists(out):
                with h5py.File(out, "r") as f:
                    have = sorted(f["data"].keys(), key=demo_key)
                assert have == keys, f"{out} exists with different demos; remove it to rebuild"
                print(f"{n} demos -> {out} (exists, verified)")
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".partial"
            with h5py.File(tmp, "w") as dst:
                g = dst.create_group("data")
                for a, v in src["data"].attrs.items():
                    g.attrs[a] = v
                for k in keys:
                    src.copy(src[f"data/{k}"], g, name=k)
                g.attrs["total"] = int(sum(src[f"data/{k}/actions"].shape[0] for k in keys))
                g.attrs["subset_provenance"] = json.dumps(
                    {"source": src_path, "n_demos": n, "seed": args.seed,
                     "construction": "first n of one numpy default_rng(seed) permutation of all demos (nested)",
                     "demos": keys})
                if "mask" in src:
                    m = dst.create_group("mask")
                    for name in src["mask"]:
                        kept = [x for x in src["mask"][name][:] if x.decode() in set(keys)]
                        m.create_dataset(name, data=np.array(kept, dtype="S"))
                total = int(g.attrs["total"])
            os.replace(tmp, out)
            print(f"{n} demos -> {out} ({total} transitions)")


if __name__ == "__main__":
    main()
