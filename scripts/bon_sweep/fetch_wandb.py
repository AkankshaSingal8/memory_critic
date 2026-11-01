"""Pull the sweep's results back from W&B into the local layout the plot scripts read:
<out_root>/mc_decision_seed<C>/square_<ds>_<n>demos_seed<S>_step<T>_sde_bon<N>.json
with success_rate and mean_length (plus "source": the W&B run path). Use it where the runs were not
submitted, e.g. to bring results from another account or machine back to this one. Existing files are
kept unless --overwrite.

    python scripts/bon_sweep/fetch_wandb.py --out_root /data/user_data/$USER/memory_critic/bon_curve
    python scripts/bon_sweep/fetch_wandb.py --out_root plots/bon/fetched --group bon_curve_mc_critic
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "akankshasingal08"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "memory-critic"))
    ap.add_argument("--group", default=os.environ.get("WANDB_GROUP", "bon_curve_mc_critic"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import wandb
    runs = wandb.Api().runs(f"{args.entity}/{args.project}", filters={"group": args.group})
    written = kept = 0
    for run in runs:
        c = run.config
        if "policy" not in c or "critic_seed" not in c:
            continue
        out_dir = os.path.join(args.out_root, f"mc_decision_seed{c['critic_seed']}")
        os.makedirs(out_dir, exist_ok=True)
        for row in run.scan_history():
            T = row.get("bc_step")
            if T is None:
                continue
            for key, val in row.items():
                if not (key.startswith("bon") and key.endswith("/success")) or val is None:
                    continue
                N = int(key[3:-len("/success")])
                p = os.path.join(out_dir, f"{c['policy']}_step{int(T)}_sde_bon{N}.json")
                if os.path.exists(p) and not args.overwrite:
                    kept += 1
                    continue
                with open(p, "w") as f:
                    json.dump({"success_rate": val, "mean_length": row.get(f"bon{N}/mean_length"),
                               "episodes": c.get("episodes"), "bon_n": N, "critic_seed": c["critic_seed"],
                               "checkpoint_step": int(T), "source": "/".join(run.path)}, f, indent=1)
                written += 1
        print(f"{run.name}: {run.summary.get('n_results')} results ({run.state})")
    print(f"wrote {written} files, kept {kept} existing, under {args.out_root}")


if __name__ == "__main__":
    main()
