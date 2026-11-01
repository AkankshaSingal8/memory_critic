"""Log one sweep task's best-of-N results to W&B: one run per (policy, critic seed).

The run id is fixed by the policy and critic seed (bonsweep-<ds>-<n>-s<S>-c<C>). If that run already
holds every (step, N) point, logging again does nothing; if it is incomplete (a task that failed part
way), the new results go to a fresh run with suffix -r1, -r2, ..., never appended to the old one. Each row of history is one BC checkpoint, with x = bc_step and
bon<N>/success, bon<N>/mean_length for every N; the per-run JSONs are attached as files and the
whole grid as a table, which scripts/bon_sweep/fetch_wandb.py reads back.

    python scripts/bon_sweep/log_wandb.py --out_dir <OUT_ROOT>/mc_decision_seed1 \
        --prefix square_ph_10demos_seed0 --steps 50000 ... --ns 8 16 32 --critic_seed 1
Environment: WANDB_ENTITY, WANDB_PROJECT, WANDB_GROUP, WANDB_MODE (as scripts/bon_sweep/config.sh).
"""
import argparse
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True, help="square_<ds>_<n>demos_seed<S>")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--critic_seed", type=int, required=True)
    ap.add_argument("--critic", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="print what would be logged")
    args = ap.parse_args()

    m = re.fullmatch(r"square_(ph|mh)_(\d+)demos_seed(\d+)", args.prefix)
    assert m, args.prefix
    ds, n_demos, bc_seed = m.group(1), int(m.group(2)), int(m.group(3))
    rows, files = [], []
    for T in sorted(args.steps):
        row = {"bc_step": T}
        for N in args.ns:
            p = os.path.join(args.out_dir, f"{args.prefix}_step{T}_sde_bon{N}.json")
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                row[f"bon{N}/success"] = d["success_rate"]
                row[f"bon{N}/mean_length"] = d["mean_length"]
                files.append(p)
        if len(row) > 1:
            rows.append(row)
    expected = len(args.steps) * len(args.ns)
    print(f"[wandb] {args.prefix} critic seed {args.critic_seed}: {len(files)}/{expected} results")
    if args.dry_run or not rows:
        for r in rows:
            print("  ", r)
        return 0

    import wandb
    entity = os.environ.get("WANDB_ENTITY", "akankshasingal08")
    project = os.environ.get("WANDB_PROJECT", "memory-critic")
    group = os.environ.get("WANDB_GROUP", "bon_curve_mc_critic")
    name = f"bon_{args.prefix}_critic{args.critic_seed}"
    base_id = f"bonsweep-{ds}-{n_demos}-s{bc_seed}-c{args.critic_seed}"
    run_id = base_id
    if os.environ.get("WANDB_MODE", "online") != "offline":
        # Leave a complete run alone; never append a second copy of the history to an existing run.
        try:
            api = wandb.Api()
            k = 0
            while True:
                try:
                    old = api.run(f"{entity}/{project}/{run_id}")
                except Exception:  # noqa: BLE001 - not found
                    break
                if old.summary.get("n_results") == expected:
                    print(f"[wandb] {run_id} already complete ({old.url}); not logging again")
                    return 0
                k += 1
                run_id = f"{base_id}-r{k}"
        except Exception as exc:  # noqa: BLE001 - no API access: just log
            print(f"[wandb] could not check existing runs ({type(exc).__name__}: {exc})")

    run = wandb.init(entity=entity, project=project, group=group, name=name, id=run_id,
                     job_type="bon_curve", dir=os.environ.get("TMPDIR"),
                     config={"dataset": ds, "n_demos": n_demos, "bc_seed": bc_seed,
                             "critic_seed": args.critic_seed, "ns": args.ns, "steps": args.steps,
                             "episodes": args.episodes, "critic": args.critic, "run_dir": args.run_dir,
                             "policy": args.prefix, "host": os.uname().nodename,
                             "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")})
    run.define_metric("bc_step")
    run.define_metric("bon*", step_metric="bc_step")
    for r in rows:
        run.log(r)
    cols = ["bc_step", "N", "success", "mean_length"]
    table = wandb.Table(columns=cols, data=[[r["bc_step"], N, r[f"bon{N}/success"], r[f"bon{N}/mean_length"]]
                                            for r in rows for N in args.ns if f"bon{N}/success" in r])
    run.log({"results": table})
    for p in files:
        run.save(p, base_path=args.out_dir, policy="now")
    run.summary["n_results"] = len(files)
    run.finish()
    print(f"[wandb] logged {len(files)} results to {entity}/{project}/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
