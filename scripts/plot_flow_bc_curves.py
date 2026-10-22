"""Task success % vs BC gradient steps for the flow-BC dataset-size sweep (scripts/submit_flow_bc.sh),
the eval_sde/success (and eval/success, ODE) vs eval/bc_step curves the CPU eval job logs, as in
SafeVaDaR qipm BC.

Pulls every run of each group from W&B (so it doubles as the W&B logging check) and, when the
run dirs are readable (/data is on compute nodes only), cross-checks each W&B point against
<run_dir>/cpu_eval/step_<N>.json.

Every finished seed of a group is used: lines are the mean over seeds, bands the min-max.

  sde_success_vs_steps.png      SDE success, one line per dataset size (each seed smoothed with the
                                centred 3-point mean select_bc_checkpoints.py uses, then averaged)
  ode_success_vs_steps.png      the same for ODE
  success_per_dataset_size.png  one panel per dataset size: unsmoothed SDE and ODE per checkpoint;
                                each seed's selected best step (selected/target_100.pkl) in the title
  wandb_check.txt               what W&B holds per run, and the cpu_eval cross-check
  curves.json                   the plotted data

    srun -p maxlab-cpu -c 1 --mem=2G --time=01:00:00 env OPENBLAS_NUM_THREADS=1 \
        .venv/bin/python scripts/plot_flow_bc_curves.py [--demos 10 20 50 100 200]
  (or: sbatch scripts/plot_flow_bc_curves.sbatch). Replot without W&B (login node is fine):
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_flow_bc_curves.py --cached
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ENTITY_PROJECT = "akankshasingal08/memory-critic"
STEP_KEY = "eval/bc_step"
KEYS = {"sde": "eval_sde/success", "ode": "eval/success"}
# Ordinal ramp for dataset size (dataviz reference sequential blue, steps 250-650; validated
# --ordinal); per-panel SDE / ODE use categorical slots 1-2.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SDE_C, ODE_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def smooth(y):
    """Centred 3-point mean; endpoints average the neighbours that exist."""
    return np.array([y[max(0, i - 1):i + 2].mean() for i in range(len(y))])


def fetch(args):
    import time
    t0 = time.time()
    import wandb
    print(f"[{time.time() - t0:.0f}s] imported wandb", flush=True)
    api = wandb.Api(timeout=120)
    data, report = {}, []
    for n in args.demos:
        group = f"{args.task}_{args.dataset_type}_flow_bc_{n}demos"
        runs = list(api.runs(ENTITY_PROJECT, filters={"group": group}))
        print(f"[{time.time() - t0:.0f}s] {group}: {len(runs)} run(s)", flush=True)
        report.append(f"== {group}: {len(runs)} W&B run(s)")
        for r in sorted(runs, key=lambda r: int(re.search(r"qipmseed(\d+)", r.name).group(1))):
            seed = re.search(r"qipmseed(\d+)", r.name).group(1)
            # history() with a key filter returns only rows holding those keys; samples is far
            # above the row counts (50 evals, ~100-200 loss rows), so nothing is subsampled.
            ev = r.history(keys=[STEP_KEY, KEYS["sde"], KEYS["ode"]], samples=100_000, pandas=False)
            ev = sorted({int(row[STEP_KEY]): row for row in ev}.values(), key=lambda row: row[STEP_KEY])
            train = r.history(keys=["offline_agent/bc_loss"], samples=100_000, pandas=False)
            print(f"[{time.time() - t0:.0f}s]   seed {seed}: {len(ev)} eval rows, {len(train)} bc_loss rows",
                  flush=True)
            prefixes = sorted({k.split("/")[0] for k in r.summary.keys() if "/" in k})
            best = r.summary.get("levels/target_100/step")
            report.append(f"   seed {seed}: run {r.id} '{r.name}' state={r.state} url={r.url}")
            report.append(f"     summary key prefixes: {prefixes}")
            report.append(f"     offline_agent/bc_loss rows: {len(train)}  eval rows: {len(ev)}"
                          + (f" (bc_step {int(ev[0][STEP_KEY])}..{int(ev[-1][STEP_KEY])})" if ev else ""))
            report.append(f"     levels/target_100/step (best checkpoint): {best}")
            d = {"run_id": r.id, "name": r.name, "best_step": best,
                 "step": [int(row[STEP_KEY]) for row in ev],
                 "sde": [float(row[KEYS["sde"]]) for row in ev],
                 "ode": [float(row[KEYS["ode"]]) for row in ev]}
            report += cross_check(args, n, seed, d)
            if best is None:
                report.append("     eval not finished (no levels/target_100 yet); not plotted")
                continue
            data.setdefault(str(n), {})[seed] = d
    return data, report


def cross_check(args, n, seed, d):
    pat = os.path.join(args.save_root, f"{args.task}_{args.dataset_type}_flow_bc_{n}demos", "*",
                       f"*_qipmseed{seed}_r*", "cpu_eval")
    dirs = glob.glob(pat)
    if len(dirs) != 1:
        return [f"     cpu_eval cross-check skipped ({len(dirs)} dirs match {pat})"]
    files = {int(os.path.basename(f)[5:-5]): f for f in glob.glob(os.path.join(dirs[0], "step_*.json"))}
    bad = []
    for i, s in enumerate(d["step"]):
        if s not in files:
            bad.append(f"{s}: not in cpu_eval")
            continue
        j = json.load(open(files[s]))
        if abs(j["sde"]["success"] - d["sde"][i]) > 1e-9 or abs(j["ode"]["success"] - d["ode"][i]) > 1e-9:
            bad.append(f"{s}: W&B sde/ode {d['sde'][i]}/{d['ode'][i]} vs file "
                       f"{j['sde']['success']}/{j['ode']['success']}")
    missing = sorted(set(files) - set(d["step"]))
    out = [f"     cpu_eval cross-check: {len(files)} files, {len(d['step'])} W&B points, "
           f"{len(bad)} mismatches, {len(missing)} steps missing from W&B"]
    return out + [f"       {b}" for b in bad[:10]] + ([f"       missing: {missing}"] if missing else [])


def seed_stats(runs, mode, smoothed):
    """Steps every seed has; mean, min and max over seeds (each seed smoothed first if asked)."""
    steps = sorted(set.intersection(*(set(d["step"]) for d in runs.values())))
    ys = []
    for d in runs.values():
        y = np.array([d[mode][d["step"].index(s)] for s in steps])
        ys.append(100 * (smooth(y) if smoothed else y))
    ys = np.array(ys)
    return np.array(steps), ys.mean(0), ys.min(0), ys.max(0)


def style(ax, ylabel=True):
    from matplotlib.ticker import FuncFormatter
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 510_000)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1e3)}k"))
    ax.set_xlabel("BC gradient steps", color=INK2)
    if ylabel:
        ax.set_ylabel("task success (%)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def seeds_label(data):
    seeds = sorted({sd for runs in data.values() for sd in runs}, key=int)
    return f"seed {seeds[0]}" if len(seeds) == 1 else f"mean of seeds {', '.join(seeds)}; band = min-max"


def plot_sweep(plt, data, mode, title, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE)
    sizes = sorted(data, key=int)
    for colour, n in zip(RAMP[len(RAMP) - len(sizes):], sizes):
        x, m, lo, hi = seed_stats(data[n], mode, smoothed=True)
        if len(data[n]) > 1:
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(x, m, color=colour, lw=2, label=f"{n} demos", solid_capstyle="round")
    style(ax)
    ax.set_title(f"{title}\n{seeds_label(data)}", color=INK, loc="left", fontsize=11)
    h, lab = ax.get_legend_handles_labels()   # largest dataset first: matches top-to-bottom order
    ax.legend(h[::-1], lab[::-1], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_panels(plt, data, out):
    sizes = sorted(data, key=int)
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 3.6), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        runs = data[n]
        for mode, colour, label in (("sde", SDE_C, "SDE"), ("ode", ODE_C, "ODE")):
            x, m, lo, hi = seed_stats(runs, mode, smoothed=False)
            if len(runs) > 1:
                ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0)
            ax.plot(x, m, color=colour, lw=1.6, label=label)
        style(ax, ylabel=ax is axes[0][0])
        best = ", ".join(f"{runs[sd]['best_step'] // 1000}k" for sd in sorted(runs, key=int))
        ax.set_title(f"{n} demos  (best: {best})", color=INK, loc="left", fontsize=10)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper right", ncol=2, labelcolor=INK2)
    fig.suptitle("Flow BC on square-ph: success per checkpoint, unsmoothed (64 episodes per seed; "
                 f"{seeds_label(data)}; best = selected checkpoint per seed)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="square")
    ap.add_argument("--dataset_type", default="ph")
    ap.add_argument("--demos", nargs="+", type=int, default=[10, 20, 50, 100, 200])
    ap.add_argument("--save_root",
                    default=f"/data/user_data/{os.environ.get('USER')}/memory_critic/flow_bc/memory-critic")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--cached", action="store_true",
                    help="replot from <out_dir>/curves.json instead of fetching from W&B")
    args = ap.parse_args()
    out_dir = args.out_dir or f"plots/flow_bc_{args.task}_{args.dataset_type}"
    os.makedirs(out_dir, exist_ok=True)

    if args.cached:
        data = json.load(open(os.path.join(out_dir, "curves.json")))
        # curves.json from the single-seed version: {demos: run} -> {demos: {seed: run}}
        data = {n: ({re.search(r"qipmseed(\d+)", v["name"]).group(1): v} if "step" in v else v)
                for n, v in data.items()}
    else:
        data, report = fetch(args)
        with open(os.path.join(out_dir, "wandb_check.txt"), "w") as f:
            f.write("\n".join(report) + "\n")
        with open(os.path.join(out_dir, "curves.json"), "w") as f:
            json.dump(data, f, indent=1)
        print("\n".join(report))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = f"{args.task}-{args.dataset_type}"
    plot_sweep(plt, data, "sde", f"Flow BC on {tag}: SDE success vs BC steps (3-point smoothed)",
               os.path.join(out_dir, "sde_success_vs_steps.png"))
    plot_sweep(plt, data, "ode", f"Flow BC on {tag}: ODE success vs BC steps (3-point smoothed)",
               os.path.join(out_dir, "ode_success_vs_steps.png"))
    plot_panels(plt, data, os.path.join(out_dir, "success_per_dataset_size.png"))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
