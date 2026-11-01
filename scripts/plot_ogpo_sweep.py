"""SDE success plots for the public-repo OGPO runs, one folder per dataset, in the style of
plots/flow_bc_square_{ph,mh}/ (scripts/plot_flow_bc_curves.py).

  plots/ogpo_square_{dt}/      OGPO with critics (KIND=ogpo), dataset-size sweep plus the full dataset
    sde_success_vs_online_steps.png  one line per dataset size, x = online env steps after the BC
                                     hand-off; each seed 50k-step moving average, then mean over seeds,
                                     band = min-max; each line stops at the step all its seeds have
                                     reached (averaging fewer seeds past that point shifts the mean)
    sde_success_vs_dataset_size.png  per dataset size: BC only (final), OGPO at the hand-off, OGPO
                                     after 2M online steps (last 50k average); dots = seeds
    curves.json / wandb_check.txt    the plotted data / what W&B and the job logs hold per run
  plots/ogpo_bc_square_{dt}/   OGPO repo BC phase only (KIND=bc), from plots/ogpo_public/curves.json
    sde_success_vs_steps.png         one line per dataset size, 3-point smoothed, mean over seeds
    sde_success_vs_dataset_size.png  at the eval step where the seed-mean curve peaks, and at 500k
  plots/ogpo_sweep/            the two OGPO figures above with ph and mh side by side

SDE evals in this repo go through sample_actions_BON (8 chunks scored by the critic). In BC-only
runs the critic is never trained, so that score is random.

Runs: groups square_{dt}_ogpo_public_{N}demos (sweep) and square_{dt}_ogpo_public (full: ph 200
demos, mh 300). A run preempted in the BC phase restarts BC in a new W&B run, so per seed the run
with the largest step is used. The hand-off step is the first "online_start step=" in the job
log (logs/ogpo_public_square_{dt}[_{N}demos]_seed{S}_<job>.out, oldest job first); online env
steps = W&B step - hand-off. For a finished run this equals the last step - 2M.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_ogpo_sweep.py [--cached]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from plot_flow_bc_curves import ENTITY_PROJECT, INK, INK2, SURFACE, plot_sweep, smooth, style
from plot_ogpo_public import history, seed_of, smooth_k

SWEEP = {"ph": [10, 20, 50, 100], "mh": [10, 20, 50, 100, 200]}
FULL_DEMOS = {"ph": 200, "mh": 300}
ONLINE_STEPS = 2_000_000
ONLINE_SMOOTH = 5   # centred window in evals; evals every 10k online steps -> 50k
# Ordinal ramp for dataset size, dataviz reference sequential blue steps 250, 300, 400, 500, 600,
# 700. Fixed per size, so a size has the same colour in every figure and panel.
SIZE_C = dict(zip([10, 20, 50, 100, 200, 300], ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]))
# Categorical slots 1-3 (dataviz reference): OGPO final, BC only, OGPO at hand-off.
OGPO_C, BC_C, HAND_C = "#2a78d6", "#eb6834", "#1baf7a"


def handoff_from_logs(dt, tag, sd):
    files = sorted(glob.glob(f"logs/ogpo_public_square_{dt}{tag}_seed{sd}_*.out"),
                   key=lambda p: int(re.search(r"_(\d+)\.out$", p).group(1)))
    for p in files:
        m = re.search(r"online_start step=(\d+)", open(p, errors="ignore").read())
        if m:
            return int(m.group(1)), [os.path.basename(f) for f in files]
    return None, [os.path.basename(f) for f in files]


def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    data, report = {}, {}
    for dt in ("ph", "mh"):
        rep = report.setdefault(dt, [])
        for n in SWEEP[dt] + [FULL_DEMOS[dt]]:
            full = n == FULL_DEMOS[dt]
            tag = "" if full else f"_{n}demos"
            group = f"square_{dt}_ogpo_public{tag}"
            runs = list(api.runs(ENTITY_PROJECT, filters={"group": group}))
            rep.append(f"== {group}: {len(runs)} W&B runs")
            best = {}
            for r in runs:
                sd = seed_of(r.name)
                if sd not in best or (r.summary.get("_step") or 0) > (best[sd].summary.get("_step") or 0):
                    best[sd] = r
            for sd in sorted(best, key=int):
                r = best[sd]
                h = history(r)
                bc_end, logs = handoff_from_logs(dt, tag, sd)
                last = h["step"][-1]
                check = ""
                if r.state == "finished":
                    check = "ok" if last - ONLINE_STEPS == bc_end else f"MISMATCH (last - 2M = {last - ONLINE_STEPS})"
                rep.append(f"   seed {sd}: {r.id} state={r.state} evals={len(h['step'])} last_step={last} "
                              f"hand-off={bc_end} {check} logs={logs} "
                              f"(other runs: {[x.id for x in runs if seed_of(x.name) == sd and x is not r]})")
                if bc_end is None:
                    rep.append("     no hand-off in the logs; not plotted")
                    continue
                data.setdefault(dt, {}).setdefault(str(n), {})[sd] = {
                    "run_id": r.id, "name": r.name, "state": r.state, "bc_end": bc_end, **h}
            print("\n".join(rep[-len(best) - 1:]), flush=True)
    return data, report


def online(d):
    """(online steps, raw SDE %) from the hand-off on."""
    step = np.array(d["step"])
    on = step >= d["bc_end"]
    return step[on] - d["bc_end"], 100 * np.array(d["sde"])[on]


def final_avg(d):
    """Mean raw SDE % over the last 50k online steps, or None if the seed is not at 2M yet."""
    x, y = online(d)
    if x[-1] < ONLINE_STEPS:
        return None
    return float(y[(x > ONLINE_STEPS - 50_000) & (x <= ONLINE_STEPS)].mean())


def online_axis(ax):
    from matplotlib.ticker import FuncFormatter
    ax.set_xlim(0, ONLINE_STEPS * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M"))
    ax.set_xlabel("online env steps after the BC hand-off", color=INK2)


def draw_online(ax, runs_by_n, dt):
    """Seed-mean online curve per dataset size on ax; each line stops where a seed is still running."""
    for n in sorted(runs_by_n, key=int):
        colour = SIZE_C[int(n)]
        curves = {}
        for sd, d in runs_by_n[n].items():
            x, y = online(d)
            keep = x <= ONLINE_STEPS
            curves[sd] = dict(zip(x[keep].tolist(), smooth_k(y[keep], ONLINE_SMOOTH).tolist()))
        steps = sorted(set().union(*curves.values()))
        cnt = np.array([sum(s in c for c in curves.values()) for s in steps])
        vals = [[c[s] for c in curves.values() if s in c] for s in steps]
        m = np.array([np.mean(v) for v in vals])
        lo, hi = np.array([min(v) for v in vals]), np.array([max(v) for v in vals])
        steps = np.array(steps)
        allin = cnt == len(curves)
        cut = int(np.flatnonzero(allin)[-1]) + 1 if allin.any() else 0
        label = f"{n} demos" + (" (full)" if int(n) == FULL_DEMOS[dt] else "")
        if len(curves) > 1:
            ax.fill_between(steps[:cut], lo[:cut], hi[:cut], color=colour, alpha=0.15, lw=0)
        ax.plot(steps[:cut], m[:cut], color=colour, lw=2, label=label, solid_capstyle="round")
        if cut < len(steps):
            print(f"  {dt} {n} demos: stops at {steps[cut - 1]:,} online steps (seeds still running)")
    style(ax)
    online_axis(ax)


def seeds_of(*runs_by_n):
    return ", ".join(sorted({sd for r in runs_by_n for runs in r.values() for sd in runs}, key=int))


def plot_online(plt, runs_by_n, dt, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE)
    draw_online(ax, runs_by_n, dt)
    ax.set_title(f"OGPO on square-{dt}: SDE success vs online steps (50k moving average)\n"
                 f"mean of seeds {seeds_of(runs_by_n)}; band = min-max; a line stops where a seed is still running",
                 color=INK, loc="left", fontsize=11)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_online_both(plt, data, out):
    """ph and mh side by side; a dataset size has the same colour in both panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, facecolor=SURFACE)
    handles = {}
    for ax, dt in zip(axes, ("ph", "mh")):
        draw_online(ax, data[dt], dt)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(f"square-{dt} (full dataset = {FULL_DEMOS[dt]} demos)", color=INK, loc="left", fontsize=10)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(int(lab.split()[0]), (h, lab.replace(" (full)", "")))
    order = sorted(handles, reverse=True)
    fig.legend([handles[n][0] for n in order], [handles[n][1] for n in order], frameon=False,
               loc="upper left", bbox_to_anchor=(0.885, 0.86), labelcolor=INK2)
    fig.suptitle("OGPO dataset-size sweep: SDE success vs online env steps (50k moving average)\n"
                 f"mean of seeds {seeds_of(data['ph'], data['mh'])}; band = min-max; "
                 "a line stops where a seed is still running", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.88, 0.93))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def bc_final(runs):
    """3-point smoothed SDE % at each seed's last eval."""
    return [100 * smooth(np.array(d["sde"]))[-1] for d in runs.values()]


def draw_scaling(ax, runs_by_n, bc_by_n, dt):
    sizes = sorted(runs_by_n, key=int)
    series = [
        ("BC only, 500k BC steps", BC_C, 0.94,
         {n: bc_final(bc_by_n[n]) for n in sorted(bc_by_n, key=int)}),
        ("OGPO at the BC hand-off", HAND_C, 1.0,
         {n: [online(d)[1][0] for d in runs_by_n[n].values()] for n in sizes}),
        ("OGPO after 2M online steps", OGPO_C, 1.06,
         {n: [v for v in (final_avg(d) for d in runs_by_n[n].values()) if v is not None] for n in sizes}),
    ]
    for label, colour, dx, per_n in series:
        xs = [int(n) for n in per_n if per_n[n]]
        ms = [float(np.mean(per_n[str(x)])) for x in xs]
        for x in xs:
            v = per_n[str(x)]
            ax.scatter(np.full(len(v), x * dx), v, s=14, color=colour, alpha=0.35, lw=0)
            if len(v) < 3:
                ax.annotate(f"{len(v)} seed{'s' * (len(v) > 1)}", (x * dx, np.mean(v)), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7, color=INK2)
        ax.plot(np.array(xs) * dx, ms, color=colour, lw=2, marker="o", ms=7, label=label,
                markeredgecolor=SURFACE, markeredgewidth=2)
        print(f"  {dt} {label:28s} " + "  ".join(f"{x}: {m:5.1f}% ({len(per_n[str(x)])})" for x, m in zip(xs, ms)))
    style(ax)
    xs = sorted({int(n) for n in sizes} | {int(n) for n in bc_by_n})
    ax.set_xscale("log")
    ax.set_xlim(8, 380)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(f"demos (log scale); {FULL_DEMOS[dt]} = full dataset", color=INK2)


def plot_scaling_online(plt, runs_by_n, bc_by_n, dt, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE)
    draw_scaling(ax, runs_by_n, bc_by_n, dt)
    ax.set_title(f"OGPO on square-{dt}: SDE success vs dataset size\nmean of seeds (dots = seeds); "
                 "final = last 50k of 2M online steps", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_scaling_both(plt, data, bc, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, facecolor=SURFACE)
    for ax, dt in zip(axes, ("ph", "mh")):
        draw_scaling(ax, data[dt], bc[dt], dt)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(f"square-{dt}", color=INK, loc="left", fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=3, labelcolor=INK2)
    fig.suptitle("OGPO dataset-size sweep: SDE success vs dataset size\nmean of seeds (dots = seeds); "
                 "final = last 50k of 2M online steps, finished seeds only", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_scaling_bc(plt, data, dt, out):
    sizes = sorted(data, key=int)
    xs = np.array([int(n) for n in sizes])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes, ("peak of the seed-mean curve", "final (500k BC steps)")):
        means, dots = [], []
        for n in sizes:
            runs = data[n]
            steps = sorted(set.intersection(*(set(d["step"]) for d in runs.values())))
            ys = np.array([100 * smooth(np.array([d["sde"][d["step"].index(s)] for s in steps])) for d in runs.values()])
            i = int(ys.mean(0).argmax()) if name.startswith("peak") else len(steps) - 1
            means.append(ys[:, i].mean())
            dots.append(ys[:, i])
            if name.startswith("peak"):
                ax.annotate(f"{steps[i] // 1000}k", (int(n), ys[:, i].mean()), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7, color=INK2)
        for x, row in zip(xs, dots):
            ax.scatter(np.full(len(row), x), row, s=14, color=OGPO_C, alpha=0.35, lw=0)
        ax.plot(xs, means, color=OGPO_C, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2)
        style(ax, ylabel=ax is axes[0])
        ax.set_xscale("log")
        ax.set_xlim(8, 320)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("demos (log scale)", color=INK2)
        ax.set_title(name + (" (label = BC step)" if name.startswith("peak") else ""), color=INK, loc="left", fontsize=10)
        print(f"  BC only {dt} {name}: " + "  ".join(f"{n}: {m:5.1f}%" for n, m in zip(sizes, means)))
    fig.suptitle(f"OGPO repo, BC only, on square-{dt}: SDE success vs dataset size\n"
                 "mean of seeds 0, 7, 42 (dots = seeds; 3-point smoothed)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true", help="replot from plots/ogpo_square_*/curves.json")
    args = ap.parse_args()
    if args.cached:
        data = {dt: json.load(open(f"plots/ogpo_square_{dt}/curves.json")) for dt in ("ph", "mh")}
    else:
        data, report = fetch()
        for dt in ("ph", "mh"):
            os.makedirs(f"plots/ogpo_square_{dt}", exist_ok=True)
            json.dump(data[dt], open(f"plots/ogpo_square_{dt}/curves.json", "w"), indent=1)
            with open(f"plots/ogpo_square_{dt}/wandb_check.txt", "w") as f:
                f.write("\n".join(report[dt]) + "\n")
    bc = json.load(open("plots/ogpo_public/curves.json"))["bc"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dt in ("ph", "mh"):
        od, bd = f"plots/ogpo_square_{dt}", f"plots/ogpo_bc_square_{dt}"
        os.makedirs(bd, exist_ok=True)
        print(f"square-{dt}:")
        plot_online(plt, data[dt], dt, f"{od}/sde_success_vs_online_steps.png")
        plot_scaling_online(plt, data[dt], bc[dt], dt, f"{od}/sde_success_vs_dataset_size.png")
        plot_sweep(plt, bc[dt], "sde", f"OGPO repo, BC only, on square-{dt}: SDE success vs BC steps (3-point smoothed)",
                   f"{bd}/sde_success_vs_steps.png")
        plot_scaling_bc(plt, bc[dt], dt, f"{bd}/sde_success_vs_dataset_size.png")
        print(f"wrote {od}/ and {bd}/")
    os.makedirs("plots/ogpo_sweep", exist_ok=True)
    plot_online_both(plt, data, "plots/ogpo_sweep/sde_success_vs_online_steps.png")
    plot_scaling_both(plt, data, bc, "plots/ogpo_sweep/sde_success_vs_dataset_size.png")
    print("wrote plots/ogpo_sweep/")


if __name__ == "__main__":
    main()
