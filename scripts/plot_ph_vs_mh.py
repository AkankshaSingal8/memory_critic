"""Flow-BC dataset-size sweep, square PH vs MH.

  flow_bc_<task>_ph_vs_mh.png     ODE success vs number of demos, mean over seeds (dots = seeds),
                                  at each seed's selected best checkpoint and the final checkpoint
  flow_bc_<task>_ph_mh_seeds.png  rows PH / MH, one column per dataset size: every seed's ODE
                                  success vs BC steps, plus the seed mean

Reads the curves.json that scripts/plot_flow_bc_curves.py writes for each dataset type, so run
that first for both (--dataset_type ph / mh). Login node is fine:

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_ph_vs_mh.py
"""
import argparse
import json
import os

import numpy as np

from plot_flow_bc_curves import INK, INK2, SURFACE, at_step, seeds_label, smooth, style

# Dataviz reference categorical slots 1-2, in fixed order.
COLOURS = {"ph": "#2a78d6", "mh": "#eb6834"}
LABELS = {"ph": "PH (proficient human)", "mh": "MH (multi-human)"}
# Slots 1-3 for seeds 0 / 7 / 42 in the per-seed grid.
SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="square")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = {t: json.load(open(f"plots/flow_bc_{args.task}_{t}/curves.json")) for t in ("ph", "mh")}
    out = args.out or f"plots/flow_bc_{args.task}_ph_vs_mh.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True, facecolor=SURFACE)
    panels = (("selected best checkpoint", lambda d: d["best_step"]),
              ("final checkpoint", lambda d: d["step"][-1]))
    for ax, (name, pick) in zip(axes, panels):
        for t, dx in (("ph", 0.97), ("mh", 1.03)):
            sizes = sorted(data[t], key=int)
            xs = np.array([int(n) for n in sizes])
            ys = np.array([[at_step(d, "ode", pick(d)) for d in data[t][n].values()] for n in sizes])
            for x, row in zip(xs, ys):
                ax.scatter(np.full(len(row), x * dx), row, s=14, color=COLOURS[t], alpha=0.35, lw=0)
            ax.plot(xs, ys.mean(1), color=COLOURS[t], lw=2, marker="o", ms=7, label=LABELS[t],
                    markeredgecolor=SURFACE, markeredgewidth=2)
        style(ax, ylabel=ax is axes[0])
        ax.set_xscale("log")
        ax.set_xlim(8, 320)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("demos (log scale)", color=INK2)
        ax.set_title(name, color=INK, loc="left", fontsize=10)
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK2)
    fig.suptitle(f"Flow BC on {args.task}: PH vs MH demos, ODE success\n"
                 f"{seeds_label(data['ph']).split(';')[0]} (dots = individual seeds; 3-point smoothed)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    plot_seed_grid(plt, data, args.task, out.replace("_ph_vs_mh.png", "_ph_mh_seeds.png"))


def plot_seed_grid(plt, data, task, out):
    """Rows PH / MH, one column per dataset size: each seed's ODE success vs BC steps (thin) and
    their mean (ink), all 3-point smoothed."""
    sizes = sorted(data["ph"], key=int)
    seeds = sorted({sd for runs in data["ph"].values() for sd in runs}, key=int)
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.0 * len(sizes), 6.4), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for row, t in zip(axes, ("ph", "mh")):
        for ax, n in zip(row, sizes):
            runs = data[t][n]
            ys = []
            for colour, sd in zip(SEED_COLOURS, seeds):
                d = runs[sd]
                y = 100 * smooth(np.array(d["ode"]))
                ys.append(y)
                ax.plot(d["step"], y, color=colour, lw=1.2, alpha=0.9, label=f"seed {sd}")
            ax.plot(runs[seeds[0]]["step"], np.mean(ys, 0), color=INK, lw=2, label="mean")
            style(ax, ylabel=ax is row[0])
            ax.xaxis.set_major_locator(plt.MultipleLocator(250_000))
            if ax is row[0]:
                ax.set_ylabel(f"{t.upper()}: task success (%)", color=INK2)
            if t == "ph":
                ax.set_xlabel("")
            ax.set_title(f"{t.upper()}, {n} demos", color=INK, loc="left", fontsize=10)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper right", ncol=len(lab), labelcolor=INK2)
    fig.suptitle(f"Flow BC on {task}: ODE success per seed vs BC steps (64 episodes per eval, "
                 "3-point smoothed)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
