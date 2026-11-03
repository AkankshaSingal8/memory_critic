"""How much best-of-N with the MC critic adds over BC alone, as N grows, per BC dataset size.

For each (dataset, BC demos, N, critic seed): the mean over BC seeds 0/7/42 and the checkpoints in --grid
of (best-of-N success - BC-alone SDE success at the same checkpoint), in percentage points. BC alone is
the eval's unsmoothed eval_sde/success at that step (plots/flow_bc_square_<ds>/curves.json), on the
same 64 start states as the best-of-N rollouts (scripts/bon_curve.sbatch, scripts/bon_sweep/).

One panel per dataset; one line per BC dataset size (blue ordinal ramp); the line is the mean over
critic seeds and the bar spans the critic seeds' min-max. Writes <out>.png and <out>.json.

    python scripts/plot_bon_gain_vs_n.py --critic_seeds 0 1 2 --ns 8 16 32 --out plots/bon/bon_gain_vs_n
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_flow_bc_curves import GRID, INK, INK2, RAMP, SURFACE  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ("0", "7", "42")
SIZES = (10, 20, 50, 100, 200)


def gain(ds, n, N, root, grid, curves):
    """Mean over (BC seed, step) of BoN - BC in percentage points, and the number of matched points."""
    d = []
    for s in SEEDS:
        run = curves[str(n)].get(s)
        if run is None:
            continue
        bc = dict(zip(run["step"], run["sde"]))
        for t in grid:
            p = os.path.join(root, f"square_{ds}_{n}demos_seed{s}_step{t}_sde_bon{N}.json")
            if os.path.exists(p) and t in bc:
                with open(p) as f:
                    d.append(100 * (json.load(f)["success_rate"] - bc[t]))
    return (float(np.mean(d)) if d else np.nan), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bon_root", default="/data/user_data/asingal2/memory_critic/bon_curve")
    ap.add_argument("--critic_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--datasets", nargs="+", default=["ph", "mh"])
    ap.add_argument("--grid", type=int, nargs="+", default=list(range(50000, 500001, 50000)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    G = {}   # (ds, n, N) -> per-critic-seed gains
    expected = len(SEEDS) * len(args.grid)
    for ds in args.datasets:
        with open(os.path.join(REPO, f"plots/flow_bc_square_{ds}/curves.json")) as f:
            curves = json.load(f)
        for n in SIZES:
            for N in args.ns:
                row = []
                for c in args.critic_seeds:
                    g, k = gain(ds, n, N, os.path.join(args.bon_root, f"mc_decision_seed{c}"), args.grid, curves)
                    if k < expected:
                        print(f"note: {ds}-{n} N={N} critic {c}: {k}/{expected} points")
                    row.append(g)
                G[(ds, n, N)] = row

    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5.6 * len(args.datasets) + 1.6, 4.6),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    xs = np.log2(args.ns)
    dodge = np.linspace(-0.08, 0.08, len(SIZES))
    for ax, ds in zip(axes[0], args.datasets):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        for k, (colour, n) in enumerate(zip(RAMP, SIZES)):
            A = np.array([G[(ds, n, N)] for N in args.ns])          # (len(ns), n_critics)
            m, lo, hi = np.nanmean(A, 1), np.nanmin(A, 1), np.nanmax(A, 1)
            x = xs + dodge[k]
            ax.vlines(x, lo, hi, color=colour, lw=2, alpha=0.45, zorder=2)
            ax.plot(x, m, color=colour, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3,
                    solid_capstyle="round")
        ax.set_xticks(xs, [str(N) for N in args.ns])
        ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
        ax.set_xlabel("N (chunks the critic picks from)", color=INK2)
        ax.set_title(f"square-{ds.upper()}", color=INK, loc="left", fontsize=11)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    axes[0][0].set_ylabel("best-of-N minus BC alone\n(success, percentage points)", color=INK2)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=6, mec=SURFACE) for c in RAMP][::-1]
    labels = [f"{n} demos" for n in SIZES][::-1]
    handles.append(Line2D([], [], color=INK2, lw=2, alpha=0.45))
    labels.append("range over critic seeds")
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.835, 0.82), labelcolor=INK2)
    crit = ", ".join(map(str, args.critic_seeds))
    fig.suptitle("Gain from best-of-N with the no-memory MC critic, by BC dataset size\n"
                 f"mean over BC seeds 0, 7, 42 and checkpoints 50k-500k, then over critic seeds {crit}; "
                 "64 episodes per point, SDE sampler", x=0.04, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.83, 0.86))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=170, facecolor=SURFACE)
    with open(args.out + ".json", "w") as f:
        json.dump({"critic_seeds": args.critic_seeds, "ns": args.ns,
                   "gain_pp": {f"{ds}/{n}/{N}": G[(ds, n, N)] for ds in args.datasets for n in SIZES for N in args.ns}},
                  f, indent=1)
    print(args.out + ".png")
    for ds in args.datasets:
        print(f"{ds}: " + "  ".join(f"{n}d " + "/".join(f"{np.nanmean(G[(ds, n, N)]):+.1f}" for N in args.ns)
                                    for n in SIZES) + f"   (N = {'/'.join(map(str, args.ns))})")


if __name__ == "__main__":
    main()
