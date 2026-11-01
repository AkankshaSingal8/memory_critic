"""Best-of-N with the MC critic over BC training, drawn like plots/flow_bc_square_<ds>/sde_success_vs_steps.png.

One panel per N. In each, one colour per BC dataset size (the same blue ordinal ramp, style and
3-point smoothing as scripts/plot_flow_bc_curves.py):
  dashed  BC alone: eval_sde/success from plots/flow_bc_square_<ds>/curves.json (every 10k steps)
  solid   best-of-N: <root>/square_<ds>_<n>demos_seed<S>_step<T>_sde_bon<N>.json from
          scripts/bon_curve.sbatch: the same checkpoint and the same 64 start states (seeds T..T+63)
          as that eval, π_base draws N SDE chunks per decision and the MC critic (ensemble mean)
          keeps one. Checkpoints every 50k, so the line has markers.
Lines are the mean over seeds 0, 7, 42 (each seed smoothed first); the band is the min-max of the
best-of-N line. Missing points are left out and counted in the title.

Writes <out>.png and <out>.json, and prints best-of-N minus BC at the matched checkpoints.

    python scripts/plot_bon_vs_bc_steps.py --dataset mh \
        --root /data/user_data/$USER/memory_critic/bon_curve/mc_decision_seed0 --out plots/bon/bon_vs_bc_steps_mh
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_flow_bc_curves import GRID, INK, INK2, RAMP, SURFACE, smooth, style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ("0", "7", "42")
SIZES = (10, 20, 50, 100, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mh")
    ap.add_argument("--root", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--grid", type=int, nargs="+", default=list(range(50000, 500001, 50000)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds, grid = args.dataset, np.array(args.grid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(os.path.join(REPO, f"plots/flow_bc_square_{ds}/curves.json")) as f:
        C = json.load(f)

    def bc(n):
        """Steps every seed has; mean over seeds of each seed's smoothed SDE success (%)."""
        runs = {s: C[str(n)][s] for s in SEEDS if s in C[str(n)]}
        steps = sorted(set.intersection(*(set(r["step"]) for r in runs.values())))
        ys = np.array([100 * smooth(np.array([r["sde"][r["step"].index(t)] for t in steps])) for r in runs.values()])
        raw = {s: dict(zip(r["step"], r["sde"])) for s, r in runs.items()}
        return np.array(steps), ys.mean(0), raw

    def bon(n, N):
        Y = np.full((len(SEEDS), len(grid)), np.nan)
        for i, s in enumerate(SEEDS):
            for j, t in enumerate(grid):
                p = os.path.join(args.root, f"square_{ds}_{n}demos_seed{s}_step{t}_sde_bon{N}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        Y[i, j] = 100 * json.load(f)["success_rate"]
        return Y

    total = len(SEEDS) * len(grid) * len(SIZES)
    data = {N: {n: bon(n, N) for n in SIZES} for N in args.ns}
    have = {N: int(sum((~np.isnan(Y)).sum() for Y in data[N].values())) for N in args.ns}
    print("points: " + ", ".join(f"N={N} {h}/{total}" for N, h in have.items()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE, sharex=True, sharey=True)
    out_json = {"grid": grid.tolist(), "seeds": SEEDS}
    for ax, N in zip(axes.ravel(), args.ns):
        for colour, n in zip(RAMP, SIZES):
            x, m, _ = bc(n)
            ax.plot(x, m, color=colour, lw=1.4, ls=(0, (4, 2.5)), alpha=0.9)
            Y = data[N][n]
            S = np.array([smooth(y) if not np.isnan(y).any() else y for y in Y])
            mean = np.nanmean(S, 0)
            ok = ~np.isnan(Y).all(0)
            if ok.any():
                ax.fill_between(grid[ok], np.nanmin(S, 0)[ok], np.nanmax(S, 0)[ok], color=colour, alpha=0.15, lw=0)
                ax.plot(grid[ok], mean[ok], color=colour, lw=2, marker="o", ms=4, mec=SURFACE, mew=1,
                        solid_capstyle="round")
            out_json[f"{n}demos/bon{N}"] = np.where(np.isnan(Y), None, Y).tolist()
        style(ax, ylabel=ax in axes[:, 0])
        if ax not in axes[1]:
            ax.set_xlabel("")
        part = "" if have[N] == total else f"  ({have[N]}/{total} points so far)"
        ax.set_title(f"Best-of-{N} vs BC alone{part}", color=INK, loc="left", fontsize=11)
    for ax in axes.ravel()[len(args.ns):]:
        ax.axis("off")
    size_h = [Line2D([], [], color=c, lw=2) for c in RAMP][::-1]
    size_l = [f"{n} demos" for n in SIZES][::-1]
    style_h = [Line2D([], [], color=INK2, lw=2, marker="o", ms=4), Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2.5)))]
    style_l = ["best-of-N, MC critic", "BC alone (N = 1)"]
    fig.legend(size_h + style_h, size_l + style_l, frameon=False, loc="upper left", bbox_to_anchor=(0.835, 0.9),
               labelcolor=INK2)
    fig.suptitle(f"Flow BC on square-{ds}: SDE success vs BC steps, best-of-N with the no-memory MC critic\n"
                 "mean of seeds 0, 7, 42 (3-point smoothed); band = min-max of best-of-N; "
                 "64 episodes per point on the eval's start states", x=0.06, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 0.83, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=170, facecolor=SURFACE)
    with open(args.out + ".json", "w") as f:
        json.dump(out_json, f)
    print(args.out + ".png")
    for n in SIZES:
        _, _, raw = bc(n)
        for N in args.ns:
            d = [(Y, raw[s].get(int(t))) for i, s in enumerate(SEEDS) for j, t in enumerate(grid)
                 for Y in [data[N][n][i, j]] if not np.isnan(Y) and raw.get(s, {}).get(int(t)) is not None]
            if d:
                b = np.array([y for y, _ in d]); a = 100 * np.array([z for _, z in d])
                print(f"{ds}-{n:<3} N={N:<3} matched {len(d):2d} pts: BC {a.mean():5.1f}  BoN {b.mean():5.1f}  diff {b.mean() - a.mean():+5.1f}")


if __name__ == "__main__":
    main()
