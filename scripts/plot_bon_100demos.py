"""Flow BC on square, 100 demos: SDE success vs BC steps for PH and MH, plus best-of-N with the MC critic.

Colour = dataset (PH blue, MH orange, as in scripts/plot_ph_vs_mh.py).
  dashed  BC alone (N = 1): eval_sde/success from plots/flow_bc_square_<ds>/curves.json, every 10k
          steps, mean over seeds 0/7/42 of each seed's 3-point-smoothed curve (as plot_flow_bc_curves.py).
  solid   best-of-N on the same checkpoint: <root>/square_<ds>_100demos_seed<S>_step<T>_sde_bon<N>.json
          from scripts/bon_curve.sbatch (64 episodes on that eval's start states, MC critic seed 0,
          ensemble-mean argmax). Checkpoints every 50k; mean over the 3 seeds, unsmoothed;
          band = min-max over seeds.

    python scripts/plot_bon_100demos.py --n 8 --out plots/bon/bon_vs_bc_steps_100demos_bon8
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_flow_bc_curves import INK, INK2, SURFACE, smooth, style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ("0", "7", "42")
COLOURS = {"ph": "#2a78d6", "mh": "#eb6834"}
LABELS = {"ph": "PH", "mh": "MH"}


def bc_curve(ds, n_demos):
    with open(os.path.join(REPO, f"plots/flow_bc_square_{ds}/curves.json")) as f:
        runs = json.load(f)[str(n_demos)]
    runs = [runs[s] for s in SEEDS]
    steps = sorted(set.intersection(*(set(r["step"]) for r in runs)))
    ys = [100 * smooth(np.array([r["sde"][r["step"].index(t)] for t in steps])) for r in runs]
    return np.array(steps), np.mean(ys, 0)


def bon_curve(root, ds, n_demos, N, grid):
    Y = np.full((len(SEEDS), len(grid)), np.nan)
    for i, s in enumerate(SEEDS):
        for j, t in enumerate(grid):
            p = os.path.join(root, f"square_{ds}_{n_demos}demos_seed{s}_step{t}_sde_bon{N}.json")
            if os.path.exists(p):
                with open(p) as f:
                    Y[i, j] = 100 * json.load(f)["success_rate"]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed0")
    ap.add_argument("--n_demos", type=int, default=100)
    ap.add_argument("--n", type=int, default=8, help="best-of-N")
    ap.add_argument("--bon", nargs="+", default=["ph", "mh"], help="datasets to draw a best-of-N line for")
    ap.add_argument("--grid", type=int, nargs="+", default=list(range(50000, 500001, 50000)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grid = np.array(args.grid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
    out = {"grid": grid.tolist(), "seeds": SEEDS, "n": args.n, "n_demos": args.n_demos}
    ends = []
    for ds in ("ph", "mh"):
        c = COLOURS[ds]
        x, m = bc_curve(ds, args.n_demos)
        ax.plot(x, m, color=c, lw=1.6, ls=(0, (4, 2.5)))
        ends.append((m[-1], f"{LABELS[ds]} BC", c))
        out[f"{ds}/bc"] = {"step": x.tolist(), "success_pct": m.tolist()}
        if ds not in args.bon:
            continue
        Y = bon_curve(args.root, ds, args.n_demos, args.n, grid)
        missing = int(np.isnan(Y).sum())
        assert missing == 0, f"{ds}: {missing} best-of-{args.n} points missing under {args.root}"
        mean = Y.mean(0)
        ax.fill_between(grid, Y.min(0), Y.max(0), color=c, alpha=0.14, lw=0)
        ax.plot(grid, mean, color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, solid_capstyle="round")
        ends.append((mean[-1], f"{LABELS[ds]} + BoN-{args.n}", c))
        out[f"{ds}/bon{args.n}"] = {"per_seed_pct": Y.tolist(), "mean_pct": mean.tolist()}
        bc_at = np.array([m[list(x).index(t)] for t in grid])
        print(f"{ds}: BC {bc_at.mean():5.1f}  BoN-{args.n} {mean.mean():5.1f}  diff {mean.mean() - bc_at.mean():+5.1f}"
              f"  (avg over the {len(grid)} checkpoints; at 500k BC {bc_at[-1]:.1f} vs BoN {mean[-1]:.1f})")

    style(ax)
    ax.set_ylabel("successful episodes (%)", color=INK2)
    # Direct labels at the right edge, nudged apart so they never overlap.
    ends.sort()
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 4.5)
    for (_, text, c), y in zip(ends, ys):
        ax.annotate(text, xy=(500_000, y), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
        ax.plot([505_000], [y], marker="s", ms=5, color=c, clip_on=False)
    ax.set_xlim(0, 500_000)
    handles = [Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 2.5))),
               Line2D([], [], color=INK2, lw=2, marker="o", ms=5)]
    ax.legend(handles, ["BC alone (N = 1)", f"best-of-{args.n}, MC critic"], frameon=False, loc="upper left",
              labelcolor=INK2)
    ax.set_title(f"Flow BC on square, {args.n_demos} demos: success vs BC steps, PH vs MH, with best-of-{args.n}",
                 color=INK, loc="left", fontsize=11)
    fig.text(0.01, 0.005, "BC: mean of seeds 0/7/42, 3-point smoothed, eval every 10k.\n"
             f"BoN-{args.n}: no-memory MC critic (seed 0), same checkpoint and 64 start states, every 50k; "
             "mean of 3 seeds, band = min-max.", color=INK2, fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 0.88, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=170, facecolor=SURFACE)
    with open(args.out + ".json", "w") as f:
        json.dump(out, f)
    print(args.out + ".png")


if __name__ == "__main__":
    main()
