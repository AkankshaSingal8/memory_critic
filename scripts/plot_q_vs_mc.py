"""Q vs MC return on the held-out test episodes, in the style of qipm's eval plot
(ogpo/utils/evaluation_qipm.visualize_per_subtask_q_vs_mc: Q on x, MC return on y, dashed y = x,
Pearson r, equal axes), one panel per policy group (square ph/mh x N demos).

Points are the critic's test decision points from scripts/train_mc_critic.py (test_predictions.npz),
colored by whether the episode succeeded. The dark line is the calibration curve: mean MC return
in 10 quantile bins of Q.

    python scripts/plot_q_vs_mc.py <critic out_dir> <out.png> [--max_points 4000]
"""

import argparse
import os

import numpy as np

SUCCESS, FAILURE = "#2a78d6", "#eb6834"  # dataviz reference palette, categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("critic_dir")
    ap.add_argument("out")
    ap.add_argument("--max_points", type=int, default=4000, help="per panel, for legibility")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = np.load(os.path.join(args.critic_dir, "test_predictions.npz"))
    q, G, group, succ = z["q"], z["G"], z["group"].astype(str), z["ep_success"].astype(bool)
    rng = np.random.default_rng(0)
    lo, hi = -100, 3

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.8), sharex=True, sharey=True)
    for r, ds in enumerate(("ph", "mh")):
        for c, n in enumerate((10, 20, 50, 100, 200)):
            ax = axes[r, c]
            m = np.flatnonzero(group == f"{ds}-{n}")
            show = rng.choice(m, min(len(m), args.max_points), replace=False)
            for flag, color in ((False, FAILURE), (True, SUCCESS)):
                s = show[succ[show] == flag]
                ax.scatter(q[s], G[s], s=4, alpha=0.25, color=color, linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.2, zorder=3)
            edges = np.quantile(q[m], np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(edges, q[m], side="right") - 1, 0, 9)
            bq = [q[m][b == i].mean() for i in range(10) if (b == i).any()]
            bG = [G[m][b == i].mean() for i in range(10) if (b == i).any()]
            ax.plot(bq, bG, "-o", color=INK, lw=2, ms=4, zorder=4)
            rr = np.corrcoef(q[m], G[m])[0, 1]
            mse = np.mean((q[m] - G[m]) ** 2)
            ax.text(0.04, 0.96, f"r = {rr:.3f}\nMSE = {mse:.0f}\nsuccess = {succ[m][z['timestep'][m] == 0].mean():.2f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRID, alpha=0.9))
            ax.set_title(f"{ds.upper()}-{n}", fontsize=11, color=INK)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 1:
                ax.set_xlabel("Q value (critic, ensemble mean)")
            if c == 0:
                ax.set_ylabel("MC return $G_t$")
    handles = [Line2D([], [], marker="o", ls="", color=SUCCESS, ms=6, label="successful episode"),
               Line2D([], [], marker="o", ls="", color=FAILURE, ms=6, label="failed episode (hit 400 steps)"),
               Line2D([], [], color=INK, marker="o", ms=4, lw=2, label="mean $G_t$ in 10 quantile bins of Q"),
               Line2D([], [], color=MUTED, ls="--", lw=1.2, label="y = x")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("MC critic (no memory): Q vs MC return on held-out test episodes, decision points",
                 y=1.04, fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
