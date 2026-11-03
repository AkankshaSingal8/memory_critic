"""PH vs MH on square: SDE success vs BC steps, BC alone and best-of-N with the MC critic, per BC dataset size.

The same chart as scripts/plot_bon_100demos.py (colour = dataset, PH blue, MH orange), generalised:
  - one panel per --n_demos (one size gives the single-panel chart with direct labels; several give
    small multiples with a shared legend),
  - several --root dirs (one per critic seed) are pooled.
  dashed  BC alone (N = 1): eval_sde/success from plots/flow_bc_square_<ds>/curves.json, every 10k
          steps, mean over seeds 0/7/42 of each seed's 3-point-smoothed curve.
  solid   best-of-N on the same checkpoint and the same 64 start states (SDE sampler, N chunks, the
          critic's ensemble mean keeps one), every 50k; mean over BC seeds x critic seeds, unsmoothed;
          band = min-max over those rows.

    python scripts/plot_bon_ph_vs_mh.py --n 8 --out plots/bon/bon8_ph_vs_mh_by_size
    python scripts/plot_bon_ph_vs_mh.py --n 8 --n_demos 100 --out plots/bon/bon8_ph_vs_mh_100demos
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_bon_100demos import COLOURS, LABELS, SEEDS, bc_curve  # noqa: E402
from plot_flow_bc_curves import INK, INK2, SURFACE, style  # noqa: E402

BON_ROOT = "/data/user_data/asingal2/memory_critic/bon_curve"


def bon_rows(roots, ds, n_demos, N, grid):
    Y = np.full((len(roots) * len(SEEDS), len(grid)), np.nan)
    for r, root in enumerate(roots):
        for i, s in enumerate(SEEDS):
            for j, t in enumerate(grid):
                p = os.path.join(root, f"square_{ds}_{n_demos}demos_seed{s}_step{t}_sde_bon{N}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        Y[r * len(SEEDS) + i, j] = 100 * json.load(f)["success_rate"]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="best-of-N")
    ap.add_argument("--n_demos", type=int, nargs="+", default=[10, 20, 50, 100, 200])
    ap.add_argument("--critic_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid", type=int, nargs="+", default=list(range(50000, 500001, 50000)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grid = np.array(args.grid)
    roots = [os.path.join(BON_ROOT, f"mc_decision_seed{c}") for c in args.critic_seeds]
    crit = (f"critic seed {args.critic_seeds[0]}" if len(roots) == 1 else
            "critic seeds " + ", ".join(map(str, args.critic_seeds)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    single = len(args.n_demos) == 1
    if single:
        fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
        axes = [ax]
    else:
        ncols = 3
        nrows = (len(args.n_demos) + ncols) // ncols          # one slot left for the legend
        fig, A = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.9 * nrows + 0.9), facecolor=SURFACE,
                              sharey=True)
        axes = list(A.ravel())

    out = {"grid": grid.tolist(), "n": args.n, "critic_seeds": args.critic_seeds, "bc_seeds": SEEDS}
    for ax, n_demos in zip(axes, args.n_demos):
        ends = []
        for ds in ("ph", "mh"):
            c = COLOURS[ds]
            x, m = bc_curve(ds, n_demos)
            ax.plot(x, m, color=c, lw=1.6, ls=(0, (4, 2.5)))
            ends.append((m[-1], f"{LABELS[ds]} BC", c))
            Y = bon_rows(roots, ds, n_demos, args.n, grid)
            missing = int(np.isnan(Y).sum())
            assert missing == 0, f"{ds}-{n_demos}: {missing} best-of-{args.n} points missing"
            mean = Y.mean(0)
            ax.fill_between(grid, Y.min(0), Y.max(0), color=c, alpha=0.14, lw=0)
            ax.plot(grid, mean, color=c, lw=2, marker="o", ms=6 if single else 4.5, mec=SURFACE, mew=1.2,
                    solid_capstyle="round")
            ends.append((mean[-1], f"{LABELS[ds]} + BoN-{args.n}", c))
            bc_at = np.array([m[list(x).index(t)] for t in grid])
            out[f"{n_demos}/{ds}"] = {"bc_step": x.tolist(), "bc_pct": m.tolist(), "bon_rows_pct": Y.tolist(),
                                      "bon_mean_pct": mean.tolist()}
            print(f"{ds}-{n_demos:<3} N={args.n:<2}: BC {bc_at.mean():5.1f}  BoN {mean.mean():5.1f}  "
                  f"diff {mean.mean() - bc_at.mean():+5.1f}  (500k: BC {bc_at[-1]:.1f} vs BoN {mean[-1]:.1f})")
        style(ax, ylabel=single or ax in A[:, 0])
        ax.set_xlim(0, 500_000)
        if single:
            ax.set_ylabel("successful episodes (%)", color=INK2)
            ends.sort()
            ys = [e[0] for e in ends]
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i - 1] + 4.5)
            for (_, text, c), y in zip(ends, ys):
                ax.annotate(text, xy=(500_000, y), xytext=(8, 0), textcoords="offset points", va="center",
                            color=INK2, fontsize=9)
                ax.plot([505_000], [y], marker="s", ms=5, color=c, clip_on=False)
        else:
            ax.set_title(f"{n_demos} demos", color=INK, loc="left", fontsize=11)
            if axes.index(ax) + ncols < len(args.n_demos):        # another panel below: drop the x labels
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    handles = [Line2D([], [], color=COLOURS["ph"], lw=2, marker="o", ms=5, mec=SURFACE),
               Line2D([], [], color=COLOURS["ph"], lw=1.6, ls=(0, (4, 2.5))),
               Line2D([], [], color=COLOURS["mh"], lw=2, marker="o", ms=5, mec=SURFACE),
               Line2D([], [], color=COLOURS["mh"], lw=1.6, ls=(0, (4, 2.5)))]
    labels = [f"PH + best-of-{args.n}", "PH BC alone (N = 1)", f"MH + best-of-{args.n}", "MH BC alone (N = 1)"]
    note = (f"BC: mean of seeds 0/7/42, 3-point smoothed, eval every 10k.\n"
            f"BoN-{args.n}: no-memory MC critic ({crit}), same checkpoint and 64 start states, every 50k;\n"
            f"mean over BC seeds x critic seeds, band = min-max. Both use the SDE sampler.")
    if single:
        axes[0].legend([Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 2.5))),
                        Line2D([], [], color=INK2, lw=2, marker="o", ms=5)],
                       ["BC alone (N = 1)", f"best-of-{args.n}, MC critic"], frameon=False, loc="upper left",
                       labelcolor=INK2)
        axes[0].set_title(f"Flow BC on square, {args.n_demos[0]} demos: success vs BC steps, PH vs MH, "
                          f"with best-of-{args.n}", color=INK, loc="left", fontsize=11)
        fig.text(0.01, 0.005, note, color=INK2, fontsize=7.5, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.075, 0.88, 1))
    else:
        for ax in axes[len(args.n_demos):]:
            ax.axis("off")
        leg_ax = axes[len(args.n_demos)]
        leg_ax.legend(handles, labels, frameon=False, loc="upper left", labelcolor=INK2, fontsize=10)
        leg_ax.text(0.0, 0.42, note, transform=leg_ax.transAxes, color=INK2, fontsize=8, va="top")
        fig.suptitle(f"Flow BC on square: success vs BC steps, PH vs MH, with best-of-{args.n} "
                     f"(no-memory MC critic, {crit})", x=0.03, ha="left", color=INK, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=170, facecolor=SURFACE)
    with open(args.out + ".json", "w") as f:
        json.dump(out, f)
    print(args.out + ".png")


if __name__ == "__main__":
    main()
