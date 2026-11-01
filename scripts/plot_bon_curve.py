"""π_base vs best-of-N with the MC critic over BC training, one panel per BC dataset size.

Same layout and conventions as SafeVaDaR scripts/safety_pretrain/plot_pibase_vs_bon.py
(plots/critic_bon/pibase_vs_bon_with_n32.png): 3 seeds (0 7 42), line = seed mean, 64 SDE episodes
per point, 3-point smoothing, π_base in categorical slot 1 (#2a78d6), BoN-N on the one-hue orange
ordinal ramp light -> dark with N, direct labels "name value%" at each line's end.

  π_base  plots/flow_bc_<task>_<ds>/curves.json: eval_sde/success of eval_bc_run.py, 64 episodes
          from seeds T..T+63 at checkpoint T
  BoN-N   <root>/square_<ds>_<n>demos_seed<S>_step<T>_sde_bon<N>.json (scripts/bon_curve.sbatch):
          the same checkpoint and the same 64 start states; π_base draws N chunks per decision and
          the MC critic (ensemble mean) keeps one

Writes <out>.png and <out>.json (the plotted per-seed values, before smoothing).

    python scripts/plot_bon_curve.py --root /data/user_data/$USER/memory_critic/bon_curve/mc_decision_seed0 \
        --out plots/bon/pibase_vs_bon_ph
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID_C, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEEDS = ("0", "7", "42")
SIZES = (10, 20, 50, 100, 200)
BASE_COLOR = "#2a78d6"
BON_RAMP = ("#f2a07c", "#e45f28", "#a83c10", "#5f1e04")  # as SafeVaDaR (validate_palette.js --ordinal passes)


def smooth(y):
    return np.array([np.nanmean(y[max(0, i - 1):i + 2]) for i in range(len(y))])


def style(ax, title):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("BC step (k)", color=INK2)
    ax.set_ylabel("Success (%)", color=INK2)


def labels_at_end(ax, items, span, lo, hi):
    """As SafeVaDaR: labels at each line's last point, >= 6.5% of the axis apart, re-centred on
    the values and kept inside [lo, hi]."""
    gap = span * 0.065
    for x in sorted({it[1] for it in items}):
        grp = sorted((y, name) for y, xx, name in items if xx == x)
        ys = [y for y, _ in grp]
        pos = []
        for y in ys:
            pos.append(max(y, pos[-1] + gap) if pos else y)
        shift = np.mean(ys) - np.mean(pos)
        pos = [p + shift for p in pos]
        pos = [p + max(0, lo + gap / 2 - pos[0]) - max(0, pos[-1] - (hi - gap / 2)) for p in pos]
        for (y, name), yy in zip(grp, pos):
            ax.annotate(f"{name}  {y:.1f}%", (x, yy), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)


def panel(ax, curves, title, grid):
    x = np.array(grid) / 1e3
    ends = []
    for label, short, color, Y in curves:
        S = np.array([smooth(y) for y in Y])
        m = np.nanmean(S, 0)
        ok = ~np.isnan(m) & ~np.isnan(Y).all(0)
        if not ok.any():
            continue
        sparse = ok.sum() < len(grid)
        ax.plot(x[ok], m[ok], color=color, lw=2.6 if short == "π_base" else 2, label=label,
                zorder=4 if short == "π_base" else 3, solid_capstyle="round",
                **({"marker": "o", "ms": 4} if sparse else {}))
        ends.append((m[ok][-1], x[ok][-1], short))
    hi = 105
    ax.set_ylim(0, hi)
    ax.set_xlim(0, x[-1] * 1.22)
    labels_at_end(ax, ends, hi, 0, hi)
    style(ax, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="ph")
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--grid", type=int, nargs="+", default=list(range(50000, 500001, 50000)))
    ap.add_argument("--critic_name", default="MC critic")
    ap.add_argument("--out", default=os.path.join(REPO, "plots/bon/pibase_vs_bon_ph"))
    args = ap.parse_args()
    grid, ds = args.grid, args.dataset
    with open(os.path.join(REPO, f"plots/flow_bc_square_{ds}/curves.json")) as f:
        C = json.load(f)

    def base(n):
        Y = np.full((len(SEEDS), len(grid)), np.nan)
        for i, s in enumerate(SEEDS):
            r = C[str(n)].get(s)
            if r:
                d = dict(zip(r["step"], r["sde"]))
                Y[i] = [100 * d.get(t, np.nan) for t in grid]
        return Y

    def bon(n, N):
        Y = np.full((len(SEEDS), len(grid)), np.nan)
        for i, s in enumerate(SEEDS):
            for j, t in enumerate(grid):
                p = os.path.join(args.root, f"square_{ds}_{n}demos_seed{s}_step{t}_sde_bon{N}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        Y[i, j] = 100 * json.load(f)["success_rate"]
        return Y

    data = {n: {"base": base(n), **{N: bon(n, N) for N in args.ns}} for n in SIZES}
    total = len(SEEDS) * len(grid) * len(SIZES)
    have = {f"N={N}": int(sum((~np.isnan(data[n][N])).sum() for n in SIZES)) for N in args.ns}
    print("points: " + ", ".join(f"{k} {v}/{total}" for k, v in have.items()))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), facecolor=BG)
    flat = axes.ravel()
    for ax, n in zip(flat, SIZES):
        curves = [("π_base (SDE sample, N = 1)", "π_base", BASE_COLOR, data[n]["base"])]
        curves += [(f"best-of-{N}: π_base draws {N}, {args.critic_name} keeps one", f"N={N}", c, data[n][N])
                   for N, c in zip(args.ns, BON_RAMP[len(BON_RAMP) - len(args.ns):])]
        panel(ax, curves, f"Square {ds.upper()}, {n} demos", grid)
    flat[-1].axis("off")
    flat[-1].legend(*flat[0].get_legend_handles_labels(), loc="center left", frameon=False, fontsize=10)
    head = (f"Flow BC, square {ds.upper()} · best-of-N with the no-memory {args.critic_name} · 3 seeds "
            "(line = seed mean) · 64 SDE episodes per point · 3-pt smoothed")
    if any(v < total for v in have.values()):
        head += " · partial: " + ", ".join(f"{k} {v}/{total}" for k, v in have.items() if v < total)
    fig.suptitle(head, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=125, facecolor=BG)
    nan_none = lambda v: np.where(np.isnan(v), None, v).tolist()
    with open(args.out + ".json", "w") as f:
        json.dump({"grid": grid, "seeds": SEEDS, "ns": args.ns,
                   **{f"{n}demos/{'pibase' if k == 'base' else f'bon{k}'}": nan_none(v)
                      for n, d in data.items() for k, v in d.items()}}, f)
    print(args.out + ".png")
    for n in SIZES:
        for N in args.ns:
            m = ~np.isnan(data[n][N]) & ~np.isnan(data[n]["base"])
            if m.any():
                print(f"{n:>3} demos N={N:<3d} matched {m.sum():2d} pts: π_base {data[n]['base'][m].mean():5.1f}  "
                      f"BoN {data[n][N][m].mean():5.1f}  diff {(data[n][N] - data[n]['base'])[m].mean():+5.1f}")


if __name__ == "__main__":
    main()
