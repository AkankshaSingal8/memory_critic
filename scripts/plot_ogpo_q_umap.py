"""Q-map of the OGPO critic: UMAP of the action chunks a trained OGPO agent chose, coloured by what its
critic says about them, next to what actually happened.

Input: the npz files from scripts/ogpo_q_diagnostics.py (one per run and sampler: every decision of 64
eval episodes, its action chunk, the target critic's Q (q_agg over the ensemble) and the discounted MC
return from that decision). The UMAP is OGPO's own (StandardScaler on the flattened chunk, n_neighbors 15,
min_dist 0.1, random_state 42), fitted once per (run, sampler) and drawn four ways:
  1  Q vs MC return, y = x dashed; marker = episode outcome (success circle, failure cross)
  2  UMAP coloured by Q            } one blue scale shared by 2 and 3, so the colours compare
  3  UMAP coloured by MC return    }
  4  UMAP coloured by Q - MC (blue = critic too low, red = too high, gray = right)
Writes <dir>/umap/<run>.png (a row per sampler), <dir>/umap/overview_<sampler>.png (UMAP coloured by Q,
PH and MH x demo count) and <dir>/umap/summary.json.

    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 $OGPO_PUBLIC/.venv/bin/python scripts/plot_ogpo_q_umap.py
"""
import argparse
import glob
import json
import os
import re

import numpy as np

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
OK, FAIL = "#2a78d6", "#eb6834"          # categorical slots 1, 2
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # blue 100..700
DIV = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2e"]                          # blue - gray - red
SAMPLER_LABEL = {"ode": "ODE (plain flow policy)", "sde": "SDE best-of-8 (deployed policy)"}


def parse(run):
    m = re.match(r"ogpo_public_square_(ph|mh)(?:_(\d+)demos)?_\d+_seed(\d+)$", run)
    ds, n, seed = m.groups()
    full = n is None
    n = {"ph": 200, "mh": 300}[ds] if full else int(n)
    return ds, n, int(seed), full


def embed(actions):
    import umap
    from sklearn.preprocessing import StandardScaler
    return umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(
        StandardScaler().fit_transform(actions))


def load(path, cache):
    z = dict(np.load(path, allow_pickle=False))
    key = os.path.basename(path)[:-4]
    cp = os.path.join(cache, key + "_umap.npy")
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(path):
        z["xy"] = np.load(cp)
    else:
        z["xy"] = embed(z["actions"]).astype(np.float32)
        np.save(cp, z["xy"])
    z["succ"] = z["episode_success"][z["episode"]] > 0.5
    return z


def umap_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("UMAP 1", color=MUTED, fontsize=8); ax.set_ylabel("UMAP 2", color=MUTED, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="plots/ogpo_public/q_diagnostics")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D

    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    div = LinearSegmentedColormap.from_list("div", DIV)
    out = os.path.join(args.dir, "umap")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir, "*_ode.npz")) + glob.glob(os.path.join(args.dir, "*_sde.npz")))
    runs = {}
    for f in files:
        run, sampler = os.path.basename(f)[:-4].rsplit("_", 1)
        runs.setdefault(run, {})[sampler] = f
    rng = np.random.default_rng(0)
    data, summary = {}, {}

    for run in sorted(runs, key=lambda r: parse(r)[:2]):
        ds, n, seed, full = parse(run)
        samplers = [s for s in ("ode", "sde") if s in runs[run]]
        fig, axes = plt.subplots(len(samplers), 4, figsize=(17, 4.2 * len(samplers) + 0.9), facecolor=SURFACE,
                                 squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1, 1]})
        for row, sampler in zip(axes, samplers):
            z = load(runs[run][sampler], out)
            data[(run, sampler)] = z
            q, mc, xy, succ = z["q"], z["mc"], z["xy"], z["succ"]
            order = rng.permutation(len(q))
            r = float(np.corrcoef(q, mc)[0, 1]) if len(q) > 2 else float("nan")
            sr = float(np.nanmean(z["episode_success"]))
            summary[f"{run}/{sampler}"] = {"decisions": int(len(q)), "episodes": int(len(z["episode_success"])),
                                           "success": sr, "pearson_q_mc": r, "mean_q": float(q.mean()),
                                           "mean_mc": float(mc.mean()), "mean_q_minus_mc": float((q - mc).mean()),
                                           "ckpt_step": int(z["ckpt_step"])}
            lo, hi = float(min(q.min(), mc.min())), float(max(q.max(), mc.max()))

            ax = row[0]
            ax.set_facecolor(SURFACE)
            ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=MUTED, lw=1.2, zorder=1)
            ax.scatter(q[succ], mc[succ], s=7, color=OK, alpha=0.45, lw=0, zorder=2)
            ax.scatter(q[~succ], mc[~succ], s=12, color=FAIL, alpha=0.7, marker="x", lw=0.9, zorder=3)
            ax.set_xlim(lo - 2, hi + 2); ax.set_ylim(lo - 2, hi + 2); ax.set_aspect("equal")
            ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_xlabel("critic Q", color=INK2); ax.set_ylabel("MC return", color=INK2)
            ax.set_title(f"{SAMPLER_LABEL[sampler]}\nQ vs MC: r = {r:.2f}, success {100 * sr:.0f}%",
                         color=INK, loc="left", fontsize=10)

            norm = Normalize(lo, hi)
            for ax, val, title in ((row[1], q, "UMAP coloured by critic Q"), (row[2], mc, "UMAP coloured by MC return")):
                umap_axes(ax)
                sc = ax.scatter(xy[order, 0], xy[order, 1], c=val[order], cmap=seq, norm=norm, s=5, lw=0, alpha=0.85)
                ax.set_title(title, color=INK, loc="left", fontsize=10)
            cb = fig.colorbar(sc, ax=row[1:3], shrink=0.8, pad=0.01, aspect=30)
            cb.set_label("value (Q or MC return)", color=INK2, fontsize=8); cb.ax.tick_params(labelsize=7, colors=INK2)
            cb.outline.set_visible(False)

            ax = row[3]
            umap_axes(ax)
            err = q - mc
            m = max(float(np.abs(err).max()), 1e-6)
            sc = ax.scatter(xy[order, 0], xy[order, 1], c=err[order], cmap=div, norm=TwoSlopeNorm(0, -m, m),
                            s=5, lw=0, alpha=0.85)
            ax.set_title("UMAP coloured by Q - MC", color=INK, loc="left", fontsize=10)
            cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01, aspect=30)
            cb.set_label("Q - MC  (red: critic too high)", color=INK2, fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=INK2); cb.outline.set_visible(False)

        handles = [Line2D([], [], color=OK, marker="o", ls="", ms=5), Line2D([], [], color=FAIL, marker="x", ls="", ms=6),
                   Line2D([], [], color=MUTED, ls=(0, (4, 3)), lw=1.2)]
        fig.legend(handles, ["successful episode", "failed episode", "y = x"], frameon=False, loc="upper right",
                   ncol=3, labelcolor=INK2, fontsize=9, bbox_to_anchor=(0.99, 0.995))
        name = f"square-{ds.upper()}, {n} demos{' (full dataset)' if full else ''}, seed {seed}"
        fig.suptitle(f"OGPO critic Q-map: {name}, final checkpoint ({int(z['ckpt_step']) // 1000}k steps)\n"
                     f"each point = one decision's action chunk ({int(z['horizon'])} steps x {int(z['action_dim'])} dims) "
                     f"from 64 eval episodes; Q = target critic, ensemble {z['q_agg']}",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.savefig(os.path.join(out, f"{run}.png"), dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(os.path.join(out, f"{run}.png"))

    # overview: UMAP coloured by Q, rows PH / MH, columns demo count
    cols = sorted({parse(r)[1] for r in runs})
    for sampler in ("ode", "sde"):
        keys = [(r, s) for (r, s) in data if s == sampler]
        if not keys:
            continue
        lo = min(float(min(data[k]["q"].min(), data[k]["mc"].min())) for k in keys)
        hi = max(float(max(data[k]["q"].max(), data[k]["mc"].max())) for k in keys)
        fig, axes = plt.subplots(2, len(cols), figsize=(2.9 * len(cols) + 1.2, 6.4), facecolor=SURFACE, squeeze=False)
        for i, ds in enumerate(("ph", "mh")):
            for j, n in enumerate(cols):
                ax = axes[i][j]
                umap_axes(ax)
                ax.set_xlabel(""); ax.set_ylabel("")
                hit = [k for k in keys if parse(k[0])[:2] == (ds, n)]
                if not hit:
                    ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
                    continue
                z = data[hit[0]]
                o = rng.permutation(len(z["q"]))
                sc = ax.scatter(z["xy"][o, 0], z["xy"][o, 1], c=z["q"][o], cmap=seq, norm=Normalize(lo, hi), s=3, lw=0)
                s = summary[f"{hit[0][0]}/{sampler}"]
                full = " (full)" if parse(hit[0][0])[3] else ""
                ax.set_title(f"{ds.upper()} {n} demos{full}\nsuccess {100 * s['success']:.0f}%, r = {s['pearson_q_mc']:.2f}",
                             color=INK, loc="left", fontsize=9)
        cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01, aspect=35)
        cb.set_label("critic Q", color=INK2); cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
        fig.suptitle(f"OGPO critic Q-map, {SAMPLER_LABEL[sampler]}: UMAP of chosen action chunks coloured by Q, "
                     "final checkpoints, seed 0 (r = Pearson Q vs MC return)", x=0.01, ha="left", color=INK, fontsize=11)
        fig.savefig(os.path.join(out, f"overview_{sampler}.png"), dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(os.path.join(out, f"overview_{sampler}.png"))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
