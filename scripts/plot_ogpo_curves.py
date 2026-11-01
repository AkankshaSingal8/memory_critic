"""OGPO on square-mh (scripts/train_ogpo_square.sbatch): ODE success per seed, BC phase and
online-RL phase side by side.

W&B logs every eval against total steps (BC gradient steps + online env steps). The phase
boundary follows the clip_bc rule the runs use: BC ends at the first eval with ODE success
>= clip_bc_threshold, else at offline_steps. A run whose logged evals never cross it and stop
before offline_steps is still in (or was cut off in) BC: bc_end is null and no hand-off is drawn.

  ogpo_success.png   left: BC phase, raw evals (x = BC steps); right: online RL (x = online env
                     steps, 0 = end of BC), 3-point smoothed within the phase.
  curves.json        the plotted data

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_ogpo_curves.py [--cached]
"""
import argparse
import json
import os
import re

import numpy as np

from plot_flow_bc_curves import ENTITY_PROJECT, INK, INK2, SURFACE, smooth, style

# Dataviz reference categorical slots 1-3, in fixed seed order.
SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


def fetch(group, clip, offline_steps):
    import wandb
    api = wandb.Api(timeout=120)
    data = {}
    for r in api.runs(ENTITY_PROJECT, filters={"group": group}):
        seed = re.search(r"ogposeed(\d+)", r.name).group(1)
        rows = sorted(r.history(keys=["eval/success"], samples=100_000, pandas=False),
                      key=lambda row: row["_step"])
        step = [int(row["_step"]) for row in rows]
        ode = [float(row["eval/success"]) for row in rows]
        bc_end = next((s for s, y in zip(step, ode) if y >= clip),
                      offline_steps if step and step[-1] >= offline_steps else None)
        data[seed] = {"run_id": r.id, "name": r.name, "state": r.state, "bc_end": bc_end,
                      "step": step, "ode": ode}
        print(f"seed {seed}: {r.id} state={r.state} evals={len(step)} bc_end={bc_end} "
              f"last_step={step[-1] if step else None}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="square_mh_ogpo")
    ap.add_argument("--clip_bc_threshold", type=float, default=0.45)
    ap.add_argument("--offline_steps", type=int, default=500_000)
    ap.add_argument("--online_steps", type=int, default=2_000_000)
    ap.add_argument("--out_dir", default="plots/ogpo_square_mh")
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "curves.json")
    if args.cached:
        data = json.load(open(path))
    else:
        data = fetch(args.group, args.clip_bc_threshold, args.offline_steps)
        json.dump(data, open(path, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, (ax_bc, ax_on) = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True, facecolor=SURFACE,
                                       gridspec_kw={"width_ratios": [1, 2.2]})
    seeds = sorted(data, key=int)
    for colour, sd in zip(SEED_COLOURS, seeds):
        d = data[sd]
        step, y = np.array(d["step"]), 100 * np.array(d["ode"])
        end = d["bc_end"] if d["bc_end"] is not None else step[-1]
        bc = step <= end
        # BC unsmoothed: the hand-off is decided on a single raw eval, which smoothing would hide.
        ax_bc.plot(step[bc], y[bc], color=colour, lw=1.6)
        if d["bc_end"] is None:
            ax_on.plot([], [], color=colour, lw=2, label=f"seed {sd} (no online evals logged yet)")
            continue
        ax_bc.scatter([end], [y[bc][-1]], s=40, color=colour, edgecolor=SURFACE, lw=2, zorder=3)
        # Online smoothed within the phase only, starting from the hand-off value at x = 0.
        on = step > end
        ax_on.plot(np.r_[0, step[on] - end], np.r_[y[bc][-1], smooth(y[on])], color=colour, lw=2,
                   label=f"seed {sd}")
        ax_on.scatter([0], [y[bc][-1]], s=40, color=colour, edgecolor=SURFACE, lw=2, zorder=3)
    bc_max = max(d["bc_end"] or d["step"][-1] for d in data.values())
    for ax, xmax, xlabel in ((ax_bc, 50_000 * np.ceil(bc_max * 1.1 / 50_000), "BC gradient steps"),
                             (ax_on, args.online_steps, "online env steps")):
        style(ax, ylabel=ax is ax_bc)
        ax.set_xlim(0, xmax * 1.02)
        ax.set_xlabel(xlabel, color=INK2)
    ax_on.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    ax_bc.axhline(100 * args.clip_bc_threshold, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax_bc.set_title(f"BC (raw; stops at \u2265 {args.clip_bc_threshold:.0%}, dashed)", color=INK,
                    loc="left", fontsize=10)
    ax_on.set_title("online RL (3-point smoothed; dot = BC hand-off)", color=INK, loc="left",
                    fontsize=10)
    ax_on.legend(frameon=False, loc="lower right", labelcolor=INK2)
    unfinished = [sd for sd in seeds if data[sd]["state"] != "finished"]
    note = (f"\nnot finished: seeds {', '.join(unfinished)} (curves end at the last logged eval)"
            if unfinished else "")
    fig.suptitle(f"OGPO on square-mh: ODE success, 64 episodes per eval{note}",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.out_dir, "ogpo_success.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
