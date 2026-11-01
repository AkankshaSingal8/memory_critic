"""Plots for the public-repo runs (scripts/train_ogpo_public.sbatch): the BC-only baseline
(KIND=bc, square ph/mh x 10/20/50/100/200 demos) and OGPO with critics (KIND=ogpo, full
square-mh / square-ph datasets), both pulled from W&B project memory-critic.

W&B steps: BC runs log eval/success (ODE) and eval_sde/success every 10k BC steps. OGPO keeps
one step counter across phases: BC gradient steps until the clip_bc hand-off (first eval with
SDE success >= 0.45, the public square.sh rule), then +1 per online env step. So online env
steps = step - bc_end. bc_end is cross-checked against "Breaking due to BC clipping" in the
job log.

A BC job that was preempted after it had finished was requeued and retrained from scratch,
so some seeds have several finished 500k runs. The first finished run per seed is plotted;
the rest are listed in wandb_check.txt.

  bc_success_vs_steps.png       BC only: success vs BC steps per dataset size (rows ph/mh,
                                cols SDE/ODE), 3-point smoothed, mean over seeds, band min-max
  bc_success_vs_dataset_size.png  BC only: success vs demos at the final and the peak eval
  ogpo_square_{mh,ph}.png       OGPO per seed: BC phase (raw) and online phase (smoothed),
                                ODE and SDE; square-mh also shows the paper's Fig. 9 targets
  ogpo_vs_bc.png                OGPO online curve (mean over seeds) vs the BC-only baseline
  wandb_check.txt / curves.json what W&B holds per run / the plotted data

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/plot_ogpo_public.py [--cached]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from plot_flow_bc_curves import ENTITY_PROJECT, INK, INK2, ODE_C, RAMP, SDE_C, SURFACE, smooth, style

DEMOS = [10, 20, 50, 100, 200]
TYPES = ["ph", "mh"]
# Full datasets (what OGPO trains on): square-ph has 200 demos, square-mh 300.
FULL_DEMOS = {"ph": 200, "mh": 300}
CLIP = 0.45
ONLINE_STEPS = 2_000_000
ONLINE_SMOOTH = 5   # centred window, in evals (eval every 10k online steps -> 50k)
SEED_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]   # dataviz reference categorical slots 1-3
OGPO_C, BC_C = "#2a78d6", "#eb6834"
# Paper Fig. 9, square-mh OGPO-CA, read off the figure: (online env steps, success %).
PAPER_MH = [(400_000, 85), (1_000_000, 95), (1_500_000, 97)]


def smooth_k(y, k):
    """Centred k-point mean; ends average the neighbours that exist."""
    h = k // 2
    return np.array([y[max(0, i - h):i + h + 1].mean() for i in range(len(y))])


def seed_of(name):
    return re.search(r"_seed(\d+)seed", name).group(1)


def history(run):
    """{step: {"ode": .., "sde": ..}} for every step holding both evals (later rows win)."""
    out = {}
    for key, mode in (("eval/success", "ode"), ("eval_sde/success", "sde")):
        for row in run.history(keys=[key], samples=100_000, pandas=False):
            out.setdefault(int(row["_step"]), {})[mode] = float(row[key])
    steps = sorted(s for s, v in out.items() if len(v) == 2)
    return {"step": steps, "ode": [out[s]["ode"] for s in steps], "sde": [out[s]["sde"] for s in steps]}


def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    data, report = {"bc": {}, "ogpo": {}}, []
    for dt in TYPES:
        for n in DEMOS:
            group = f"square_{dt}_bc_public_{n}demos"
            runs = list(api.runs(ENTITY_PROJECT, filters={"group": group}))
            finished = sorted((r for r in runs if r.state == "finished"), key=lambda r: r.created_at)
            report.append(f"== {group}: {len(runs)} W&B runs, {len(finished)} finished")
            by_seed = {}
            for r in finished:
                by_seed.setdefault(seed_of(r.name), []).append(r)
            for sd in sorted(by_seed, key=int):
                r, extra = by_seed[sd][0], by_seed[sd][1:]
                d = {"run_id": r.id, "name": r.name, **history(r)}
                report.append(f"   seed {sd}: plotted {r.id} ({r.created_at}), {len(d['step'])} evals "
                              f"{d['step'][0]}..{d['step'][-1]}; duplicate finished runs: "
                              f"{[x.id for x in extra] or 'none'}")
                data["bc"].setdefault(dt, {}).setdefault(str(n), {})[sd] = d
            print(report[-1 - len(by_seed)], flush=True)
        group = f"square_{dt}_ogpo_public"
        runs = list(api.runs(ENTITY_PROJECT, filters={"group": group}))
        report.append(f"== {group}: {len(runs)} W&B runs")
        for r in sorted(runs, key=lambda r: int(seed_of(r.name))):
            sd = seed_of(r.name)
            d = {"run_id": r.id, "name": r.name, "state": r.state, **history(r)}
            # The online phase logs a fresh eval at the hand-off step, overwriting the BC eval
            # there in W&B, so the job log is authoritative; the W&B rule is a cross-check.
            hits = [i for i, y in enumerate(d["sde"]) if y >= CLIP]
            wb_end = d["step"][hits[0]] if hits else None
            log_end, log_vals = log_handoff(dt, sd)
            d["bc_end"] = log_end if log_end is not None else wb_end
            d["bc_end_log"] = log_vals   # {"ode", "sde"} of the last BC eval, from the log
            report.append(f"   seed {sd}: {r.id} state={r.state} evals={len(d['step'])} "
                          f"bc_end={d['bc_end']} (log {log_end}, {log_vals}; W&B first SDE>=0.45: "
                          f"{wb_end} -> {'agree' if wb_end == log_end else 'DISAGREE'}) "
                          f"last_step={d['step'][-1] if d['step'] else None}")
            data["ogpo"].setdefault(dt, {})[sd] = d
        print("\n".join(report[-len(runs) - 1:]), flush=True)
    return data, report


def log_handoff(dt, sd):
    """(BC step, {"ode", "sde"}) of the eval that triggered the first 'Breaking due to BC
    clipping' in the job log."""
    logs = glob.glob(f"logs/ogpo_public_square_{dt}_seed{sd}_*.out")
    if len(logs) != 1:
        return None, None
    last, vals = None, {}
    for line in open(logs[0], errors="ignore"):
        m = re.match(r"\[EVAL (ODE|SDE)\] step=\s*(\d+) \| success=([0-9.]+)", line)
        if m:
            last = int(m.group(2))
            vals[m.group(1).lower()] = float(m.group(3))
        if line.startswith("Breaking due to BC clipping"):
            return last, dict(vals)
    return None, None


def bc_stats(runs, mode, k=3):
    steps = sorted(set.intersection(*(set(d["step"]) for d in runs.values())))
    ys = np.array([100 * smooth_k(np.array([d[mode][d["step"].index(s)] for s in steps]), k)
                   for d in runs.values()])
    return np.array(steps), ys


def finish(fig, out, rect=(0, 0, 1, 0.95)):
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


def legend_right(ax):
    h, lab = ax.get_legend_handles_labels()   # largest dataset first: matches top-to-bottom order
    ax.legend(h[::-1], lab[::-1], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK2)


def plot_bc_steps(plt, bc, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True, facecolor=SURFACE)
    for row, dt in zip(axes, TYPES):
        for ax, (mode, name) in zip(row, (("sde", "SDE"), ("ode", "ODE"))):
            for colour, n in zip(RAMP, sorted(bc[dt], key=int)):
                x, ys = bc_stats(bc[dt][n], mode)
                ax.fill_between(x, ys.min(0), ys.max(0), color=colour, alpha=0.15, lw=0)
                ax.plot(x, ys.mean(0), color=colour, lw=2, label=f"{n} demos", solid_capstyle="round")
            style(ax, ylabel=ax is row[0])
            if dt == TYPES[0]:
                ax.set_xlabel("")
            ax.set_title(f"square-{dt}, {name}", color=INK, loc="left", fontsize=10)
        legend_right(row[1])
    fig.suptitle("BC only (public OGPO repo, BC phase): success vs BC steps\n3-point smoothed, mean of "
                 "seeds 0, 7, 42, band = min-max, 64 episodes per eval", x=0.01, ha="left", color=INK, fontsize=11)
    finish(fig, out, rect=(0, 0, 1, 0.93))
    plt.close(fig)


def plot_bc_scaling(plt, bc, out):
    xs = np.array(DEMOS)
    picks = (("final eval (500k)", lambda y: y[-1]), ("peak eval", lambda y: y.max()))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), sharey=True, facecolor=SURFACE)
    lines = []
    for row, dt in zip(axes, TYPES):
        for ax, (pname, pick) in zip(row, picks):
            for mode, colour, label, dx in (("sde", SDE_C, "SDE", 0.97), ("ode", ODE_C, "ODE", 1.03)):
                vals = np.array([[pick(y) for y in bc_stats(bc[dt][str(n)], mode)[1]] for n in xs])
                for x, v in zip(xs, vals):
                    ax.scatter(np.full(len(v), x * dx), v, s=14, color=colour, alpha=0.35, lw=0)
                ax.plot(xs, vals.mean(1), color=colour, lw=2, marker="o", ms=7, label=label,
                        markeredgecolor=SURFACE, markeredgewidth=2)
                lines.append(f"square-{dt} {pname:<17} {label}: " + "  ".join(
                    f"{n}:{m:5.1f}" for n, m in zip(xs, vals.mean(1))))
            style(ax, ylabel=ax is row[0])
            ax.set_xscale("log")
            ax.set_xlim(8, 260)
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.minorticks_off()
            ax.set_xlabel("demos (log scale)", color=INK2)
            ax.set_title(f"square-{dt}, {pname}", color=INK, loc="left", fontsize=10)
    axes[0][0].legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.suptitle("BC only (public OGPO repo): success vs dataset size\nmean of seeds 0, 7, 42 "
                 "(dots = seeds), 3-point smoothed", x=0.01, ha="left", color=INK, fontsize=11)
    finish(fig, out, rect=(0, 0, 1, 0.94))
    plt.close(fig)
    print("\n".join(lines))


def online_xy(d, mode):
    """Online env steps (0 = hand-off, value = the hand-off eval) and smoothed success (%)."""
    step, y = np.array(d["step"]), 100 * np.array(d[mode])
    on = step > d["bc_end"]
    y0 = y[step == d["bc_end"]][0]
    return np.r_[0, step[on] - d["bc_end"]], np.r_[y0, smooth_k(y[on], ONLINE_SMOOTH)]


def online_axis(ax, xlabel=True):
    from matplotlib.ticker import FuncFormatter
    ax.set_xlim(0, ONLINE_STEPS * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    ax.set_xlabel("online env steps" if xlabel else "", color=INK2)


def paper_marks(ax, label=True):
    ax.scatter(*zip(*PAPER_MH), marker="D", s=36, facecolor=SURFACE, edgecolor=INK, lw=1.4, zorder=4,
               label="paper Fig. 9 (approx.)" if label else None)


def status_note(runs):
    parts = []
    for sd in sorted(runs, key=int):
        d = runs[sd]
        if d["state"] != "finished":
            parts.append(f"seed {sd} at {(d['step'][-1] - d['bc_end']) / 1e6:.2f}M")
    return f"\nstill running: {', '.join(parts)}" if parts else "\nall seeds finished"


def plot_ogpo(plt, runs, dt, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 2.4]})
    bc_max = max(d["bc_end"] for d in runs.values())
    for row, (mode, name) in zip(axes, (("ode", "ODE"), ("sde", "SDE"))):
        ax_bc, ax_on = row
        for colour, sd in zip(SEED_COLOURS, sorted(runs, key=int)):
            d = runs[sd]
            step, y = np.array(d["step"]), 100 * np.array(d[mode])
            bc = step < d["bc_end"]
            # BC unsmoothed: the hand-off is decided on a single raw eval. W&B holds the online
            # phase's re-eval at bc_end, so the BC eval there comes from the job log.
            xb, yb = np.r_[step[bc], d["bc_end"]], np.r_[y[bc], 100 * d["bc_end_log"][mode]]
            ax_bc.plot(xb, yb, color=colour, lw=1.6)
            ax_bc.scatter([xb[-1]], [yb[-1]], s=40, color=colour, edgecolor=SURFACE, lw=2, zorder=3)
            x, ys = online_xy(d, mode)
            ax_on.plot(x, ys, color=colour, lw=2, label=f"seed {sd}")
            ax_on.scatter([0], [ys[0]], s=40, color=colour, edgecolor=SURFACE, lw=2, zorder=3)
        if dt == "mh":
            paper_marks(ax_on)
        style(ax_bc)
        ax_bc.set_xlim(0, 50_000 * np.ceil(bc_max * 1.15 / 50_000))
        ax_bc.set_xlabel("BC gradient steps" if mode == "sde" else "", color=INK2)
        style(ax_on, ylabel=False)
        online_axis(ax_on, xlabel=mode == "sde")
        if mode == "sde":
            ax_bc.axhline(100 * CLIP, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax_bc.set_title(f"{name}: BC phase (raw)" + ("; dashed = 45%" * (mode == "sde")),
                        color=INK, loc="left", fontsize=10)
        ax_on.set_title(f"{name}: online RL ({ONLINE_SMOOTH * 10}k-step moving average; dot = hand-off)",
                        color=INK, loc="left", fontsize=10)
    axes[0][1].legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.suptitle(f"OGPO with critics (public repo) on square-{dt}, {FULL_DEMOS[dt]} demos: success, "
                 f"64 episodes per eval{status_note(runs)}", x=0.01, ha="left", color=INK, fontsize=11)
    finish(fig, out, rect=(0, 0, 1, 0.94))
    plt.close(fig)


def plot_vs(plt, data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True, facecolor=SURFACE)
    grid = np.arange(0, ONLINE_STEPS + 1, 10_000)
    for col, dt in enumerate(TYPES):
        runs = data["ogpo"][dt]
        bc_n = str(max(DEMOS))
        for row, (mode, name) in enumerate((("ode", "ODE"), ("sde", "SDE"))):
            ax = axes[row][col]
            curves = [online_xy(d, mode) for d in runs.values()]
            end = min(x[-1] for x, _ in curves)
            g = grid[grid <= end]
            ys = np.array([np.interp(g, x, y) for x, y in curves])
            ax.fill_between(g, ys.min(0), ys.max(0), color=OGPO_C, alpha=0.15, lw=0)
            ax.plot(g, ys.mean(0), color=OGPO_C, lw=2, label=f"OGPO with critics ({FULL_DEMOS[dt]} demos)")
            _, bys = bc_stats(data["bc"][dt][bc_n], mode)
            final = np.array([y[-1] for y in bys])
            ax.axhspan(final.min(), final.max(), color=BC_C, alpha=0.15, lw=0)
            ax.axhline(final.mean(), color=BC_C, lw=2, ls=(0, (5, 3)),
                       label=f"BC only, {bc_n} demos, final (500k BC steps)")
            if dt == "mh":
                paper_marks(ax, label=row == 0)
            ax.text(g[-1], ys.mean(0)[-1] + 2.5, f"{ys.mean(0)[-1]:.0f}%", color=INK, ha="right", fontsize=9)
            ax.text(ONLINE_STEPS, final.mean() - 6, f"{final.mean():.0f}%", color=INK, ha="right", fontsize=9)
            style(ax, ylabel=col == 0)
            online_axis(ax, xlabel=row == 1)
            ax.set_title(f"square-{dt}, {name}", color=INK, loc="left", fontsize=10)
    h, lab = axes[0][1].get_legend_handles_labels()
    fig.legend(h, [l.replace(" (300 demos)", "") for l in lab], frameon=False, loc="upper left",
               bbox_to_anchor=(0.01, 0.915), ncol=3, labelcolor=INK2, fontsize=9)
    fig.suptitle("OGPO with critics vs BC only (public repo). OGPO: mean of seeds over the online steps all "
                 f"seeds have reached ({ONLINE_SMOOTH * 10}k moving average), band = min-max;\nBC only: "
                 "mean of seeds 0, 7, 42 at the final eval (3-point smoothed), band = min-max. "
                 "Square-mh BC uses 200 of the 300 demos.", x=0.01, ha="left", color=INK, fontsize=10)
    finish(fig, out, rect=(0, 0, 1, 0.88))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="plots/ogpo_public")
    ap.add_argument("--cached", action="store_true", help="replot from <out_dir>/curves.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "curves.json")
    if args.cached:
        data = json.load(open(path))
    else:
        data, report = fetch()
        json.dump(data, open(path, "w"), indent=1)
        with open(os.path.join(args.out_dir, "wandb_check.txt"), "w") as f:
            f.write("\n".join(report) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_bc_steps(plt, data["bc"], os.path.join(args.out_dir, "bc_success_vs_steps.png"))
    plot_bc_scaling(plt, data["bc"], os.path.join(args.out_dir, "bc_success_vs_dataset_size.png"))
    for dt in TYPES:
        plot_ogpo(plt, data["ogpo"][dt], dt, os.path.join(args.out_dir, f"ogpo_square_{dt}.png"))
    plot_vs(plt, data, os.path.join(args.out_dir, "ogpo_vs_bc.png"))


if __name__ == "__main__":
    main()
