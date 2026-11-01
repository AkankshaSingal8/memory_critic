"""Q estimate vs MC return in the layout of OGPO paper Fig. 7 (arXiv 2605.03065, q_vs_mc_panel_*.png):
three panels min(Q), mean(Q), max(Q) over the critic ensemble, Q estimate on x, MC return on y,
dashed y = x, one colour per policy, the policy's success rate in the legend.

The paper colours early / mid / late OGPO checkpoints. Here the critic is the no-memory MC critic
from scripts/train_mc_critic.py and the policies are flow-BC policies of increasing quality (BC
dataset size), so the colours are dataset sizes. Points are the decision points (chunk_pos == 0, the
critic's training points) of EPISODES held-out test episodes (index >= 320) of one policy seed;
MC return = discounted return to the episode end, gamma from the critic's config. One figure per
dataset (ph, mh).

Colour: the dataviz reference categorical slots in fixed order (blue, orange, aqua, ...), as the
paper uses three distinct hues; on one hue the overlapping clouds could not be told apart. The
legend lists the sizes in order.

--replot redraws from <out>.npz / <out>.json without the critic or /data.

    python scripts/plot_q_vs_mc_fig7.py --critic_dir /data/user_data/$USER/memory_critic/critics/mc_decision_seed0 \
        --out plots/mc_critic/q_vs_mc_fig7
"""
import argparse
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np

from train_mc_critic import chunk_actions, discounted_returns  # noqa: E402

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic_dir", required=True)
    ap.add_argument("--data_dir", default="/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--policy_seed", default="0")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--test_start", type=int, default=320)
    ap.add_argument("--out", required=True, help="prefix; writes <out>_<ds>.png and <out>.npz")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    rows = load(args) if args.replot else evaluate(args)
    plot(args, rows)


def load(args):
    z = np.load(args.out + ".npz")
    with open(args.out + ".json") as f:
        meta = json.load(f)
    return {(ds, n): {**{k: z[f"{ds}-{n}/{k}"] for k in ("G", "min", "mean", "max")},
                      "success": meta[f"{ds}-{n}"]["success"]}
            for ds in ("ph", "mh") for n in args.sizes}


def evaluate(args):

    import jax
    import jax.numpy as jnp
    from ogpo.networks.critics import Value

    with open(os.path.join(args.critic_dir, "critic_best.pkl"), "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    critic = Value(hidden_dims=tuple(int(x) for x in cfg["hidden_dims"].split(",")), layer_norm=True,
                   num_ensembles=cfg["num_qs"], critic_loss_type="mse", num_subtasks=1)
    apply = jax.jit(lambda o, a: critic.apply(ck["params"], o, a))  # (M, B)

    rows = {}
    for ds in ("ph", "mh"):
        for n in args.sizes:
            z = np.load(os.path.join(args.data_dir, f"square_{ds}_{n}demos_seed{args.policy_seed}_sde.npz"))
            ep = z["episode_id"]
            eps = np.arange(args.test_start, args.test_start + args.episodes)
            keep = (z["chunk_pos"] == 0) & np.isin(ep, eps)
            G = discounted_returns(z["reward"].astype(np.float64), ep, cfg["discount"])[keep]
            act = chunk_actions(z["action"], ep, cfg["horizon"])[keep]
            Q = np.asarray(apply(jnp.asarray(z["obs"][keep]), jnp.asarray(act)))
            rows[(ds, n)] = {"G": G, "min": Q.min(0), "mean": Q.mean(0), "max": Q.max(0),
                             "success": float(z["ep_success"][eps].mean())}
            print(f"{ds}-{n}: {keep.sum()} points, success {rows[(ds, n)]['success']:.2f}, "
                  f"r(mean Q, G) {np.corrcoef(rows[(ds, n)]['mean'], G)[0, 1]:.3f}", flush=True)
    np.savez(args.out + ".npz", **{f"{ds}-{n}/{k}": v for (ds, n), r in rows.items()
                                   for k, v in r.items() if k != "success"})
    with open(args.out + ".json", "w") as f:
        json.dump({f"{ds}-{n}": {"success": r["success"], "n_points": int(len(r["G"])),
                                 "r_mean": float(np.corrcoef(r["mean"], r["G"])[0, 1])}
                   for (ds, n), r in rows.items()}, f, indent=1)
    return rows


def plot(args, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ramp = CATEGORICAL[:len(args.sizes)]
    allv = np.concatenate([np.concatenate([r["G"], r["min"], r["max"]]) for r in rows.values()])
    lo, hi = np.floor(allv.min() / 10) * 10 - 5, 5
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ds in ("ph", "mh"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True, facecolor=SURFACE)
        for ax, stat in zip(axes, ("min", "mean", "max")):
            for colour, n in zip(ramp, args.sizes):
                r = rows[(ds, n)]
                ax.scatter(r[stat], r["G"], s=9, color=colour, alpha=0.4, linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], ls=(0, (4, 2.5)), color=INK2, lw=1.3, zorder=3)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
            ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=10)
            ax.set_title(f"{stat}(Q)", color=INK, fontsize=13, fontweight="bold")
            ax.set_xlabel("Q estimate", color=INK2, fontsize=11)
        axes[0].set_ylabel("MC return", color=INK2, fontsize=11)
        h = [Line2D([], [], ls="", marker="o", ms=9, color=c) for c in ramp]
        l = [f"{n} demos  ({100 * rows[(ds, n)]['success']:.0f}% success)" for n in args.sizes]
        h.append(Line2D([], [], ls=(0, (4, 2.5)), color=INK2, lw=1.3)); l.append("y = x")
        fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=11, labelcolor=INK,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Square {ds.upper()}: no-memory MC critic, Q vs MC return on {args.episodes} held-out "
                     f"episodes per flow-BC policy (seed {args.policy_seed}), decision points",
                     color=INK, fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
        fig.savefig(f"{args.out}_{ds}.png", dpi=150, facecolor=SURFACE)
        print(f"{args.out}_{ds}.png", flush=True)


if __name__ == "__main__":
    main()
