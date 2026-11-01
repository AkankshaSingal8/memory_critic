"""Q vs MC return with OGPO's own plotter, ogpo.utils.evaluation.visualize_q_accuracy (the same
function OGPO_public calls when eval.plot_q_vs_mc is on): top, Q prediction and MC return-to-go at
each decision point of each episode; bottom, all (Q, MC) pairs against y = x, with the correlation.

Input is a critic's test_predictions.npz from scripts/train_mc_critic.py (Q = ensemble mean at the
decision points of held-out test episodes; MC = discounted return to the episode end, gamma 0.99,
the same quantity OGPO's discounted_returns_from_prefix gives). One figure per policy group, from
EPISODES test episodes of one policy seed, then a contact sheet of all groups.

    python scripts/plot_q_accuracy_ogpo.py <critic dir> <out dir> [--policy_seed 0] [--episodes 10]
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("critic_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--policy_seed", default="0")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    from ogpo.utils.evaluation import visualize_q_accuracy

    z = np.load(os.path.join(args.critic_dir, "test_predictions.npz"))
    with open(os.path.join(args.critic_dir, "metrics.json")) as f:
        policies = json.load(f)["policies"]
    paths = []
    for ds in ("ph", "mh"):
        for n in (10, 20, 50, 100, 200):
            pid = policies.index(f"square_{ds}_{n}demos_seed{args.policy_seed}_sde")
            m = z["policy"] == pid
            eps = sorted(set(z["episode"][m].tolist()))[:args.episodes]
            ts, sc = [], []
            for e in eps:
                k = np.flatnonzero(m & (z["episode"] == e))
                k = k[np.argsort(z["timestep"][k])]
                q, g = z["q"][k].astype(float).tolist(), z["G"][k].astype(float).tolist()
                ts.append({"q_preds": q, "mc_returns": g})
                sc.extend(zip(q, g))
            suffix = f"square_{ds}_{n}demos_seed{args.policy_seed}"
            visualize_q_accuracy(ts, sc, "test", suffix, dir_suffix="mc_critic", save_dir=args.out_dir)
            paths.append(os.path.join(args.out_dir, "q_accuracy_plots", "mc_critic", f"q_accuracy_step_test_{suffix}.png"))

    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    sheet = Image.new("RGB", (5 * w, 2 * h), "white")
    for j, im in enumerate(ims):
        sheet.paste(im, ((j % 5) * w, (j // 5) * h))
    sheet.save(os.path.join(args.out_dir, f"q_accuracy_all_groups_seed{args.policy_seed}.png"))
    print("saved", len(paths), "figures and the contact sheet")


if __name__ == "__main__":
    main()
