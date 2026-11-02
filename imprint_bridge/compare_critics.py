"""Compare critic arms (nomem / clock / mem_*) from their metrics.json: test MSE, R2, AUC, mean +- std over seeds.

    python -m imprint_bridge.compare_critics /data/user_data/$USER/memory_critic/imprint/critics [--by n_demos]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("critics_dir")
    ap.add_argument("--by", choices=("overall", "n_demos", "group"), default="overall")
    ap.add_argument("--out", default=None, help="optional JSON with the table")
    args = ap.parse_args(argv)

    rows = defaultdict(lambda: defaultdict(list))   # (arm, slice) -> metric -> [per seed]
    for path in sorted(glob.glob(os.path.join(args.critics_dir, "*_seed*", "metrics.json"))):
        arm = re.sub(r"_seed\d+$", "", os.path.basename(os.path.dirname(path)))
        test = json.load(open(path))["test"]
        slices = {"all": test["overall"]} if args.by == "overall" else test[f"by_{args.by}"]
        for sl, m in slices.items():
            for k in ("mse", "r2", "auc_success", "ece"):
                if m.get(k) is not None:
                    rows[(arm, sl)][k].append(m[k])
    if not rows:
        raise SystemExit(f"no */metrics.json under {args.critics_dir}")

    def fmt(v):
        return f"{np.mean(v):7.3f}±{np.std(v):.3f}" if v else "      -      "

    print(f"{'arm':>22} {'slice':>6} {'seeds':>5} {'test mse':>14} {'r2':>14} {'auc_success':>14} {'ece':>14}")
    table = []
    for (arm, sl), m in sorted(rows.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
        print(f"{arm:>22} {sl:>6} {len(m['mse']):>5} {fmt(m['mse'])} {fmt(m['r2'])} {fmt(m['auc_success'])} "
              f"{fmt(m['ece'])}")
        table.append({"arm": arm, "slice": sl, **{k: v for k, v in m.items()}})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
