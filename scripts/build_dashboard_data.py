"""Datasets for the project dashboard artifact (https://claude.ai/artifact/YKqC7RPua3YxBpBp6u7DnR).

Reads the curves.json files written by plot_flow_bc_curves.py and plot_ogpo_public.py and writes
one JSON file of row objects per dashboard dataset into OUT_DIR:
  flow_bc.json   Flow BC (this repo) per dt/demos/mode/step: mean, lo, hi over seeds (3-pt smoothed)
  ogpo_bc.json   OGPO repo BC-only, same shape
  ogpo_rl.json   OGPO with critics, online steps from the hand-off (5-eval smoothed)
  ogpo_handoff.json, paper_mh.json
The page itself is dashboard/index.html.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/build_dashboard_data.py OUT_DIR
"""
import json, numpy as np, sys
OUT = sys.argv[1]
def sm(y, k):
    h = k // 2; y = np.asarray(y, float)
    return np.array([y[max(0, i-h):i+h+1].mean() for i in range(len(y))])

def bc_rows(per_dt, method):
    rows = []
    for dt, sizes in per_dt.items():
        for n, seeds in sizes.items():
            steps = sorted(set.intersection(*[set(s["step"]) for s in seeds.values()]))
            for mode in ("sde", "ode"):
                ys = np.array([100*sm([s[mode][s["step"].index(t)] for t in steps], 3) for s in seeds.values()])
                for i, t in enumerate(steps):
                    rows.append(dict(dt=dt, demos=int(n), mode=mode, step=t, seeds=len(seeds),
                                     mean=round(float(ys[:, i].mean()), 2), lo=round(float(ys[:, i].min()), 2), hi=round(float(ys[:, i].max()), 2)))
    print(method, len(rows), {dt: {n: len(s) for n, s in sz.items()} for dt, sz in per_dt.items()})
    return rows

flow = {dt: json.load(open(f"plots/flow_bc_square_{dt}/curves.json")) for dt in ("ph", "mh")}
og = json.load(open("plots/ogpo_public/curves.json"))
flow_rows = bc_rows(flow, "flow")
ogbc_rows = bc_rows(og["bc"], "ogpo_bc")

rl_rows, hand_rows = [], []
for dt, seeds in og["ogpo"].items():
    per = {}
    for sd, d in seeds.items():
        step = np.array(d["step"]); on = step > d["bc_end"]
        hand_rows.append(dict(dt=dt, seed=int(sd), bc_end=d["bc_end"], ode=round(100*d["bc_end_log"]["ode"], 1), sde=round(100*d["bc_end_log"]["sde"], 1), final_step=int(step[-1]-d["bc_end"])))
        for mode in ("sde", "ode"):
            y = np.array(d[mode])
            x = np.r_[0, step[on]-d["bc_end"]]; v = np.r_[d["bc_end_log"][mode], sm(y[on], 5)]
            per.setdefault(mode, []).append(dict(zip(x.tolist(), (100*v).tolist())))
    for mode, curves in per.items():
        common = sorted(set.intersection(*[set(c) for c in curves]))
        for t in common:
            vals = [c[t] for c in curves]
            rl_rows.append(dict(dt=dt, mode=mode, step=int(t), seeds=len(vals), mean=round(float(np.mean(vals)), 2), lo=round(float(min(vals)), 2), hi=round(float(max(vals)), 2)))
    print(dt, "rl common steps", len(common), common[-1])
paper = [dict(dt="mh", step=s, success=v) for s, v in [(400_000, 85), (1_000_000, 95), (1_500_000, 97)]]
for name, rows in [("flow_bc", flow_rows), ("ogpo_bc", ogbc_rows), ("ogpo_rl", rl_rows), ("ogpo_handoff", hand_rows), ("paper_mh", paper)]:
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
