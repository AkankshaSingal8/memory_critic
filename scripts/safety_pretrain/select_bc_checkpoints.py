"""Pick, per seed, the BC checkpoint whose training-time success is closest to each
target success level.

Success comes from <run_dir>/eval.csv, smoothed with a centred 3-point moving average
over consecutive eval steps (endpoints average the neighbours that exist). Candidates
are eval steps with a params_<step>.pkl in the run dir and step > 0; ties go to the
earliest step. Writes a selection.json for the evaluation step.

    uv run --locked python scripts/safety_pretrain/select_bc_checkpoints.py \
        --run_dir_pattern /data/.../bc_seedSEED --seeds 0 7 42 \
        --targets 0.25 0.5 0.75 1.0 --out selection.json
"""
import argparse
import csv
import json
import os
import sys

WARN_GAP = 0.10


def read_eval_csv(run_dir):
    """{step: success} from eval.csv; a requeue can repeat a step, keep the last.
    Rows that fail to parse (torn writes) are skipped."""
    out = {}
    with open(os.path.join(run_dir, "eval.csv"), newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[int(float(row["step"]))] = float(row["success"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def smooth(raw):
    """Centred 3-point moving average over steps sorted ascending."""
    steps = sorted(raw)
    out = {}
    for i, s in enumerate(steps):
        window = [raw[t] for t in steps[max(0, i - 1):i + 2]]
        out[s] = sum(window) / len(window)
    return out


def select_levels(raw, candidates, targets):
    """Per target, the candidate step whose smoothed success is closest (ties: earliest).

    raw: {step: success} over ALL evaluated steps (the smoothing window runs over
    consecutive evaluated steps); candidates: the steps that may be selected.
    Returns [{"target", "step", "smoothed_success", "raw_success"}] in target order.
    """
    smoothed = smooth(raw)
    cands = sorted(s for s in candidates if s in smoothed and s > 0)
    if not cands:
        raise ValueError("no candidate step with an eval result")
    out = []
    for t in targets:
        step = min(cands, key=lambda s: (abs(smoothed[s] - t), s))
        out.append({"target": t, "step": step, "smoothed_success": smoothed[step],
                    "raw_success": raw[step]})
    return out


def select_for_seed(run_dir, targets):
    raw = read_eval_csv(run_dir)
    cands = [s for s in raw if os.path.exists(os.path.join(run_dir, f"params_{s}.pkl"))]
    if not [s for s in cands if s > 0]:
        raise ValueError(f"no eval step with a params file in {run_dir}")
    out = select_levels(raw, cands, targets)
    for p in out:
        p["actor"] = os.path.join(run_dir, f"params_{p['step']}.pkl")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir_pattern", required=True,
                    help="run dir with the literal SEED replaced per seed")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--targets", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    selections = []
    for seed in args.seeds:
        run_dir = args.run_dir_pattern.replace("SEED", str(seed))
        try:
            picks = select_for_seed(run_dir, args.targets)
        except (OSError, ValueError) as e:
            sys.exit(f"seed {seed}: {e}")
        for p in picks:
            selections.append({"seed": seed, **p})
            print(f"seed {seed} target {p['target']:.2f} -> step {p['step']} "
                  f"smoothed {p['smoothed_success']:.4f} raw {p['raw_success']:.4f}")
            if abs(p["smoothed_success"] - p["target"]) > WARN_GAP:
                print(f"WARNING: seed {seed} target {p['target']:.2f}: closest "
                      f"checkpoint (step {p['step']}) has smoothed success "
                      f"{p['smoothed_success']:.4f}, more than {WARN_GAP} away")
        by_step = {}
        for p in picks:
            by_step.setdefault(p["step"], []).append(p["target"])
        for step, ts in by_step.items():
            if len(ts) > 1:
                if len(ts) == 2:
                    names = f"{ts[0]} and {ts[1]}"
                else:
                    names = ", ".join(str(t) for t in ts)
                print(f"WARNING: seed {seed} targets {names} share step {step}")

    with open(args.out, "w") as f:
        json.dump({"targets": args.targets, "selections": selections}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
