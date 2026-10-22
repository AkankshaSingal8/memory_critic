#!/bin/bash
# End-to-end check: dataset info + plain BC training (2 tiny epochs) with a short low-dim env rollout.
set -euo pipefail
REPO=/home/asingal2/memory_critic
DATA_DIR=${ROBOMIMIC_DATA:-/data/user_data/$USER/robomimic}
OUT=${SMOKE_OUT:-/scratch/$USER/robomimic_smoke}
source "$REPO/.venv-robomimic/bin/activate"
export MUJOCO_GL=${MUJOCO_GL:-egl}
RM=$REPO/third_party/robomimic/robomimic
DS=${1:-$DATA_DIR/square/ph/low_dim_v15.hdf5}
python "$RM/scripts/get_dataset_info.py" --dataset "$DS" | head -40
mkdir -p "$OUT"
python - "$RM/exps/templates/bc.json" "$DS" "$OUT" > "$OUT.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["experiment"].update(name="smoke", validate=False, render_video=False)
c["experiment"]["save"]["enabled"] = False
c["experiment"]["logging"].update(log_wandb=False, log_tb=False)
c["experiment"]["epoch_every_n_steps"] = 20
c["experiment"]["rollout"].update(enabled=True, n=2, horizon=50, rate=2, warmstart=0)
c["train"].update(data=[{"path": sys.argv[2]}], output_dir=sys.argv[3], num_epochs=2, cuda=False, num_data_workers=0)
print(json.dumps(c, indent=2))
PY
rm -rf "$OUT/smoke"  # disposable smoke output; train.py prompts to overwrite otherwise
python "$RM/scripts/train.py" --config "$OUT.json"
