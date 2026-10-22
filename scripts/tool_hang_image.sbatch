#!/bin/bash
#SBATCH --job-name=tool-hang-image
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
# Renders tool_hang/ph image_v15.hdf5 from raw (robomimic recipe: sideview + eye_in_hand, 240x240),
# then runs a tiny image-BC smoke test on GPU. Idempotent on requeue: extraction writes a .partial
# file and is skipped once image_v15.hdf5 exists.
set -euo pipefail
REPO=/home/asingal2/memory_critic
D=${ROBOMIMIC_DATA:-/data/user_data/$USER/robomimic}/tool_hang/ph
OUT=/scratch/$USER/robomimic_smoke_image
source "$REPO/.venv-robomimic/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
RM=$REPO/third_party/robomimic/robomimic
echo "== $(date) start on $(hostname), restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name --format=csv,noheader

if [ ! -f "$D/image_v15.hdf5" ]; then
  python "$RM/scripts/dataset_states_to_obs.py" --done_mode 2 --dataset "$D/demo_v15.hdf5" \
    --output_name image_v15.hdf5.partial \
    --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240
  mv "$D/image_v15.hdf5.partial" "$D/image_v15.hdf5"
fi
ls -lh "$D/image_v15.hdf5"
python "$RM/scripts/get_dataset_info.py" --dataset "$D/image_v15.hdf5" | grep -E "total|obs key|num_demos" | head -20

mkdir -p "$OUT"; rm -rf "$OUT/smoke_image"
python "$REPO/scripts/make_image_smoke_config.py" "$D/image_v15.hdf5" "$OUT" > "$OUT.json"
python "$RM/scripts/train.py" --config "$OUT.json"
echo "== $(date) done"
