#!/bin/bash
#SBATCH --job-name=robomimic-dl
#SBATCH --partition=maxlab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Downloads robomimic v1.5 datasets (raw + low_dim) for square (ph, mh) and tool_hang (ph).
# Existing files are kept; every file is size-checked against HF.
# Image hdf5s are not hosted for v1.5; render them from raw with dataset_states_to_obs.py.
set -euo pipefail
REPO=/home/asingal2/memory_critic
DATA_DIR=${ROBOMIMIC_DATA:-/data/user_data/$USER/robomimic}
source "$REPO/.venv-robomimic/bin/activate"
export TMPDIR=/scratch/$USER/robomimic_dl_tmp; mkdir -p "$TMPDIR" "$DATA_DIR"
python "$REPO/scripts/download_robomimic.py" --download_dir "$DATA_DIR" --spec \
  square:ph:raw square:ph:low_dim square:mh:raw square:mh:low_dim \
  tool_hang:ph:raw tool_hang:ph:low_dim
