#!/bin/bash
#SBATCH --job-name=plot_flow_bc
#SBATCH --partition=maxlab-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/asingal2/memory_critic/logs/%x_%j.out
#SBATCH --error=/home/asingal2/memory_critic/logs/%x_%j.out
# W&B check + success-vs-steps plots for the flow-BC sweep (scripts/plot_flow_bc_curves.py).
# CPU only, one core: it only reads W&B and the cpu_eval JSONs on /data.
set -euo pipefail
cd /home/asingal2/memory_critic
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python scripts/plot_flow_bc_curves.py "$@"
