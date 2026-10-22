#!/bin/bash
# Rebuild the Python env: uv venv (py3.10) + pinned deps + editable robosuite/robomimic submodules.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs  # sbatch --output target; slurm fails silently if missing
git submodule update --init third_party/robomimic third_party/robosuite
uv venv --python 3.10 .venv-robomimic  # .venv is the uv project env (ogpo, JAX)
source .venv-robomimic/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e third_party/robosuite -e third_party/robomimic
