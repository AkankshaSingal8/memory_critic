# memory_critic

## Environments

Two separate envs:

| Env | Path | Used by |
|---|---|---|
| ogpo (JAX flow policies) | `.venv/` — uv project env from `pyproject.toml` + `uv.lock` | `ogpo/main.py`, `scripts/train_flow_bc.sbatch` |
| robomimic (PyTorch) | `.venv-robomimic/` | robomimic scripts, downloads, image rendering, smoke tests |

### ogpo env

`ogpo/`, `envs/`, `pyproject.toml`, `uv.lock` and `run_scripts/_forward_term.sh` are unmodified copies of
SafeVaDaR @ `2c4454c` (branch `safety-critic-pretrain`). robosuite resolves to the editable
`~/safe-robosuite` (shared with SafeVaDaR). Build:

```bash
uv sync --locked --python 3.12 --extra cuda12 --extra robomimic
```

Third-party packages are identical to `~/SafeVaDaR/.venv` (jax 0.9.0.1 cuda12, flax 0.12.4, mujoco 3.3.7).

### robomimic env

| Component | Version | Source |
|---|---|---|
| Python | 3.10 (uv venv at `.venv-robomimic/`) | |
| robomimic | 0.5.0 | submodule `third_party/robomimic` (editable) |
| robosuite | 1.5.1 | submodule `third_party/robosuite`, `v1.5.1` branch (editable) |
| mujoco | **3.2.6** (pinned) | newer mujoco (e.g. 3.16) breaks robosuite 1.5.1 at env reset |
| torch | 2.5.1 + CUDA 12.4 | |
| numpy / mink / opencv | 1.26.4 / 0.0.5 / 4.10 | pinned for compatibility with mujoco 3.2.6 |

Full pins are in `requirements.lock`. Rebuild with:

```bash
scripts/setup_env.sh
source .venv-robomimic/bin/activate
```

## Datasets (robomimic v1.5)

Stored at `/data/user_data/$USER/robomimic/<task>/<type>/` (compute nodes only):

| Task | Type | Files |
|---|---|---|
| square | ph, mh | `demo_v15.hdf5` (raw), `low_dim_v15.hdf5` |
| tool_hang | ph (only type available) | `demo_v15.hdf5` (raw), `low_dim_v15.hdf5` |

Download / verify (skips existing files, checks sizes against HF):

```bash
mkdir -p logs && sbatch scripts/download_robomimic.sbatch
```

Image datasets are not hosted for v1.5. Render them from `demo_v15.hdf5` on a GPU node with
`third_party/robomimic/robomimic/scripts/dataset_states_to_obs.py` (see
`robomimic/scripts/extract_obs_from_raw_datasets.sh` for the exact flags per task).

## Smoke test

```bash
srun -p maxlab-cpu --cpus-per-task=4 --mem=16G --time=00:30:00 scripts/smoke_test.sh [dataset.hdf5]
```

Runs `get_dataset_info.py`, then 2 tiny BC epochs plus a 2-episode low-dim rollout (defaults to square/ph).

## GPU

Verified on `preempt` (L40S, driver 610.43): `torch.cuda.is_available() == True`.
`preempt` accepts only `sbatch` (no `srun`) and needs `--qos=preempt_qos`; preempt jobs must checkpoint and auto-resume.

## Flow-policy BC (ogpo, `algo=qipm`, BC only)

```bash
.venv/bin/python scripts/make_demo_subsets.py --task square --dataset_type ph --sizes 10 20 50 100 200  # on a compute node
scripts/submit_flow_bc.sh                      # square ph, 10/20/50/100/200 demos, seed 0: train + CPU eval, chained
DRY_RUN=1 TASK=tool_hang DEMOS="50 200" SEEDS="0 7 42" scripts/submit_flow_bc.sh
```

- `scripts/train_flow_bc.sbatch`: port of SafeVaDaR `run_scripts/train_bc_full.sh` (header lists every change). Its
  resolved config (via `ogpo.utils.config_loader`) for tool_hang at 500k steps differs from the source only in
  project, save_dir and repeat_id. Logs `offline_agent/*` and `memory/*`; no in-training eval (`EVAL_EVERY=-1`).
- `scripts/eval_flow_bc.sbatch`: port of `run_scripts/eval_bc_run.sh`. `scripts/safety_pretrain/eval_bc_run.py` and
  `select_bc_checkpoints.py` are verbatim. Evaluates every `actor_<step>.pkl` on CPU into the same W&B run against
  `eval/bc_step` (`eval/*` ODE, `eval_sde/*` SDE), then writes `selected/target_{25,50,75,100}.pkl`;
  **`selected/target_100.pkl` is the best checkpoint** (highest 3-point-smoothed ODE success), `levels/*` in the W&B summary.
- `scripts/_flow_bc_names.sh`: the one place names are built. W&B project `memory-critic`, group
  `<task>_<type>_flow_bc_<N>demos`, run `flow_bc_<task>_<type>_<N>demos_h<H>_<RUN_TAG>_qipmseed<S>_r<repeat_id>`,
  run dir `/data/user_data/$USER/memory_critic/flow_bc/memory-critic/<group>/<env>/<run>`.
- Subsets: `/data/user_data/$USER/robomimic_subsets/<task>_<type>_<N>` (nested, `default_rng(0)` permutation;
  demo list in `data.attrs["subset_provenance"]`). `N_DEMOS=all` uses the full download.
- Rolling checkpoints: `/data/user_data/$USER/ogpo_checkpoints/qipm/` (repeat_id `memcritic_<RUN_TAG>`).

## Best-of-N sweep

Best-of-N with the MC critic over BC training (critic seeds 1-2, N = 8/16/32): `scripts/bon_sweep/submit.sh` from the asingal2 account submits all 30 CPU jobs. Details in [`docs/bon_sweep.md`](docs/bon_sweep.md).
