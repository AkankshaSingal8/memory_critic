# Best-of-N sweep over BC training (critic seeds 1 and 2)

This sweep runs best-of-N with the no-memory MC critic on every flow-BC checkpoint from 50k to 500k
steps (every 50k).

- **Policies:** square PH and MH × 10/20/50/100/200 demos × BC seeds 0/7/42.
- **N:** 8, 16, 32.
- **Critics:** MC critic seeds 1 and 2. Seed 0 was already run with `scripts/bon_curve.sbatch` (N = 4/8/16/32). Results are in `/data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed0`.
- **Episodes:** 64 per point, from seeds T..T+63. These are the same start states as the SDE eval at step T, so the BC curve in `plots/flow_bc_square_<ds>/curves.json` is the N = 1 baseline.
- **Size:** 60 array tasks, one per (dataset, size, BC seed, critic seed). Each task runs 10 checkpoints × 3 values of N = 30 rollouts, 3 at a time on one GPU.
- **Logging:** each task writes JSON results and logs one W&B run to `akankshasingal08/memory-critic`, group `bon_curve_mc_critic`.

| File | What it does |
|---|---|
| `scripts/bon_sweep/config.sh` | Every setting (what to run, input/output paths, W&B, Slurm); each can be overridden from the environment |
| `scripts/bon_sweep/submit.sh` | Submits the array (`DRY_RUN=1` prints, `CHECK=1` runs `sbatch --test-only`) |
| `scripts/bon_sweep/bon_sweep.sbatch` | One task: checks inputs, runs the rollouts, logs to W&B |
| `scripts/bon_sweep/stage_inputs.sh` | Copies the inputs into one directory, for another account or machine |
| `scripts/bon_sweep/log_wandb.py` | Logs one task's results to W&B (the job calls it; it can also be run by hand) |
| `scripts/bon_sweep/fetch_wandb.py` | Pulls results from W&B into the layout the plot scripts read |

## 1. Stage the inputs (on the asingal2 account)

The sweep reads, for each policy:
- the run's `flags.json` and `actor_<T>.pkl` at the 10 steps,
- the critic `critic_best.pkl` for seeds 1 and 2,
- the robomimic `low_dim_v15.hdf5` for each dataset subset. The env reads only its metadata.

`/data` is only mounted on compute nodes, so copy them from one:

```bash
cd ~/memory_critic
srun -p maxlab-cpu -q maxlab_cpu_qos -c 2 --mem 4G -t 2:00:00 \
    scripts/bon_sweep/stage_inputs.sh /data/user_data/asingal2/bon_sweep_inputs
# add TAR=1 in front to also write bon_sweep_inputs.tar for scp
```

Then make that directory reachable from the account that will submit. Either:
- give that account read access, e.g. `setfacl -R -m u:<their_user>:rX /data/user_data/asingal2/bon_sweep_inputs` plus `setfacl -m u:<their_user>:X /data/user_data/asingal2`, if ACLs work on that filesystem, or
- copy it across: `scp bon_sweep_inputs.tar <host>:<dir>/`, then `tar -xf` it there.

## 2. Set up the repo on the other server

```bash
git clone https://github.com/AkankshaSingal8/memory_critic.git
# robosuite comes from the sibling ../safe-robosuite (pyproject.toml [tool.uv.sources]), the branch this repo was run with:
git clone -b deadbeat-one-step-tracking https://github.com/cpranavvignesh/safe-robosuite.git
cd memory_critic
uv sync --locked --python 3.12 --extra cuda12 --extra robomimic
.venv/bin/wandb login          # paste YOUR API key (wandb.ai/authorize), so the runs land in your account
```

## 3. Submit

Point the paths at the staged inputs and at an output directory that account can write:

```bash
export INPUT_ROOT=<dir>/bon_sweep_inputs
export DATASET_ROOT=<dir>/bon_sweep_inputs/robomimic_subsets
export OUT_ROOT=<somewhere writable>/memory_critic/bon_curve

DRY_RUN=1 scripts/bon_sweep/submit.sh      # task table and the sbatch line, nothing queued
TASK_ID=0 PLAN_ONLY=1 REPO=$PWD bash scripts/bon_sweep/bon_sweep.sbatch   # checks task 0's inputs exist (run where they are mounted)
CHECK=1 scripts/bon_sweep/submit.sh        # sbatch --test-only: is the request valid, when would it start
scripts/bon_sweep/submit.sh                # submit all 60 tasks
```

Defaults are Babel `general`: `--qos=normal --gres=gpu:1 -c 24 --mem 64G -t 1-00:00:00`, 3 rollouts at a time. Override any of them:

```bash
PARTITION=preempt QOS=preempt_qos scripts/bon_sweep/submit.sh           # Babel preempt instead
GRES=gpu:L40S:1 EXTRA_SBATCH="--exclude=babel-x-1" scripts/bon_sweep/submit.sh
ARRAY=0-1 scripts/bon_sweep/submit.sh                                    # try two tasks first
CRITIC_SEEDS="0 1 2" NS="4 8 16 32" scripts/bon_sweep/submit.sh          # a different sweep (180 tasks)
```

Logs go to `logs/bon_sweep/bon_sweep_<job>_<task>.out`. Each rollout also has its own log in `logs/bon_sweep/runs/`.

**Resuming.** A task skips any (T, N) whose JSON already exists, so resubmitting the same array index finishes only what's missing. A task exits non-zero if any rollout failed. Resubmit failed tasks with `ARRAY=<ids> scripts/bon_sweep/submit.sh`.

## 4. Get the results back here

The results are small JSONs under `$OUT_ROOT/mc_decision_seed{1,2}/`. Two ways to bring them back:

```bash
# a. from W&B, on this account (no file transfer needed)
.venv/bin/python scripts/bon_sweep/fetch_wandb.py --out_root /data/user_data/asingal2/memory_critic/bon_curve
# b. copy the files
rsync -av <host>:<OUT_ROOT>/ /data/user_data/asingal2/memory_critic/bon_curve/
```

A task logs to W&B once all its rollouts are done, including when some have failed. If that step failed (for example, no network), run it again by hand. It's safe to repeat: a complete W&B run isn't logged twice.

```bash
.venv/bin/python scripts/bon_sweep/log_wandb.py --out_dir $OUT_ROOT/mc_decision_seed1 \
    --prefix square_ph_10demos_seed0 --ns 8 16 32 --critic_seed 1 \
    --steps 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000
```

## 5. Plot

These are the same plots as for critic seed 0. Point `--root` at a critic seed's directory:

```bash
python scripts/plot_bon_vs_bc_steps.py --dataset ph --ns 8 16 32 \
    --root /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed1 --out plots/bon/bon_vs_bc_steps_ph_critic1
python scripts/plot_bon_curve.py --dataset mh --ns 8 16 32 \
    --root /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed1 --out plots/bon/pibase_vs_bon_mh_critic1
```
