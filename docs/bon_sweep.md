# Best-of-N sweep over BC training (critic seeds 1 and 2)

Best-of-N with the no-memory MC critic on every flow-BC checkpoint from 50k to 500k steps (every 50k).

- **Policies:** square PH and MH × 10/20/50/100/200 demos × BC seeds 0/7/42 (30 policies).
- **N:** 8, 16, 32. **Critics:** MC critic seeds 1 and 2. Seed 0 (N = 4/8/16/32) is already in `/data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed0`.
- **Episodes:** 64 per point, from the same start states as the SDE eval at step T, so the BC curve in `plots/flow_bc_square_<ds>/curves.json` is the N = 1 baseline.
- **Jobs:** 30 array tasks, one per policy. Each runs both critic seeds × 10 checkpoints × 3 N = 60 rollouts, 3 at a time, on CPU (about 1.5 min per rollout, so 30–45 min per task).
- **Results:** `/data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed{1,2}/*.json`, plus one W&B run per (policy, critic seed) in `akankshasingal08/memory-critic`, group `bon_curve_mc_critic`.

## Run it (asingal2 account, Babel login node)

No GPU, no staging, nothing to copy. It uses your own `/data` files, the `maxlab-cpu` partition and your existing `wandb login`.

```bash
cd ~/memory_critic
git pull
DRY_RUN=1 scripts/bon_sweep/submit.sh   # optional: prints the 30 tasks and the sbatch line
scripts/bon_sweep/submit.sh             # submit
```

`maxlab_cpu_qos` runs 10 tasks at a time, so the whole sweep takes about 2–3 hours.

- Progress: `squeue --me`; job logs in `logs/bon_sweep/bon_sweep_<job>_<task>.out`; one log per rollout in `logs/bon_sweep/runs/`.
- Count results: `ls /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed{1,2} | wc -l` from a compute node (600 when done, 300 per seed).
- **Resume / retry.** A task skips every rollout whose JSON exists, and exits non-zero if any rollout failed. Resubmit failed tasks with `ARRAY=<ids> scripts/bon_sweep/submit.sh`.

All settings live in `scripts/bon_sweep/config.sh`, and each can be overridden from the environment:

```bash
ARRAY=0 scripts/bon_sweep/submit.sh                                  # one task first, as a test
PARTITION=preempt QOS=preempt_cpu_qos scripts/bon_sweep/submit.sh    # CPU on preempt instead (max 64 CPUs at once)
PARTITION=general QOS=normal GRES=gpu:1 scripts/bon_sweep/submit.sh  # on a GPU (needs the normal QOS)
CRITIC_SEEDS="0 1 2" NS="4 8 16 32" scripts/bon_sweep/submit.sh      # a different sweep
```

If W&B logging failed for a task (for example, no network), rerun it by hand on a compute node. It's safe to repeat, since a complete run isn't logged twice:

```bash
.venv/bin/python scripts/bon_sweep/log_wandb.py --out_dir /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed1 \
    --prefix square_ph_10demos_seed0 --ns 8 16 32 --critic_seed 1 --steps $(seq 50000 50000 500000)
```

## Plot

The same plots as for critic seed 0, with `--root` pointed at a critic seed's directory:

```bash
python scripts/plot_bon_vs_bc_steps.py --dataset ph --ns 8 16 32 \
    --root /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed1 --out plots/bon/bon_vs_bc_steps_ph_critic1
python scripts/plot_bon_curve.py --dataset mh --ns 8 16 32 \
    --root /data/user_data/asingal2/memory_critic/bon_curve/mc_decision_seed1 --out plots/bon/pibase_vs_bon_mh_critic1
```

## From another account or machine (only if needed)

1. On asingal2, copy the inputs (about 4.5 GB) into one folder from a compute node: `srun -p maxlab-cpu -q maxlab_cpu_qos -c 2 --mem 4G -t 2:00:00 scripts/bon_sweep/stage_inputs.sh /data/user_data/asingal2/bon_sweep_inputs`. Then give the other user read access (`setfacl -R -m u:<user>:rX,d:u:<user>:rX` on that folder), or add `TAR=1` and copy the tar.
2. On the other account, clone this repo and `../safe-robosuite` (branch `deadbeat-one-step-tracking`), then run `uv sync --locked --python 3.12 --extra cuda12 --extra robomimic`.
3. Set `INPUT_ROOT=<staged>`, `DATASET_ROOT=<staged>/robomimic_subsets`, `OUT_ROOT=<writable dir>`, plus `PARTITION`/`QOS` for that account. Set `WANDB_MODE=offline`, so no W&B key is needed there. Don't share your API key: it would sit in plain text in their account.
4. Copy the result JSONs back into `bon_curve/mc_decision_seed{1,2}`, then log them to W&B from asingal2 with `log_wandb.py` as above.
