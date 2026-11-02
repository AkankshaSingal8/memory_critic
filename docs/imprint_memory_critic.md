# IMPRINT memory critic

A critic Q(s_t, m_t, a_{t:t+8}) where m_t is a frozen IMPRINT memory of the episode so far
(arXiv 2610.14406, `third_party/imprint`). This builds arm A5 of `docs/bon_memory_critic.md`, plus
its controls A3 (no memory) and A4 (clock). Background on IMPRINT is in `docs/imprint.md`.

## 1. Do we need to fine-tune IMPRINT on square? Yes.

No released IMPRINT encoder can read our data as is.

| | Released encoders (RMBench / RoboCasa) | Our square rollouts |
|---|---|---|
| Long-term visual stream | WAN2.2 VAE latents, 48-d, 3 views × 2×2 tokens per chunk | none (no images, no sim states saved) |
| Recent visual stream | DINOv3 patch features, 768-d | none |
| Action | 12-d (RoboCasa) | 7-d OSC |
| Proprio | 16-d | 9-d (eef_pos, eef_quat, gripper_qpos) |

The input projections, tokenizers and decoder heads have the wrong shapes for our data, and the
visual streams don't exist for it. So Stage 1 has to run on square. The real question is how to
initialise it, and we run both options:

- **scratch**: IMPRINT's architecture and Stage-1 recipe, trained only on our data.
- **ft**: fine-tune the released RoboCasa recurrent Stage-1 encoder
  (`Amazon-FAR/imprint_models`, `simulation/robocasa/recurrent/stage1/stage1_step160000.npz`).
  Every tensor whose shape matches is copied: the gated-delta core, the 256 learned queries, the
  read-out, recent fusion and the decoder trunk. In a shape check against a RoboCasa-shaped
  checkpoint, that was 98.6% of encoder parameters and 99.2% of decoder parameters. Re-initialised
  tensors: `proj_cosmo`, `proj_recent`, `tok_action/l1`, `tok_proprio/l1`, `event_slot` (20 → 9
  events per chunk), `patch_emb`, and the align/action/proprio heads. Then everything trains.
  The run logs the exact list (`init_report` in the W&B config and in `stage1_summary.json`).

Which init is better is an empirical question, so both feed the critic. Pick on held-out Stage-1
reconstruction (`val/r2_*`) and on the critic's test metrics.

## 2. How Stage 1 runs on square (`imprint_bridge/`)

This is the low-dim shortcut from `docs/imprint.md` §4.2: the 14-d `object` state stands in for vision.

| IMPRINT stream | Square input |
|---|---|
| long-term visual, one token per finalized chunk of 4 steps | concat of the 4 frames' object state (56-d) |
| recent visual, frames after the last finalized chunk | object state per frame (14-d) |
| action stream | causal executed action a_{f-1}, zero at f = 0 (7-d) |
| proprio stream | obs[:9] |

All streams are z-scored with train-split statistics (`norm.npz`).

Stage-1 targets are the paper's composite loss with the paper's weights. At 16 random past steps,
the decoder reconstructs the object state (align, 1.0), the causal action (0.5) and the proprio (0.5).
A progress scalar for the current step (0.5) uses the mean of the cumulative subtask flags
(grasped, lifted near the peg, success). Those flags are sticky, so progress depends on the history,
not only on s_t.

- **Model, loss, optimizer, checkpoint format:** IMPRINT's own (`make_stage1_modules`,
  `composite_alignment_loss`, `build_optimizer`, `_save_checkpoint`). The config is
  `imprint_bridge/square_lowdim_gated_delta.yaml`: the recurrent `gated_delta_hybrid` encoder, with
  every width of the released RoboCasa config.
- **Only the dataloader is ours** (`imprint_bridge/data.py`). IMPRINT's loader reads a WAN/DINOv3
  feature store that we don't have.
- **Data:** the 30 flow-BC rollout files (`docs/collect_flow_bc_rollouts.md`). Episodes < 288 train
  and 288–319 validate. Test episodes (≥ 320) are never seen by Stage 1, the cache fit or the critic.
- **Defaults:** 40k steps, batch 64, lr 1e-4 (cosine, 1k warmup), bf16 compute with chunkwise-WY scan,
  checkpoint and validation every 2k steps. It resumes automatically after preemption. The best
  validation checkpoint is `stage1_stepbest.npz`.

## 3. The memory critic

`imprint_bridge/cache_memory.py` freezes the encoder. At every decision point (chunk_pos == 0) it
encodes the history 0..t and stores two 1024-d pools:

- `mean`: the mean of the 256 memory tokens.
- `progress`: the Stage-1 progress query's read-out, a trained attention pool.

Candidate action chunks never go through the encoder.

The critic is `scripts/train_mc_critic.py`, the no-memory critic itself, with `--mem_dir`. **The
architecture does not change**: the same `Value` ensemble (10 × MLP 4×512, LayerNorm), optimizer,
batch, steps, Monte Carlo targets, split and metrics. The memory is appended to the obs vector as 64
whitened PCA components fit on the train split (`--mem_pca`). So only the first layer's input width
grows, from 79 to 143. `critic_best.pkl` stores the memory transform
(`input_layout`), so a best-of-N consumer can rebuild the inputs from a memory cache. The clock control appends t/400 instead (`--add_timestep`).

Arms, with critic seeds 0/1/2 each:

| Arm | Input | Purpose |
|---|---|---|
| `nomem` | obs, a | A3 baseline (rerun here, same code path) |
| `clock` | obs, t/400, a | is memory just a clock? |
| `mem_scratch_mean`, `mem_scratch_progress` | obs, m_t, a | memory critic, Stage 1 from scratch |
| `mem_ft_mean`, `mem_ft_progress` | obs, m_t, a | memory critic, Stage 1 fine-tuned from the release |

## 4. Run it

### Setup (once, on the submitting account)

```bash
git clone <this repo> ~/memory_critic && cd ~/memory_critic
git submodule update --init third_party/imprint
# critic env (.venv); needs ../safe-robosuite on branch deadbeat-one-step-tracking
uv sync --locked --python 3.12 --extra cuda12 --extra robomimic
# IMPRINT env (.venv-imprint), see docs/imprint.md §0
cd third_party/imprint && RAYON_NUM_THREADS=2 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 \
  UV_CONCURRENT_INSTALLS=2 UV_PROJECT_ENVIRONMENT=$PWD/../../.venv-imprint uv sync --python 3.11 --frozen && cd -
wandb login        # with a key that can write to the akankshasingal08 entity
```

**Rollouts.** By default they are read from asingal2's `/data`. From another account (for example
pchellap), first grant that account read access from asingal2:

```bash
setfacl -R -m u:pchellap:rX,d:u:pchellap:rX /data/user_data/asingal2/memory_critic/rollouts
setfacl -m u:pchellap:X /data/user_data/asingal2 /data/user_data/asingal2/memory_critic
```

Alternatively, copy the rollouts and set `DATA_DIR`.

### Submit

```bash
DRY_RUN=1 scripts/memory_critic/submit.sh   # print the plan
scripts/memory_critic/submit.sh
```

The jobs never use more than `MAX_GPUS=4` GPUs at once:

- **Stage 1 (scratch, ft):** 2 GPUs. The baseline critics run alongside on the other 2.
- **Cache:** after Stage 1.
- **Memory critics:** after the cache and the baselines, 4 at a time.

Every W&B run goes to `akankshasingal08/memory-critic`, in the groups `imprint_stage1` and
`critic_mc_imprint`. Outputs are under `/data/user_data/$USER/memory_critic/imprint/`. All settings
(partition, QoS, GPU type, steps, arms) are in `scripts/memory_critic/config.sh`, and each can be
overridden from the environment. `SKIP_STAGE1=1` restarts from the cache. A task that finished is
skipped, and Stage 1 resumes from its last checkpoint.

### Read the result

```bash
python -m imprint_bridge.compare_critics /data/user_data/$USER/memory_critic/imprint/critics
python -m imprint_bridge.compare_critics /data/user_data/$USER/memory_critic/imprint/critics --by n_demos
```

The decision rule (`docs/bon_memory_critic.md` §5.4): memory earns its place only if the `mem_*`
arms beat both `nomem` and `clock` by more than the seed spread, first on test MSE and success AUC,
then in best-of-N.

## 5. Checks done, and not done

**Checked locally on CPU,** with synthetic rollouts in the real file format
(`python -m imprint_bridge.make_fake_rollouts <dir>`) and a shrunken model:

- Stage-1 training and validation.
- Preemption resume.
- Fine-tune init from a checkpoint with RoboCasa shapes.
- The bf16 + chunkwise-WY path.
- The cache.
- All critic arms through `run.sbatch`.
- Identical critic parameter trees apart from the first layer's input width.

**Not run yet:**

- The full-size model on GPU, so Stage-1 time per step is unknown. Look at `sec_per_step` in W&B,
  and lower `STAGE1_STEPS` if 40k steps won't fit in `STAGE1_TIME`.
- Loading the real released checkpoint.
- Two speedups were left out on purpose; add them if the GPU numbers call for it:
  - **Length-bucketed Stage-1 batches.** Every history is padded to 400 steps today. Bucketing could
    save about 1.5–2× encoder compute, if the scan cost scales with padded length.
  - **A single recurrent pass per episode in the cache.** It would replace about 25 prefix re-encodes
    per episode, but needs per-chunk read-outs from the encoder.
- Online best-of-N with the memory critic, which needs a per-env memory state at eval (§5.3 of
  `docs/bon_memory_critic.md`).
