# IMPRINT in memory_critic

This doc explains what [amazon-far/imprint](https://github.com/amazon-far/imprint) is and how its
memory module could connect to our robomimic flow-BC policy (`ogpo/`, `algo=qipm`,
`training.bc_only=true`). No IMPRINT code is wired into `ogpo/` yet. This is a plan.

Paths below are relative to the repo root. IMPRINT paths are under `third_party/imprint/`,
shortened to `imprint/` in the text.

---

## 0. Setup status

| Item | State |
|---|---|
| Source | Git submodule `third_party/imprint`, pinned at `e89a16e` (the only commit, "Publish simulation-only IMPRINT Pi0.5 release"). Staged in `.gitmodules`, **not committed**. |
| Env | `.venv-imprint/` (7.7 GB): Python 3.11, jax 0.5.3, flax 0.10.2, torch 2.7.1+cu128, built from `third_party/imprint/uv.lock` (see §0.1). Verified on 2026-10-23: the smoke check below gave **26 passed** on `maxlab-cpu` (job 10902036). On the login node, pytest dumps core (thread limits), so run it on a compute node. |
| Models / data | Not downloaded. See §0.2. |

### 0.1 Why a third env

IMPRINT cannot share either existing env:

| | `.venv` (ogpo) | `.venv-robomimic` | `.venv-imprint` |
|---|---|---|---|
| Python | 3.12 | 3.10 | 3.11 |
| JAX / flax | 0.9.0.1 / 0.12.4 | none | **0.5.3 / 0.10.2** |
| torch | none | 2.5.1+cu124 | **2.7.1+cu128** |

The JAX/flax gap between `.venv` and `.venv-imprint` is the main constraint on the design below.
IMPRINT's modules use `flax.nnx` (in flax 0.10.2). ogpo's networks use `flax.linen` (in flax
0.12.4). Neither package can be imported into the other's env without a port.

Build (re-run from the submodule; the login node's thread limit needs the caps):

```bash
cd third_party/imprint
RAYON_NUM_THREADS=2 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2 \
UV_PROJECT_ENVIRONMENT=$PWD/../../.venv-imprint uv sync --python 3.11 --frozen
```

Smoke check (CPU, no models needed):

```bash
cd third_party/imprint
srun -p maxlab-cpu --cpus-per-task=4 --mem=16G --time=00:30:00 env JAX_PLATFORMS=cpu \
  PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src \
  ../../.venv-imprint/bin/python -m pytest -q -p no:cacheprovider \
  src/mme_vla_suite/tests/cosmo/test_cosmo_buffer.py \
  src/mme_vla_suite/tests/cosmo/test_serving.py \
  src/mme_vla_suite/tests/cosmo/test_gated_delta_mem.py
```

### 0.2 External assets (download only when needed)

| Asset | Needed for | Source |
|---|---|---|
| WAN2.2 VAE (`vae/` only, ~2.8 GB) | long-term visual stream (offline latents) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff73…` |
| DINOv3 ViT-B/16 (gated on HF) | recent-frame stream and Stage-1 align target | `facebook/dinov3-vitb16-pretrain-lvd1689m` @ `5931719…` |
| Pi0.5 base params | only IMPRINT's own Stage 2 (the VLA). **We don't need it.** | OpenPI |
| IMPRINT checkpoints/datasets | only to reproduce the paper (RMBench/RoboCasa) | `Amazon-FAR/imprint_models`, `Amazon-FAR/imprint_datasets` (hundreds of GB) |

Put large files under `/data/user_data/$USER/imprint/`, not in `$HOME` (100 GB quota).

---

## 1. What IMPRINT is

IMPRINT is a **policy-agnostic long-horizon memory encoder**. It compresses a whole episode
history into a fixed set of **256 tokens × 1024 dims**, and a policy reads those tokens. The
public release attaches it to Pi0.5 (default branch), Pi0 (`pi0` branch), and MolmoAct2
(`molmoact2` branch). The code lives in the vendored package `imprint/src/mme_vla_suite/`, next
to a pinned OpenPI fork in `imprint/src/openpi/`.

### 1.1 The one interface that matters

```
memory_encoder(
    cosmo_emb,    # long-term visual: finalized WAN2.2 latent chunks   (B, Nc*tokens, 48)
    recent_emb,   # recent visual: DINOv3 tokens for not-yet-finalized frames (B, Nr, 768)
    action_emb,   # causal executed actions, action_causal[t] = action[t-1]  (B, Ns, A)
    proprio_emb,  # proprio per step                                     (B, Ns, P)
    cosmo_mask, lang_emb=None, recent_mask, action_mask, proprio_mask,
) -> tokens (B, 256, 1024), mask (B, 256), stats
```

- `lang_emb=None` is an explicit identity path. Language enters only through FiLM on the
  learned queries, so with no language there is no FiLM (`imprint/src/mme_vla_suite/models/representation/cosmo_mem.py:50`).
  We don't need language for square or tool_hang.
- Every stream has its own validity mask, so variable-length history is padded and masked.
- History is **strictly causal**: the action stream at step t is the action executed at t-1, with
  zero at t=0 (`imprint/docs/TRAINING.md`, "Data Contract").

### 1.2 Two interchangeable encoders

Built by `create_cosmo_memory_encoder` in
`imprint/src/mme_vla_suite/models/representation/memory_factory.py`. It is selected by
`cosmo_bottleneck_memory.memory_encoder_type` in the history YAML.

| Encoder | Class | How it works | Online use |
|---|---|---|---|
| Full attention (`perceiver`) | `CosmoBottleneckMemory` (`cosmo_mem.py:50`) | 256 learned queries cross-attend over the concatenation of all four streams, then self-attend (`num_cross_blocks` blocks) | Re-runs over the whole history at every query |
| Recurrent (`gated_delta_hybrid`) | `GatedDeltaBottleneckMemory` (`gated_delta_mem.py:578`) | A 2-layer gated DeltaNet compresses the chronological sequence of [action/proprio events, finalized visual chunk] into a fixed matrix state. Language-conditioned queries read that state, then cross-attend the exact recent window. | `init_incremental_state` / `update_incremental` / `read_incremental` (`gated_delta_mem.py:996-1076`). State is O(1) per chunk, with no rescan. |

The recurrent encoder is the natural fit for an RL/BC loop. Its state is a small per-env carry
that can be updated once per finalized chunk.

### 1.3 Visual streams (the expensive part)

- **Long-term:** WAN2.2 VAE latents (`models/representation/wan_frontend.py`). The VAE is
  temporally causal with temporal factor 4 and 16× spatial downsampling, giving 48-dim latents.
  Frames are first subsampled by `downsample=2`, so one chunk covers 8 env steps. Latents are
  pooled to `pool_hw × pool_hw` tokens per view, and a chunk is consolidated only once it is
  *finalized*. (The README says "Cosmos" in places. The paper configs use `variant: Wan2.2-TI2V-5B`;
  `cosmo_frontend.py` is the older Cosmos tokenizer.)
- **Recent:** DINOv3 ViT-B/16 features, pooled to 64 tokens per view at width 768, for the frames
  since the last finalized chunk.
- **Training uses precomputed features only.** `latent_source: offline` reads
  `features/episode_<id>/latents_v{0..7}.npy`. Those are eight pre-rendered, causally valid,
  frame-jittered subsamplings of each episode. At every sample, one variant whose first frame is
  ≤ t is drawn at random (`training/dataset.py:440-500`). **The script that generates the eight
  variants is not in the public release.** Only consumers exist (`assemble_var8_base.py`,
  `precompute_mem.py`, `dataset.py`).
- **Inference uses an online buffer.** `CosmoMemoryBuffer` (`shared/cosmo_buffer.py:64`) appends
  frames, actions, and proprio every step and encodes newly finalized chunks. Its tested
  invariant is that it reproduces exactly the offline store's prefix at step t
  (`tests/cosmo/test_wan_store.py`).

### 1.4 The two-stage recipe

**Stage 1: bootstrap memory (no policy).** `imprint/scripts/reproduce/train_stage1.sh` calls
`mme_vla_suite.scripts.pretrain_bottleneck`, which trains the encoder plus a Perceiver-IO
reconstruction decoder (`cosmo_decoder.py`). The decoder queries the 256 tokens at 16 random
*past* steps and reconstructs (`training/pretrain_loss.py:48`):

| Target | Weight |
|---|---|
| DINOv3 patch features at that step (per-token LayerNorm, then MSE) | 1.0 |
| causal action at that step | 0.5 |
| proprio at that step | 0.5 |
| normalized progress, a scalar "reward" for the current step | 0.5 |

Paper settings: AdamW, lr 1e-4, wd 1e-4, 1k warmup, cosine, batch 64, 160k steps, 3 views.
The decoder is thrown away after Stage 1. A separate ablation config can retrain a fresh auxiliary
decoder during Stage 2 (`mme_wan_stage2_pi05_gated_delta_memft_aux`, model `aux_weight=0.25`).
The four `imprint_*_memft` paper configs leave the model's `aux_weight` at 0, so they train the
action loss only. Their YAMLs still say `aux_weight: 0.25`, which only makes the *dataset* load
aux targets.

**Stage 2: adapt the policy.** `imprint/scripts/reproduce/train_stage2.sh` calls
`scripts/train_mme.py`. It loads Pi0.5 and the Stage-1 encoder, discards the decoder, and trains
the flow-matching action loss. The encoder is **fine-tuned**: the paper's "memft" configs freeze
only the image tower. A frozen-encoder recipe also exists
(`get_stage2_frozen_bottleneck_freeze_filter`, `history_pi0.py:353`), as does
`dataset_builder/precompute_mem.py`. That script caches frozen `(256, 1024)` fp16 tokens per
(episode, step, variant), so Stage 2 never runs the encoder.

**How memory enters the policy (`integration_type: modulation`).** In every action-expert
transformer block, just before the FFN, the action tokens cross-attend the memory tokens
(`MemoryAttention`, 4 query heads, 1 KV head, head_dim 256, RoPE; hardcoded for width 1024). The result produces a scale and shift
through `MemoryRMSNorm` (`imprint/src/mme_vla_suite/models/integration/history_gemma.py:45,178`).
The VLM prefix never sees memory; only the action denoiser does. This is FiLM by attention, and
§4.3 maps it onto our MLP actor.

### 1.5 Compute reality

Paper Stage 2 runs Pi0.5 (~3B) with FSDP over **4×80 GB GPUs**. We won't do that on L40S, and we
don't need to: **our "Stage 2" is the 4×512 flow MLP.**

Stage 1 alone is small: a 2-layer gated delta with 256×1024 tokens and a 2-block decoder. Our
episodes are short. Square ph is roughly 150 steps, about 19 chunks of 8 steps. Tool_hang is
roughly 500–700 steps, about 60–90 chunks. The paper uses caps of 1244–1776 steps and buckets of
up to 4096 latent tokens. Check the real lengths with `get_dataset_info.py`. Stage 1 should fit
on one L40S with smaller caps and bucket lists.

---

## 2. Our flow-BC policy, as IMPRINT sees it

| | ogpo flow BC (today) | IMPRINT policy |
|---|---|---|
| Framework | JAX + `flax.linen` | JAX + `flax.nnx` (+ linen via `nnx_bridge`) |
| Obs | low-dim only: `eef_pos(3) + eef_quat(4) + gripper_qpos(2) + object(14 square / 44 tool_hang)` | 3 RGB views + proprio + language |
| Actor | `ActorVectorField`, 4×512 MLP, input = concat(obs, noisy action chunk, t) (`ogpo/networks/actors.py:102`) | Pi0.5 action expert |
| Chunk | `horizon_length=8` × 7-dim OSC action | 20 (RoboCasa) / 50 (RMBench) |
| Flow steps | 10 | 10 |
| History | **none.** `observations` is a single frame | 256 memory tokens |

Two facts matter for the integration:

1. **ogpo's `Dataset` is flat transitions** with `terminals`/`masks`. It has no episode id or step
   index per transition, and memory needs both. The `history_length` argument of
   `Dataset.sample_sequence` (`ogpo/utils/datasets.py:137,264`) returns *future* chunks starting
   at t…t+n-1 for the temporal-bias safety arm. It is **not** past history and can't be reused
   for memory.
2. **Eval is vectorized and chunked.** `evaluate_qipm_parallel` runs 32 envs, each with its own
   `action_queues[i]`, and queries the actor only when a queue empties, which is every 8 steps
   (`ogpo/utils/evaluation_qipm.py:304-375`). A memory buffer must be **per env**, **reset at
   `envs.reset`**, and **fed every env step** (frame, executed action, proprio), not only at
   decision points.

---

## 3. Step 1: flow BC without memory (the baseline)

The no-memory baseline is our existing pipeline, unchanged:

```bash
scripts/submit_flow_bc.sh       # square ph, demos 10/20/50/100/200, seed 0; train + CPU eval chained
```

(`scripts/train_flow_bc.sbatch`: `--algo=qipm --training.bc_only=true`, 500k steps, horizon 8,
flow_steps 10, actor lr 3e-4 cosine, batch 256, W&B project `memory-critic`.) Those runs are
already the "no memory" arm. Keep every flag identical when the memory arm is added, so the only
difference is the memory input.

To make the ablation clean later, the memory arm should be a **strict superset** of this run:
the same dataset, normalization, actor, and eval seeds. The obs vector gets extra memory features
appended, and nothing else changes.

For reference only: IMPRINT's own no-memory baseline is the `pi05_baseline` config
(`training/config.py:771`, `use_history=False`). That is a Pi0.5 VLA without memory, not
comparable to our flow MLP, so we won't run it.

Before any memory run, render images. Memory's visual streams need RGB, and our flow BC uses
`low_dim_v15.hdf5`. Reuse the recipe in `scripts/tool_hang_image.sbatch` (robomimic
`dataset_states_to_obs.py` on `demo_v15.hdf5`), and choose the cameras (see §6). Check that the
no-memory baseline on the image-rendered HDF5's low-dim keys gives the same numbers as the
existing runs. Rendering must not change `obs`/`actions`.

---

## 4. Step 2: adding IMPRINT memory to flow BC

### 4.1 Pick the integration route

| Route | What | Pros | Cons |
|---|---|---|---|
| **A. Frozen memory tokens (recommended first)** | Stage 1 in `.venv-imprint` → `precompute_mem.py`-style cache of `(256,1024)` tokens per (episode, step) → ogpo loads plain arrays | No env merge. ogpo code changes are confined to data loading and the actor input. This is IMPRINT's own frozen-bottleneck recipe. The BC arm stays bit-comparable. | Memory can't adapt to the BC loss. Eval needs the encoder online, so it runs as a separate process (IMPRINT already splits policy and simulator processes the same way). |
| B. Port the encoder into ogpo | Copy `gated_delta_mem.py` (+ `utils.kernel_init`) into `ogpo/`, run it under flax 0.12.4 nnx, bridge to linen | Joint fine-tuning ("memft", as in the paper). One process at eval. | nnx API drift from 0.10.2 to 0.12.4. Checkpoint loading must be remapped. Breaks the "ogpo/ is byte-identical to SafeVaDaR" rule (see the README). |
| C. Port the flow actor into IMPRINT's env | Re-implement the 4×512 flow MLP in `.venv-imprint` | Small actor code. Fine-tuning is available. | A different JAX/flax stack from the SafeVaDaR baselines, so numbers aren't directly comparable. |

**Recommendation:** start with **A** to learn whether memory helps at all on our tasks. Fine-tuning
memory with the actor requires B or C, so decide on that after A's result.

### 4.2 Data pipeline (robomimic → IMPRINT store)

For each demo in `demo_v15.hdf5`:

1. **Frames.** Render V views (for example `agentview`, `robot0_eye_in_hand`, or `sideview` for
   tool_hang) at 256×256 (IMPRINT resizes or pads to 256 for feature extraction).
2. **`action_proprio.npy`.** `action` (T, 7) from the HDF5, and `proprio` (T, 9) =
   eef_pos + eef_quat + gripper_qpos. Add `action_causal` = shift-by-one with zero at t=0.
   IMPRINT's `pretrain_bottleneck.py` can z-score these with `meta/assets/robomme/norm_stats.json`
   (`NORM_TARGETS=1`). Reuse ogpo's own normalization stats so that both sides agree.
3. **WAN latents.** `WanFrontend` over the downsample-2 frames gives
   `latents.npy (num_chunks, V*pool², 48)` plus `chunk_index.json`. Start with **one** variant
   (v0 = canonical, every frame) and set `n_variants: 1`. The jittered 8-variant augmentation
   needs a generator that isn't released, so write it later if needed.
4. **DINOv3.** `build_dinov3_store.py` gives `dinov3.npy (T, V, 64, 768)`. This file serves as both
   the recent stream and the Stage-1 align target.
5. **Index.** `meta/samples.json` lists (episode, t), and `meta/` holds normalization stats.
   `scripts/reproduce/validate_store.py` checks the layout.

A cheaper variant for a first look: the encoder only needs input *widths*. `proj_cosmo` is
`Linear(cosmos_latent_dim, 1024)` and `proj_recent` is `Linear(siglip_dim, 1024)`. You could feed
the low-dim `object` vector as a one-token "visual" stream (`num_views: 1`, `pool_hw: 1`,
`cosmos_latent_dim = siglip_dim = object_dim`) and use it as the align target. This skips
rendering, WAN, and DINOv3 completely. It departs from the paper, so label it that way.

### 4.3 Stage 1 for robomimic

Copy `imprint/src/mme_vla_suite/models/config/robomme/cosmo-wan-gated-delta-robocasa-combined600-dinov3-memft-optimized.yaml`
and change:

| Field | Paper (RoboCasa) | robomimic |
|---|---|---|
| `num_views` | 3 | 2 (our choice) |
| `cosmo_bottleneck_memory.action_in_dim` / `cosmo_decoder.action_dim` | 12 | 7 |
| `cosmo_bottleneck_memory.proprio_in_dim` / `cosmo_decoder.proprio_dim` / `memory_feature.state.input_dim` | 16 | 9 |
| `n_variants` | 8 | 1 (until a variant generator exists) |
| `latent_cap`, `latent_buckets`, `max_step_tokens`, `max_stream_steps`, `delta_max_chunks`, `cosmo_decoder.max_query_steps` | 4096 / [256…4096] / 1776 / 1776 / 384 / 2048 | sized to the longest episode (for example 512–1024 steps for tool_hang) |
| `cosmo_decoder.align_num_patches` | 192 (= 3 views × 64) | V × 64 |
| `action_horizon` | 20 | 8 (only used by Stage 2) |

Also set `aux_weight: 0` in our YAML (§1.4: in Stage 2 it only drives extra dataset IO).
Action and proprio widths are read from the YAML (`pretrain_bottleneck.py:516-518`,
`dataset.py:408`). The one hardcoded RoboCasa assumption is `_load_norm`
(`pretrain_bottleneck.py:45`). It slices state to 16 and actions to 12 from
`meta/assets/robomme/norm_stats.json`, and only runs with `NORM_TARGETS=1`. Either leave
that off or patch the slice. The `latents_v*.npy` / `variant_index.json` generator is
confirmed absent: only readers exist, plus a test fixture.

Then run `train_stage1.sh` with our own `experiment.env`, or call
`python -m mme_vla_suite.scripts.pretrain_bottleneck` directly with `--fsdp_devices 1`, on one
L40S under `.venv-imprint`. Use `--w_reward 0.5` only if we provide a progress target. robomimic
has a natural one: t/T, or ogpo's subtask status from `envs/robomimic_utils.py`.

### 4.4 Feeding memory to the flow actor (route A)

- **Training.** For each sampled transition (episode e, step t), load the cached tokens
  `M[e,t] ∈ R^{256×1024}`. Reduce them to a vector with a mean-pool or a small learned
  attention-pool (one query cross-attends M, like IMPRINT's reward head in `cosmo_decoder.py`),
  then project to d, for example 64–256. Then use one of these:
  - **concat** to `observations` before `ActorVectorField` (simplest, and leaves the MLP unchanged), or
  - **FiLM** of the hidden layers from the memory vector. This is the closest analog to IMPRINT's
    modulation. Note that `ActorVectorField(use_film=True)` constructs `MLPWithFiLM` with
    MLP-style arguments (`actors.py:126`), but `MLPWithFiLM`'s fields are
    `act_dim/Ta/obs_dim/To/...` (`ogpo/networks/modules/mlp.py:125`). That path looks unused, so
    verify it before relying on it.
- **Bookkeeping in ogpo.** Add `episode_id` and `step_idx` arrays (or a precomputed `mem_feat`
  array aligned to transitions) to the dataset built in `ogpo/utils/datasets.py`, and gather them
  in `sample_sequence` at `idxs`, the chunk start. The memory at t must see only steps < t for
  actions and ≤ t for frames. This is exactly IMPRINT's causal contract.
- **Eval.** Each of the 32 envs needs its own memory state: a `CosmoMemoryBuffer` plus the
  gated-delta incremental state. Reset it on `envs.reset` and feed it every env step. At each
  decision point (queue empty), read the tokens, pool them, and pass them to `agent.sample_actions`
  with the obs. With route A, the encoder runs in the `.venv-imprint` process, and the obvious
  boundary is IMPRINT's websocket server (`imprint/scripts/serve_imprint_policy.py`,
  `mme_vla_suite/serving/websocket_policy_server.py`) or a small RPC.
  `scripts/eval_flow_bc.sbatch` currently evaluates on CPU, and memory eval needs a GPU for
  WAN/DINOv3.

### 4.5 Later: memory for the critic

The same pooled memory vector can condition the critic (`ogpo/networks/critics.py`). That is the
`memory_critic` direction. Route A makes it a data-loading change only.

---

## 5. Suggested order of work

1. Render images for square ph (and tool_hang ph). Confirm the no-memory BC numbers are unchanged.
2. Build a robomimic IMPRINT store (§4.2), first with the low-dim "visual" shortcut, then with WAN+DINOv3.
3. Stage 1 on one L40S (§4.3). Check that reconstruction loss of past actions and proprio falls
   (a sanity check that memory holds history).
4. Precompute frozen tokens. Add the memory vector to the flow actor (concat). Train BC with every
   other flag identical to `train_flow_bc.sbatch`.
5. Online eval with a per-env memory buffer. Compare against §3.
6. Only if memory helps: route B/C for joint fine-tuning, the critic, and the 8-variant jitter.

---

## 6. Open questions for you

- **Is memory needed on these tasks?** With the low-dim `object` state, square and tool_hang are
  close to Markov, so the BC arm may see no gain. A memory-dependent variant (an occluded object,
  a hidden subtask phase, image-only obs) may be needed to show an effect.
- **Which camera views,** and how many (the cost scales with V)?
- **Frozen or fine-tuned memory** for the main result? This decides route A vs B/C.
- **Horizon:** keep 8 (comparable to the baselines) or move toward IMPRINT's 20?
- **Variants:** is single-variant Stage 1 acceptable, or do we write the jitter generator?

---

## 7. File map (IMPRINT)

| Concern | File |
|---|---|
| Encoder factory | `src/mme_vla_suite/models/representation/memory_factory.py` |
| Full-attention encoder | `src/mme_vla_suite/models/representation/cosmo_mem.py` |
| Recurrent encoder (+ incremental API) | `src/mme_vla_suite/models/representation/gated_delta_mem.py` |
| Stage-1 decoder / loss / trainer | `models/representation/cosmo_decoder.py`, `training/pretrain_loss.py`, `scripts/pretrain_bottleneck.py` |
| Pi0.5 + memory model | `models/integration/history_pi0.py` (Pi0.5), `memory_pi0.py` (Pi0) |
| Memory modulation in the action expert | `models/integration/history_gemma.py:45,178` |
| Stage-2 data gather (variants, causal action) | `training/dataset.py:384-560` |
| Train configs (`pi05_baseline`, `imprint_*_memft`) | `training/config.py:771,1798-1955` |
| Online buffer | `shared/cosmo_buffer.py` |
| WAN / DINOv3 frontends | `models/representation/wan_frontend.py`, `dataset_builder/build_dinov3_store.py` |
| Store builders / frozen-token cache | `dataset_builder/build_wan_store.py`, `build_stage2_index.py`, `precompute_mem.py` |
| Launchers / paper profiles | `scripts/reproduce/train_stage{1,2}.sh`, `configs/paper/<bench>/<arch>/experiment.env` |
