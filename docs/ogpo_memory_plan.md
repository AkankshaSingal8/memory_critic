# Experiment B: OGPO with an IMPRINT memory critic (plan)

Experiment A (docs/imprint_memory_critic.md) fits a memory critic offline, after the fact. Experiment B puts
the memory into OGPO's own critic. That critic is trained online by TD, drives the actor's policy-gradient
advantage and does the best-of-8 (BoN) pick.

## Arms

| Arm | Critic sees | Actor sees | Runs |
|---|---|---|---|
| B1 | obs + `m_t` | obs | square ph, mh × seeds 0, 7, 42, 100 demos |
| B2 | obs + `m_t` | obs + `m_t` | same |
| baseline | obs | obs | **open: rerun, or the existing OGPO 100-demo runs (Q4)** |

At most 4 GPUs at once. W&B entity `akankshasingal08`.

## How OGPO uses its critic in the square config

Code map from public OGPO_public `0b3be41`:

- **Critic.** `Value`, a 10-member ensemble of MLP 4×512 + LayerNorm. Input is obs (23) + chunk (4×7).
  Trained only in the online phase.
  - TD target: `r + γ⁴ · mean_k Q_target,k(s_{t+4}, a')`, with `a'` from the target actor.
  - Run once on the replay batch and once on the success-buffer batch.
  - Learning rate after the BC→online hand-off: `ppo_lr` = 4.5e-5.
- **Actor.** PPO-style update. The advantage comes from 32 sampled chunks per state, scored by `target_critic`.
  The conservative max/min over members is used. There is also a BC term.
- **BoN-8.** Min of 2 random ensemble members, then argmax. Used in online rollouts and SDE evals,
  including BC-phase evals, where the critic is still random.
- **Privileged critic input.** `full_states` / `next_full_states` already reach every critic call.
  `_get_critic_obs`, BoN's `full_state` kwarg, `sample_sequence` and eval padding all handle them,
  while the actor keeps `observations`.

## What gets built

1. **Memory tracker** (`imprint_bridge/tracker.py`, in this repo).
   - Batched, per-env incremental IMPRINT state.
   - Reset at env reset with frame 0 and a zero causal action.
   - `step(action, next_obs)` after every env step. Every 4 frames the finished chunk goes into the gated-delta
     state (`update_incremental`). `read()` gives `m_t` for the current frame (`read_incremental` + pooling).
   - Same history contract and normalization as `imprint_bridge/data.py`, so `m_t` equals what
     `cache_memory.py` computes with a full forward pass.
   - Test: tracker vs full forward on the same episodes.
2. **flax 0.12 compatibility.** OGPO pins jax 0.10.1 / flax 0.12.7, where IMPRINT's modules (plain Python
   lists of submodules) fail to build. Fix: a `pytree=False` subclass, plus a stand-alone loader for the
   Stage-1 `.npz` and the history YAML. That avoids IMPRINT's trainer and omegaconf in OGPO's env.
3. **Memory → network input.** **Open (Q2):** 64 whitened PCs fit on A's cache (as in A), the full 1024-d,
   or a learned projection.
4. **OGPO patch** (`ogpo_memory/ogpo_public.patch`, applied to a checkout). **Base is open (Q1):** public
   `0b3be41` or your `memcritic-preempt`.
   - Replay buffer: store `full_states = [obs, m_t]` and `next_full_states = [next_obs, m_{t+1}]` per step.
     A sampled 4-step window can cross an episode end, so the next memory is stored, not looked up.
   - Online loop: tracker on the training env (reset with the env, step every env step). Pass `full_state` to BoN.
   - Eval (`evaluate_parallel`): a tracker over the 32 envs, reset at `envs.reset`, stepped after `envs.step`.
     `full_state` is padded like the existing code does.
   - Critic init with the widened example input.
   - B2 only: observations themselves become `[obs, m_t]` for the actor and critic, online and in eval.
5. **Demo memory (B2 only).** The B2 actor trains on the 100 demos during the 500k BC steps, so at dataset
   load the tracker runs over every demo episode and appends `m_t` to `observations` / `next_observations`.
6. **Encoder and feature.** **Open (Q3):** wait for A and use the winning Stage-1 init and pooling, or fix now.
   The encoder stays frozen throughout B (no TD fine-tuning).
7. **Jobs** (`scripts/ogpo_memory/`).
   - A setup script: checkout + patch + `uv sync`.
   - A run script and a submit script: one array over arm × dataset × seed, throttled to 4 GPUs,
     depending on A's Stage 1 + cache.
   - Partition and time limit depend on Q1: without resume fixes, a non-preempt partition.
8. **Tests (CPU).**
   - Tracker equivalence.
   - Patched `sample_sequence` returns the memory columns.
   - Critic init and BoN with the widened input.
   - A tiny OGPO run (a few hundred BC + online steps) on a small square file if it can be downloaded;
     otherwise the first cluster job is the integration test.

## Open questions (asked, not assumed)

1. OGPO base: public `0b3be41` or `memcritic-preempt`.
2. Memory input: 64 PCs / full 1024 / learned projection.
3. Encoder: wait for A's winner, or fix it now.
4. Baseline: rerun no-memory OGPO on the patched code, or the existing runs.

## Status

| Item | State | Where |
|---|---|---|
| 1 tracker | done, tested | `imprint_bridge/tracker.py`, `test_tracker.py` |
| 2 flax 0.12 compat | done, tested | `tracker._compat`, `_load_params`, `load_history_config` |
| 3 memory input | code supports all three options; **choice open (Q2)** | `imprint_bridge/fit_projection.py`, `MEMORY_PROJECTION` |
| 4 OGPO patch | done, 26 lines; applies on `0b3be41`; **base open (Q1)** | `scripts/ogpo_memory/ogpo_public.patch`, `imprint_bridge/ogpo_hooks.py` |
| 5 demo memory | done, tested | `ogpo_hooks.demo_memory` |
| 6 encoder choice | configurable; **choice open (Q3)** | `MEMORY_STAGE1_DIR`, `MEMORY_FEATURE` |
| 7 jobs | done; open choices are required settings | `scripts/ogpo_memory/{config.sh,setup.sh,run.sbatch,submit.sh}` |
| 8 tests | done (CPU), except a real OGPO run | `test_tracker.py`, `test_ogpo_hooks.py` |

**Tests that passed (CPU, synthetic episodes, tiny encoder):**

- Tracker vs full forward pass: max error 6e-7 under IMPRINT's flax 0.10, and 1.3e-6 under OGPO's jax 0.10.1 / flax 0.12.7.
- Tracker vs `cache_memory.py`'s saved features: max error 1e-3, which is the cache's float16 rounding.
- `test_ogpo_hooks` against OGPO's own code (locked env, jax 0.10.1 / flax 0.12.7), both arms:
  - Demo bounds found, and next memory equals the memory one step later.
  - Online env and vector env memory match demo memory to 5e-7.
  - `OGPOAgent.create` from OGPO's config gives the expected input widths. Critic: obs + mem + chunk.
    Actor: no memory in B1, + mem in B2.
  - A ReplayBuffer filled as `run_online_rl` fills it, followed by `batch_update`, gives finite losses.
  - `evaluate_parallel` with BoN completes.
- The patch applies to a fresh `0b3be41` checkout, and the patched `square.sh` parses.

**Not run:**

- OGPO's `main.py` end to end. It needs robomimic, MuJoCo and the square demos, which aren't on this machine.
- The CLI path `--memory.mode=none` → `None` (fixed in code, not rerun).
- The `test_ogpo_hooks` rerun after that fix. Running OGPO's code was blocked here by a permission check.
