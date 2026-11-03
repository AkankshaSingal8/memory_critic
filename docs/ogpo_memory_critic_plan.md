# Memory-conditioned OGPO-style critic for best-of-N on square (plan)

Nothing here is built yet. This refines [`bon_memory_critic.md`](bon_memory_critic.md); the changes to that doc are listed at the end. It was drafted by a planning agent on 2026-11-03. File:line claims were spot-checked where marked ✓.

## Key constraint

All N candidates at a decision share the same history. So any term in Q that depends only on the history gets added equally to every candidate and cannot change which one wins:
- a learned V(h),
- a clock offset,
- IMPRINT's progress head.

**Memory can only change best-of-N if it interacts with the action chunk.**

## 1. Architecture

| | Where memory enters | Verdict |
|---|---|---|
| **C1 (recommended)** | One shared history encoder gives m_t. A Dense layer maps it to 64 dims, concatenated to obs, then the unchanged `ogpo.networks.critics.Value` (10 × 4×512, LN). | A strict superset of the Markov critic. The existing BoN loader (`scripts/collect_flow_bc_rollouts.py:123-132` ✓) only needs a wider input. The MLP lets memory and action interact. |
| C2 | (obs, chunk) embedding cross-attends to per-chunk history tokens, then the MLP. | IMPRINT-style read. Try only if C1 shows a signal. |
| C3 | GRU over one event per completed chunk: obs and actions of that chunk (240 dims, ≤ 50 events per episode). | The encoder to use inside C1. O(1) state at eval time. |

**Causal contract.**
- At decision t, the memory sees obs_0..obs_{t-1} and the executed, clipped actions a_0..a_{t-1}.
- obs_t enters only through the critic's state input.
- The candidate chunk never enters the memory.

**Where the memory is computed.**
- **First:** a GRU in linen inside the ogpo `.venv`. No new environment, and it trains end to end on the value loss.
- **Second:** IMPRINT, run through `jax.export`. `.venv-imprint` emits calling convention 9 and `.venv` accepts 9–10 ✓. Export `update_incremental` / `read_incremental` (`gated_delta_mem.py:1012,1048` ✓) as pure functions and load them in-process.
  - Fallback: a numpy-pipe sidecar process.
  - A linen port of the 1,127-line module isn't worth it.
  - **Check before relying on it:** IMPRINT's chunk length is `temporal_factor` from its config (`gated_delta_mem.py:608`). It is not fixed at 8 steps. Make it 8 so the memory updates once per decision point.

## 2. Training

**Targets.**
- MC returns, gamma 0.99, at decision points only, with the 8-step chunk zero-padded. This matches `scripts/train_mc_critic.py`.
- No target critic is needed for MC.
- If TD is added later: SARSA on the logged next chunk. The next-state memory is the prefix plus one event, so it's available offline. The target network is an EMA of the whole module, encoder included.

**Setup.**
- Same uniform decision-point sampler as the baseline; each sample gathers its padded prefix (≤ 50 events).
- Same episode split: < 288 train, 288–319 val, ≥ 320 test.
- Reuse `load_dataset`, `metrics` and `grouped_metrics`.
- 200k steps, best-val checkpoint, seeds 0–2.
- History window: the full episode (≤ 50 events).

**Arms.** All use the same head, data and optimizer.

| Arm | Input | What it isolates |
|---|---|---|
| M0 | none (rerun of `mc_decision_seed{0,1,2}`) | loader parity |
| CLK | t/400 (the time limit; episode length T is unknown at decision time) | a clock |
| GRU | full history | the claim |
| SHUF | history from another episode of the same policy, same t | content vs clock and policy identity |
| K1 | last chunk only | short context (velocity) vs long memory |
| IMP | frozen IMPRINT pooled m_t | does pretraining beat recurrence |

Training pools 30 policies, so the history can identify which policy is acting. Always report metrics within each policy group (`grouped_metrics` already does this).

## 3. Best-of-N at eval time

**Stateful scorer.** `make_bon_actor` becomes stateful, chosen by `ck["config"]["memory"]`. The Markov path stays unchanged. It has three hooks:
- `reset(n_envs)` after `envs.reset`.
- `observe(obs, acts, active)` around each `envs.step`.
- At a decision: push the completed chunk event for the envs that need an action, then pad memory rows the same way obs is padded.

**Scoring.** Keep the ensemble mean, so results compare with the existing sweep. OGPO's min-over-2-of-10 becomes an optional flag.

**Sweep.** Add `CRITIC_TAG` (default `mc_decision`) to `scripts/bon_sweep/config.sh`, used by `critic()` / `out_dir()` in `bon_sweep.sbatch`. The existing plots then take `--root .../gru_seed{0,1,2}` unchanged. Common random numbers already hold (`RNG_KEY=3000`, `seed_base=T`).

**Caveat.** At best-of-N time the history holds the chosen actions, so it is shifted from the data the critic was trained on.

## 4. Files, jobs and order

**New files:**
- `memcritic/history.py`: chunk events, prefix index, SHUF permutation.
- `memcritic/critics.py`: `MemoryValue` (encoder → 64 → `Value`).
- `scripts/train_memory_critic.{py,sbatch}`: same shape as `train_mc_critic.sbatch`, about 2× its wall time. Array of arms × 3 seeds, GPU.
- `scripts/bon_disagreement.py`: CPU, minutes per policy.
- `scripts/plot_bon_memory_vs_markov.py`: paired differences.
- Later: `imprint_bridge/{build_lowdim_store.py, robomimic_lowdim.yaml, export_encoder.py}` (in `.venv-imprint`) and `memcritic/imprint_memory.py` (in `.venv`).

**Modified, backward compatible:**
- `scripts/collect_flow_bc_rollouts.py`
- `scripts/bon_sweep/{config.sh,bon_sweep.sbatch}`

OGPO_public and `ogpo/` stay byte-unchanged.

**Order, with a go/no-go after each step:**
0. **Parity.** M0 must match the existing `metrics.json` within the seed spread. Otherwise the loader is a confound.
1. **Offline arms** (CLK, GRU, SHUF, K1).
   - Go if GRU beats CLK and SHUF on within-group test MSE and AUC.
   - This is necessary but not sufficient, since history-only terms don't rank candidates.
2. **Disagreement gate.**
   - Re-sample 8 SDE chunks at recorded test observations with the frozen actor.
   - Go if GRU-vs-M0 argmax disagreement is greater than the disagreement between M0 seed 0 and M0 seed 1. If not, online best-of-N will tie, so stop.
   - This replaces the oracle check, which is impossible because the rollout npz has no sim state.
3. **Online best-of-N.**
   - N = 8, then 16 and 32, on CPU: about 20 minutes per task and about 1 hour of total wall time per arm.
   - Decision rule as in §5.4 of `bon_memory_critic.md`, with CLK and SHUF as controls.
4. **IMPRINT.**
   1. Export smoke test (CPU, about 30 minutes). Pass: `.venv` output matches `.venv-imprint` to 1e-5.
   2. Low-dim store, then Stage 1 at reduced width (GPU, a few hours).
   3. m_t cache at reduced width.
   4. Run IMP through steps 1–3.

## 5. Open questions for the user

1. Keep ensemble-mean scoring (comparable to the existing sweep), or switch every arm to min-over-2?
2. Is GRU-first acceptable, or must the headline result use IMPRINT's encoder? If IMPRINT, step 4 is on the critical path.
3. Low-dim only, or re-collect rollouts with images and sim state? That would allow image memory and the oracle check.
4. First online grid: all 30 policies × 10 BC steps, or 10 and 20 demos only?

## Changes to `bon_memory_critic.md`

- §3/§5.0: the policy set is the flow-BC rollouts (horizon 8, qipm `Value`). A3 already exists as `train_mc_critic` seeds 0–2. The OGPO finals (horizon 4) are deferred.
- §2/§4: scoring is the ensemble mean, not min-over-2.
- §4/§6: A6 (GRU) moves ahead of A5 (IMPRINT), and SHUF and K1 are added. With the low-dim shortcut IMPRINT loses its visual streams, so the GRU tests the same hypothesis more cheaply.
- §4: the clock input is t/400, not t/T.
- §5.2: the oracle check is replaced by the disagreement gate.
- §5.3: `jax.export` replaces the websocket server.
- §5.1: m_t is cached at reduced width, not as 1,024-dim tokens.
