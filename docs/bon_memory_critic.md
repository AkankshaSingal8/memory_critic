# Best-of-N with and without an IMPRINT memory critic

This plan covers using the IMPRINT memory encoder inside a critic that reranks best-of-N (BoN)
action chunks from the policies we have already trained. Nothing is built yet. Background on
IMPRINT, its envs and its file map is in `docs/imprint.md`. This doc extends §4.5 there.

The dashboard has the same plan, with a block diagram, under the "Plan: memory critic for BoN"
tab: https://claude.ai/artifact/YKqC7RPua3YxBpBp6u7DnR

---

## 1. What IMPRINT gives us, and what it doesn't

IMPRINT ("Memory Without Curation", arXiv 2610.14406) trains a memory encoder that compresses an
episode's history (frames, executed actions, proprio) into 256 tokens. Stage 1 trains it with a
decoder that reconstructs, at random past steps, the DINOv3 features, the executed action, the
proprio, and one current-step progress scalar. The progress head is a single learned query that
cross-attends the memory tokens, followed by a sigmoid (`cosmo_decoder.py:219`). Stage 2 feeds the
tokens to the action expert.

The paper never uses memory for a critic, reward model or reranker. Its value-like piece is the
progress head, and that is V(history): it has no action input, so it cannot rank 8 candidate chunks
that share the same history. Using memory as a critic is our extension, built on frozen Stage-1
tokens.

Candidate chunks must not go through the encoder's action stream. That stream was trained on
executed actions only, and the candidates have no matching future proprio or frames. Memory is
computed once per decision point from steps 0..t and shared by all N candidates:

```
Q_mem(s_t, m_t, a_{t:t+H})     m_t = pool(IMPRINT_tokens(history_{0..t}))
```

---

## 2. BoN in OGPO_public today

`OGPOAgent.sample_actions_BON` (`OGPO_public/ogpo/agents/ogpo.py:1880`) does BoN as follows:

1. Repeat the obs N times (`best_of_n=8` in `scripts/ogpo/square.sh`).
2. Sample N chunks from the EMA target actor with the SDE sampler (`horizon_length=4`, so 4×7).
3. Score each chunk with `target_critic`, a 10-member MLP ensemble (4×512, LayerNorm). With
   `subsample_bon=true`, each chunk's score is the min over 2 members drawn at random (lines
   1955-1961). `q_agg` is not used here.
4. Execute the argmax chunk.

Online rollouts and every `eval_sde` log go through this path, and `eval` (ODE) does not. So the
full-dataset SDE finals on the dashboard are already "BoN-8 with a Markov critic", and the ODE
finals have no critic at all. A memory critic changes only step 3. The actor stays frozen.

The BC phase never trains the critic (`bc_update` only updates the actor, and `calql_steps` and
`q_warmup_steps` are 0), yet `bc_runner.py:171` still evaluates SDE with `sample_actions_BON`. The
SDE curves of the OGPO BC-only runs are therefore BoN-8 scored by a randomly initialised critic,
which is close to a random pick among 8 samples, so they serve as A1 below for those policies.

---

## 3. Which policies we can use

Checked on a compute node on 2026-10-26:

| Policy set | Checkpoint | Has a critic? | Usable |
|---|---|---|---|
| OGPO finals, square {ph, mh} × {10, 20, 50, 100, 200 (mh only), full} × seeds {0, 7, 42} | `/data/user_data/asingal2/memory_critic/ogpo_public/ckpt/<run>/ogpo/<seed>_square-<dt>-low_dim/<step>_agent.pkl` | yes: `critic` + `target_critic` | yes, first choice |
| OGPO repo BC-only sweep (`KIND=bc`) | none saved (`save_interval_bc=-1`) | no | only after a rerun with `save_interval_bc` set |
| This repo's flow BC (`train_flow_bc.sbatch`) | `<run_dir>/selected/target_100.pkl` | no | later; different actor and env (`.venv`), so it needs its own BoN wrapper |

Each OGPO checkpoint holds `actor`, `target_actor`, `noise_net`, `target_noise_net`, `critic`,
`target_critic`, and a `ReplayBuffer` of about 2M online transitions with keys `observations` (23),
`actions` (7, one env step each), `rewards`, `masks`, `terminals`, `is_success` and `score`. It has
no episode id, images or sim state. The online loop steps a single env (`online_rl_runner.py:571`),
so episodes are contiguous in the buffer and can be cut at `terminals`. The checkpoint doesn't keep
a copy of the BC-phase actor.

Arms 3 to 5 below train their own critic from rollouts, so they work for any policy, including
the BC ones once they're saved. Only arm 2 needs the checkpoint's own critic.

---

## 4. The comparison

All arms use the same frozen actor, N=8, and the same eval seeds. Only the scorer changes.

| Arm | Scorer | Exists today? | What it isolates |
|---|---|---|---|
| A0 | none, ODE sampler | yes (`eval` in W&B) | deterministic baseline |
| A1 | none, SDE with N=1 (`sample_actions`) | no, but it's a flag | what BoN gains at all |
| A2 | checkpoint's own Markov critic, `Q(s, a)` | yes (`eval_sde` in W&B) | the existing BoN |
| A3 | Markov critic retrained on our rollout data | no | control for "trained on different data" |
| A4 | clock critic, `Q(s, t/T, a)` | no | control for "memory is mostly a clock" |
| A5 | memory critic, `Q(s, m_t, a)` with frozen IMPRINT tokens | no | the claim |
| A6 (optional) | GRU history critic, trained end to end | no | does IMPRINT pretraining beat plain recurrence |

A3, A4 and A5 share data, targets, architecture (OGPO's 10×MLP 4×512 with LayerNorm), ensemble
size, BoN aggregation (min over 2 random members) and training steps. They differ only in the extra input. Without A3, any gain
from A5 over A2 could come from the new training data. Without A4, it could come from knowing the
step index: the episode has a time cap and the reward is −1 per step, so the true Q depends on t.

### Expected headroom

The policy is Markov in the 23-dim obs and the simulator is Markov in its full state. History can
only help the critic through what the obs leaves out (velocities, contact, a failed grasp the
object pose no longer shows) and through the time cap. On low-dim square that may be small.
Full-dataset ODE and SDE finals are within a few points of each other and both above 90%, which
leaves little to rerank. The 10 and 20-demo checkpoints have weaker critics and more failures,
so start there.

---

## 5. Pipeline

### 5.0 First: the no-memory critic (A3)

Train A3 before anything touches IMPRINT. It needs no new env, and every later arm reuses its data,
targets and training loop. A4 and A5 add one input each.

OGPO_public already has the training step. `OGPOAgent.bc_q_update` (`ogpo/agents/ogpo.py:1530`,
used by the `bc_q` phase in `bc_runner.py:345`, which square.sh turns off with `bc_q_steps=0`)
trains only the critic with

```
loss = TD( Q(s_t, a_t:t+4),  r_t:t+4 + γ · mean_k Q_target,k(s_t+4, a_t+4:t+8) )
     + mc_regression_coeff · MSE( Q(s_t, a_t:t+4),  G_t )
```

where the next chunk is the one logged in the data (so the TD part evaluates the data's policy),
and `G_t` comes from `Dataset.compute_mc_returns`. Batches come from
`Dataset.sample_sequence(256, sequence_length=4, discount, ret_next_act=True, ret_mc=True)`.

Steps for one checkpoint (start with a 10 or 20-demo OGPO final):

1. Load the checkpoint and keep the actor frozen.
2. Roll it out with plain SDE sampling and one sample per decision (`sample_actions`, A1), not
   BoN, so the data comes from the same policy that will produce the candidates. Use 32 envs and
   about 1000 episodes. Save per env step: obs (23), action (7), reward (−1, or 0 at success),
   success, whether the episode ended by success or by the time cap, episode id, step index, and
   the MuJoCo sim state (for the memory critic and the oracle check later).
3. Build an OGPO `Dataset` from those arrays. Set `terminals = 1` at the last step of every
   episode, since that's how `compute_mc_returns` finds episode boundaries. Set `masks = 0` only
   where the episode ended in success, and keep `masks = 1` at a time-cap ending. Run
   `compute_mc_returns(discount)`. Hold out 10% of episodes, split by episode.
4. Build an agent with square.sh's agent config and the checkpoint's actor params, then
   reinitialise the critic: 10 MLPs, 4×512, LayerNorm, input = obs plus the flattened 4×7 chunk.
5. Train with `bc_q_update`, keeping `discount`, `horizon_length=4`, `q_agg=mean`,
   `critic_lr=3e-4` and `tau` from square.sh. Start with 100k steps and stop when the held-out
   loss flattens.
6. Check it on held-out episodes: MSE of Q against `G_t`, success AUC of Q at matched steps, and
   the same two numbers for the checkpoint's own critic (A2) on the same episodes.
7. For BoN, put the new critic params into the loaded agent (`agent.replace(critic_network=...)`)
   and run `sample_actions_BON` with the same seeds as A1 and A2.

A faster first pass needs no rollouts: train on the checkpoint's replay buffer, whose episodes are
contiguous. It's off-policy, since it mixes every policy from training, so treat that run as a
pipeline test, not as A3. Check first how the buffer marks time-cap endings in `masks`.

Two things to keep in mind. The data comes from π with one sample, so the critic estimates Q of π,
not of the BoN policy that will act at later steps. That makes BoN a one-step improvement on π.
Without a t input, the critic can't see how close the time cap is, which is what A4 measures.

### 5.1 Offline: build the critics

1. Rollouts. For each chosen checkpoint, run K episodes (start with 500) with A1 (SDE, N=1)
   and record obs, executed action, reward, success, episode id and step index, plus the
   MuJoCo sim state every step for the optional oracle check in 5.2. Collect these on-policy for
   the actor whose candidates will be scored. Also cut the replay buffer into episodes. Use those
   for Stage 1, and for critic pretraining only if the on-policy rollouts are too few.
2. IMPRINT store, low-dim first. Use the shortcut in `docs/imprint.md` §4.2: the 14-dim
   `object` vector stands in as a one-token visual stream (`num_views: 1`, `pool_hw: 1`,
   `cosmos_latent_dim = siglip_dim = 14`), proprio = the first 9 obs dims, action = 7, and
   `action_causal` shifted by one. That needs no rendering, WAN or DINOv3. Images come later,
   rendered during new rollouts, since the buffer has no sim states.
3. Stage 1 in `.venv-imprint` on one L40S, recurrent (`gated_delta_hybrid`) encoder, on demos,
   rollouts and buffer episodes from both ph and mh. Train it once and share it across all
   checkpoints. Mask the progress target on failed episodes, where t/T has no meaning. To check
   it, reconstruction error on past actions and proprio should fall well below a no-history
   baseline.
4. Cache memory. Freeze the encoder and write, for every rollout step, two 1024-dim vectors:
   the mean of the 256 tokens, and the output of the Stage-1 progress query (a trained pool).
   fp16 on `/data`.
5. Train critics A3 to A5 (and A6) in OGPO_public's `.venv` on the cached arrays, with Monte
   Carlo targets from the rollouts under OGPO's reward (−1 per step, 0 at success) and discount.
   Monte Carlo targets from on-policy rollouts estimate Q of that policy without a bootstrap.
   The memory vector goes through a learned projection to 64 dims and is concatenated to the obs.

### 5.2 Gate: offline checks before any online eval

Run these on held-out rollout episodes:

- Success AUC of each critic's Q at matched step t (memory should separate eventual failures
  earlier than the Markov critic).
- Argmax agreement: at recorded decision points, draw the same 8 candidates and count how often
  A5 picks the same chunk as A3. If they agree almost always, the online BoN numbers will match,
  and the online path isn't worth building.
- Optional oracle: restore the sim state, roll each of the 8 candidates out to the end with the
  policy, and use the outcomes as true Q to measure ranking accuracy per critic.

### 5.3 Online: BoN eval

- One memory state per env, reset at `envs.reset` and fed every env step, not only at decision
  points (`docs/imprint.md` §2).
- The encoder runs in `.venv-imprint` as a small server. The eval process (OGPO_public `.venv`)
  sends each step's obs and executed action for the 32 envs and asks for `m_t` at decision points.
  The two envs can't share a process (jax 0.5.3/flax nnx vs jax 0.9/flax linen).
- Use common random numbers: the same env reset seeds and the same actor noise keys for every arm,
  so the first 8 candidates are identical across arms and the arms only diverge after picking
  differently.
- 256 episodes per checkpoint per arm (8 × 32 envs), for seeds 0, 7 and 42. At p ≈ 0.8, one seed's
  standard error is about 2.5 points.

### 5.4 Decision rule

Memory earns its cost on this task only if A5 beats both A3 and A4 at 10 and 20 demos by more than
the spread across seeds. Otherwise stop on low-dim square and move to a setting where history
carries information the obs lacks: image-only obs, an occluded object, or tool_hang.

---

## 6. Order of work

| Step | Output | Gate to continue |
|---|---|---|
| 1. Load each OGPO final, run A0, A1, A2 with the new eval script | a table per checkpoint; A2 should match the W&B `eval_sde` final within noise | A2 − A1 shows BoN helps somewhere (likely low demos) |
| 2. Rollouts for 10/20-demo ph and mh, then train A3 and run it in BoN (5.0) | per-step arrays with episode ids on `/data`; the A3 critic; BoN results for A1, A2, A3 | enough failures to train on; A3 fits held-out returns at least as well as A2 |
| 3. Low-dim IMPRINT store + Stage 1 | frozen encoder, reconstruction curves | past-action/proprio reconstruction beats no-history |
| 4. Cache `m_t`, train A4 and A5 (+A6) on the A3 data and loop | critics with configs identical to A3 | |
| 5. Offline checks (5.2) | AUC and argmax-agreement tables | A5 ranks differently from A3 |
| 6. Memory server + online BoN eval | success per arm per checkpoint | decision rule (5.4) |
| 7. Add results to the dashboard | a `bon_eval` dataset and chart | |

New code would live in this repo (for example `bon/` for rollouts, critics and the BoN evaluator,
importing OGPO_public, and `imprint_bridge/` for the store builder, Stage-1 config, token cache and
memory server). OGPO_public's `memcritic-preempt` branch and `ogpo/` here stay unchanged.

---

## 7. Open questions

- Which policies first? The recommendation is the OGPO finals at 10 and 20 demos (ph and mh, 3 seeds).
- Do you want BoN on pure BC policies too? That needs either a BC-only rerun with
  `save_interval_bc` set, or a wrapper around this repo's flow-BC `target_100.pkl` checkpoints.
- Low-dim memory only for the first result, or render images during the new rollouts so an image
  memory can follow without re-collecting?
- Monte Carlo targets (simple, on-policy) or TD targets (closer to how A2 was trained)?
