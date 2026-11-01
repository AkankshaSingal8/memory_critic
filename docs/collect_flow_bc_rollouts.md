# Rollout collection from the frozen flow-BC checkpoints

Goal: 300 to 500 episodes per flow-BC policy, every env-step transition
(s_t, a_t, r_t, s_{t+1}, d_t) saved with policy_id, episode_id, success and timestep.
These rollouts are the training and test data for the critics in [bon_memory_critic.md](bon_memory_critic.md).

## Policies

There are 30 qipm flow-BC runs (`scripts/train_flow_bc.sbatch`, RUN_TAG 20261019, h8):
square-ph and square-mh, 10/20/50/100/200 demos, seeds 0/7/42. Each run's frozen policy is
`<run_dir>/selected/target_100.pkl`, which is the best checkpoint from `eval_flow_bc.sbatch`
(highest 3-point smoothed ODE success). It's a copy of `actor_<step>.pkl`, so it has the online
actor and the EMA target actor and noise nets, and it loads with `restore_actor_checkpoint`
the same way the eval loads it. `selection.json` gives the source step.

`policy_id = square_{ph|mh}_{N}demos_seed{S}_{sampler}`. The checkpoint step goes in the metadata.

## Sampler

The default is SDE (`agent.sample_actions`, target actor plus noise net, one sample per chunk,
no best-of-N), the same sampler as the `eval_sde/*` numbers. BoN reranks SDE samples, and a
critic needs outcomes that vary from the same start state. ODE only varies through its initial
noise. Policy noise comes from `PRNGKey(1000 + policy index)` and never reuses the eval keys
0/1. `--sampler ode` exists if we want ODE data too.

## Episodes and start states

There are 400 episodes per policy: 50 batches of 8 parallel envs. Episode e resets with seed
`10_000_000 + e`, so the start states are disjoint from every eval seed (those are at most
500_063) and the same across all 30 policies. Episode e is the same initial scene for every
policy, which pairs the comparisons across dataset sizes.

## What gets saved per policy

One `<policy_id>.npz` under `/data/user_data/$USER/memory_critic/rollouts/flow_bc_target100/`,
written in shards of one 8-episode batch, so a preempted job resumes where it stopped. Then a
merge step writes the final file and `<policy_id>.json` metadata.

Per transition (length T = total env steps):

| key | shape | meaning |
|---|---|---|
| obs | T x obs_dim | s_t (low-dim state; obs_dim is read from the env) |
| action | T x 7 | a_t, the clipped action that was passed to `env.step` |
| reward | T | r_t (the run's `rew_fn`) |
| next_obs | T x obs_dim | s_{t+1}; on the last step this is the true final state (autoreset is DISABLED) |
| terminated | T | 1 when the env ended the episode on success |
| truncated | T | 1 at the 400-step time limit |
| done | T | terminated or truncated (d_t) |
| is_success | T | info["success"] at this step |
| episode_id | T | 0..399 |
| timestep | T | t within the episode |
| chunk_pos | T | position of a_t inside its 8-action chunk; 0 marks a policy decision point |
| subtask_status | T x 3 | per-subtask flags from info["reward_info"] when the env reports them |

Per episode: `ep_success`, `ep_length`, `ep_return`, `ep_seed`.

Metadata json: policy_id, run_dir, checkpoint path and step, sampler, rng key, seed base,
episode count, obs_dim, controller and dataset root from flags.json, and the success rate.

## Code

`scripts/collect_flow_bc_rollouts.py` imports `apply_run_environment`, `build_eval_envs` and
`build_agent_template` from `scripts/safety_pretrain/eval_bc_run.py`, so env and agent
construction match the eval exactly. The stepping loop copies `evaluate_qipm_parallel`
(action queue per env, zero actions for finished envs, clip to [-1, 1]) and drops the Q
tracking and shield diagnostics, which don't change the actions. `ogpo/` and `envs/` stay
unchanged.

`scripts/collect_flow_bc_rollouts.sbatch` is a CPU array job on preempt with one task per
policy. It runs with `JAX_PLATFORMS=cpu`, 8 envs, and requeue on preemption. The eval took
about 46 s per 64 episodes, so a policy should take about 5 minutes.

## Checks

1. Smoke: one policy, 8 episodes. Check that T equals the sum of ep_length, that
   next_obs[t] == obs[t+1] inside each episode, that chunk_pos cycles 0..7, and that the
   last step of each episode has done=1.
2. Full run: the success rate per policy should be close to that run's `eval_sde/success` at
   the same step (64-episode estimate, so expect ±0.1).
