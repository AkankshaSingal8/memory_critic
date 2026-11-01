"""Roll out one frozen flow-BC checkpoint and save every env-step transition.

Plan and output format: docs/collect_flow_bc_rollouts.md. Env and agent are built exactly as
scripts/safety_pretrain/eval_bc_run.py builds them (same flags.json, controller, dataset root);
the stepping loop is ogpo/utils/evaluation_qipm.evaluate_qipm_parallel without the Q tracking.

Episodes run in batches of n_envs. Batch b resets with seed seed_base + b*n_envs (env i gets
seed + i, as in the eval), so episode e always starts from seed seed_base + e, the same scene
for every policy. Policy noise for batch b is fold_in(PRNGKey(rng_key), b), so a resumed job
draws the same noise. Each finished batch is written atomically to
<out_dir>/shards/<policy_id>/batch_<b>.npz and skipped on restart; once all batches exist they
are merged into <out_dir>/<policy_id>.npz plus <out_dir>/<policy_id>.json.

    python scripts/collect_flow_bc_rollouts.py --run_dir <training run dir> \
        --policy_id square_ph_10demos_seed0_sde --out_dir <dir> --episodes 400
"""

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts", "safety_pretrain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

EXIT_INCOMPLETE = 3  # the sbatch wrapper requeues on this
STEP_KEYS = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "done",
             "is_success", "episode_id", "timestep", "chunk_pos", "subtask_status")
EP_KEYS = ("ep_success", "ep_length", "ep_return", "ep_seed")


def _atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _subtask_status(infos, i, n_sub):
    ri = infos.get("reward_info")
    if isinstance(ri, dict) and "subtask_status" in ri:
        return np.asarray(ri["subtask_status"][i], dtype=np.int8)[:n_sub]
    return np.full(n_sub, -1, dtype=np.int8)


def rollout_batch(envs, actor_fn, rng, seed, n_envs, action_dim, n_sub, first_episode):
    """One batch of n_envs episodes; returns (per-step dict, per-episode dict)."""
    import jax
    raw, _ = envs.reset(seed=seed)
    obs = raw["state"] if isinstance(raw, dict) else raw
    assert not isinstance(raw, dict), "image observations are not supported"
    queues = [[] for _ in range(n_envs)]
    active = np.ones(n_envs, dtype=bool)
    steps = np.zeros(n_envs, dtype=int)
    pos = np.zeros(n_envs, dtype=int)
    rec = [{k: [] for k in STEP_KEYS} for _ in range(n_envs)]
    ep_ret = np.zeros(n_envs)
    ep_succ = np.zeros(n_envs, dtype=np.int8)

    while np.any(active):
        need = [i for i, q in enumerate(queues) if not q and active[i]]
        if need:
            o = obs[need]
            if len(need) < n_envs:  # pad to a fixed batch to avoid recompiling
                o = np.concatenate([o, np.zeros((n_envs - len(need),) + o.shape[1:], o.dtype)])
            rng, key = jax.random.split(rng)
            chunks = np.array(actor_fn(observations=o, rng=key)).reshape(n_envs, -1, action_dim)
            for j, i in enumerate(need):
                queues[i].extend(chunks[j])
                pos[i] = 0
        acts = np.zeros((n_envs, action_dim))
        cpos = np.full(n_envs, -1)
        for i, q in enumerate(queues):
            if active[i] and q:
                acts[i] = q.pop(0)
                cpos[i] = pos[i]
                pos[i] += 1
        acts = np.clip(acts, -1, 1)
        raw, rewards, terms, truncs, infos = envs.step(acts)
        next_obs = raw
        for i in range(n_envs):
            if not active[i]:
                continue
            done = bool(terms[i] or truncs[i])
            r = rec[i]
            r["obs"].append(obs[i]); r["action"].append(acts[i]); r["reward"].append(rewards[i])
            r["next_obs"].append(next_obs[i]); r["terminated"].append(terms[i])
            r["truncated"].append(truncs[i]); r["done"].append(done)
            r["is_success"].append(float(infos["success"][i]) if "success" in infos else 0.0)
            r["episode_id"].append(first_episode + i); r["timestep"].append(steps[i])
            r["chunk_pos"].append(cpos[i]); r["subtask_status"].append(_subtask_status(infos, i, n_sub))
            ep_ret[i] += float(rewards[i])
            steps[i] += 1
            if done:
                ep_succ[i] = int(float(infos["success"][i]) > 0) if "success" in infos else 0
                active[i] = False
        obs = next_obs

    dtypes = {"obs": np.float32, "action": np.float32, "reward": np.float32,
              "next_obs": np.float32, "terminated": np.int8, "truncated": np.int8,
              "done": np.int8, "is_success": np.int8, "episode_id": np.int32,
              "timestep": np.int16, "chunk_pos": np.int8, "subtask_status": np.int8}
    out = {k: np.concatenate([np.asarray(r[k], dtype=dtypes[k]) for r in rec]) for k in STEP_KEYS}
    ep = {"ep_success": ep_succ, "ep_length": steps.astype(np.int16), "ep_return": ep_ret.astype(np.float32),
          "ep_seed": (seed + np.arange(n_envs)).astype(np.int64)}
    return out, ep


def make_bon_actor(sampler, critic_path, n, horizon_dim):
    """Best-of-n over sampler's chunks, scored by the MC critic's ensemble mean (scripts/train_mc_critic.py)."""
    import pickle
    import jax
    import jax.numpy as jnp
    from ogpo.networks.critics import Value
    with open(critic_path, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    critic = Value(hidden_dims=tuple(int(x) for x in cfg["hidden_dims"].split(",")), layer_norm=True,
                   num_ensembles=cfg["num_qs"], critic_loss_type="mse", num_subtasks=1)
    cparams = jax.device_put(ck["params"])

    @jax.jit
    def bon(observations, rng):
        b = observations.shape[0]
        obs_rep = jnp.repeat(observations, n, axis=0)
        chunks = jnp.clip(sampler(obs_rep, rng=rng).reshape(b * n, horizon_dim), -1, 1)
        q = critic.apply(cparams, obs_rep, chunks).mean(0).reshape(b, n)
        return chunks.reshape(b, n, horizon_dim)[jnp.arange(b), jnp.argmax(q, axis=1)]

    return lambda observations, rng: bon(jnp.asarray(observations), rng)


def merge(shard_dir, n_batches):
    parts = [np.load(os.path.join(shard_dir, f"batch_{b:04d}.npz")) for b in range(n_batches)]
    return {k: np.concatenate([p[k] for p in parts]) for k in STEP_KEYS + EP_KEYS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--policy_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="selected/target_100.pkl", help="relative to run_dir")
    ap.add_argument("--sampler", choices=("sde", "ode"), default="sde")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--seed_base", type=int, default=10_000_000)
    ap.add_argument("--rng_key", type=int, default=1000)
    ap.add_argument("--bon_n", type=int, default=1, help="best-of-n chunks per decision (1 = plain sampler)")
    ap.add_argument("--critic", default=None, help="critic_best.pkl from scripts/train_mc_critic.py (bon_n > 1)")
    args = ap.parse_args(argv)
    assert args.bon_n == 1 or args.critic, "--bon_n > 1 needs --critic"
    assert args.episodes % args.n_envs == 0, "episodes must be a multiple of n_envs"
    n_batches = args.episodes // args.n_envs

    from eval_bc_run import apply_run_environment, build_agent_template, build_eval_envs
    from ogpo.utils.main_helper import install_preemption_handler, is_preemption_requested
    install_preemption_handler()

    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "flags.json")) as f:
        flags = json.load(f)
    ctrl, root, sidecar = apply_run_environment(flags)
    mp = flags["experiment"].get("matmul_precision")
    import jax
    if mp:
        jax.config.update("jax_default_matmul_precision", mp)
    from ogpo.agents.modules.flax_utils import restore_actor_checkpoint, restore_agent_with_file

    ckpt = os.path.join(run_dir, args.checkpoint)
    import pickle
    with open(ckpt, "rb") as f:
        ckpt_step = pickle.load(f).get("step")
    ckpt_step = None if ckpt_step is None else int(ckpt_step)

    shard_dir = os.path.join(args.out_dir, "shards", args.policy_id)
    os.makedirs(shard_dir, exist_ok=True)
    todo = [b for b in range(n_batches)
            if not os.path.exists(os.path.join(shard_dir, f"batch_{b:04d}.npz"))]
    print(f"[collect] {args.policy_id}: run_dir={run_dir}\n[collect] checkpoint={ckpt} "
          f"(step {ckpt_step}) sampler={args.sampler} controller={ctrl or '<dataset OSC>'} "
          f"dataset_root={root}\n[collect] {n_batches - len(todo)}/{n_batches} batches already done",
          flush=True)

    obs_dim = action_dim = None
    if todo:
        envs = build_eval_envs(flags, args.n_envs)
        obs_dim = envs.single_observation_space.shape[-1]
        action_dim = envs.single_action_space.shape[-1]
        template = build_agent_template(flags, obs_dim, action_dim)
        if os.path.basename(ckpt).startswith("params_"):
            agent = restore_agent_with_file(template, ckpt)
        else:
            agent = restore_actor_checkpoint(template, ckpt)
        if args.sampler == "sde":
            actor_fn = agent.sample_actions
        else:
            actor_fn = getattr(agent, "compute_flow_actions_corrected", agent.compute_flow_actions)
        if args.bon_n > 1:
            horizon = int(flags["agent"].get("horizon_length", 8))
            actor_fn = make_bon_actor(actor_fn, args.critic, args.bon_n, horizon * action_dim)
        n_sub = flags["agent"].get("no_of_subtasks", 3)
        base = jax.random.PRNGKey(args.rng_key)
        try:
            for b in todo:
                if is_preemption_requested():
                    print("[PREEMPTION] stopping between batches; exiting for requeue", flush=True)
                    return EXIT_INCOMPLETE
                t0 = time.time()
                out, ep = rollout_batch(envs, actor_fn, jax.random.fold_in(base, b),
                                        args.seed_base + b * args.n_envs, args.n_envs,
                                        action_dim, n_sub, b * args.n_envs)
                _atomic_savez(os.path.join(shard_dir, f"batch_{b:04d}.npz"), **out, **ep)
                print(f"[collect] batch {b + 1}/{n_batches}: success {ep['ep_success'].mean():.3f} "
                      f"mean len {ep['ep_length'].mean():.0f} ({time.time() - t0:.0f}s)", flush=True)
        finally:
            envs.close()

    data = merge(shard_dir, n_batches)
    T = len(data["reward"])
    assert T == int(data["ep_length"].astype(np.int64).sum()), "transition count != sum of episode lengths"
    _atomic_savez(os.path.join(args.out_dir, f"{args.policy_id}.npz"),
                  policy_id=np.array(args.policy_id), **data)
    meta = {"policy_id": args.policy_id, "run_dir": run_dir, "checkpoint": ckpt,
            "checkpoint_step": ckpt_step, "sampler": args.sampler, "rng_key": args.rng_key,
            "seed_base": args.seed_base, "n_envs": args.n_envs, "episodes": args.episodes,
            "transitions": T, "obs_dim": int(data["obs"].shape[1]),
            "action_dim": int(data["action"].shape[1]), "env": flags["env"]["name"],
            "controller": ctrl, "dataset_root": root, "bon_n": args.bon_n, "critic": args.critic,
            "horizon": flags["agent"].get("horizon_length"),
            "success_rate": float(data["ep_success"].mean()),
            "mean_length": float(data["ep_length"].mean())}
    with open(os.path.join(args.out_dir, f"{args.policy_id}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[collect] done: {T} transitions, success {meta['success_rate']:.3f}, "
          f"mean length {meta['mean_length']:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
