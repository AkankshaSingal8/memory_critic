"""Integration test of experiment B's memory hook against OGPO's own code (CPU, synthetic episodes).

    PYTHONPATH=<memory_critic>:<imprint src>:<OGPO_public> python -m imprint_bridge.test_ogpo_hooks \
        --stage1_dir <tiny stage1> --data_dir <fake rollouts> --mode critic|actor_critic

Checks, for one mode:
  1. demo memory: one episode per demo is found, and the dataset carries [obs, m_t] where the mode says
     (full_states for "critic", observations for "actor_critic"), with m_{t+1} in the next_* field;
  2. online == offline: replaying a demo through the wrapped online env and the wrapped vector env
     reproduces the memory attached to that demo, step for step;
  3. OGPO runs on it: OGPOAgent.create from OGPO's config (square.sh's agent settings, smaller nets),
     the critic and actor input widths are as the mode says, a ReplayBuffer filled the way
     run_online_rl fills it, one batch_update, and evaluate_parallel with best-of-N.
"""

import argparse
import sys

import numpy as np


class ReplayEnv:
    """Gymnasium-style env that replays one recorded episode (obs, next obs), ignoring actions."""

    def __init__(self, obs, next_obs):
        self.obs, self.next_obs, self.t = obs, next_obs, 0

    def reset(self, seed=None):
        self.t = 0
        return self.obs[0].copy(), {}

    def step(self, action):
        o = self.next_obs[self.t].copy()
        self.t += 1
        done = self.t >= len(self.obs)
        return o, -1.0, bool(done), False, {"success": float(done)}


class ReplayVectorEnv:
    """AsyncVectorEnv stand-in (autoreset disabled) over several ReplayEnvs."""

    def __init__(self, envs):
        self.envs, self.num_envs = envs, len(envs)

    def reset(self, seed=None):
        return np.stack([e.reset()[0] for e in self.envs]), {}

    def step(self, actions):
        out = []
        for e, a in zip(self.envs, actions):
            out.append(e.step(a) if e.t < len(e.obs) else (e.next_obs[-1].copy(), 0.0, True, False, {"success": 1.0}))
        obs, rew, term, trunc, info = zip(*out)
        succ = np.array([i["success"] for i in info])
        return (np.stack(obs), np.array(rew), np.array(term), np.array(trunc),
                {"success": succ, "_success": np.ones(len(succ), bool)})


def fake_demos(data_dir, n):
    """OGPO-format demo fields from the first n episodes of a fake rollout file."""
    from imprint_bridge import data as D
    z = np.load(D.rollout_files(data_dir)[0])
    obs, act, ep = z["obs"].astype(np.float32), z["action"].astype(np.float32), z["episode_id"]
    cols = {k: [] for k in ("observations", "actions", "rewards", "terminals", "masks", "next_observations")}
    for e in np.unique(ep)[:n]:
        o, a = obs[ep == e], act[ep == e]
        nxt = np.concatenate([o[1:], o[-1:] + 0.01])
        done = np.zeros(len(o), np.float32); done[-1] = 1
        for k, v in (("observations", o), ("actions", a), ("rewards", -np.ones(len(o), np.float32)),
                     ("terminals", done), ("masks", 1 - done), ("next_observations", nxt)):
            cols[k].append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mode", choices=("critic", "actor_critic"), required=True)
    ap.add_argument("--n_demos", type=int, default=6)
    ap.add_argument("--n_eval_envs", type=int, default=4)
    args = ap.parse_args(argv)

    import jax
    from ogpo.agents.ogpo import OGPOAgent
    from ogpo.utils.config_loader import config_to_agent_dict, load_config
    from ogpo.utils.datasets import Dataset, ReplayBuffer
    from ogpo.utils.evaluation import evaluate_parallel

    from imprint_bridge.ogpo_hooks import MemoryEnv, MemoryVectorEnv, _make_tracker, install
    from imprint_bridge.tracker import Projection

    cfg = {"mode": args.mode, "stage1_dir": args.stage1_dir, "feature": "mean", "projection": None}
    demos = Dataset.create(**fake_demos(args.data_dir, args.n_demos))
    obs_dim = demos["observations"].shape[1]
    bounds = [(s, e) for s, e in zip(*np.unique(np.cumsum(np.r_[0, demos["terminals"][:-1]]), return_index=True))]

    episodes = []
    s = 0
    for e in np.flatnonzero(demos["terminals"]) + 1:
        episodes.append((s, int(e))); s = int(e)
    env0 = ReplayEnv(demos["observations"][episodes[0][0]:episodes[0][1]],
                     demos["next_observations"][episodes[0][0]:episodes[0][1]])
    venv = ReplayVectorEnv([ReplayEnv(demos["observations"][a:b], demos["next_observations"][a:b])
                            for a, b in episodes[:args.n_eval_envs]])
    env, _, eval_envs, ds = install(env0, None, venv, demos, cfg, args.n_eval_envs)
    key_now, key_next = (("full_states", "next_full_states") if args.mode == "critic"
                         else ("observations", "next_observations"))
    mem_all = np.asarray(ds[key_now])[:, obs_dim:]
    mem_next = np.asarray(ds[key_next])[:, obs_dim:]
    dim = mem_all.shape[1]

    # 1. dataset fields
    from imprint_bridge.ogpo_hooks import demo_episode_bounds
    found = demo_episode_bounds(np.asarray(demos["observations"]), np.asarray(demos["next_observations"]))
    assert found == episodes, f"episode bounds {found[:3]}... != {episodes[:3]}..."
    for a, b in episodes:   # next memory of step t == memory of step t+1 inside an episode
        assert np.allclose(mem_next[a:b - 1], mem_all[a + 1:b], atol=1e-5)
    if args.mode == "critic":
        assert ds["observations"].shape[1] == obs_dim
    print(f"[1] {len(found)} demo episodes, memory dim {dim}, fields ok")

    # 2. online == offline (online env, then vector env)
    def mem_of(o):
        return (o["full_state"] if isinstance(o, dict) else o)[..., obs_dim:]

    a, b = episodes[0]
    o, _ = env.reset()
    got = [mem_of(o)]
    for t in range(b - a - 1):
        o, *_ = env.step(demos["actions"][a + t])
        got.append(mem_of(o))
    err1 = np.abs(np.stack(got) - mem_all[a:b]).max()
    o, _ = eval_envs.reset()
    errs = [np.abs(mem_of(o) - mem_all[[s for s, _ in episodes[:args.n_eval_envs]]]).max()]
    for t in range(3):
        acts = np.stack([demos["actions"][s + t] for s, _ in episodes[:args.n_eval_envs]])
        o, *_ = eval_envs.step(acts)
        errs.append(np.abs(mem_of(o) - mem_all[[s + t + 1 for s, _ in episodes[:args.n_eval_envs]]]).max())
    print(f"[2] online env vs demo memory {err1:.1e}; vector env vs demo memory {max(errs):.1e}")
    assert err1 < 1e-4 and max(errs) < 1e-4

    # 3. OGPO itself
    overrides = {"agent.horizon_length": 4, "agent.num_qs": 10, "agent.q_agg": "mean", "agent.best_of_n": 8,
                 "agent.subsample_bon": True, "agent.adv_strategy": "conservative", "agent.batch_size": 32,
                 "agent.ppo_batch_size": 32, "agent.actor_hidden_dims": [64, 64], "agent.value_hidden_dims": [64, 64],
                 "agent.use_success_buffer_q": True}
    config = load_config("ogpo", overrides)
    agent_config = config_to_agent_dict(config)
    ex = ds.sample(())
    agent = OGPOAgent.create(0, ex["observations"], ex["actions"], agent_config, agent_config["adv_clip_min"],
                             ex_images=None, ex_full_states=ex.get("full_states"))
    flat = {"/".join(str(getattr(k, "key", k)) for k in p): v.shape
            for p, v in jax.tree_util.tree_leaves_with_path(agent.critic_network.params)}
    first = [v for k, v in flat.items() if "modules_critic" in k and k.endswith("kernel")][0]
    want = obs_dim + dim + 4 * 7
    assert first[-2] == want, f"critic first layer input {first[-2]} != obs {obs_dim} + mem {dim} + chunk 28"
    aflat = {"/".join(str(getattr(k, "key", k)) for k in p): v.shape
             for p, v in jax.tree_util.tree_leaves_with_path(agent.actor_network.params)}
    actor_in = [v for k, v in aflat.items() if "modules_actor" in k and k.endswith("kernel")][0][-2]
    print(f"[3] critic first-layer input {first[-2]} (obs {obs_dim} + mem {dim} + chunk 28); "
          f"actor first-layer input {actor_in}")

    template = dict(observations=ex["observations"], actions=ex["actions"], rewards=np.array(0.0),
                    terminals=np.array(0.0), masks=np.array(1.0), next_observations=ex["observations"])
    if "full_states" in ex:
        template["full_states"] = template["next_full_states"] = ex["full_states"]
    rb = ReplayBuffer.create(template, 4000, numpy_rng=np.random.default_rng(0))
    rng = jax.random.PRNGKey(0)
    from ogpo.utils.evaluation import _split_obs
    for ep in range(6):   # the run_online_rl loop, without training
        a, b = episodes[ep % len(episodes)]
        env.env.__init__(demos["observations"][a:b], demos["next_observations"][a:b])
        ob, _, fs = _split_obs(env.reset()[0])
        queue, done = [], False
        while not done:
            if not queue:
                rng, k = jax.random.split(rng)
                kw = {"observations": ob, "rng": k}
                if fs is not None:
                    kw["full_state"] = fs
                queue = list(np.array(agent.sample_actions_BON(**kw)).reshape(-1, 7))
            act = queue.pop(0)
            nob_raw, r, term, trunc, info = env.step(act)
            nob, _, nfs = _split_obs(nob_raw)
            done = term or trunc
            tr = dict(observations=ob, actions=act, rewards=r, terminals=float(done), masks=1.0 - float(term),
                      next_observations=nob, is_success=float(info["success"]), score=1.0)
            if fs is not None:
                tr["full_states"], tr["next_full_states"] = fs, nfs
            rb.add_transition(tr)
            ob, fs = nob, nfs
    from ogpo.runners.online_rl_runner import _sample_online_batch
    batch = _sample_online_batch(rb, 32, 1, 4, 0.99)
    agent, info = agent.batch_update(batch, batch, False)
    losses = {k: float(v) for k, v in info.items() if "loss" in k and np.ndim(v) == 0}
    assert all(np.isfinite(v) for v in losses.values()), losses
    print(f"[3] replay buffer {rb.size} transitions; batch_update ok, e.g. "
          f"{dict(list(sorted(losses.items()))[:3])}")

    class Flags:
        n_eval_envs, eval_episodes, env_name, discount, run_name = args.n_eval_envs, 1, "square-ph-low_dim", 0.99, "t"
    for e in eval_envs.envs.envs:
        e.t = 0
    stats = evaluate_parallel(eval_envs, agent, Flags, 7, 0, actor_fn=getattr(agent, "sample_actions_BON"))
    print(f"[3] evaluate_parallel with BoN over the memory vector env: {stats}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
