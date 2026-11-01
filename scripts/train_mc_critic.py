"""No-memory baseline: one Q(s_t, a_t) fit by Monte Carlo regression on the pooled flow-BC rollouts.

Data: the 30 rollout files from scripts/collect_flow_bc_rollouts.py (docs/collect_flow_bc_rollouts.md).
Critic: the qipm critic module, ogpo.networks.critics.Value, built with qipm's own critic settings
(value_hidden_dims 4 x 512, layer_norm, num_qs=10 ensemble, MSE) and num_subtasks=1, so every
ensemble member outputs one value. Optimizer as qipm's critic_tx: clip_by_global_norm(1000) then
AdamW(lr 3e-4, weight decay 1e-5, eps 1e-8), constant schedule, batch 256.

The action input is the policy's 8-step chunk a_{t:t+8} (56-d), as the qipm critic sees it; steps
after the episode end are zero-padded. Samples are the policy decision points (chunk_pos == 0) by
default, the states where best-of-N would score a chunk. The target is
G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k with the logged per-step reward and gamma = 0.99; the episode
ends at success or at the 400-step limit, and no bootstrap is added past a truncation.

Split by episode index, the same for every policy (episode e has the same start state for every
policy): episodes < 288 train, 288..319 validation, >= 320 test. The checkpoint with the lowest
validation MSE is the one reported. Test metrics are per group (square ph/mh x N demos, the three
policy seeds pooled), per N with ph and mh pooled, and overall:
  mse, rmse, var_G (MSE of predicting the group's mean return), r2, bias = mean(Q - G),
  calibration: 10 quantile bins of Q, ece = count-weighted mean |mean Q - mean G| per bin,
  slope and intercept of the least-squares fit G ~ Q (1 and 0 when calibrated),
  auc_success: AUC of Q at the episode's first decision for predicting episode success.

    python scripts/train_mc_critic.py --data_dir <rollouts> --out_dir <dir> --seed 0
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

POLICY_RE = re.compile(r"square_(ph|mh)_(\d+)demos_seed(\d+)_(sde|ode)")


# --------------------------------------------------------------------------- data

def discounted_returns(reward, episode_id, gamma):
    G = np.zeros(len(reward), np.float64)
    run = 0.0
    for t in range(len(reward) - 1, -1, -1):
        if t == len(reward) - 1 or episode_id[t + 1] != episode_id[t]:
            run = 0.0
        run = reward[t] + gamma * run
        G[t] = run
    return G.astype(np.float32)


def chunk_actions(action, episode_id, horizon):
    """(T, horizon * act_dim): a_{t:t+horizon} inside the episode, zeros after its end."""
    T, A = action.shape
    out = np.zeros((T, horizon, A), np.float32)
    idx = np.arange(T)
    for k in range(horizon):
        j = np.minimum(idx + k, T - 1)
        valid = (idx + k < T) & (episode_id[j] == episode_id)
        out[valid, k] = action[j[valid]]
    return out.reshape(T, horizon * A)


def load_dataset(data_dir, gamma, horizon, points):
    cols = {k: [] for k in ("obs", "act", "G", "episode", "timestep", "ep_success", "group", "n_demos",
                            "dataset", "policy")}
    policies = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        m = POLICY_RE.fullmatch(os.path.basename(p)[:-4])
        if not m:
            continue
        ds, n, seed, _ = m.groups()
        z = np.load(p)
        ep = z["episode_id"]
        G = discounted_returns(z["reward"].astype(np.float64), ep, gamma)
        acts = chunk_actions(z["action"], ep, horizon)
        keep = z["chunk_pos"] == 0 if points == "decision" else np.ones(len(ep), bool)
        succ = z["ep_success"][ep]
        pi = len(policies)
        policies.append(os.path.basename(p)[:-4])
        cols["obs"].append(z["obs"][keep]); cols["act"].append(acts[keep]); cols["G"].append(G[keep])
        cols["episode"].append(ep[keep]); cols["timestep"].append(z["timestep"][keep])
        cols["ep_success"].append(succ[keep])
        k = int(keep.sum())
        cols["group"].append(np.full(k, f"{ds}-{int(n)}"))
        cols["n_demos"].append(np.full(k, int(n))); cols["dataset"].append(np.full(k, ds))
        cols["policy"].append(np.full(k, pi))
    assert policies, f"no rollout files in {data_dir}"
    return {k: np.concatenate(v) for k, v in cols.items()}, policies


# --------------------------------------------------------------------------- metrics

def auc(score, label):
    label = label.astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(q, G, timestep, ep_success, n_bins=10):
    err = q - G
    mse = float(np.mean(err ** 2)); var = float(np.var(G))
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, n_bins - 1)
    bins = []
    for i in range(n_bins):
        m = b == i
        if m.any():
            bins.append({"n": int(m.sum()), "mean_q": float(q[m].mean()), "mean_G": float(G[m].mean())})
    ece = sum(x["n"] * abs(x["mean_q"] - x["mean_G"]) for x in bins) / len(q)
    slope, intercept = np.polyfit(q, G, 1) if np.std(q) > 0 else (float("nan"), float("nan"))
    first = timestep == 0
    return {"n": int(len(q)), "mse": mse, "rmse": float(np.sqrt(mse)), "var_G": var,
            "r2": float(1 - mse / var) if var > 0 else None, "bias": float(err.mean()),
            "mean_G": float(G.mean()), "ece": float(ece), "slope": float(slope),
            "intercept": float(intercept), "auc_success": auc(q[first], ep_success[first]),
            "success_rate": float(ep_success[first].mean()) if first.any() else None, "bins": bins}


def grouped_metrics(q, d, idx):
    out = {"overall": metrics(q, d["G"][idx], d["timestep"][idx], d["ep_success"][idx])}
    for key, col in (("group", d["group"][idx]), ("n_demos", d["n_demos"][idx])):
        out[f"by_{key}"] = {}
        for g in sorted(set(col.tolist()), key=lambda s: (str(s).split("-")[0], int(str(s).split("-")[-1]))):
            m = col == g
            out[f"by_{key}"][str(g)] = metrics(q[m], d["G"][idx][m], d["timestep"][idx][m],
                                               d["ep_success"][idx][m])
    return out


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip_grad_norm", type=float, default=1000.0)
    ap.add_argument("--num_qs", type=int, default=10)
    ap.add_argument("--hidden_dims", default="512,512,512,512")
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--points", choices=("decision", "all"), default="decision")
    ap.add_argument("--val_start", type=int, default=288)
    ap.add_argument("--test_start", type=int, default=320)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--wandb_project", default="memory-critic")
    ap.add_argument("--wandb_group", default="critic_mc_nomem")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import optax
    from ogpo.networks.critics import Value

    t0 = time.time()
    d, policies = load_dataset(args.data_dir, args.discount, args.horizon, args.points)
    ep = d["episode"]
    tr = np.flatnonzero(ep < args.val_start)
    va = np.flatnonzero((ep >= args.val_start) & (ep < args.test_start))
    te = np.flatnonzero(ep >= args.test_start)
    print(f"[mc-critic] {len(policies)} policies, {len(ep)} samples ({args.points} points): "
          f"train {len(tr)} val {len(va)} test {len(te)}; obs {d['obs'].shape[1]} act {d['act'].shape[1]} "
          f"G in [{d['G'].min():.1f}, {d['G'].max():.1f}] ({time.time() - t0:.0f}s)", flush=True)

    critic = Value(hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")), layer_norm=True,
                   num_ensembles=args.num_qs, critic_loss_type="mse", num_subtasks=1)
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    params = critic.init(k_init, jnp.asarray(d["obs"][:1]), jnp.asarray(d["act"][:1]))
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm),
                     optax.adamw(args.lr, weight_decay=args.weight_decay, eps=1e-8))
    opt_state = tx.init(params)

    obs_tr, act_tr, G_tr = (jax.device_put(d[k][tr]) for k in ("obs", "act", "G"))

    def loss_fn(p, o, a, g):
        q = critic.apply(p, o, a)  # (M, B)
        return jnp.mean((q - g[None]) ** 2)

    @jax.jit
    def train_step(p, s, k):
        i = jax.random.randint(k, (args.batch_size,), 0, obs_tr.shape[0])
        loss, grads = jax.value_and_grad(loss_fn)(p, obs_tr[i], act_tr[i], G_tr[i])
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def predict_batch(p, o, a):
        return critic.apply(p, o, a).mean(0)

    def predict(p, idx, bs=65536):
        return np.concatenate([np.asarray(predict_batch(p, jnp.asarray(d["obs"][idx[i:i + bs]]),
                                                        jnp.asarray(d["act"][idx[i:i + bs]])))
                               for i in range(0, len(idx), bs)])

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                             name=args.run_name or os.path.basename(os.path.normpath(args.out_dir)),
                             config={**vars(args), "policies": policies, "n_train": len(tr),
                                     "n_val": len(va), "n_test": len(te)})
        except Exception as exc:  # noqa: BLE001
            print(f"[mc-critic] W&B disabled ({type(exc).__name__}: {exc})", flush=True)

    best = {"val_mse": float("inf"), "step": None}
    history = []
    losses = []
    for step in range(1, args.steps + 1):
        key, k = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, k)
        losses.append(loss)
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = float(np.mean(jax.device_get(losses))); losses = []
            qv = predict(params, va); val_mse = float(np.mean((qv - d["G"][va]) ** 2))
            qt = predict(params, te); test_mse = float(np.mean((qt - d["G"][te]) ** 2))
            history.append({"step": step, "train_loss": train_loss, "val_mse": val_mse, "test_mse": test_mse})
            if val_mse < best["val_mse"]:
                best = {"val_mse": val_mse, "step": step}
                with open(os.path.join(args.out_dir, "critic_best.pkl"), "wb") as f:
                    pickle.dump({"params": jax.device_get(params), "step": step, "config": vars(args)}, f)
            print(f"[mc-critic] step {step}: train {train_loss:.3f} val {val_mse:.3f} test {test_mse:.3f} "
                  f"(best val step {best['step']}, {time.time() - t0:.0f}s)", flush=True)
            if run is not None:
                run.log({"train/loss": train_loss, "val/mse": val_mse, "test/mse": test_mse}, step=step)

    with open(os.path.join(args.out_dir, "critic_best.pkl"), "rb") as f:
        best_params = pickle.load(f)["params"]
    qt = predict(best_params, te)
    res = {"best_step": best["step"], "best_val_mse": best["val_mse"], "history": history,
           "test": grouped_metrics(qt, d, te), "config": vars(args), "policies": policies}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(args.out_dir, "test_predictions.npz"), q=qt, G=d["G"][te], group=d["group"][te],
             policy=d["policy"][te], episode=d["episode"][te], timestep=d["timestep"][te],
             ep_success=d["ep_success"][te])

    rows = res["test"]["by_group"]
    print(f"\n[mc-critic] test metrics at best val step {best['step']}")
    print(f"{'group':>8} {'n':>7} {'mse':>8} {'var_G':>8} {'r2':>6} {'bias':>7} {'ece':>6} {'slope':>6} {'auc':>6}")
    for g, r in list(rows.items()) + [("all", res["test"]["overall"])]:
        a = "  -  " if r["auc_success"] is None else f"{r['auc_success']:.3f}"
        print(f"{g:>8} {r['n']:>7} {r['mse']:>8.2f} {r['var_G']:>8.2f} {r['r2']:>6.3f} {r['bias']:>7.2f} "
              f"{r['ece']:>6.2f} {r['slope']:>6.3f} {a:>6}", flush=True)
    if run is not None:
        for g, r in rows.items():
            for k in ("mse", "r2", "bias", "ece", "slope", "auc_success"):
                if r[k] is not None:
                    run.summary[f"test/{g}/{k}"] = r[k]
        run.summary["best_step"] = best["step"]
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
