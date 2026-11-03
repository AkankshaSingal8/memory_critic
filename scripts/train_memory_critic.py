"""Memory-conditioned MC critic: the arms of docs/ogpo_memory_critic_plan.md (section 2), steps 0-1.

Same data, targets, split, sampler, optimizer, head and metrics as the no-memory baseline
(scripts/train_mc_critic.py); only the memory input differs. Arms (--arm):
  m0    no memory: Value(obs, chunk). Parity check against critics/mc_decision_seed{0,1,2}.
  clk   obs + Dense(t / 400)
  gru   obs + Dense(GRU over the events of decisions 0..k-1 of this episode), at most 50 events
  shuf  as gru, but the events come from another episode of the same policy and split that also reached
        decision k (a fixed derangement, seeded by --seed); separates content from clock and policy id
  k1    as gru, last event only
An event is the chunk executed from a decision: its 8 obs and 8 executed actions (memcritic/history.py),
standardized with the train events' mean and std. The candidate chunk never enters the memory.

    python scripts/train_memory_critic.py --arm gru --out_dir <dir> --seed 0
"""
import argparse
import json
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [p for p in (REPO, os.path.join(REPO, "scripts")) if p not in sys.path]

import numpy as np

from memcritic import history
from train_mc_critic import grouped_metrics, load_dataset

ARMS = {"m0": ("none", 0, False), "clk": ("clk", 0, False), "gru": ("gru", history.MAX_EVENTS, False),
        "shuf": ("gru", history.MAX_EVENTS, True), "k1": ("gru", 1, False)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--data_dir", default="/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100")
    ap.add_argument("--policy_glob", default="*", help="rollout files to use (smoke tests)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip_grad_norm", type=float, default=1000.0)
    ap.add_argument("--num_qs", type=int, default=10)
    ap.add_argument("--hidden_dims", default="512,512,512,512")
    ap.add_argument("--mem_dim", type=int, default=64)
    ap.add_argument("--gru_dim", type=int, default=256)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--val_start", type=int, default=288)
    ap.add_argument("--test_start", type=int, default=320)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--pred_batch", type=int, default=8192)
    ap.add_argument("--wandb_project", default="memory-critic")
    ap.add_argument("--wandb_group", default="critic_memory")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args(argv)
    memory, window, shuf = ARMS[args.arm]
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import optax
    from memcritic.critics import MemoryValue

    t0 = time.time()
    d, policies = load_dataset(args.data_dir, args.discount, args.horizon, "decision", events=True,
                               policy_glob=args.policy_glob)
    ep = d["episode"]
    split = np.where(ep < args.val_start, 0, np.where(ep < args.test_start, 1, 2))
    tr, va, te = (np.flatnonzero(split == s) for s in range(3))
    ep_start, k = history.episode_index(d["policy"], ep, d["timestep"], args.horizon)
    alone = 0
    if shuf:
        ep_start, alone = history.shuffle_donors(ep_start, k, d["policy"], split, args.seed)
    start, length = history.prefix(ep_start, k, max(window, 1))
    ev_mean = d["event"][tr].mean(0)
    ev_std = d["event"][tr].std(0) + 1e-6
    events = ((d["event"] - ev_mean) / ev_std).astype(np.float32)
    print(f"[mem-critic] arm {args.arm} (memory {memory}, window {window}, shuf {shuf}): {len(policies)} "
          f"policies, {len(ep)} decisions: train {len(tr)} val {len(va)} test {len(te)}; obs "
          f"{d['obs'].shape[1]} act {d['act'].shape[1]} event {events.shape[1]}; max k {k.max()}; "
          f"shuf kept own history for {alone} ({time.time() - t0:.0f}s)", flush=True)

    critic = MemoryValue(memory, hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
                         num_qs=args.num_qs, mem_dim=args.mem_dim, gru_dim=args.gru_dim)
    W = max(window, 1)
    dev = {n: jax.device_put(x) for n, x in (("obs", d["obs"]), ("act", d["act"]), ("G", d["G"]),
                                             ("t", d["timestep"].astype(np.int32)), ("start", start),
                                             ("length", length), ("events", events))}

    def inputs(i):
        hist = history.gather(dev["events"], dev["start"][i], dev["length"][i], W, xp=jnp)
        return dev["obs"][i], dev["act"][i], hist, dev["length"][i], dev["t"][i]

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    params = critic.init(k_init, *inputs(jnp.arange(2)))
    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm),
                     optax.adamw(args.lr, weight_decay=args.weight_decay, eps=1e-8))
    opt_state = tx.init(params)
    tr_dev = jax.device_put(tr)

    def loss_fn(p, i):
        q = critic.apply(p, *inputs(i))  # (M, B)
        return jnp.mean((q - dev["G"][i][None]) ** 2)

    @jax.jit
    def train_step(p, s, kk):
        i = tr_dev[jax.random.randint(kk, (args.batch_size,), 0, tr_dev.shape[0])]
        loss, grads = jax.value_and_grad(loss_fn)(p, i)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def predict_batch(p, i):
        return critic.apply(p, *inputs(i)).mean(0)

    def predict(p, idx):
        return np.concatenate([np.asarray(predict_batch(p, jnp.asarray(idx[i:i + args.pred_batch])))
                               for i in range(0, len(idx), args.pred_batch)])

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                             name=args.run_name or os.path.basename(os.path.normpath(args.out_dir)),
                             config={**vars(args), "policies": policies, "n_train": len(tr),
                                     "n_val": len(va), "n_test": len(te)})
        except Exception as exc:  # noqa: BLE001
            print(f"[mem-critic] W&B disabled ({type(exc).__name__}: {exc})", flush=True)

    meta = {"config": vars(args), "memory": memory, "window": window, "shuf": shuf,
            "event_mean": ev_mean, "event_std": ev_std, "horizon": args.horizon}
    best = {"val_mse": float("inf"), "step": None}
    history_rows, losses = [], []
    for step in range(1, args.steps + 1):
        key, kk = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, kk)
        losses.append(loss)
        if step % args.eval_every == 0 or step == args.steps:
            train_loss = float(np.mean(jax.device_get(losses))); losses = []
            qv = predict(params, va); val_mse = float(np.mean((qv - d["G"][va]) ** 2))
            qt = predict(params, te); test_mse = float(np.mean((qt - d["G"][te]) ** 2))
            history_rows.append({"step": step, "train_loss": train_loss, "val_mse": val_mse, "test_mse": test_mse})
            if val_mse < best["val_mse"]:
                best = {"val_mse": val_mse, "step": step}
                with open(os.path.join(args.out_dir, "critic_best.pkl"), "wb") as f:
                    pickle.dump({"params": jax.device_get(params), "step": step, **meta}, f)
            print(f"[mem-critic] step {step}: train {train_loss:.3f} val {val_mse:.3f} test {test_mse:.3f} "
                  f"(best val step {best['step']}, {time.time() - t0:.0f}s)", flush=True)
            if run is not None:
                run.log({"train/loss": train_loss, "val/mse": val_mse, "test/mse": test_mse}, step=step)

    with open(os.path.join(args.out_dir, "critic_best.pkl"), "rb") as f:
        best_params = pickle.load(f)["params"]
    qt = predict(best_params, te)
    res = {"arm": args.arm, "best_step": best["step"], "best_val_mse": best["val_mse"], "history": history_rows,
           "test": grouped_metrics(qt, d, te), "config": vars(args), "policies": policies,
           "shuf_own_history": int(alone)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(args.out_dir, "test_predictions.npz"), q=qt, G=d["G"][te], group=d["group"][te],
             policy=d["policy"][te], episode=d["episode"][te], timestep=d["timestep"][te],
             ep_success=d["ep_success"][te])

    rows = res["test"]["by_group"]
    print(f"\n[mem-critic] arm {args.arm}: test metrics at best val step {best['step']}")
    print(f"{'group':>8} {'n':>7} {'mse':>8} {'var_G':>8} {'r2':>6} {'bias':>7} {'ece':>6} {'slope':>6} {'auc':>6}")
    for g, r in list(rows.items()) + [("all", res["test"]["overall"])]:
        a = "  -  " if r["auc_success"] is None else f"{r['auc_success']:.3f}"
        print(f"{g:>8} {r['n']:>7} {r['mse']:>8.2f} {r['var_G']:>8.2f} {r['r2']:>6.3f} {r['bias']:>7.2f} "
              f"{r['ece']:>6.2f} {r['slope']:>6.3f} {a:>6}", flush=True)
    if run is not None:
        for g, r in rows.items():
            for m in ("mse", "r2", "bias", "ece", "slope", "auc_success"):
                if r[m] is not None:
                    run.summary[f"test/{g}/{m}"] = r[m]
        run.summary["best_step"] = best["step"]
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
