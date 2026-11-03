"""Episode history for the memory critics (docs/ogpo_memory_critic_plan.md, section 1).

Samples are decision points (chunk_pos == 0) from scripts/train_mc_critic.load_dataset, ordered by
policy, then episode, then time. Decision i of an episode is at timestep horizon * k_i. Its event is
the chunk executed from it: obs_{t..t+h-1} and the executed actions a_{t..t+h-1}, flattened and
zero-padded past the episode end. So the event never holds obs_{t+h}, the next decision's obs.

The memory of decision k is the events of decisions 0..k-1 of the same episode (or, for SHUF, of a
donor episode of the same policy and split that also reached decision k), at most `window` of the
most recent. Decision 0 has an empty memory.
"""
import numpy as np

MAX_EVENTS = 50          # 400-step limit / 8-step chunks


def episode_index(policy, episode, timestep, horizon):
    """Per sample: global index of its episode's first decision, and its decision index k."""
    n = len(policy)
    idx = np.arange(n)
    new = np.ones(n, bool)
    new[1:] = (policy[1:] != policy[:-1]) | (episode[1:] != episode[:-1])
    ep_start = np.maximum.accumulate(np.where(new, idx, 0))
    k = idx - ep_start
    n_eps = len(set(zip(policy.tolist(), episode.tolist())))
    assert int(new.sum()) == n_eps, "an episode's decisions are not contiguous"
    assert (timestep == horizon * k).all(), "decisions are not every `horizon` steps from t = 0"
    return ep_start, k


def prefix(start_of_episode, k, window):
    """(start, length) of the memory: the last min(k, window) events before decision k."""
    length = np.minimum(k, window)
    return start_of_episode + k - length, length


def shuffle_donors(ep_start, k, policy, split, seed):
    """SHUF: for each (policy, split, k), a fixed derangement of the episodes that reached decision k.

    Returns the donor episode's start index per sample (its own when it is alone in its group) and the
    number of samples left with their own history.
    """
    donor = ep_start.copy()
    order = np.lexsort((k, split, policy))
    key = np.stack([policy[order], split[order], k[order]], 1)
    cut = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    rng = np.random.default_rng(seed)
    alone = 0
    for g in np.split(order, cut):
        if len(g) < 2:
            alone += len(g)
            continue
        perm = rng.permutation(len(g))
        donor[g[perm]] = ep_start[g[np.roll(perm, -1)]]
    return donor, alone


def gather(events, start, length, window, xp=np):
    """(B, window, E) memory, zero past `length`; works with numpy or jax.numpy as `xp`."""
    j = xp.arange(window)
    idx = xp.minimum(start[:, None] + j[None], events.shape[0] - 1)
    mask = (j[None] < length[:, None])[..., None]
    return xp.where(mask, events[idx], 0.0)
