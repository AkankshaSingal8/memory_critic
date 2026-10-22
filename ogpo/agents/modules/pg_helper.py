from typing import Tuple, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
import distrax

def sample_flow_actions_ode(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    """Sample actions using deterministic ODE flow (Euler integration)."""
    batch_size = observations.shape[0]

    # Initialize with noise
    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    dt = 1.0 / flow_steps

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels = actor_fn(observations, actions, t, is_encoded=is_encoded)
        actions = actions + vels * dt

        if clip_intermediate:
            actions = jnp.clip(actions, -clip_value, clip_value)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def sample_flow_actions_ode_with_correction(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        constant_noise_std: float,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    """
    Sample actions using ODE flow with SDE-to-ODE drift correction.
    The correction term is: v_ode = v_sde + (sigma / (2 * dt)) * z_pred
    """
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    dt = 1.0 / flow_steps
    correction_coeff = constant_noise_std / (2 * dt) if dt > 0 else 0.0

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels, z_pred = actor_fn(observations, actions, t, is_encoded=is_encoded, return_denoiser=True)

        # SDE-to-ODE correction
        vels = vels + correction_coeff * z_pred
        actions = actions + vels * dt

        if clip_intermediate:
            actions = jnp.clip(actions, -clip_value, clip_value)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def sample_flow_actions_sde(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        randn_clip_value: float = 3.0,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        is_encoded: bool = False,
        params: Any = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample actions using stochastic SDE flow with noise injection.

    OGPO/BPTT-specific: Injects noise at each step using a learned noise network
    or constant noise. Returns the full chain and log probabilities for
    policy gradient computation.
    """
    batch_size = observations.shape[0]

    rng, key = jax.random.split(rng)
    actions = jax.random.normal(key, (batch_size, act_dim))
    chains = [actions]
    sigmas = []

    # Initial log prob: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob = init_dist.log_prob(actions).sum(-1)

    dt = 1.0 / flow_steps

    for i in range(flow_steps):
        rng, key = jax.random.split(rng)
        t = jnp.full((batch_size, 1), i / flow_steps)

        # Get velocity
        if params is not None:
            vels = actor_fn(observations, actions, t, params=params, is_encoded=is_encoded)
        else:
            vels = actor_fn(observations, actions, t, is_encoded=is_encoded)

        # Compute mean of next action
        mean_next = actions + vels * dt

        if clip_intermediate:
            mean_next = jnp.clip(mean_next, -clip_value, clip_value)

        # Get noise std
        if use_constant_noise:
            sigma = constant_noise_std * jnp.ones(actions.shape)
        else:
            if params is not None:
                sigma = noise_fn(observations, t, params=params)
            else:
                sigma = noise_fn(observations, t)
            sigma = jnp.maximum(sigma, min_noise_std)

        sigmas.append(sigma)

        # Sample from distribution
        step_dist = distrax.Normal(mean_next, sigma)
        rng, noise_key = jax.random.split(rng)
        actions = step_dist.sample(seed=noise_key)

        actions = jnp.clip(actions, mean_next - randn_clip_value * sigma, mean_next + randn_clip_value * sigma)
        logprob = logprob + step_dist.log_prob(actions).sum(-1)
        
        if i == flow_steps - 1:
            actions = jnp.clip(actions, act_min, act_max)

        chains.append(actions)

    chains = jnp.stack(chains, axis=1)
    sigmas = jnp.stack(sigmas, axis=1)

    return actions, chains, logprob, sigmas


def compute_flow_log_prob(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        flow_steps: int,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        normalize_horizon: bool = False,
        normalize_dim: bool = False,
        is_encoded: bool = False,
        params: Any = None,
        get_entropy: bool = False,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    """ Compute log probability for a flow chain (given actions).
    Used for policy gradient computation - given a chain of actions,
    compute the log probability under the current policy.
    """
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]
    dt = 1.0 / flow_steps

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    # Initial probability: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob_init = init_dist.log_prob(chains[:, 0]).sum(-1)
    logprob = logprob + logprob_init
    logprob_steps += 1

    if get_entropy:
        entropy_init = init_dist.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_init

    # Pre-compute chain slices
    chains_prev = chains[:, :-1, :]  # [B, flow_steps, act_dim]
    chains_next = chains[:, 1:, :]   # [B, flow_steps, act_dim]

    # Compute velocities and sigmas for all steps
    chains_vel = jnp.zeros_like(chains_prev)
    chains_stds = jnp.zeros_like(chains_prev)

    for i in range(flow_steps):
        t = jnp.full((batch_size, 1), i / flow_steps)
        x_t = chains[:, i, :]

        if params is not None:
            v = actor_fn(observations, x_t, t, params=params, is_encoded=is_encoded)
        else:
            v = actor_fn(observations, x_t, t, is_encoded=is_encoded)

        if use_constant_noise:
            sigma = constant_noise_std * jnp.ones((batch_size, act_dim))
        else:
            if params is not None:
                sigma = noise_fn(observations, t, params=params)
            else:
                sigma = noise_fn(observations, t)
            sigma = jnp.maximum(sigma, min_noise_std)

        chains_vel = chains_vel.at[:, i, :].set(v)
        chains_stds = chains_stds.at[:, i, :].set(sigma)
        logprob_steps += 1

    # Compute means for all transitions
    chains_mean = chains_prev + chains_vel * dt

    # Clip intermediate means to match SDE sampling behavior
    if clip_intermediate:
        chains_mean = jnp.clip(chains_mean, -clip_value, clip_value)

    # Create distribution for all transitions
    chains_dist = distrax.Normal(chains_mean, chains_stds)

    # Compute log probabilities
    logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
    logprob = logprob + logprob_trans.sum(axis=-1)

    if get_entropy:
        entropy_trans = chains_dist.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_trans.sum(axis=-1)
        entropy_rate = joint_entropy / logprob_steps / act_dim
    else:
        entropy_rate = None

    # Normalization
    if normalize_horizon:
        logprob = logprob / logprob_steps

    if normalize_dim:
        logprob = logprob / act_dim

    info = {
        'vel_mean': chains_vel.mean(),
        'sigma_mean': chains_stds.mean(),
        'logprob_mean': logprob.mean(),
        'logprob_steps': logprob_steps,
    }

    if get_entropy:
        info['joint_entropy'] = joint_entropy.mean()
        info['entropy_rate'] = entropy_rate.mean()

    return logprob, entropy_rate, info


def compute_ppo_loss(log_probs: jnp.ndarray, old_log_probs: jnp.ndarray, advantages: jnp.ndarray, config) -> Tuple[jnp.ndarray, Dict]:
    """ Compute PPO clipped surrogate loss """
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    lower_bound = 1.0 - config.clip_epsilon * config.clip_min_epsilon_multiplier
    upper_bound = 1.0 + config.clip_epsilon * config.clip_min_epsilon_multiplier

    ratio_clipped_upper = jnp.mean(ratio > upper_bound)
    ratio_clipped_lower = jnp.mean(ratio < lower_bound)
    ratio_clipped_total = ratio_clipped_upper + ratio_clipped_lower

    clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)

    if config.adv_clip_min is not None:
        advantages = jnp.clip(advantages, config.adv_clip_min)

    # PPO loss
    pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    stats = {
        'ratio': ratio.mean(),
        'ratio_std': ratio.std(),
        'ratio_min': ratio.min(),
        'ratio_max': ratio.max(),
        'approx_kl': approx_kl,
        'ratio_clipped_upper': ratio_clipped_upper,
        'ratio_clipped_lower': ratio_clipped_lower,
        'ratio_clipped_total': ratio_clipped_total,
    }

    return pg_loss, stats


def _safe_max(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Conservative aggregation across Q-ensemble (branch-free).

    safeMax(x_1,...,x_k) =
        min_i x_i   if  min_i x_i > 0   (all agree positive)
        max_i x_i   if  max_i x_i < 0   (all agree negative)
        0           otherwise            (disagreement on sign)
    """
    x_min = x.min(axis=axis)
    x_max = x.max(axis=axis)
    return x_min * (x_min > 0) + x_max * (x_max < 0)

# TODO: Penultimate min, max 
def _penultimate_safe_max(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Conservative aggregation across Q-ensemble using penultimate values.

    penultimateSafeMax(x_1,...,x_k) =
        second_min_i x_i   if  min_i x_i > 0   (all agree positive)
        second_max_i x_i   if  max_i x_i < 0   (all agree negative)
        0                  otherwise            (disagreement on sign)
    """
    x_sorted = jnp.sort(x, axis=axis)  # sorted ascending along M
    x_min = x_sorted.take(0, axis=axis)
    x_max = x_sorted.take(-1, axis=axis)
    x_min2 = x_sorted.take(1, axis=axis)    # second smallest
    x_max2 = x_sorted.take(-2, axis=axis)  # second largest
    return x_min2 * (x_min > 0) + x_max2 * (x_max < 0)

def compute_ogpo_advantages(q_agg: jnp.ndarray, q_full: jnp.ndarray, config) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute advantages with different strategies for OGPO."""
    if config.adv_strategy == 'vanilla':
        if config.normalize_group:
            advantages = (q_agg - q_agg.mean(axis=0, keepdims=True)) / (q_agg.std(axis=0, keepdims=True) + 1e-8)
        else:
            advantages = q_agg - q_agg.mean(axis=0, keepdims=True)
        return advantages, q_agg

    elif config.adv_strategy == 'conservative':
        # Per-Q-network advantage: A_{i,m} = Q_m(s,a_i) - (1/G) sum_{i'} Q_m(s,a_{i'})
        adv_per_q = q_full - q_full.mean(axis=0, keepdims=True)  # [G, M, batch]
        return _safe_max(adv_per_q, axis=1), q_full.mean(axis=1)
        
    elif config.adv_strategy == 'penultimate':
        # Per-Q-network advantage: A_{i,m} = Q_m(s,a_i) - (1/G) sum_{i'} Q_m(s,a_{i'})
        adv_per_q = q_full - q_full.mean(axis=0, keepdims=True)  # [G, M, batch]
        return _penultimate_safe_max(adv_per_q, axis=1), q_full.mean(axis=1)

    if config.adv_strategy == 'awr':
        advantages = q_agg - q_agg.mean(axis=0, keepdims=True)
        return jnp.exp(advantages / config.awr_beta), q_agg
    else:
        raise ValueError(f"Unknown adv_strategy: {config.adv_strategy}")


# ===================== FPO / FPO++ =====================

def compute_spo_penalty(
    ratios: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_epsilon: float,
) -> jnp.ndarray:
    """SPO quadratic restoring penalty for ratios outside the trust region.

    Unlike PPO which zeros the gradient when clipped, SPO provides a quadratic
    penalty that pushes the ratio back toward the trust region.
    """
    abs_adv = jnp.abs(advantages)
    upper_excess = jnp.maximum(0.0, ratios - (1.0 + clip_epsilon))
    lower_excess = jnp.maximum(0.0, (1.0 - clip_epsilon) - ratios)
    excess = upper_excess + lower_excess  # one is always 0
    return jnp.square(excess) * abs_adv / (4.0 * clip_epsilon)


def compute_fpo_cfm_ratios(
    old_actor_fn: Callable,
    new_actor_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    n_mc: int,
    rng: jnp.ndarray,
    per_sample: bool,
    cfm_loss_clamp: float,
    cfm_diff_clamp: float,
) -> Tuple[jnp.ndarray, dict]:
    """Compute FPO importance ratio via CFM loss difference: exp(L_old - L_new).

    Args:
        old_actor_fn: callable(obs, x_tau, t) -> velocity, uses target (frozen) params.
        new_actor_fn: callable(obs, x_tau, t) -> velocity, uses current (grad) params.
        observations: [batch, obs_dim], pre-encoded.
        actions: [batch, act_dim], clean actions sampled via ODE.
        n_mc: number of (tau, eps) Monte Carlo pairs.
        rng: JAX PRNGKey.
        per_sample: if True, return per-MC-sample ratios [n_mc, batch].
        cfm_loss_clamp: clamp individual CFM losses for stability.
        cfm_diff_clamp: clamp loss difference before exp.

    Returns:
        ratios: [n_mc, batch] if per_sample else [batch].
        stats: dict with cfm_old_mean, cfm_new_mean, ratio_mean, ratio_std.
    """
    batch_size = observations.shape[0]
    act_dim = actions.shape[-1]

    # 1. Draw shared MC samples
    rng, tau_key, eps_key = jax.random.split(rng, 3)
    taus = jax.random.uniform(tau_key, (n_mc,))                      # [n_mc]
    eps = jax.random.normal(eps_key, (n_mc, batch_size, act_dim))    # [n_mc, batch, act_dim]

    # 2. Compute noised actions and velocity targets
    # x_tau = tau * actions + (1 - tau) * eps
    taus_expanded = taus[:, None, None]                              # [n_mc, 1, 1]
    x_tau = taus_expanded * actions[None] + (1.0 - taus_expanded) * eps   # [n_mc, batch, act_dim]
    vel_target = actions[None] - eps                                      # [n_mc, batch, act_dim]

    # 3. Compute CFM loss for each MC sample
    def single_mc_cfm_loss(x_tau_i, vel_target_i, tau_i, actor_fn):
        t = jnp.full((batch_size, 1), tau_i)
        v_pred = actor_fn(observations, x_tau_i, t)
        return jnp.sum(jnp.square(v_pred - vel_target_i), axis=-1)  # [batch]

    vmapped_cfm = jax.vmap(single_mc_cfm_loss, in_axes=(0, 0, 0, None))

    cfm_old = vmapped_cfm(x_tau, vel_target, taus, old_actor_fn)    # [n_mc, batch]
    cfm_new = vmapped_cfm(x_tau, vel_target, taus, new_actor_fn)    # [n_mc, batch]

    # 4. Clamp individual losses
    cfm_old = jnp.clip(cfm_old, 0.0, cfm_loss_clamp)
    cfm_new = jnp.clip(cfm_new, 0.0, cfm_loss_clamp)

    # 5. Compute ratios
    if per_sample:
        diff = cfm_old - cfm_new                                    # [n_mc, batch]
        diff = jnp.clip(diff, -cfm_diff_clamp, cfm_diff_clamp)
        ratios = jnp.exp(diff)                                      # [n_mc, batch]
    else:
        diff = cfm_old.mean(axis=0) - cfm_new.mean(axis=0)          # [batch]
        diff = jnp.clip(diff, -cfm_diff_clamp, cfm_diff_clamp)
        ratios = jnp.exp(diff)                                      # [batch]

    stats = {
        'cfm_old_mean': cfm_old.mean(),
        'cfm_new_mean': cfm_new.mean(),
        'ratio_mean': ratios.mean(),
        'ratio_std': ratios.std(),
    }

    return ratios, stats


def compute_fpo_ppo_loss(
    ratios: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_epsilon: float,
    per_sample: bool,
    use_aspo: bool,
    clip_min_epsilon_multiplier: float = 1.0,
    adv_clip_min: float = None,
) -> Tuple[jnp.ndarray, dict]:
    """PPO loss with FPO ratios, optional per-sample clipping and ASPO.

    Args:
        ratios: [n_mc, G*batch] if per_sample, else [G*batch]. Direct ratios (not log).
        advantages: [G*batch].
        clip_epsilon: PPO clip epsilon.
        per_sample: whether ratios have an n_mc leading dimension.
        use_aspo: if True, use asymmetric SPO for negative advantages.
        clip_min_epsilon_multiplier: multiplier for clip bounds.
        adv_clip_min: optional lower clamp for advantages.
    """
    lower_bound = 1.0 - clip_epsilon * clip_min_epsilon_multiplier
    upper_bound = 1.0 + clip_epsilon * clip_min_epsilon_multiplier

    if adv_clip_min is not None:
        advantages = jnp.clip(advantages, adv_clip_min)

    if per_sample:
        adv = advantages[None, :]                                    # [1, G*batch]
    else:
        adv = advantages

    clipped_ratios = jnp.clip(ratios, lower_bound, upper_bound)

    if use_aspo:
        # PPO branch for positive advantages
        ppo_obj = jnp.minimum(ratios * adv, clipped_ratios * adv)
        # SPO branch for negative advantages
        spo_penalty = compute_spo_penalty(ratios, adv, clip_epsilon)
        spo_obj = ratios * adv - spo_penalty
        # Select based on advantage sign
        pos_mask = adv > 0
        objective = jnp.where(pos_mask, ppo_obj, spo_obj)
    else:
        objective = jnp.minimum(ratios * adv, clipped_ratios * adv)

    pg_loss = -jnp.mean(objective)

    # Stats (compatible with compute_ppo_loss output keys)
    ratio_flat = ratios.reshape(-1)
    approx_kl = jnp.float32(0.0)  # Not meaningful for FPO (no log-prob ratio)
    stats = {
        'ratio': ratio_flat.mean(),
        'ratio_std': ratio_flat.std(),
        'ratio_min': ratio_flat.min(),
        'ratio_max': ratio_flat.max(),
        'approx_kl': approx_kl,
        'ratio_clipped_upper': jnp.mean(ratio_flat > upper_bound),
        'ratio_clipped_lower': jnp.mean(ratio_flat < lower_bound),
        'ratio_clipped_total': jnp.mean(ratio_flat > upper_bound) + jnp.mean(ratio_flat < lower_bound),
    }

    return pg_loss, stats


# MIP Stuff
def sample_mip_actions_sde(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        mip_t_star: float,
        act_dim: int,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        randn_clip_value: float = 3.0,
        act_min: float = -1.0,
        act_max: float = 1.0,
        is_encoded: bool = False,
        params: Any = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample actions using MIP with stochastic noise injection.
    MIP (Minimal Iterative Policy) uses 2 steps with noise injection.
    """
    batch_size = observations.shape[0]

    rng, key = jax.random.split(rng)
    x_0 = jax.random.normal(key, (batch_size, act_dim))
    chains = [x_0]
    sigmas = []

    # Initial log prob: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob = init_dist.log_prob(x_0).sum(-1)

    # Step 1: t=0
    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_0 = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_0 = noise_fn(observations, t_0, params=params)
        else:
            sigma_0 = noise_fn(observations, t_0)
        sigma_0 = jnp.maximum(sigma_0, min_noise_std)

    sigmas.append(sigma_0)

    step_0_dist = distrax.Normal(mean_0, sigma_0)
    rng, noise_key = jax.random.split(rng)
    a_0_hat = step_0_dist.sample(seed=noise_key)

    a_0_hat = jnp.clip(
        a_0_hat,
        mean_0 - randn_clip_value * sigma_0,
        mean_0 + randn_clip_value * sigma_0
    )

    logprob = logprob + step_0_dist.log_prob(a_0_hat).sum(-1)
    chains.append(a_0_hat)

    # Step 2: t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(observations, a_0_hat, t_star, params=params, is_encoded=is_encoded)
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_final = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_final = noise_fn(observations, t_star, params=params)
        else:
            sigma_final = noise_fn(observations, t_star)
        sigma_final = jnp.maximum(sigma_final, min_noise_std)

    sigmas.append(sigma_final)

    step_final_dist = distrax.Normal(mean_final, sigma_final)
    rng, noise_key = jax.random.split(rng)
    actions = step_final_dist.sample(seed=noise_key)

    actions = jnp.clip(
        actions,
        mean_final - randn_clip_value * sigma_final,
        mean_final + randn_clip_value * sigma_final
    )

    logprob = logprob + step_final_dist.log_prob(actions).sum(-1)

    actions = jnp.clip(actions, act_min, act_max)
    chains.append(actions)

    chains = jnp.stack(chains, axis=1)
    sigmas = jnp.stack(sigmas, axis=1)

    return actions, chains, logprob, sigmas


def sample_mip_actions_ode(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        mip_t_star: float,
        act_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    """
    Sample actions using MIP (Minimal Iterative Policy) - 2-step flow.
    MIP uses only two steps: t=0 and t=t* (typically 0.9).
    """
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    # Step 1: t=0
    t_0 = jnp.zeros((batch_size, 1))
    a_0_hat = actor_fn(observations, actions, t_0, is_encoded=is_encoded)

    # Step 2: t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    actions = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    actions = jnp.clip(actions, act_min, act_max)
    return actions


def compute_mip_log_prob(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        mip_t_star: float,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        normalize_horizon: bool = False,
        normalize_dim: bool = False,
        is_encoded: bool = False,
        params: Any = None,
        get_entropy: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    """
    Compute log probability for an MIP chain.
    MIP chains have shape [B, 3, act_dim] -> (x_0, a_0_hat, a_final)
    """
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]

    x_0 = chains[:, 0, :]
    a_0_hat = chains[:, 1, :]
    a_final = chains[:, 2, :]

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    # Initial probability: p(x_0) ~ N(0, I)
    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob_init = init_dist.log_prob(x_0).sum(-1)
    logprob = logprob + logprob_init
    logprob_steps += 1

    if get_entropy:
        entropy_init = init_dist.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_init

    # Step 1: Transition from x_0 to a_0_hat at t=0
    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_0 = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_0 = noise_fn(observations, t_0, params=params)
        else:
            sigma_0 = noise_fn(observations, t_0)
        sigma_0 = jnp.maximum(sigma_0, min_noise_std)

    dist_0 = distrax.Normal(mean_0, sigma_0)
    logprob_0 = dist_0.log_prob(a_0_hat).sum(-1)
    logprob = logprob + logprob_0
    logprob_steps += 1

    if get_entropy:
        entropy_0 = dist_0.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_0

    # Step 2: Transition from a_0_hat to a_final at t=t*
    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(observations, a_0_hat, t_star, params=params, is_encoded=is_encoded)
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_final = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_final = noise_fn(observations, t_star, params=params)
        else:
            sigma_final = noise_fn(observations, t_star)
        sigma_final = jnp.maximum(sigma_final, min_noise_std)

    dist_final = distrax.Normal(mean_final, sigma_final)
    logprob_final = dist_final.log_prob(a_final).sum(-1)
    logprob = logprob + logprob_final
    logprob_steps += 1

    if get_entropy:
        entropy_final = dist_final.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_final
        entropy_rate = joint_entropy / logprob_steps / act_dim
    else:
        entropy_rate = None

    # Normalization
    if normalize_horizon:
        logprob = logprob / logprob_steps

    if normalize_dim:
        logprob = logprob / act_dim

    info = {
        'mean_0': mean_0.mean(),
        'mean_final': mean_final.mean(),
        'sigma_0': sigma_0.mean(),
        'sigma_final': sigma_final.mean(),
        'logprob_mean': logprob.mean(),
        'logprob_steps': logprob_steps,
    }

    if get_entropy:
        info['joint_entropy'] = joint_entropy.mean()
        info['entropy_rate'] = entropy_rate.mean()

    return logprob, entropy_rate, info

