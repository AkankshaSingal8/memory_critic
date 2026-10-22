import copy
from typing import Any, Optional, Tuple, List, NamedTuple
from functools import partial
import numpy as np

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import distrax
import pickle

from ogpo.networks.encoders import encoder_modules, vit_encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import (
    ActorVectorField, Value, ActorVectorFieldTF, ValueTF,
    ActorVectorFieldSimBa, ValueSimBa, NoiseInjectionNetwork,
)
from ogpo.networks.goal_net import GoalConditionedNetwork
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets, get_shortcut_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    get_mip_targets,
    apply_chunking_mask,
    compute_flow_bc_loss,
    compute_mip_bc_loss,
    compute_flow_bc_loss_with_denoiser,
)
from ogpo.agents.modules.maintenance import (
    create_lr_schedule,
    compute_param_norm,
    compute_param_diff_norm,
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
    create_optimizer,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    reduce_q_over_samples,
    compute_td_target,
    compute_td_loss,
    compute_calql_regularizer,
    get_q_values as _get_q_values,
    get_mip_q_targets,
    sample_mip_q_values,
    compute_mip_q_predictions,
    sample_mip_q_ensemble_values,
    compute_mip_q_ensemble_predictions,
)
from ogpo.agents.modules.fql_helper import (
    create_one_step_network,
    one_step_distillation_loss,
    sample_one_step,
    sample_one_step_BON,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
    sample_mip_actions_ode,
    sample_flow_actions_ode_with_correction,
    sample_flow_actions_sde,
    sample_mip_actions_sde,
    compute_flow_log_prob,
    compute_mip_log_prob,
    compute_ppo_loss,
    compute_ogpo_advantages,
    compute_fpo_cfm_ratios,
    compute_fpo_ppo_loss,
)

def _split_keys(rng: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Split a PRNGKey into one key and a list of n subkeys."""
    keys = jax.random.split(rng, n + 1)
    return keys[0], list(keys[1:])


class OGPOAgent(flax.struct.PyTreeNode):
    """ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning."""
    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    config: Any = nonpytree_field()
    one_step_network: Optional[TrainState] = None

    # --- Vision encoding helpers ---

    def _encode_for_actor(self, observations, images=None, params=None, use_target=False):
        """Encode observations for actor. Returns encoded obs (flat features)."""
        if not self.config.encoder:
            return observations
        module = 'target_actor' if use_target else 'actor'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        return self.actor_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)

    def _encode_for_critic(self, observations, images=None, params=None, use_target=False):
        """Encode observations for critic. Returns encoded obs (flat features)."""
        if not self.config.encoder:
            return observations
        module = 'target_critic' if use_target else 'critic'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        return self.critic_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)

    def compute_log_probs_and_entropy(
        self,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        params: flax.core.FrozenDict,
        get_entropy: bool = False,
        use_target: bool = False,
        is_encoded: bool = False,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute log-probabilities and optionally entropy using pg_helper functions.
        """
        actor_module_name = 'target_actor' if use_target else 'actor'
        noise_module_name = 'target_noise_net' if use_target else 'noise_net'

        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, params=params, use_target=use_target)

        # Create wrapper functions for the network calls
        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'mip':
            logprob, entropy_rate_est, info = compute_mip_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                mip_t_star=self.config.mip_t_star,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
            )
        else:
            logprob, entropy_rate_est, info = compute_flow_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                flow_steps=self.config.flow_steps,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
            )

        if get_entropy:
            return logprob, entropy_rate_est, info
        else:
            return logprob, None

    def compute_log_probs(self, observations: jnp.ndarray, chains: jnp.ndarray, params: flax.core.FrozenDict, use_target: bool = False,
                          is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """ Compute sum of log-probabilities over the denoising chain. """
        logprob, _ = self.compute_log_probs_and_entropy(observations, chains, params, get_entropy=False, use_target=use_target, is_encoded=is_encoded, images=images)
        return logprob

    def _sample_actions(self, obs: jnp.ndarray, rng: jnp.ndarray, use_target: bool, num_samples: int, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Sample actions from TARGET policy """
        sample_rngs = jax.random.split(rng, num_samples)
        vmapped_sample = jax.vmap(
            lambda obs_g, rng_g: self.sample_actions_with_noise(obs_g, rng=rng_g, use_target=use_target, is_encoded=True),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, use_target=use_target)
        actions, chains, old_lp, sigmas = vmapped_sample(obs_encoded, sample_rngs)

        return actions, chains, old_lp, sigmas

    def _sample_actions_ode(self, obs_encoded: jnp.ndarray, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample actions from target policy via ODE (no logprob, no chains). For FPO."""
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('target_actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        def sample_ode_single(rng_g):
            if self.config.error_correct_sde_to_ode and self.config.use_denoiser:
                return sample_flow_actions_ode_with_correction(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    constant_noise_std=self.config['constant_noise_std'],
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )
            else:
                return sample_flow_actions_ode(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )

        sample_rngs = jax.random.split(rng, num_samples)
        return jax.vmap(sample_ode_single)(sample_rngs)  # [G, batch, act_dim]

    def _compute_current_log_probs(self, obs: jnp.ndarray, chains: jnp.ndarray, grad_params: flax.core.FrozenDict, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """ Compute log probabilities under CURRENT policy for actions sampled from target policy """
        vmapped_logprob = jax.vmap(
            lambda obs_g, chains_g: self.compute_log_probs_and_entropy(
                obs_g, chains_g, grad_params, get_entropy=True, is_encoded=True
            ),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)
        lp, entropy_rate_est, info_vmapped = vmapped_logprob(obs_encoded, chains)

        return lp, entropy_rate_est, info_vmapped

    def _compute_advantages(self, obs: jnp.ndarray, actions: jnp.ndarray, rng: jnp.ndarray, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Compute GRPO-style advantages from Q-values - supports MIP-Q and ensemble-Q. """
        critic_obs_enc = self._encode_for_critic(obs, images=images, use_target=True)

        if self.config.get('use_mip_q', False):
            # MIP-Q path: two-step sampling
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = critic_obs_enc.shape[0]

            if self.config.get('mip_q_ensemble', False):
                # Explicit ensemble: Single noise for all actions
                rng, noise_rng = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_rng,
                    (batch_size, 1),
                    minval=-noise_scale,
                    maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_ensemble_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_sample=noise_sample,
                        mip_t_star=mip_t_star,
                        is_encoded=bool(self.config.encoder)
                    )  # [num_ensemble_members, batch]

                    if self.config['q_agg'] == 'min':
                        q_agg = q_vals.min(axis=0)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                        q_agg = q_vals[subsample_idxs].min(axis=0)
                    else:
                        q_agg = q_vals.mean(axis=0)

                    return q_agg, q_vals  # [batch], [num_ensemble_members, batch]
            else:
                # Implicit ensemble: Multiple noises for all actions
                rng, noise_rng = jax.random.split(rng)
                noise_samples = jax.random.uniform(
                    noise_rng,
                    (num_ensemble_members, batch_size, 1),
                    minval=-noise_scale,
                    maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_samples=noise_samples,
                        mip_t_star=mip_t_star,
                        is_encoded=bool(self.config.encoder)
                    )  # [num_ensemble_members, batch]

                    if self.config['q_agg'] == 'min':
                        q_agg = q_vals.min(axis=0)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                        q_agg = q_vals[subsample_idxs].min(axis=0)
                    else:
                        q_agg = q_vals.mean(axis=0)

                    return q_agg, q_vals  # [batch], [num_ensemble_members, batch]
        else:
            # Original ensemble-Q path
            def compute_q_for_group(actions_g, rng_g):
                q_vals = self.critic_network.select('target_critic')(critic_obs_enc, actions_g, is_encoded=bool(self.config.encoder))  # [num_qs, batch]

                if self.config['q_agg'] == 'min':
                    q_agg = q_vals.min(axis=0)
                elif self.config['q_agg'] == 'subsample':
                    subsample_idxs = jax.random.randint(rng_g, (2,), 0, self.config.num_qs)
                    q_agg = q_vals[subsample_idxs].min(axis=0)
                else:
                    q_agg = q_vals.mean(axis=0)

                return q_agg, q_vals  # [batch], [num_qs, batch]

        subsample_rngs = jax.random.split(rng, actions.shape[0])
        vmapped_critic = jax.vmap(compute_q_for_group, in_axes=(0, 0))
        q_agg, q_full = vmapped_critic(actions, subsample_rngs)  # [G, batch], [G, M, batch]

        adv, adv_pre = compute_ogpo_advantages(q_agg, q_full, self.config)
        return adv, adv_pre

    def _compute_bc_loss(self, rng, grad_params, batch, batch_success, success_flag, obs_encoded=None, batch_success_encoded=None, images=None, images_success=None):
        rng, key = jax.random.split(rng)

        def true_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch_success, grad_params, key, obs_encoded=batch_success_encoded, images=images_success)
            return bc_loss

        def false_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch, grad_params, key, obs_encoded=obs_encoded, images=images)
            return bc_loss

        bc_loss = jax.lax.cond(success_flag, true_branch, false_branch,
            (batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success),
        )
        return bc_loss

    def actor_loss(self, batch: dict, batch_success: dict, grad_params: flax.core.FrozenDict, success_flag: bool, rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """ Compute PPO surrogate loss + entropy + BC regularization with OGPO advantage calculation. """
        # FQL mode: skip PG loss when one-step policy handles RL (flow actor only gets BC)
        skip_pg_loss = not self.config.get('fql_train_main_policy', True)

        # Default values (used when PG is skipped in FQL mode)
        pg_loss = jnp.float32(0.0)
        ent_loss = jnp.float32(0.0)
        ent = jnp.float32(0.0)
        lp = jnp.float32(0.0)
        old_lp = jnp.float32(0.0)
        adv = jnp.zeros(1)
        adv_pre = jnp.zeros(1)
        sigmas = jnp.float32(0.0)
        actions = jnp.zeros(1)
        pg_stats = {}
        info_extra = {}
        G = self.config.group_num_samples

        if not skip_pg_loss:
            # Subsample observations if needed
            obs = batch['observations']
            images = batch.get('images')
            if self.config.ppo_batch_size < self.config.batch_size:
                rng, key = jax.random.split(rng)
                idxs = jax.random.choice(key, obs.shape[0], (self.config.ppo_batch_size,), replace=False)
                obs = obs[idxs]
                if images is not None:
                    images = images[idxs]

            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)

            rng, sample_rng = jax.random.split(rng)

            if self.config.get('use_fpo', False):
                # ===== FPO PATH =====
                # Step 1: Sample actions via ODE from target policy (no chains, no logprob)
                actions = self._sample_actions_ode(obs_encoded, sample_rng, G)  # [G, batch, act_dim]

                # Step 2: Compute advantages (identical to OGPO)
                rng, adv_rng = jax.random.split(rng)
                adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, images=images)

                # Step 3: Compute FPO CFM ratios (vmapped over G groups)
                def old_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('target_actor')(obs_a, x_tau, t, is_encoded=True)
                def new_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('actor')(obs_a, x_tau, t, params=grad_params, is_encoded=True)

                rng, ratio_rng = jax.random.split(rng)
                def compute_ratio_single(actions_g, rng_g):
                    return compute_fpo_cfm_ratios(
                        old_actor_fn=old_actor_fn, new_actor_fn=new_actor_fn,
                        observations=obs_encoded, actions=actions_g,
                        n_mc=self.config.fpo_n_mc, rng=rng_g,
                        per_sample=self.config.fpo_per_sample,
                        cfm_loss_clamp=self.config.fpo_cfm_loss_clamp,
                        cfm_diff_clamp=self.config.fpo_cfm_diff_clamp,
                    )
                ratio_rngs = jax.random.split(ratio_rng, G)
                ratios, fpo_stats = jax.vmap(compute_ratio_single)(actions, ratio_rngs)
                # ratios: [G, n_mc, batch] if per_sample, [G, batch] if not

                # Step 4: PPO loss with FPO ratios
                adv_flat = adv.reshape(-1)  # [G * batch]
                if self.config.fpo_per_sample:
                    # Reshape: [G, n_mc, batch] -> [n_mc, G*batch]
                    n_mc = self.config.fpo_n_mc
                    ratios_reshaped = ratios.transpose(1, 0, 2).reshape(n_mc, -1)
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_reshaped, adv_flat, self.config.clip_epsilon,
                        per_sample=True, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )
                else:
                    ratios_flat = ratios.reshape(-1)  # [G*batch]
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_flat, adv_flat, self.config.clip_epsilon,
                        per_sample=False, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )

                # FPO has no tractable entropy
                ent = jnp.float32(0.0)
                ent_loss = jnp.float32(0.0)
                sigmas = jnp.float32(0.0)
                lp = jnp.float32(0.0)
                old_lp = jnp.float32(0.0)

                # Merge FPO-specific stats into pg_stats
                pg_stats.update({f'fpo_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in fpo_stats.items()})
                info_extra = {}

            else:
                # ===== ORIGINAL OGPO PATH (unchanged) =====
                actions, chains, old_lp, sigmas = self._sample_actions(obs, sample_rng, use_target=True, num_samples=G, obs_encoded=obs_encoded, images=images)

                rng, adv_rng = jax.random.split(rng)
                adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, images=images)
                lp, entropy_rate_est, info_vmapped = self._compute_current_log_probs(obs, chains, grad_params, obs_encoded=obs_encoded)

                lp = lp.reshape(-1)  # [G * batch]
                old_lp = old_lp.reshape(-1)  # [G * batch]
                adv = adv.reshape(-1)  # [G * batch]
                pg_loss, pg_stats = compute_ppo_loss(lp, old_lp, adv, self.config)

                ent = jnp.mean(entropy_rate_est)
                ent_loss = -ent
                info_extra = {f'info_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in info_vmapped.items()}

        # BC regularization (always runs, including FQL mode)
        bc_loss = 0.0
        if self.config.use_bc_regularization:
            rng, key = jax.random.split(rng)

            batch_images = batch.get('images')
            batch_success_images = batch_success.get('images')
            batch_obs_encoded = self._encode_for_actor(batch['observations'], images=batch_images, params=grad_params)
            batch_success_encoded = self._encode_for_actor(batch_success['observations'], images=batch_success_images, params=grad_params)

            bc_loss = self._compute_bc_loss(rng, grad_params, batch, batch_success, success_flag, obs_encoded=batch_obs_encoded, batch_success_encoded=batch_success_encoded, images=batch_images, images_success=batch_success_images)

        total_loss = pg_loss + self.config.entropy_coeff * ent_loss + self.config.bc_coeff * bc_loss
        info = {
            'pg_loss': pg_loss,
            'ent_loss': ent_loss,
            'bc_loss': bc_loss,
            'lp': lp.mean() if hasattr(lp, 'mean') else lp,
            'lp_min': lp.min() if hasattr(lp, 'min') else lp,
            'lp_max': lp.max() if hasattr(lp, 'max') else lp,
            'old_lp': old_lp.mean() if hasattr(old_lp, 'mean') else old_lp,
            'entropy': ent,
            'adv': adv.mean(),
            'adv_min': adv.min(),
            'adv_max': adv.max(),
            'adv_std': adv.std(),
            'adv_pre_norm': adv_pre.mean(),
            'adv_pre_min': adv_pre.min(),
            'adv_pre_max': adv_pre.max(),
            'q_values': adv_pre.mean(),
            'sigmas': sigmas.mean() if hasattr(sigmas, 'mean') else sigmas,
            'actions': actions.mean(),
            'actions_min': actions.min(),
            'actions_max': actions.max(),
            'actions_std': actions.std(),
            'group_num_samples': G,
            'grpo_adv_group_std': jnp.std(adv_pre, axis=0).mean(),
            'grpo_adv_between_group_std': jnp.std(adv_pre.mean(axis=0)),
            **pg_stats,
            **info_extra,
        }

        return total_loss, info

    def get_td_loss(self, batch, batch_actions, next_actions, grad_params, rng, obs_encoded=None, next_obs_encoded=None, images=None, next_images=None):
        """Compute TD loss using q_helper functions - supports both ensemble-Q and MIP-Q."""
        # Get next Q-values from target network
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        if self.config.get('use_mip_q', False):
            # MIP-Q path: two-step sampling
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = next_obs_encoded.shape[0]

            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                # Explicit ensemble: Single noise, multiple networks
                rng, noise_rng = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_rng, (batch_size, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                next_qs = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=next_obs_encoded,
                    actions=next_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # [num_ensemble_members, batch]
            else:
                # Implicit ensemble: Multiple noises, single network
                rng, noise_rng = jax.random.split(rng)
                noise_samples = jax.random.uniform(
                    noise_rng, (num_ensemble_members, batch_size, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                next_qs = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=next_obs_encoded,
                    actions=next_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # [num_ensemble_members, batch]

            rng, agg_rng = jax.random.split(rng)
            next_q = aggregate_q_values(
                next_qs,
                method=self.config['q_agg'],
                rng=agg_rng,
                num_qs=num_ensemble_members,
            )
        else:
            # Original ensemble-Q path
            next_qs = self.critic_network.select('target_critic')(next_obs_encoded, actions=next_actions, is_encoded=bool(self.config.encoder))

            rng, agg_rng = jax.random.split(rng)
            next_q = aggregate_q_values(
                next_qs,
                method=self.config['q_agg'],
                rng=agg_rng,
                num_qs=self.config.num_qs,
            )

        # Compute TD target
        target_q = compute_td_target(
            rewards=batch['rewards'],
            masks=batch['masks'],
            next_q=next_q,
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
        )

        # Get current Q-values and compute loss
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        is_enc = bool(self.config.encoder)

        if self.config.get('use_mip_q', False):
            # MIP-Q: Use centralized training function
            critic_base_fn = self.critic_network.select('critic')
            def critic_fn_with_params(obs, actions, scalar_input, time, is_encoded, return_logits):
                return critic_base_fn(
                    obs, actions=actions, scalar_input=scalar_input, time=time,
                    params=grad_params, is_encoded=is_encoded, return_logits=return_logits
                )

            if self.config.get('mip_q_ensemble', False):
                if self.config["critic_loss_type"] == "hlgauss":
                    q_0, q, q_0_logits, q_logits = compute_mip_q_ensemble_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc,
                        return_logits=True
                    )
                else:
                    q_0, q = compute_mip_q_ensemble_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc,
                        return_logits=False
                    )
                    q_0_logits = None
                    q_logits = None
            else:
                if self.config["critic_loss_type"] == "hlgauss":
                    q_0, q, q_0_logits, q_logits = compute_mip_q_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        num_ensemble_members=num_ensemble_members,
                        is_encoded=is_enc,
                        return_logits=True
                    )
                else:
                    q_0, q = compute_mip_q_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        num_ensemble_members=num_ensemble_members,
                        is_encoded=is_enc,
                        return_logits=False
                    )
                    q_0_logits = None
                    q_logits = None
        else:
            # Original ensemble-Q path
            if self.config["critic_loss_type"] == "hlgauss":
                q, q_logits = self.critic_network.select('critic')(
                    obs_encoded, actions=batch_actions,
                    params=grad_params, return_logits=True, is_encoded=is_enc
                )
            else:
                q = self.critic_network.select('critic')(
                    obs_encoded, actions=batch_actions,
                    params=grad_params, is_encoded=is_enc
                )
                q_logits = None

        # Compute loss
        valid = batch.get('valid')
        if valid is not None and valid.ndim > 1:
            valid = valid[..., -1]

        if self.config.get('use_mip_q', False):
            # MIP-Q: Two-term loss (regression at t=0 + denoising at t=t*)
            if self.config["critic_loss_type"] == "hlgauss":
                td_loss_0, stats_0 = compute_td_loss(
                    q_pred=q_0, target_q=target_q, valid_mask=valid,
                    loss_type="hlgauss",
                    q_min=self.config['q_min'], q_max=self.config['q_max'],
                    num_bins=self.config['num_bins'], q_logits=q_0_logits,
                )
                td_loss_t_star, stats_t_star = compute_td_loss(
                    q_pred=q, target_q=target_q, valid_mask=valid,
                    loss_type="hlgauss",
                    q_min=self.config['q_min'], q_max=self.config['q_max'],
                    num_bins=self.config['num_bins'], q_logits=q_logits,
                )
            else:
                td_loss_0, stats_0 = compute_td_loss(
                    q_pred=q_0, target_q=target_q, valid_mask=valid, loss_type="mse",
                )
                td_loss_t_star, stats_t_star = compute_td_loss(
                    q_pred=q, target_q=target_q, valid_mask=valid, loss_type="mse",
                )

            td_loss = td_loss_0 + td_loss_t_star
            stats = {
                'td_loss': td_loss,
                'td_loss_regression_t0': td_loss_0,
                'td_loss_denoising_tstar': td_loss_t_star,
                **{f'q_0_{k}': v for k, v in stats_0.items()},
                **{f'q_tstar_{k}': v for k, v in stats_t_star.items()},
            }
        else:
            # Original ensemble-Q path: single loss
            if self.config["critic_loss_type"] == "hlgauss":
                td_loss, stats = compute_td_loss(
                    q_pred=q, target_q=target_q, valid_mask=valid,
                    loss_type="hlgauss",
                    q_min=self.config['q_min'], q_max=self.config['q_max'],
                    num_bins=self.config['num_bins'], q_logits=q_logits,
                )
            else:
                td_loss, stats = compute_td_loss(
                    q_pred=q, target_q=target_q, valid_mask=valid, loss_type="mse",
                )

        return td_loss, q, next_q, stats

    def get_calql_diff(self, batch, batch_actions, grad_params, rng, obs_encoded=None, images=None):
        B = batch['observations'].shape[0]
        is_enc = bool(self.config.encoder)
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        q_pred = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=is_enc)

        # CQL Part
        rng, action_rng = jax.random.split(rng)
        cql_random_actions = jax.random.uniform(action_rng, shape=(B, self.config["cql_n_actions"], self.config["full_act_dim"]),
                                                minval=-1.0, maxval=1.0,)
        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        sample_rngs = jax.random.split(current_a_rng, self.config.cql_n_actions)
        vmapped_sample = jax.vmap(
            lambda obs, rng_i: self.sample_actions(obs, rng=rng_i),
            in_axes=(None, 0)
        )

        vmapped_q_fn = jax.vmap(
            lambda a: self.critic_network.select('critic')(obs_encoded, actions=a, params=grad_params, is_encoded=is_enc),
            in_axes=1, out_axes=-1
        )
        
        cql_current_actions = vmapped_sample(batch['observations'], sample_rngs)
        cql_current_actions = jnp.transpose(cql_current_actions, (1, 0, 2))
        cql_next_actions = vmapped_sample(batch['next_observations'], sample_rngs)
        cql_next_actions = jnp.transpose(cql_next_actions, (1, 0, 2))
        all_actions = jnp.concatenate([cql_random_actions, cql_current_actions, cql_next_actions], axis=1)

        cql_qs = vmapped_q_fn(all_actions)
        chex.assert_shape(cql_qs,(self.config.num_qs, B, self.config["cql_n_actions"]*3))
        
        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(subsample_key, (self.config.calql_q_subsample,), 0, self.config.num_qs,)
        cql_qs = cql_qs[subsample_idcs]
        q_pred = q_pred[subsample_idcs]
        
        # CalQL Part
        n_actions_for_calql = self.config.cql_n_actions * 3
        mc_lower_bound = jnp.repeat(batch["mc_returns"].reshape(-1, 1), n_actions_for_calql, axis=1,)
        cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
        cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)],axis=-1)
        cql_qs -= jnp.log(cql_qs.shape[-1]) * self.config.cql_temp
        
        cql_ood_values = jax.scipy.special.logsumexp(cql_qs / self.config.cql_temp, axis=-1)* self.config.cql_temp
        calql_regularizer = (cql_ood_values - q_pred).mean()
        return calql_regularizer
        
    def bc_critic_loss(self, batch, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """Standard TD loss with support for hlgauss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            next_actions = jnp.reshape(batch["next_actions"], (batch["next_actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
            next_actions = batch["next_actions"][..., 0, :] # take the first action
        rng, sample_rng = jax.random.split(rng)

        images = batch.get('images')
        next_images = batch.get('next_images')
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=images, next_images=next_images)
        calql_regularizer = self.get_calql_diff(batch, batch_actions, grad_params, rng, obs_encoded=obs_encoded, images=images)
        critic_loss = td_loss + self.config.cql_alpha * calql_regularizer
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_q_mean': next_q.mean(),
            'calql_regularizer': calql_regularizer.mean(),
            **stats
        }

    def critic_loss(self, batch: dict, grad_params: flax.core.FrozenDict,
                    rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Standard TD loss with support for hlgauss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)

        images = batch.get('images')
        next_images = batch.get('next_images')

        # Pre-encode observations
        obs_encoded = self._encode_for_critic(batch['observations'], images=images, params=grad_params)
        next_obs_encoded = self._encode_for_critic(batch['next_observations'], images=next_images, use_target=True)

        # Sample next actions from target actor (or one-step policy for FQL)
        next_actor_obs_encoded = self._encode_for_actor(batch['next_observations'], images=next_images, use_target=True)
        if self.config.get('use_one_step_for_targets', False) and self.one_step_network is not None:
            next_actions = self.sample_actions_one_step(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)
        else:
            next_actions = self.sample_actions(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)

        td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=images, next_images=next_images)
        return td_loss, {
            'td_loss': td_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_actions': next_actions.mean(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_actions_std': next_actions.std(),
            'next_q_mean': next_q.mean(),
            **stats
        }

    def actor_total_loss(
        self,
        batch: dict,
        batch_success: dict, 
        grad_params: flax.core.FrozenDict,
        success_flag, 
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Actor loss for separate optimization."""
        a_loss, a_info = self.actor_loss(batch, batch_success, grad_params, success_flag, rng)
        info = {f'actor/{k}': v for k,v in a_info.items()}
        return a_loss, info

    def critic_total_loss(self, batch: dict, grad_params: flax.core.FrozenDict, rng: jnp.ndarray):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k,v in c_info.items()}
        return c_loss, info

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['OGPOAgent', dict]:
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.bc_loss(batch, p, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        agent.target_update(new_actor_state, 'actor')

        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # FQL one-step distillation (BC-only, no Q loss)
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_offline', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng2, use_q_loss=0.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        bc_info = {**actor_info, **one_step_info}
        new_rng, _ = jax.random.split(agent.rng)  # match upstream: re-split from agent.rng
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), bc_info
    
    @staticmethod
    def _update_offline_calql(agent, batch: dict) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to both networks."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        
        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.critic_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_critic_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.bc_critic_loss(batch, p, rng1)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        new_rng, _ = jax.random.split(new_rng)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info
        
    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)
    
    @jax.jit
    def calql_update(self, batch):
        return self._update_offline_calql(self, batch)
    
    @staticmethod
    def _update(agent, batch_tuple, success_flag) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to both networks."""
        batch, batch_success = batch_tuple
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        # Update actor network

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=agent.actor_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.actor_total_loss(batch, batch_success, p, success_flag, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Update actor target networks
        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # Update critic network

        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.critic_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.critic_total_loss(batch, p, rng2)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        # FQL one-step distillation (online, with Q loss)
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng3, _ = jax.random.split(rng2)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng3, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        # Combine info
        info = {**actor_info, **critic_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, critic_network=new_critic_state, one_step_network=new_one_step_state, rng=new_rng), info
        
    @jax.jit
    def batch_update(self, batch, batch_success, success_flag):
        policy_delay = self.config.get('policy_delay', 1)

        if policy_delay <= 1:
            # Original: joint actor+critic updates every step
            batch_tuple = (batch, batch_success)
            def scan_update(agent, batch_tuple):
                return self._update(agent, batch_tuple, success_flag)
            agent, infos = jax.lax.scan(scan_update, self, batch_tuple)
            return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

        # Policy delay > 1: two-phase scan
        # Phase 1: critic updates on ALL utd slices
        agent, critic_infos = jax.lax.scan(self._update_critic_only, self, batch)

        # Phase 2: actor updates on every policy_delay-th slice
        actor_batch = jax.tree.map(
            lambda x: x[policy_delay - 1::policy_delay], batch)
        actor_success = jax.tree.map(
            lambda x: x[policy_delay - 1::policy_delay], batch_success)

        def scan_actor(agent, inputs):
            return self._update_actor_only(agent, inputs, success_flag)
        agent, actor_infos = jax.lax.scan(
            scan_actor, agent, (actor_batch, actor_success))

        return agent, {
            **jax.tree_util.tree_map(lambda x: x.mean(), critic_infos),
            **jax.tree_util.tree_map(lambda x: x.mean(), actor_infos),
        }

    @staticmethod
    def _update_critic_only(agent, batch_tuple) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to critic network only."""
        batch = batch_tuple
        new_rng, sample_rng = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @staticmethod
    def _update_actor_only(agent, batch_tuple, success_flag) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to actor network only (used with policy delay)."""
        batch, batch_success = batch_tuple
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables, batch, batch_success,
                    rngs={'dropout': rng1},
                    method=agent.actor_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.actor_total_loss(batch, batch_success, p, success_flag, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # FQL one-step distillation
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return one_step_distillation_loss(agent, batch, p, rng2, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        info = {**actor_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        batch_tuple = (batch)
        agent, infos = jax.lax.scan(self._update_critic_only, self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    batch_critic_only_update = batch_q_warmup_update

    @staticmethod
    def _bc_refine_update(agent, batch_success) -> Tuple['OGPOAgent', dict]:
        """Apply BC loss update to actor network only using success buffer."""
        new_rng, rng1 = jax.random.split(agent.rng)

        # Update actor network with BC loss only
        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=lambda batch, params: agent.bc_loss(batch, params, rng1),
                    mutable=['batch_stats']
                )
                bc_info = {f'actor/bc_{k}': v for k, v in info.items()}
                return (loss, bc_info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                bc_loss_val, bc_info = agent.bc_loss(batch_success, p, rng1)
                info = {f'actor/bc_{k}': v for k, v in bc_info.items()}
                return bc_loss_val, info
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Update actor target networks
        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        return agent.replace(actor_network=new_actor_state, rng=new_rng), actor_info

    @jax.jit
    def batch_bc_refine_update(self, batch_success):
        """BC refinement update using only the success buffer."""
        agent, infos = jax.lax.scan(self._bc_refine_update, self, batch_success)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # --- FQL one-step sampling ---

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions_one_step(self, observations, rng=None, is_encoded=False, images=None):
        return sample_one_step(self, observations, rng, is_encoded, images)

    @jax.jit
    def sample_actions_one_step_BON(self, observations, rng=None, images=None):
        return sample_one_step_BON(self, observations, rng, images)

    # --- FQL distillation loss (called from update methods) ---

    def one_step_distillation_loss(self, batch, grad_params, rng, use_q_loss=1.0):
        return one_step_distillation_loss(self, batch, grad_params, rng, use_q_loss)

    def target_update(self, network, module_name):
        """Update the target network."""
        # Use different tau for actor vs critic
        if module_name in ['actor', 'noise_net']:
            tau = self.config['actor_tau']
        else:
            tau = self.config['tau']
            
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def sample_actions_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None, images: jnp.ndarray = None) -> jnp.ndarray:
        """Fully vectorized Best-of-N action selection.
        Best action -> max ( min ( Q(s, a)))
        """
        # Approach B: observations are always flat state, images are separate
        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[None]
            if images is not None:
                images = images[None]

        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n

        # 1. Generate all B*N candidate actions in one forward pass
        rng, action_key = jax.random.split(rng)
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
        images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None
        sampled_actions, _, _, _ = self.sample_actions_with_noise(
            obs_expanded, rng=action_key, use_target=True, is_encoded=False, images=images_expanded
        )

        # 2. Evaluate Q-ensemble for all candidates
        critic_obs = self._encode_for_critic(obs_expanded, images=images_expanded, use_target=True)

        if self.config.get('use_mip_q', False):
            # MIP-Q path: two-step sampling for all candidates
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            total_candidates = batch_size * num_samples

            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                rng, noise_key = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_key, (total_candidates, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                q_values_flat = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs,
                    actions=sampled_actions,
                    noise_sample=noise_sample,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # [num_ensemble_members, B*N]
            else:
                rng, noise_key = jax.random.split(rng)
                noise_samples_base = jax.random.uniform(
                    noise_key, (num_ensemble_members, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                noise_samples = jnp.broadcast_to(
                    noise_samples_base[:, None, :],
                    (num_ensemble_members, total_candidates, 1)
                )
                q_values_flat = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs,
                    actions=sampled_actions,
                    noise_samples=noise_samples,
                    mip_t_star=mip_t_star,
                    is_encoded=bool(self.config.encoder)
                )  # [M, B*N]

            q_values = q_values_flat.reshape(num_ensemble_members, batch_size, num_samples)
            num_ensemble_or_noise = num_ensemble_members
        else:
            # Original ensemble-Q path
            q_ensemble = self.critic_network.select('target_critic')(
                critic_obs, actions=sampled_actions, is_encoded=bool(self.config.encoder)
            )
            q_values = q_ensemble.reshape(self.config.num_qs, batch_size, num_samples)
            num_ensemble_or_noise = self.config.num_qs

        # 3. Aggregate Q-ensemble per candidate (Python if — static at trace time)
        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, num_ensemble_or_noise)
            qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)  # (2, B, N)
            min_qs = jnp.min(qs_sub, axis=0)                       # (B, N)
        else:
            min_qs = jnp.min(q_values, axis=0)                     # (B, N)

        # 4. Select the best candidate per observation
        best_idx = jnp.argmax(min_qs, axis=1)                      # (B,)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)  # (B, act_dim)

        return best_actions

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None, is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """Sample final actions via the flow policy with noise."""
        add_batch_dim = False
        if not is_encoded:
            if observations.ndim == 1:
                add_batch_dim = True
                observations = observations[None]
                if images is not None:
                    images = images[None]

        actions, chains, _, _ = self.sample_actions_with_noise(observations, use_target=True, rng=rng, is_encoded=is_encoded, images=images)

        if add_batch_dim:
            actions = actions[0]
        return actions

    def _check_add_batch_dim(self, observations, is_encoded):
        """Check if batch dimension needs to be added (based on static ndim).

        This is NOT jitted - it just checks static shape properties.
        Returns a Python bool that can be used in Python conditionals.
        """
        if is_encoded:
            return False
        # Approach B: observations are always flat state vectors
        return observations.ndim == 1

    def _encode_observations_for_flow(self, observations, is_encoded, images=None):
        """Encode observations for flow sampling. Called from within jitted functions."""
        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images)
        return observations

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions() instead."""
        # Add batch dim if needed (static argument)
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper function for actor network
        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'mip':
            # Use pg_helper for MIP ODE sampling
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            # Shortcut logic (OGPO-specific, kept inline)
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            # Use pg_helper for standard ODE flow (no correction)
            actions = sample_flow_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,  # Match old code: no intermediate clipping in ODE inference
                clip_value=self.config['denoised_clip_value'],
                noises=ode_noises,
                is_encoded=True,
            )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions(self, observations, rng=None, noises=None, is_encoded=False, images=None):
        """Compute actions using pure ODE flow (no SDE-to-ODE correction).

        This is the standard evaluation method for BC phase.
        Uses pg_helper functions for ODE flow and MIP modes.

        For online RL evaluation (after SDE training), use compute_flow_actions_corrected instead.
        """
        # Compute add_batch_dim based on static ndim (before tracing)
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_corrected_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions_corrected() instead."""
        # Add batch dim if needed (static argument)
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper function for actor network (with denoiser support)
        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'mip':
            # MIP doesn't use correction (different formulation)
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            # Shortcut doesn't use correction
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            # Use pg_helper for ODE with SDE-to-ODE correction
            actions = sample_flow_actions_ode_with_correction(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                constant_noise_std=self.config['constant_noise_std'],
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,  # Match old code: no intermediate clipping in ODE inference
                clip_value=self.config['denoised_clip_value'],
                noises=ode_noises,
                is_encoded=True,
            )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions_corrected(self, observations, rng=None, noises=None, is_encoded=False, images=None):
        """Compute actions using ODE flow with SDE-to-ODE correction."""
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_corrected_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    # @jax.jit
    def sample_actions_with_noise(
            self,
            observations: jnp.ndarray,
            grad_params: flax.core.FrozenDict = None,
            rng: jnp.ndarray = None,
            use_target: bool = True,
            is_encoded: bool = False,
            images: jnp.ndarray = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate actions, full chains, and log_probs using pg_helper SDE sampling."""
        params = grad_params or self.actor_network.params
        actor_module_name = 'target_actor'
        noise_module_name = 'target_noise_net'

        # Encode observations
        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, use_target=use_target)

        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        # Create wrapper functions for the network calls
        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'mip':
            actions, chains, logprob, sigmas = sample_mip_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                is_encoded=True,
                params=params,
            )
        else:
            actions, chains, logprob, sigmas = sample_flow_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
                is_encoded=True,
                params=params,
            )

        # Apply normalization options (matching compute_log_probs_and_entropy)
        logprob_steps = 3 if self.config.policy_type == 'mip' else self.config.flow_steps + 1
        if self.config['normalize_denoising_horizon']:
            logprob = logprob / logprob_steps

        if self.config['normalize_act_space_dimension']:
            logprob = logprob / act_dim

        return actions, chains, logprob, sigmas

    @jax.jit
    def bc_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        obs_encoded: jnp.ndarray = None,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute flow matching BC loss.

        Supports three modes:
        1. MIP mode: Two-term loss (regression at t=0 + denoising at t=t*)
        2. Flow + Denoiser mode: Velocity + noise prediction (OGPO-specific SDE-ODE correction)
        3. Standard flow matching: Velocity prediction only

        Uses bc_helper functions for common operations.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        # Encoder handling
        if images is None:
            images = batch.get('images')
        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(batch['observations'], images=images, params=grad_params)
        actor_obs = obs_encoded if self.config.encoder else batch['observations']

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))

        if self.config.policy_type == 'mip':
            # ========== MIP Mode: Two-term loss (uses bc_helper) ==========
            x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(
                batch_actions, rng, self.config.mip_t_star
            )

            # Term 1: Regression at t=0
            pred_0 = self.actor_network.select('actor')(
                actor_obs, x_0, t_0_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_regression_raw = jnp.square(pred_0 - batch_actions)

            # Term 2: Denoising at t=t*
            pred_t_star = self.actor_network.select('actor')(
                actor_obs, x_t_star, t_star_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_denoising_raw = jnp.square(pred_t_star - batch_actions)

            combined_loss_raw = loss_regression_raw + loss_denoising_raw

            bc_loss = apply_chunking_mask(
                combined_loss_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_regression = apply_chunking_mask(
                loss_regression_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_denoising = apply_chunking_mask(
                loss_denoising_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            info = {
                'bc_loss': bc_loss,
                'bc_mip_loss_regression': loss_regression,
                'bc_mip_loss_denoising': loss_denoising,
            }
        else:
            # ========== Standard Flow Matching Mode ==========
            rng, targets_rng = jax.random.split(rng)

            if self.config.use_shortcut:
                # OGPO-specific: Shortcut targets with bootstrapping
                def model_call_fn(obs, x_t, t, dt_base):
                    return self.actor_network.select('target_actor')(obs, x_t, t, dt_base)

                x_t, vel, t, dt_base, _ = get_shortcut_targets(
                    batch['observations'],
                    batch_actions,
                    model_call_fn,
                    targets_rng,
                    denoise_timesteps=self.config.shortcut_denoising_steps,
                    bootstrap_ratio=self.config.bootstrap_ratio,
                )
                pred = self.actor_network.select('actor')(
                    batch['observations'], x_t, t, dt_base, params=grad_params
                )
                loss_denoising = 0.0
            else:
                x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)
                loss_denoising = 0.0

                if self.config.use_denoiser:
                    # OGPO-specific: Denoiser for SDE-to-ODE correction
                    rng, z_rng = jax.random.split(rng)
                    z = jax.random.normal(z_rng, x_t.shape)

                    sigma = self.config.constant_noise_std
                    x_t_noisy = x_t + sigma * z

                    pred, pred_z = self.actor_network.select('actor')(
                        actor_obs, x_t_noisy, t,
                        params=grad_params,
                        is_encoded=self.config.encoder,
                        return_denoiser=True
                    )

                    loss_denoising = apply_chunking_mask(
                        jnp.square(pred_z - z), valid_mask, batch_size,
                        self.config["horizon_length"], self.config["action_dim"],
                        self.config["action_chunking"]
                    )
                else:
                    pred = self.actor_network.select('actor')(
                        actor_obs, x_t, t,
                        params=grad_params,
                        is_encoded=self.config.encoder
                    )

            # Velocity loss
            loss_vel = apply_chunking_mask(
                jnp.square(pred - vel), valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            bc_loss = loss_vel + loss_denoising
            info = {
                'bc_loss': bc_loss,
                'bc_flow_loss_vel': loss_vel,
                'bc_flow_loss_denoising': loss_denoising
            }

        return bc_loss, info
    
    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False, images=None):
        """Computes the Q-value for a given state-action pair for logging."""
        if not is_encoded:
            observations = self._encode_for_critic(observations, images=images, use_target=True)
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=bool(self.config.encoder),
        )
    
    def _get_lr_info(self, actor_state=None, critic_state=None):
        """Get learning rate information and parameter difference norms for logging."""
        actor_state = actor_state or self.actor_network
        critic_state = critic_state or self.critic_network

        # Parameter norms
        actor_norm = compute_param_norm(actor_state.params['modules_actor'])
        target_actor_norm = compute_param_norm(actor_state.params['modules_target_actor'])
        critic_norm = compute_param_norm(critic_state.params['modules_critic'])
        target_critic_norm = compute_param_norm(critic_state.params['modules_target_critic'])

        # Parameter difference norms
        actor_diff_norm = compute_param_diff_norm(
            actor_state.params['modules_actor'],
            actor_state.params['modules_target_actor']
        )
        critic_diff_norm = compute_param_diff_norm(
            critic_state.params['modules_critic'],
            critic_state.params['modules_target_critic']
        )

        # OGPO-specific: noise network norms
        noise_info = {}
        if not self.config.use_constant_noise:
            noise_norm = compute_param_norm(actor_state.params['modules_noise_net'])
            target_noise_norm = compute_param_norm(actor_state.params['modules_target_noise_net'])
            noise_diff_norm = compute_param_diff_norm(
                actor_state.params['modules_noise_net'],
                actor_state.params['modules_target_noise_net']
            )
            noise_info = {
                'noise_param_norm': noise_norm,
                'target_noise_param_norm': target_noise_norm,
                'noise_target_diff_norm': noise_diff_norm,
            }

        return {
            'actor_lr': actor_state.opt_state.hyperparams['learning_rate'],
            'critic_lr': critic_state.opt_state.hyperparams['learning_rate'],
            'actor_step': actor_state.step,
            'critic_step': critic_state.step,
            'actor_param_norm': actor_norm,
            'target_actor_param_norm': target_actor_norm,
            'critic_param_norm': critic_norm,
            'target_critic_param_norm': target_critic_norm,
            'actor_target_diff_norm': actor_diff_norm,
            'critic_target_diff_norm': critic_diff_norm,
            **noise_info,
        }

    def reset_optimizers_with_lr(self,) -> 'OGPOAgent':
        """Reset optimizers with new LR."""
        actor_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.actor_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.actor_warmup_steps,
            decay_steps=self.config.actor_decay_steps,
            end_value=self.config.actor_end_value
        )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.critic_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.critic_warmup_steps,
            decay_steps=self.config.critic_decay_steps,
            end_value=self.config.critic_end_value
        )

        new_actor_tx = create_optimizer(
            lr_schedule=actor_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.actor_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )
        new_critic_tx = create_optimizer(
            lr_schedule=critic_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.critic_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )

        opt_type = "Muon" if self.config.use_muon else "AdamW"
        print(f"Done: reset actor optimizer with {opt_type} (LR: {self.config.ppo_lr}, scheduler: {self.config.actor_scheduler})")
        print(f"Done: reset critic optimizer with {opt_type} (LR: ppo_lr={self.config.ppo_lr}, scheduler: {self.config.critic_scheduler})")

        new_actor_opt_state = new_actor_tx.init(self.actor_network.params)
        new_actor_network = self.actor_network.replace(tx=new_actor_tx, opt_state=new_actor_opt_state, step=1)

        new_critic_opt_state = new_critic_tx.init(self.critic_network.params)
        new_critic_network = self.critic_network.replace(tx=new_critic_tx, opt_state=new_critic_opt_state, step=1)

        return self.replace(actor_network=new_actor_network, critic_network=new_critic_network)
    
    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=bool(self.config.encoder))

    @classmethod
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config: Any, adv_clip_min=None, ex_images=None) -> 'OGPOAgent':
        """Initialize a OGPOAgent with network states."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        # derive dims
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        # full action dim for chunking
        if config.action_chunking:
            full_actions = jnp.concatenate([ex_actions] * config.horizon_length, axis=-1)
        else:
            full_actions = ex_actions
        full_act_dim = full_actions.shape[-1]
        config['full_act_dim'] = full_act_dim
        config.adv_clip_min = adv_clip_min
        config['proprio_dim'] = ex_observations.shape[-1]
        # MIP config validation
        if config['policy_type'] == 'mip':
            assert 0.0 < config.mip_t_star < 1.0, "mip_t_star must be in (0, 1)"
            print(f"[MIP Mode] Using Minimal Iterative Policy with t*={config.mip_t_star}")
        else:
            print(f"[Flow Mode] Using Flow Matching with {config.flow_steps} steps")

        # encoders
        encoders = dict()
        # Vision path: VisionProprioEncoder for actor_obs/critic_obs == 'image'
        if config.actor_obs == 'image' or config.critic_obs == 'image':
            from ogpo.networks.encoders import VisionProprioEncoder
            assert ex_images is not None, "ex_images required when actor_obs or critic_obs is 'image'"
            # Infer image dims from example data: ex_images shape is (H, W, C)
            vit_img_h, vit_img_w, vit_num_ch = ex_images.shape
            print(f"[Vision] ViT input: {vit_img_h}x{vit_img_w}, {vit_num_ch} channels")
        if config.actor_obs == 'image':
            actor_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['actor'] = VisionProprioEncoder(
                vit=actor_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
            )
            config['encoder'] = 'vision'  # sentinel for existing boolean checks
        if config.critic_obs == 'image':
            critic_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['critic'] = VisionProprioEncoder(
                vit=critic_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
            )
            config['encoder'] = 'vision'
        # Legacy encoder path (existing CNN/ViT encoders without Approach B)
        if config.encoder and config.actor_obs == 'state' and config.critic_obs == 'state':
            if 'vit' in config.encoder:
                encoder_module = vit_encoder_modules[config.encoder]
            else:
                encoder_module = encoder_modules[config.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
        # Resolve backbone per-network (useTF is a convenience shortcut)
        a_backbone = config['actor_backbone']
        c_backbone = config['critic_backbone']
        if config.useTF:
            a_backbone = 'tf'
            c_backbone = 'tf'
        config['actor_backbone'] = a_backbone
        config['critic_backbone'] = c_backbone

        # --- Time embedding (sinusoidal or None for raw scalar) ---
        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(
                embed_dim=config.get('time_embedding_dim', 32),
            )
            print(f"[time_embedding] sinusoidal (dim={config.get('time_embedding_dim', 32)})")

        # --- Actor definition ---
        if a_backbone == 'tf':
            actor_def = ActorVectorFieldTF(
                hidden_dim=config.tf_pi_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=False,
                num_layers=config.tf_pi_layers,
                num_heads=config.tf_pi_heads,
                dropout_rate=config.tf_pi_dropout,
                use_denoiser=config.use_denoiser,
                encoder=encoders.get('actor'),
            )
        elif config.useSimBa:
            actor_def = ActorVectorFieldSimBa(
                hidden_dims=config['simba_actor_hidden_dims'],
                action_dim=full_act_dim,
                encoder=encoders.get('actor'),
                rs_norm_momentum=0.99,
                rs_norm_epsilon=1e-8,
                time_embedding=time_emb_module,
            )
        else:
            actor_def = ActorVectorField(
                hidden_dims=config.actor_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.actor_layer_norm,
                encoder=encoders.get('actor'),
                use_film=config.useFiLM,
                use_denoiser=config.use_denoiser,
                time_embedding=time_emb_module,
            )

        # --- Critic definition ---
        if c_backbone == 'tf':
            critic_def = ValueTF(
                hidden_dim=config.tf_q_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=True,
                num_ensembles=config.num_qs,
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                num_layers=config.tf_q_layers,
                num_heads=config.tf_q_heads,
                dropout_rate=config.tf_q_dropout,
            )
        elif config.useSimBa:
            critic_def = ValueSimBa(
                hidden_dims=config['simba_critic_hidden_dims'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                rs_norm_momentum=0.99,
                rs_norm_epsilon=1e-8,
            )
        elif config.get('use_mip_q', False):
            # MIP-Q: Choose between explicit ensemble or implicit ensemble
            if config.get('mip_q_ensemble', False):
                from ogpo.networks import ValueMIPEnsemble
                critic_def = ValueMIPEnsemble(
                    hidden_dims=config['value_hidden_dims'],
                    num_ensembles=config.get('num_ensemble_members', 10),
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config['num_bins'],
                    q_min=config['q_min'],
                    q_max=config['q_max'],
                )
                print(f"[MIP-Q Ensemble Mode] Using EXPLICIT ensemble with {config.get('num_ensemble_members', 10)} networks, "
                      f"noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
            else:
                from ogpo.networks import ValueMIP
                critic_def = ValueMIP(
                    hidden_dims=config['value_hidden_dims'],
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config['num_bins'],
                    q_min=config['q_min'],
                    q_max=config['q_max'],
                )
                print(f"[MIP-Q Mode] Using IMPLICIT ensemble with {config.get('num_ensemble_members', 10)} noise samples, "
                      f"noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
        else:
            critic_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                action_repeat=config['action_repeat'],
                use_film=config.useFiLM,
            )

        # --- Goal conditioning wrappers ---
        if config.get('goal_conditioning_mode'):
            goal_dim = config['no_of_subtasks']
            goal_embed_dim = config.get('goal_embed_dim', 32)
            actor_def = GoalConditionedNetwork(
                inner=actor_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_actor', True),
            )
            critic_def = GoalConditionedNetwork(
                inner=critic_def,
                goal_dim=goal_dim,
                goal_embed_dim=goal_embed_dim,
                use_goals=config.get('goal_conditioned_critic', True),
            )
            print(f"[GC] Wrapped actor (use_goals={config.get('goal_conditioned_actor', True)}) "
                  f"and critic (use_goals={config.get('goal_conditioned_critic', True)}) "
                  f"with GoalConditionedNetwork (goal_dim={goal_dim}, embed_dim={goal_embed_dim})")

        # Add batch dimension if missing
        ex_observations = ex_observations[None, ...]
        full_actions = full_actions[None, ...]
        ex_times = ex_times[None, ...]
        if ex_images is not None:
            ex_images = ex_images[None, ...]

        print(ex_observations.shape, full_actions.shape, ex_times.shape)

        # --- Build init args using dicts (ModuleDict Mapping dispatch) ---
        actor_nets = {'actor': actor_def, 'target_actor': copy.deepcopy(actor_def)}

        actor_init_kw = dict(
            observations=ex_observations,
            actions=full_actions,
            times=ex_times,
            dt=jnp.ones_like(ex_times) if config.use_shortcut else None,
        )
        if config.use_denoiser:
            actor_init_kw['is_encoded'] = False
            actor_init_kw['return_denoiser'] = True
            if config['actor_backbone'] == 'tf':
                actor_init_kw['deterministic'] = True
        if config.actor_obs == 'image' and ex_images is not None:
            actor_init_kw['images'] = ex_images

        actor_args = {
            'actor': actor_init_kw,
            'target_actor': actor_init_kw,
        }

        critic_nets = {'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)}
        critic_init_kw = dict(observations=ex_observations, actions=full_actions)
        if config.critic_obs == 'image' and ex_images is not None:
            critic_init_kw['images'] = ex_images
        critic_args = {'critic': critic_init_kw, 'target_critic': critic_init_kw}

        if config.use_constant_noise:
            noise_def = None
        else:
            noise_def = NoiseInjectionNetwork(
                hidden_dims=config.noise_net_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.noise_net_layer_norm,
                min_noise_std=config.min_noise_std,
                max_noise_std=config.max_noise_std,
            )
            actor_nets['noise_net'] = noise_def
            actor_nets['target_noise_net'] = copy.deepcopy(noise_def)
            if config.encoder:
                if config.actor_obs == 'image':
                    # VisionProprioEncoder output dim
                    vit = encoders['actor'].vit
                    img_dim = config.vit_img_proj_dim if config.vit_img_proj_dim > 0 else (vit.num_patches * vit.embed_dim)
                    state_dim = config.vit_state_proj_dim if config.vit_state_proj_dim > 0 else config.proprio_dim
                    ex_noise_obs_shape = (img_dim + state_dim,)
                elif 'vit' in config.encoder:
                    ex_noise_obs_shape = (encoders['actor'].num_patches * encoders['actor'].embed_dim,)
                else:
                    ex_noise_obs_shape = (encoders['actor'].mlp_hidden_dims[-1],)
                ex_noise_obs = jnp.zeros((1, *ex_noise_obs_shape))
            else:
                ex_noise_obs = ex_observations
            actor_args['noise_net'] = (ex_noise_obs, ex_times)
            actor_args['target_noise_net'] = (ex_noise_obs, ex_times)
            
        # Create separate network definitions
        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        
        # Define optimizer with hyperparameter injection
        
        # Create learning rate schedules
        # For BC training, use constant scheduler if flag is set
        # Ensure all numeric values are Python scalars (not arrays)
        actor_lr = float(config['actor_lr'])
        critic_lr = float(config['critic_lr'])
        clip_grad_norm = float(config.clip_grad_norm)
        actor_weight_decay = float(config.actor_weight_decay)
        critic_weight_decay = float(config.critic_weight_decay)

        if config.use_constant_scheduler_for_bc:
            actor_lr_schedule = actor_lr  # constant for BC
        else:
            actor_lr_schedule = create_lr_schedule(
                scheduler_type=config.actor_scheduler,
                base_lr=actor_lr,
                warmup_steps=int(config.actor_warmup_steps),
                decay_steps=int(config.actor_decay_steps),
                end_value=float(config.actor_end_value)
            )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=config.critic_scheduler,
            base_lr=critic_lr,
            warmup_steps=int(config.critic_warmup_steps),
            decay_steps=int(config.critic_decay_steps),
            end_value=float(config.critic_end_value)
        )

        # Create separate optimizers with injected hyperparameters
        if config.use_muon:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=actor_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=actor_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=critic_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=critic_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
        else:
            @optax.inject_hyperparams
            def adam_optimizer(learning_rate, weight_decay):
                return optax.adamw(learning_rate, weight_decay=weight_decay, eps=1e-8)
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=actor_lr_schedule, weight_decay=actor_weight_decay)
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=critic_lr_schedule, weight_decay=critic_weight_decay)
            )

        # Initialize parameters
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # MIP-Q needs rng for initialization (ValueMIP/ValueMIPEnsemble can sample noise internally)
        if config.get('use_mip_q', False):
            critic_init_rng, target_critic_init_rng = jax.random.split(critic_rng)
            # Ensure separate dicts so rng doesn't leak between critic and target_critic
            critic_args = {
                'critic': {**critic_args['critic'], 'rng': critic_init_rng},
                'target_critic': {**critic_args['target_critic'], 'rng': target_critic_init_rng},
            }

        if config.useSimBa:
            actor_variables = actor_net_def.init(actor_rng, **actor_args)
            critic_variables = critic_net_def.init(critic_rng, **critic_args)

            # Separate trainable params from non-trainable batch_stats
            actor_params = actor_variables.pop('params')
            actor_batch_stats = actor_variables.pop('batch_stats', None) # Use.pop to handle cases without RSNorm
            critic_params = critic_variables.pop('params')
            critic_batch_stats = critic_variables.pop('batch_stats', None)

            # Create train states, now including the batch_stats
            actor_state = TrainState.create(
                apply_fn=actor_net_def.apply, # Ensure apply_fn is passed
                params=actor_params, 
                tx=actor_tx,
                batch_stats=actor_batch_stats # Store the stats here
            )
            critic_state = TrainState.create(
                apply_fn=critic_net_def.apply, # Ensure apply_fn is passed
                params=critic_params, 
                tx=critic_tx,
                batch_stats=critic_batch_stats # And here
            )
        else:
            actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
            critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
            # Create separate train states
            actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
            critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        
        # Initialize target networks
        actor_params = actor_state.params
        actor_params['modules_target_actor'] = actor_params['modules_actor']
        if not config.use_constant_noise:
            actor_params['modules_target_noise_net'] = actor_params['modules_noise_net']
        
        params = critic_state.params
        params['modules_target_critic'] = params['modules_critic']
        
        # FQL one-step policy
        one_step_state = create_one_step_network(
            config, encoders, full_act_dim, ex_observations, ex_images,
            init_rng, clip_grad_norm, actor_weight_decay,
        )

        # store dims
        config.ob_dims = ob_dims
        config.action_dim = action_dim
        return cls(rng, actor_state, critic_state, config, one_step_network=one_step_state)