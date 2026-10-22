import copy
from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import Actor, Value, ActorVectorField, LogParam
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
)
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    compute_flow_bc_loss_online,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    compute_td_target,
    compute_td_loss,
    get_q_values as _get_q_values,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
)

class DSRLBestOfNAgent(flax.struct.PyTreeNode):
    """Soft Actor-Critic (SAC) noise agent + base flow matching actor agent with Best-of-N sampling.
    Uses separate TrainState objects for actor, critic, and BC flow networks.
    """

    rng: Any
    actor_network: TrainState  # SAC actor + temperature
    critic_network: TrainState  # Critic + target critic
    bc_flow_network: TrainState  # Flow matching actor (for BC only)
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss with Best-of-N sampling using q_helper functions."""
        # Sample N actions from actor for each next_observation (Best-of-N)
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.actor_network.select('actor')(batch['next_observations'])

        # Sample num_action_samples actions per observation
        batch_size = batch['next_observations'].shape[0]
        sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
        next_actions_all = jnp.stack([next_dist.sample(seed=key) for key in sample_keys], axis=0)

        # Evaluate Q-values for all samples
        batch_obs = jnp.repeat(batch['next_observations'][None, ...], self.config["num_action_samples"], axis=0)
        next_qs_all = self.critic_network.select('target_critic')(batch_obs, actions=next_actions_all)

        # Aggregate Q-values across ensemble
        next_qs_all = aggregate_q_values(next_qs_all, method=self.config['q_agg'])

        # Best-of-N: select action with highest Q-value
        best_idx = jnp.argmax(next_qs_all, axis=0)
        next_q = jnp.take_along_axis(next_qs_all, best_idx[None, :], axis=0).squeeze(0)
        next_actions = jnp.take_along_axis(next_actions_all, best_idx[None, :, None], axis=0).squeeze(0)

        # Get temperature for entropy bonus
        temp = self.actor_network.select('temp')()

        # Compute TD target with optional entropy backup
        if self.config['entropy_backup']:
            next_log_probs = next_dist.log_prob(next_actions)
            next_q_with_entropy = next_q - temp * next_log_probs
            target_q = compute_td_target(
                rewards=batch['rewards'],
                masks=batch['masks'],
                next_q=next_q_with_entropy,
                discount=self.config['discount'],
                horizon_length=self.config['horizon_length'],
            )
        else:
            target_q = compute_td_target(
                rewards=batch['rewards'],
                masks=batch['masks'],
                next_q=next_q,
                discount=self.config['discount'],
                horizon_length=self.config['horizon_length'],
            )

        # Current Q-values
        q = self.critic_network.select('critic')(batch['observations'], actions=batch["actions"], params=grad_params)

        # MSE loss
        critic_loss, _ = compute_td_loss(q_pred=q, target_q=target_q, loss_type="mse")

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'temp': temp,
            'next_actions_mean': next_actions.mean(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_actions_std': next_actions.std(),
            'rewards_mean': batch['rewards'].mean(),
            'rewards_min': batch['rewards'].min(),
            'rewards_max': batch['rewards'].max(),
        }
        if self.config['entropy_backup']:
            info['next_log_probs_mean'] = next_log_probs.mean()

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss with integrated temperature loss."""
        # Sample actions with current policy
        rng, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        
        # Get Q-values for sampled actions
        qs = self.critic_network.select('critic')(batch['observations'], actions=actions)
        if self.config['q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)
        
        # Get temperature
        temp = self.actor_network.select('temp')()
        
        # SAC actor loss: maximize Q - temperature * entropy
        actor_loss = jnp.mean((temp * log_probs - q))

        # Temperature loss
        temp_param = self.actor_network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()
        
        total_loss = actor_loss + temp_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(),
            'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(),
            'temp': temp,
            'actor_mode': dist.mode().mean(),
        }
    
    def actor_total_loss(self, batch, grad_params, rng):
        """Actor loss for separate optimization."""
        a_loss, a_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info
    
    def actor_bc_loss(self, batch, grad_params, rng):
        """Compute BC loss on actor network using successful noise samples.

        The success buffer contains transitions where batch["actions"] is the noise
        that led to successful outcomes. We train the actor to imitate this successful noise.

        Uses MSE loss between actor output and successful noise to avoid issues with
        computing log_prob on tanh-transformed distributions.

        Args:
            batch: Batch with successful transitions where batch["actions"] contains noise
            grad_params: Parameters for gradient computation
            rng: Random number generator

        Returns:
            Weighted BC loss and info dict
        """
        # batch["actions"] contains the noise that led to successful outcomes (tanh-squashed)
        successful_noise = batch["actions"]

        # Get distribution from current actor
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)

        # Get the mode/mean of the distribution (already tanh-squashed)
        if hasattr(dist, 'mode'):
            actor_noise = dist.mode()
        else:
            actor_noise = dist.mean()

        # Compute MSE loss between actor output and successful noise
        bc_loss_unweighted = jnp.mean(jnp.square(actor_noise - successful_noise))

        # Apply bc_coeff weighting
        bc_loss = self.config["bc_coeff"] * bc_loss_unweighted

        return bc_loss, {
            'actor_bc_loss': bc_loss,
            'actor_bc_loss_unweighted': bc_loss_unweighted,
            'actor_noise_mean': actor_noise.mean(),
            'actor_noise_std': actor_noise.std(),
            'successful_noise_mean': successful_noise.mean(),
            'successful_noise_std': successful_noise.std(),
            'bc_mse_error': bc_loss_unweighted,
        }

    def critic_total_loss(self, batch, grad_params, rng):
        """Critic loss for separate optimization."""
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info
        
    def bc_loss(self, batch, grad_params, rng):
        """Compute the Flow Matching actor loss for offline pretraining.

        Uses 'valid' mask from offline dataset.
        Uses bc_helper for standard flow matching BC loss computation.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        rng, targets_rng = jax.random.split(rng)
        x_t, vel, t = get_flow_targets(
            batch['observations'],
            batch_actions,
            targets_rng,
        )
        pred = self.bc_flow_network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }

    def bc_loss_online(self, batch, grad_params, rng):
        """Compute the Flow Matching actor loss for online RL success buffer.

        Uses bc_coeff weighting and treats all timesteps as valid.
        Note: batch["actions"] contains noise, batch["true_actions"] contains actual executed actions.
        Uses bc_helper for online BC loss computation.
        """
        def model_fn(obs, x_t, t):
            return self.bc_flow_network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)

        return compute_flow_bc_loss_online(
            batch, rng, model_fn,
            self.config["action_chunking"],
            self.config["horizon_length"],
            self.config["action_dim"],
            bc_coeff=self.config["bc_coeff"],
            action_key="true_actions"
        )
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(
        agent,
        batch: dict
    ) -> Tuple['DSRLBestOfNAgent', dict]:
        """Apply gradient update to actor and critic networks (online RL)."""
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)
        
        # Update actor network
        def actor_loss_fn(p):
            return agent.actor_total_loss(batch, p, rng1)
        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        
        # Update critic network
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')
        
        # Combine info
        info = {**actor_info, **critic_info}
        
        return agent.replace(
            actor_network=new_actor_state,
            critic_network=new_critic_state,
            rng=new_rng
        ), info

    @staticmethod
    def _update_offline(
        agent,
        batch: dict
    ) -> Tuple['DSRLBestOfNAgent', dict]:
        """Apply gradient update to BC flow network only (offline training)."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        
        def bc_loss_fn(p):
            return agent.bc_loss(batch, p, rng1)  # Offline BC loss (with 'valid' mask)
        
        new_bc_flow_state, info = agent.bc_flow_network.apply_loss_fn(bc_loss_fn)
        
        return agent.replace(bc_flow_network=new_bc_flow_state, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @staticmethod
    def _update_with_bc(
        agent,
        batch_tuple: Tuple[dict, dict, bool]
    ) -> Tuple['DSRLBestOfNAgent', dict]:
        """Apply gradient update to actor (with BC on success buffer) and critic.

        The BC flow network is NOT updated during online RL - it remains frozen after
        offline pretraining. Only the actor (noise policy) is adapted using successful
        noise samples from the success buffer.
        """
        batch, batch_success, success_flag = batch_tuple
        new_rng, rng1, rng2, rng3 = jax.random.split(agent.rng, 4)

        # Update actor network with RL loss + optional BC loss on successful noise
        def actor_loss_fn(p):
            # RL loss on regular batch
            rl_loss, rl_info = agent.actor_total_loss(batch, p, rng1)

            # Conditionally add BC loss on success batch
            def add_bc_loss():
                bc_loss, bc_info = agent.actor_bc_loss(batch_success, p, rng3)
                combined_info = {**rl_info, **bc_info}
                return rl_loss + bc_loss, combined_info

            def no_bc_loss():
                # Add dummy BC metrics for consistent logging
                dummy_bc_info = {
                    'actor_bc_loss': 0.0,
                    'actor_bc_loss_unweighted': 0.0,
                    'actor_noise_mean': 0.0,
                    'actor_noise_std': 0.0,
                    'successful_noise_mean': 0.0,
                    'successful_noise_std': 0.0,
                    'bc_mse_error': 0.0,
                }
                return rl_loss, {**rl_info, **dummy_bc_info}

            return jax.lax.cond(success_flag, add_bc_loss, no_bc_loss)

        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Update critic network
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        # BC flow network is NOT updated during online RL (frozen after offline training)
        # This preserves the original DSRL philosophy: flow refinement stays fixed,
        # only the noise distribution is adapted

        # Combine info
        info = {**actor_info, **critic_info}

        return agent.replace(
            actor_network=new_actor_state,
            critic_network=new_critic_state,
            # bc_flow_network remains unchanged during online RL
            rng=new_rng
        ), info

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        """Update the agent and return a new agent with information dictionary."""
        if batch_success is None:
            # Original behavior: only update critic
            agent, infos = jax.lax.scan(self._update, self, batch)
        else:
            # New behavior: update critic + optionally BC flow on success buffer
            # For each UTD step, create tuple of (batch, batch_success, success_flag)
            def scan_fn(agent, inputs):
                batch_i, batch_success_i = inputs
                return self._update_with_bc(agent, (batch_i, batch_success_i, success_flag))
            
            agent, infos = jax.lax.scan(scan_fn, self, (batch, batch_success))
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_noise(
        self,
        observations,
        rng=None,
        deterministic=False,
    ):
        """Sample best of N noise from the actor policy with BON Q-value selection."""
        if deterministic:
            # For deterministic mode, just use mode/mean (no Best-of-N)
            dist = self.actor_network.select('actor')(observations)
            if hasattr(dist, 'mode'):
                noise = dist.mode()
            else:
                noise = dist.mean()
        else:
            # Sample N actions from actor and select best based on Q-value
            dist = self.actor_network.select('actor')(observations)
            rng, sample_rng = jax.random.split(rng)
            sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
            noises = jnp.stack([dist.sample(seed=key) for key in sample_keys], axis=0)  # (num_samples, batch_size, action_dim)

            # Evaluate Q-values for all samples
            batch_obs = jnp.repeat(observations[None, ...], self.config["num_action_samples"], axis=0)
            qs = self.critic_network.select('target_critic')(batch_obs, actions=noises)  # (num_qs, num_samples, batch_size)

            # BON Q-aggregation: always min or subsample-min (never mean)
            if self.config['subsample_bon']:
                rng, subsample_key = jax.random.split(rng)
                subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                qs = qs[subsample_idxs].min(axis=0)  # (num_samples, batch_size)
            else:
                qs = qs.min(axis=0)  # (num_samples, batch_size)

            # Best-of-N: select action with highest Q-value
            best_idx = jnp.argmax(qs, axis=0, keepdims=True)  # (1, batch_size)
            noise = jnp.take_along_axis(noises, best_idx[:, None], axis=0).squeeze(0)  # (batch_size, action_dim)
        return noise
    
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
        deterministic=False,
    ):
        """Sample actions from the actor policy with Best-of-N sampling."""
        dist = self.actor_network.select('actor')(observations)

        if deterministic:
            # For deterministic mode, just use mode/mean (no Best-of-N)
            if hasattr(dist, 'mode'):
                actions = dist.mode()
            else:
                actions = dist.mean()
        else:
            # Sample N actions from actor and select best based on Q-value
            rng, sample_rng = jax.random.split(rng)
            sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
            noises = jnp.stack([dist.sample(seed=key) for key in sample_keys], axis=0)  # (num_samples, batch_size, action_dim)

            # Evaluate Q-values for all samples
            batch_obs = jnp.repeat(observations[None, ...], self.config["num_action_samples"], axis=0)
            qs = self.critic_network.select('target_critic')(batch_obs, actions=noises)  # (num_qs, num_samples, batch_size)

            # BON Q-aggregation: always min or subsample-min (never mean)
            if self.config['subsample_bon']:
                rng, subsample_key = jax.random.split(rng)
                subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                qs = qs[subsample_idxs].min(axis=0)  # (num_samples, batch_size)
            else:
                qs = qs.min(axis=0)  # (num_samples, batch_size)

            # Best-of-N: select action with highest Q-value
            best_idx = jnp.argmax(qs, axis=0, keepdims=True)  # (1, batch_size)
            actions = jnp.take_along_axis(noises, best_idx[:, None], axis=0).squeeze(0)  # (batch_size, action_dim)

        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actions = jnp.tile(actions, self.config["horizon_length"])

        actions = self.compute_flow_actions(observations, actions)
        return actions
    
    @jax.jit
    def sample_actions_normal(
        self,
        observations,
        rng=None,
    ):
        """Sample actions from the actor policy."""
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions = self.compute_flow_actions(observations, noises)
        return actions

    def get_q_values(self, observations, actions, is_encoded=False):
        """Computes the Q-value for a given state-action pair for logging."""
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=is_encoded,
        )

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises=None,
        params=None,
        rng=None,  # Accept rng for compatibility with evaluation code
    ):
        """Compute actions from the BC flow model using pg_helper ODE function."""
        # Handle unbatched observations (e.g., from runner's data collection)
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['noise_dim']

        # Handle encoder
        if self.config['encoder'] is not None:
            if params is not None:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations, params=params)
            else:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations)

        # Create wrapper function for actor network
        def actor_fn(obs, actions, t, is_encoded=True):
            if params is not None:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded, params=params)
            else:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn,
            observations=observations,
            rng=rng,
            flow_steps=self.config['flow_steps'],
            act_dim=act_dim,
            act_min=-1.0,
            act_max=1.0,
            clip_intermediate=False,  # DSRL doesn't clip intermediate actions
            noises=noises,
            is_encoded=True,
        )

        # Remove batch dimension if it was added
        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=bool(self.config.get('encoder')))

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        
        if config["noise_chunk_length"] == -1:
            noise_dim = full_action_dim
            ex_noise = full_actions
        elif config["noise_chunk_length"] == 1:
            noise_dim = action_dim * config["noise_chunk_length"]
            ex_noise = jnp.concatenate([ex_actions] * config["noise_chunk_length"], axis=-1)
        else:
            raise ValueError(f"Invalid noise chunk length: {config['noise_chunk_length']}")
        
        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            
        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
        )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=noise_dim,  # This handles both chunked and non-chunked actions
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,  # Enable tanh squashing for bounded actions
            state_dependent_std=True,  # Enable state-dependent std for SAC
            low=-config["action_magnitude"],
            high=config["action_magnitude"],
        )
        
        temp_def = LogParam(init_value=config['init_temperature'])
        
        # Define flow matching actor network.
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        
        # Separate network definitions
        actor_nets = {
            'actor': actor_def,
            'temp': temp_def,
        }
        actor_args = {
            'actor': (ex_observations,),
            'temp': (),
        }
        
        critic_nets = {
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
        }
        critic_args = {
            'critic': (ex_observations, ex_noise),
            'target_critic': (ex_observations, ex_noise),
        }
        
        bc_flow_nets = {
            'actor_bc_flow': actor_bc_flow_def,
        }
        bc_flow_args = {
            'actor_bc_flow': (ex_observations, full_actions, ex_times),
        }
        
        # Create separate network definitions
        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        bc_flow_net_def = ModuleDict(bc_flow_nets)
        
        # Create separate optimizers
        # Ensure learning rates are Python floats
        actor_lr = float(config['lr_actor'])
        critic_lr = float(config['lr_critic'])
        bc_flow_lr = float(config['lr_actor_bc_flow'])

        actor_opt_type = config['optimizer_actor']
        critic_opt_type = config['optimizer_critic']
        bc_flow_opt_type = config['optimizer_actor_bc_flow']

        def get_optimizer(lr, opt_type):
            # lr is already a Python float
            if opt_type == 'adam':
                return optax.adam(learning_rate=lr)
            elif opt_type == 'adamw':
                return optax.adamw(learning_rate=lr)
            elif opt_type == 'sgd':
                return optax.sgd(learning_rate=lr)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")

        actor_tx = get_optimizer(actor_lr, actor_opt_type)
        critic_tx = get_optimizer(critic_lr, critic_opt_type)
        bc_flow_tx = get_optimizer(bc_flow_lr, bc_flow_opt_type)
        
        # Initialize parameters
        rng, actor_rng, critic_rng, bc_flow_rng = jax.random.split(rng, 4)
        actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
        critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
        bc_flow_params = bc_flow_net_def.init(bc_flow_rng, **bc_flow_args)['params']
        
        # Create separate train states
        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        bc_flow_state = TrainState.create(bc_flow_net_def, bc_flow_params, bc_flow_tx)
        
        # Initialize target networks
        critic_state.params[f'modules_target_critic'] = critic_state.params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config["noise_dim"] = noise_dim
        
        # Set target entropy if not specified (using action_dim / 2)
        if config['target_entropy'] is None:
            config['target_entropy'] = -float(noise_dim) / 2

        return cls(rng, actor_state, critic_state, bc_flow_state, config=flax.core.FrozenDict(**config))