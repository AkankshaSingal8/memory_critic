from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules.mlp import MLP, default_init

class GoalConditionedNetwork(nn.Module):
    """Optionally projects the goal suffix of observations before the inner network.

    The last `goal_dim` entries are a binary goal vector from the data pipeline.
    When enabled, they are replaced by a learned `goal_embed_dim`-d projection.
    When disabled (`use_goals=False`), the suffix is stripped so the inner network
    sees only the main state — useful for non-GC ablations.
    """
    inner: nn.Module          # wrapped actor or critic
    goal_dim: int             # size of the binary goal suffix in observations
    goal_embed_dim: int       # output dim of the learned goal projection
    use_goals: bool = True    # False → strip goals instead of projecting

    def setup(self):
        if self.use_goals:
            self.proj = nn.Dense(self.goal_embed_dim)

    def _project(self, observations: jnp.ndarray) -> jnp.ndarray:
        main_obs = observations[..., :-self.goal_dim]
        if not self.use_goals:
            return main_obs
        goal_vec = observations[..., -self.goal_dim:].astype(jnp.float32)
        return jnp.concatenate([main_obs, nn.gelu(self.proj(goal_vec))], axis=-1)

    def encode(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Project goals then run the inner network's encoder."""
        return self.inner.encode(self._project(observations))

    def __call__(self, observations, *args, is_encoded=False, **kwargs):
        return self.inner(self._project(observations), *args, is_encoded=is_encoded, **kwargs)