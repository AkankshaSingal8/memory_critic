"""Attention and transformer components."""

from typing import Callable
import flax.linen as nn
import jax.numpy as jnp
import jax
from einops import rearrange


def default_init(scale=1.0):
    """Default weight initialization using variance scaling.

    Args:
        scale: Scaling factor for initialization.

    Returns:
        Initialization function.
    """
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def modulate(x, shift, scale):
    """Apply affine modulation (shift and scale) to input.

    This is commonly used in FiLM (Feature-wise Linear Modulation) and
    adaptive layer normalization.

    Args:
        x: Input tensor of shape (batch, ..., features).
        shift: Shift parameter of shape (batch, features).
        scale: Scale parameter of shape (batch, features).

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale[:, None]) + shift[:, None]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module.

    Attributes:
        hidden_dim: Hidden dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
    """
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        head_dim = self.hidden_dim // self.num_heads

        qkv = nn.Dense(3 * self.hidden_dim, kernel_init=default_init())(x)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d', three=3, h=self.num_heads)

        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('b h q d, b h k d -> b h q k', q, k) * scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum('b h q k, b h k d -> b h q d', attn_weights, v)
        output = rearrange(attn_output, 'b h l d -> b l (h d)')

        output = nn.Dense(self.hidden_dim, kernel_init=default_init())(output)
        return output


class FeedForward(nn.Module):
    """Feedforward network with expansion and contraction.

    Attributes:
        hidden_dim: Hidden dimension of the input and output.
        ff_dim: Dimension of the intermediate layer.
        dropout_rate: Dropout rate.
        activation: Activation function.
    """
    hidden_dim: int
    ff_dim: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.ff_dim, kernel_init=default_init())(x)
        x = self.activation(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        return x


class AdaLNLayer(nn.Module):
    """Adaptive Layer Normalization transformer layer.

    This layer uses conditioning information to modulate the layer normalization
    parameters (shift, scale) and gating parameters for residual connections.

    Attributes:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        ff_dim: Feedforward intermediate dimension.
        dropout_rate: Dropout rate.
        activation: Activation function for feedforward network.
    """
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        cond_processed = nn.silu(cond)
        adaLN_params = nn.Dense(
            6 * self.hidden_dim,
            kernel_init=nn.initializers.constant(0.0)
        )(cond_processed)

        # Each has shape (batch_size, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaLN_params, 6, axis=-1)

        # Modulate and apply attention
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        modulated_x = modulate(norm_x, shift_msa, scale_msa)
        attn_output = MultiHeadAttention(
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate
        )(modulated_x, mask, deterministic)

        x = x + gate_msa[:, None] * attn_output
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        modulated_x = modulate(norm_x, shift_mlp, scale_mlp)
        ff_output = FeedForward(
            self.hidden_dim,
            self.ff_dim,
            self.dropout_rate,
            self.activation
        )(modulated_x, deterministic)

        x = x + gate_mlp[:, None] * ff_output
        return x


class AdaLNFinalLayer(nn.Module):
    """Adaptive Layer Normalization final output layer.

    This layer applies adaptive layer normalization and projects to the output dimension.

    Attributes:
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
    """
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x, cond):
        cond_processed = nn.silu(cond)
        adaLN_params = nn.Dense(
            2 * self.hidden_dim,
            kernel_init=nn.initializers.constant(0.0)
        )(cond_processed)

        shift, scale = jnp.split(adaLN_params, 2, axis=-1)
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        x = modulate(norm_x, shift, scale)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.constant(0.0)
        )(x)
        return x


class AdaLNTransformer(nn.Module):
    """Complete Adaptive Layer Normalization Transformer.

    This transformer uses adaptive layer normalization with conditioning
    throughout all layers.

    Attributes:
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ff_dim: Feedforward intermediate dimension.
        output_dim: Output dimension.
        dropout_rate: Dropout rate.
        activation: Activation function.
    """
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    output_dim: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        # Project input and conditioning to hidden_dim
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        cond = nn.Dense(self.hidden_dim, kernel_init=default_init())(cond)

        # Apply transformer layers
        for _ in range(self.num_layers):
            x = AdaLNLayer(
                self.hidden_dim,
                self.num_heads,
                self.ff_dim,
                self.dropout_rate,
                self.activation
            )(x, cond, mask, deterministic)

        # Final output layer with AdaLN
        x = AdaLNFinalLayer(self.hidden_dim, self.output_dim)(x, cond)

        return x
