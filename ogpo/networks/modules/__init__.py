"""Neural network modules for OGPO.

This package contains organized neural network components including:
- MLP variants (MLP, MLPCond, MLPWithFiLM, SimBaMLP)
- Attention and transformer components
- Distribution wrappers
- Miscellaneous utilities
"""

# MLP components
from ogpo.networks.modules.mlp import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    MLPResidualBlock,
    SimBaMLP,
    SimBaInternalMLP,
    RSNorm,
    default_init,
)

# Attention and transformer components
from ogpo.networks.modules.attention import (
    MultiHeadAttention,
    FeedForward,
    AdaLNLayer,
    AdaLNFinalLayer,
    AdaLNTransformer,
    modulate,
)

# Distribution components
from ogpo.networks.modules.distributions import (
    TanhMultivariateNormalDiag,
    TransformedWithMode,
    RescaleFromTanhBijector,
)

# Miscellaneous utilities
from ogpo.networks.modules.misc import (
    NoiseInjectionNetwork,
    LogParam,
    SoftplusMatrixParam,
    Identity,
    orthogonal_init,
    zeros_init,
)

__all__ = [
    # MLP components
    'MLP',
    'MLPCond',
    'MLPWithFiLM',
    'MLPResidualBlock',
    'SimBaMLP',
    'SimBaInternalMLP',
    'RSNorm',
    'default_init',
    # Attention components
    'MultiHeadAttention',
    'FeedForward',
    'AdaLNLayer',
    'AdaLNFinalLayer',
    'AdaLNTransformer',
    'modulate',
    # Distribution components
    'TanhMultivariateNormalDiag',
    'TransformedWithMode',
    'RescaleFromTanhBijector',
    # Misc utilities
    'NoiseInjectionNetwork',
    'LogParam',
    'SoftplusMatrixParam',
    'Identity',
    'orthogonal_init',
    'zeros_init',
]
