"""OGPO Networks Module.

Neural network architectures for reinforcement learning including:
- Actors: Policy networks (flow matching, Gaussian, edit policies)
- Critics: Q-function and value networks
- Encoders: Vision encoders (IMPALA, MinViT)
- Modules: Building blocks (MLP, attention, distributions)

All networks are implemented in JAX/Flax.
"""

# Actor networks
from ogpo.networks.actors import (
    Actor,
    ActorVectorField,
    ActorVectorFieldTF,
    ActorVectorFieldSimBa,
    EditPolicy,
    EditActor,
)

# Critic networks
from ogpo.networks.critics import (
    Value,
    ValuePerSubtask,
    ScalarGoalConditionedCritic,
    ValueTF,
    ValueSimBa,
    ValueMIP,
    ValueMIPEnsemble,
    ensemblize,
)

# FQL one-step policy
from ogpo.networks.actors_fql import OneStepPolicy

# Miscellaneous utilities
from ogpo.networks.modules.misc import (
    NoiseInjectionNetwork,
    LogParam,
    SoftplusMatrixParam,
    Identity,
    orthogonal_init,
    zeros_init,
)

# Encoder registries
from ogpo.networks.encoders import (
    encoder_modules,
    ImpalaEncoder,
    MinViT,
    MinVitEncoder,
)

__all__ = [
    # Actors
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
    'EditPolicy',
    'EditActor',
    # FQL
    'OneStepPolicy',
    # Critics
    'Value',
    'ValuePerSubtask',
    'ScalarGoalConditionedCritic',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
    'ensemblize',
    # Utilities
    'NoiseInjectionNetwork',
    'LogParam',
    'SoftplusMatrixParam',
    'Identity',
    'orthogonal_init',
    'zeros_init',
    # Encoders
    'encoder_modules',
    'ImpalaEncoder',
    'MinViT',
    'MinVitEncoder',
]
