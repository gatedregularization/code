from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AgentConfig(NamedTuple):
    """Configuration for the agent.

    Default values set to a neutral value such that only required parameters
    need to be set.
    """

    agents: jax.Array = jnp.array([])
    alpha: float = 0.0
    gamma: float = 0.0
    beta: float = 0.0
    gate_floor: float = 0.0
    behavioral_policy: jax.Array = jnp.array([])


def convex_combination_uniform(
    x: jax.Array,
    weight: float,
) -> jax.Array:
    """Computes a convex combination of the input vector and a corresponding
    uniform distribution.
    """
    assert 0 <= weight <= 1, f"Weight must be between 0 and 1; received {weight}"
    uniform = jnp.ones(x.shape) / x.shape[-1]
    return (1 - weight) * x + weight * uniform
