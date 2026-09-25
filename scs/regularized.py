from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.special import rel_entr

if TYPE_CHECKING:
    import jax


def regularization(
    q_values: jax.Array,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculates the maximizing policy under regularization as well as the
    resulting regularization penalty.

    Utilizes "Nestrov (2005) 'Smooth minimization of non-smooth functions', p.148"
    suggestion to improve numerical stability
    """
    max_q = jnp.max(q_values, axis=-1, keepdims=True)
    # Avoid zero in denominator by scaling a beta of zero to a tiny value
    exp_scaled_q_values = jnp.exp((q_values - max_q) / (beta + 1e-8))
    weighted_q_values = exp_scaled_q_values * behavioral_policy
    policy = weighted_q_values / jnp.sum(weighted_q_values, axis=-1, keepdims=True)
    regularization_value = jnp.sum(rel_entr(policy, behavioral_policy), axis=-1)
    return policy, beta * regularization_value
