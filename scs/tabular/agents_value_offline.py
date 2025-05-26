from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
)

import jax
import jax.numpy as jnp

from scs.gated import gated_regularization
from scs.qlearning import unregularized_greedy
from scs.regularized import regularization

if TYPE_CHECKING:
    from scs.utils import AgentConfig


def update_step(
    q_values: jax.Array,
    timestep: jax.Array,
    config: AgentConfig,
    policy_and_regularization: Callable,
) -> tuple[jax.Array, jax.Array]:
    """Update step for greedy sampled value iteration.

    If 'policy_and_regularization' is an `argmax` function, this update step
    corresponds to Watkins' q-learning. Else, any maximizing function,
    regularized or gated for example, matching the function signature can be used.
    """
    state, action, next_state, terminal = timestep[:4].astype(jnp.uint32)
    reward = timestep[4]
    max_policy, regularization = policy_and_regularization(
        q_values[next_state], config.behavioral_policy[next_state], config.beta
    )
    next_state_value = max_policy @ q_values[next_state]
    next_state_value = jax.lax.select(
        terminal, 0.0, config.gamma * next_state_value - regularization
    )
    td = reward + next_state_value - q_values[state, action]
    q_values = q_values.at[state, action].add(config.alpha * td)
    return q_values, td


@partial(jax.jit, static_argnums=(3,))
def train(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
    policy_and_regularization: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Trains the q-values on 'timesteps' using the specified maximizing policy
    and regularization as well as the 'timesteps' data.
    """
    if policy_and_regularization == "regularized":
        get_policy_regularization = regularization
    elif policy_and_regularization == "gated":
        get_policy_regularization = gated_regularization
    elif policy_and_regularization == "greedy":
        get_policy_regularization = unregularized_greedy
    else:
        raise ValueError(
            f"Unknown maximizing policy: {policy_and_regularization}; "
            "Expected either 'regularized', 'gated', or 'greedy'."
        )
    q_values, td = jax.lax.scan(
        partial(
            update_step,
            config=config,
            policy_and_regularization=get_policy_regularization,
        ),
        q_values,
        timesteps,
    )
    policy, _regularization = get_policy_regularization(
        q_values, config.behavioral_policy, config.beta
    )
    return q_values, policy, td


def train_greedy(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train(q_values, timesteps, config, policy_and_regularization="greedy")


def train_regularized(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train(q_values, timesteps, config, policy_and_regularization="regularized")


def train_gated(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train(q_values, timesteps, config, policy_and_regularization="gated")


def update_step_parallel(
    q_values: jax.Array,
    timestep: jax.Array,
    config: AgentConfig,
    policy_and_regularization: Callable,
) -> tuple[jax.Array, jax.Array]:
    """Update step for greedy sampled value iteration, for `config.agents` sets
    of q-values in parallel.

    If 'policy_and_regularization' just contains an `argmax` function, this update
    step corresponds to Watkins' q-learning. Else, any maximizing function,
    regularized or gated for example, matching the function signature can be used.
    """
    states, actions, next_states, terminals = timestep[:4].astype(jnp.uint32)
    rewards = timestep[4]
    max_policies, regularization = policy_and_regularization(
        q_values[config.agents, next_states],
        config.behavioral_policy[config.agents, next_states],
        config.beta,
    )
    next_state_values = jnp.einsum(
        "ij,ij->i", max_policies, q_values[config.agents, next_states]
    )
    next_state_values = jax.lax.select(
        terminals,
        jnp.zeros(config.agents.shape),
        config.gamma * next_state_values - regularization,
    )
    td = rewards + next_state_values - q_values[config.agents, states, actions]
    q_values = q_values.at[config.agents, states, actions].add(config.alpha * td)
    return q_values, td


@partial(jax.jit, static_argnums=(3,))
def train_parallel(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
    policy_and_regularization: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Trains the q-values on 'timesteps' using the specified maximizing policy
    and regularization for multiple sets of q-values in parallel.

    Updates `config.agents` q-values in parallel based on a 'timesteps' dataset
    that contains trajectories for all agents.
    """
    if policy_and_regularization == "regularized":
        get_policy_regularization = regularization
    elif policy_and_regularization == "gated":
        get_policy_regularization = gated_regularization
    elif policy_and_regularization == "greedy":
        get_policy_regularization = unregularized_greedy
    else:
        raise ValueError(
            f"Unknown maximizing policy: {policy_and_regularization}; "
            "Expected either 'regularized', 'gated', or 'greedy'."
        )
    q_values, mean_td = jax.lax.scan(
        partial(
            update_step_parallel,
            config=config,
            policy_and_regularization=get_policy_regularization,
        ),
        q_values,
        timesteps,
    )
    policies, _regularization = jax.vmap(
        partial(
            get_policy_regularization,
            beta=config.beta,
        ),
        in_axes=0,
        out_axes=0,
    )(q_values, config.behavioral_policy)
    return q_values, policies, mean_td


def train_parallel_greedy(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train_parallel(
        q_values, timesteps, config, policy_and_regularization="greedy"
    )


def train_parallel_regularized(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train_parallel(
        q_values, timesteps, config, policy_and_regularization="regularized"
    )


def train_parallel_gated(
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return train_parallel(
        q_values, timesteps, config, policy_and_regularization="gated"
    )
