from __future__ import annotations

from functools import partial
from typing import (
    Callable,
    NamedTuple,
)

import jax
from jax import random
import jax.numpy as jnp

from scs.tabular.utils_tabular import (
    random_action,
    random_action_parallel,
)


class MDPplan(NamedTuple):
    """Template from which to construct random MDP.

    Attributes:
        n_states: Number of states in the MDP.
        n_actions: Number of actions in the MDP.
        dalpha: Dirichlet alpha parameter for transition matrix.
    """

    n_states: int
    n_actions: int
    dalpha: float = 0.1


class MDPparameters(NamedTuple):
    """Parameterization of an MDP.

    JITable parameters of an MDP that can be passed and used in the MDP functions of
    this module.
    """

    states: jax.Array
    n_actions: jax.Array
    transition_matrix: jax.Array
    rewards: jax.Array


def mdp_from_plan(
    plan: MDPplan, key: jax.Array, reward_function: None | Callable = None
) -> tuple[MDPparameters, jax.Array]:
    """Generates a random MDP from an MDPplan and returns it in MDPparameters format.

    A random transition matrix is generated using a Dirichlet distribution where
    the alpha parameter is set to the value of dalpha in the MDPplan. If no reward
    function is provided, uniform random rewards are generated.

    Args:
        plan: MDPplan object based on which to generate the MDP.
        key: JAX random key for generating random numbers.
        reward_function: Optional callable that generates rewards. If None,
            uniform random rewards are generated.
    """
    transition_key, rewards_key, key = random.split(key, 3)
    transition_matrix = random.dirichlet(
        transition_key,
        jnp.ones(plan.n_states) * plan.dalpha,
        (plan.n_states, plan.n_actions),
    )
    if reward_function is None:
        rewards = random.uniform(rewards_key, (plan.n_states, plan.n_actions))
    else:
        rewards = reward_function(rewards_key, plan.n_states, plan.n_actions)
    return (
        MDPparameters(
            states=jnp.arange(plan.n_states),
            n_actions=jnp.asarray(plan.n_actions),
            transition_matrix=transition_matrix,
            rewards=rewards,
        ),
        key,
    )


def reset(MDPparameters: MDPparameters, key: jax.Array) -> jax.Array:
    """Resets the MDP environment by sampling a random initial state."""
    state = random.randint(key, (), 0, MDPparameters.states.shape[0])
    return state


def step(
    state: jax.Array,
    action: jax.Array,
    MDPparameters: MDPparameters,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single transition step in the MDP environment."""
    state = random.choice(
        key,
        MDPparameters.states,
        p=MDPparameters.transition_matrix[state, action],
    )
    reward = MDPparameters.rewards[state, action]
    return state, reward


def scan_step(
    state: jax.Array, keys: jax.Array, policy: Callable, MDPparameters: MDPparameters
) -> tuple[jax.Array, jax.Array]:
    """Performs a single step in the MDP environment for JAX scan operation."""
    action_key, transition_key = keys
    action = policy(state, MDPparameters.n_actions, action_key)
    next_state, reward = step(state, action, MDPparameters, transition_key)
    return next_state, jnp.array([state, action, next_state, 0, reward])


@partial(jax.jit, static_argnums=(0, 3))
def generate_trajectory(
    n_steps: int,
    MDPparameters: MDPparameters,
    key: jax.Array,
    policy: None | Callable = None,
) -> tuple[jax.Array, jax.Array]:
    """Generates a trajectory in the MDP environment using the provided policy.
    Since the `MDPparameters` MDP does not have terminal states, and is therefore
    non-episodic, there is no masking required the trajectory can be generated
     based on the desired number of steps.
    """
    if policy is None:
        policy = random_action
    keys = random.split(key, (n_steps + 1, 2))
    training_keys, initial_state_key, key = keys[:-1], keys[-1, 0], keys[-1, 1]
    state = reset(MDPparameters, initial_state_key)
    _, timesteps = jax.lax.scan(
        partial(scan_step, policy=policy, MDPparameters=MDPparameters),
        state,
        training_keys,
    )
    return timesteps, key


def reset_parallel(
    MDPparameters: MDPparameters, agents: int, key: jax.Array
) -> jax.Array:
    """Returns initial states for multiple agents in parallel."""
    states = random.randint(key, (agents,), 0, MDPparameters.states.shape[0])
    return states


def step_parallel(
    states: jax.Array,
    actions: jax.Array,
    MDPparameters: MDPparameters,
    transition_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs transition steps in `MDPparameters` for multiple agents in parallel."""
    transition_probs = MDPparameters.transition_matrix[states, actions]
    random_transitions = random.uniform(transition_key, (states.shape[0], 1))
    cumulative_probs = jnp.cumsum(transition_probs, axis=1)
    next_states = jnp.argmax(cumulative_probs > random_transitions, axis=1)
    rewards = MDPparameters.rewards[states, actions]
    return next_states, rewards


def scan_step_parallel(
    states: jax.Array, keys: jax.Array, policy: Callable, MDPparameters: MDPparameters
) -> tuple[jax.Array, jax.Array]:
    """Performs a single step for multiple agents in parallel for JAX scan operation."""
    actions_key, tranistions_key = keys
    actions = policy(states, MDPparameters.n_actions, actions_key)
    next_states, rewards = step_parallel(
        states, actions, MDPparameters, tranistions_key
    )
    terminals = jnp.zeros(states.shape)
    return next_states, jnp.array([states, actions, next_states, terminals, rewards])


@partial(jax.jit, static_argnums=(0, 1, 4))
def generate_trajectories(
    n_steps: int,
    n_trajectories: int,
    MDPparameters: MDPparameters,
    key: jax.Array,
    policy: None | Callable = None,
) -> tuple[jax.Array, jax.Array]:
    """Generates multiple trajectories in parallel using the provided policy.
    Since `MDPparameters` is non-episodic, each of the `n_trajectories` will
    contain `n_steps` steps.
    """
    if policy is None:
        policy = random_action_parallel
    keys = random.split(key, (n_steps + 1, 2))
    training_keys, initial_state_key, key = keys[:-1], keys[-1, 0], keys[-1, 1]
    states = reset_parallel(MDPparameters, n_trajectories, initial_state_key)
    _, timesteps = jax.lax.scan(
        partial(scan_step_parallel, policy=policy, MDPparameters=MDPparameters),
        states,
        training_keys,
    )
    return timesteps, key


@partial(jax.jit, static_argnums=(0, 2))
def solve_qvalues(
    n_steps: int,
    MDPparameters: MDPparameters,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solves for the optimal q-values in the MDP using value iteration.

    Returns the converged q-values function and the sum of squared TD errors at
    each iteration.
    """
    q_values = jnp.zeros(MDPparameters.rewards.shape)
    q_values, errors = jax.lax.scan(
        partial(value_iteration_step, MDPparameters=MDPparameters, gamma=gamma),
        q_values,
        jnp.arange(n_steps),
    )
    return q_values, errors


def value_iteration_step(
    carry: jax.Array,
    _step: jax.Array,
    MDPparameters: MDPparameters,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single step of greedy (synchronous q-learning) value iteration
    update.
    """
    q_values = carry
    max_q_values = jnp.max(q_values, axis=1)
    new_q_values = MDPparameters.rewards + gamma * jnp.einsum(
        "ijk, k -> ij", MDPparameters.transition_matrix, max_q_values
    )
    return new_q_values, jnp.sum((new_q_values - q_values) ** 2)


def get_trajectory_rewards(
    states: jax.Array, actions: jax.Array, MDPparameters: MDPparameters
) -> jax.Array:
    """Retrieves rewards for a given trajectory based on states and actions."""
    return MDPparameters.rewards[states, actions]


def get_policies(states: jax.Array, policies: jax.Array) -> jax.Array:
    """Gets policy actions for given states from a policy array.

    Depending on the shape of the policies array, slicing is either done for
    one agents or across the indices for all agents.
    """
    if len(policies.shape) < 3:
        return policies[states]
    else:
        return policies[jnp.arange(policies.shape[0]), states]
