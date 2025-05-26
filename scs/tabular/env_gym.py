from __future__ import annotations

from functools import partial
from typing import (
    Callable,
    NamedTuple,
)

import gymnasium as gym
import jax
from jax import random
import jax.numpy as jnp

from scs.gated import gated_regularization
from scs.qlearning import unregularized_greedy
from scs.regularized import regularization


class TabularGymParameters(NamedTuple):
    """JIT-compatible parameters for OpenAI Gym tabular environments."""

    states: jax.Array
    n_actions: jax.Array
    initial_state_distribution: jax.Array
    transition_dynamic: jax.Array
    max_steps: int

    def __hash__(self) -> int:
        """Generates a hash based on the static properties relevant to JAX's
        compilation cache.

        JAX uses the hash of static arguments to determine whether a precompiled
        computation graph can be reused. For this class, only the `max_steps`
        attribute and the shapes of the arrays (`states`, `n_actions`,
        `initial_state_distribution`, and `transition_dynamic`) influence the
        structure of the computation graph.

        The exact values of the arrays can be treated as dynamic inputs during
        runtime and do not need to remain static. Given the nature of
        TabularGymParameters, this would not happen anyway, but is how it could
        be used in theory.
        """
        return hash(
            (
                self.states.shape,
                int(self.n_actions),
                self.initial_state_distribution.shape,
                self.transition_dynamic.shape,
                self.max_steps,
            )
        )


def _scan_vi_step(
    q_values: jax.Array,
    _timestep: int,
    env: TabularGymParameters,
    gamma: float,
    policy_and_regularization: Callable,
) -> tuple[jax.Array, jax.Array]:
    """Performs a single step of synchronous policy value iteration."""
    # Since transitions are deterministic, next state values can be selected directly
    policy, regularization = policy_and_regularization(q_values)
    state_values = jnp.sum(q_values * policy, axis=-1)
    next_values = state_values[env.transition_dynamic[..., 0]] * (
        1 - env.transition_dynamic[..., 2]  # Terminal transitions have next value of 0
    )
    # Rewards are on second axis of transition dynamic array
    new_q_values = (
        env.transition_dynamic[..., 1]
        + gamma * next_values
        - regularization[:, jnp.newaxis]
    )
    stde = (new_q_values - q_values) ** 2
    return new_q_values, stde.sum()


def solve_taxi_vi(
    env: TabularGymParameters,
    gamma: float,
    iterations: int,
    policy_and_regularization: Callable,
) -> tuple[jax.Array, jax.Array]:
    """Solves the tabular environment using policy value iteration.

    Returns the fixed point q-values of the environment and the given policy
    as well as the sum of squared temporal difference errors.
    """
    q_values, sum_squared_td_errors = jax.lax.scan(
        partial(
            _scan_vi_step,
            env=env,
            gamma=gamma,
            policy_and_regularization=policy_and_regularization,
        ),
        jnp.zeros((env.states.shape[0], env.n_actions), dtype=jnp.float32),
        jnp.arange(iterations),
    )
    return q_values, sum_squared_td_errors


def solve_taxi_greedy(
    env: TabularGymParameters, gamma: float, iterations: int
) -> tuple[jax.Array, jax.Array]:
    """Solves the tabular environment using greedy policy value iteration.

    Returns the optimal q-values, obtained from synchromous q-learning, and the
    sum of squared temporal difference errors.
    """
    return solve_taxi_vi(
        env,
        gamma,
        iterations,
        partial(unregularized_greedy, beta=0.0, behavioral_policy=jnp.array(0.0)),
    )


def solve_tabular_regularized(
    env: TabularGymParameters,
    gamma: float,
    iterations: int,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Solves the tabular environment using regularized policy value iteration given
    a behavioral policy.
    """
    return solve_taxi_vi(
        env,
        gamma,
        iterations,
        partial(
            regularization,
            behavioral_policy=behavioral_policy,
            beta=beta,
        ),
    )


def solve_tabular_gated(
    env: TabularGymParameters,
    gamma: float,
    iterations: int,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Solves the tabular environment using gated policy value iteration given
    a behavioral policy.
    """
    return solve_taxi_vi(
        env,
        gamma,
        iterations,
        partial(gated_regularization, behavioral_policy=behavioral_policy, beta=beta),
    )


def create_JAXTabGym(env_name: str, max_steps: int) -> TabularGymParameters:
    """Creates a JIT-compatible Gym tabular environment based on the
    `TabularGymParameters`.
    """
    env = gym.make(env_name)

    states = jnp.arange(
        env.observation_space.n,  # type: ignore[attr-defined]
    )
    n_actions = jnp.asarray(
        env.action_space.n,  # type: ignore[attr-defined]
    )
    initial_state_distribution = jnp.asarray(
        env.unwrapped.initial_state_distrib,  # type: ignore[attr-defined]
        dtype=jnp.float32,
    )
    transition_dynamic = _create_transition_dynamics_array(env)

    return TabularGymParameters(
        states=states,
        n_actions=n_actions,
        initial_state_distribution=initial_state_distribution,
        transition_dynamic=transition_dynamic,
        max_steps=max_steps,
    )


def _create_transition_dynamics_array(env: gym.Env) -> jax.Array:
    """Creates a transition dynamics array for the tabular environment by
    extracting the dynamics from the environment transition functions.
    """
    dynamics = jnp.zeros(
        (
            int(env.observation_space.n),  # type: ignore[attr-defined]
            int(env.action_space.n),  # type: ignore[attr-defined]
            3,
        ),
        dtype=jnp.int32,
    )
    for s, state in env.unwrapped.P.items():  # type: ignore[attr-defined]
        for a, outcome in state.items():
            dynamics = dynamics.at[s, a].set(
                jnp.array(
                    [
                        outcome[0][1],  # next state
                        outcome[0][2],  # reward
                        outcome[0][3],  # terminal
                    ],
                    dtype=jnp.int32,
                )
            )
    return dynamics


def create_Taxi(max_steps: int = 100) -> TabularGymParameters:
    return create_JAXTabGym("Taxi-v3", max_steps)


def create_CliffWalker(max_steps: int = 100) -> TabularGymParameters:
    return create_JAXTabGym("CliffWalking-v0", max_steps)


def create_FrozenLake(max_steps: int = 100) -> TabularGymParameters:
    return create_JAXTabGym("FrozenLake-v1", max_steps)


def reset(env: TabularGymParameters, key: jax.Array) -> jax.Array:
    """JIT-compatible reset function for the `TabularGymParameters` representation
    of the tabular environment.
    """
    state = random.choice(key, env.states, p=env.initial_state_distribution)
    return state


def step(
    env: TabularGymParameters, state: jax.Array, action: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compatible step function for the `TabularGymParameters` representation
    of the tabular environment.
    """
    next_state, reward, terminal = env.transition_dynamic[state, action]
    return next_state, reward, terminal


def scan_timestep(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    key: jax.Array,
    env: TabularGymParameters,
    policy: Callable,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Performs a single timestep of the environment, ensuring that a
    previously terminated trajectory is continued to be masked out.
    """
    state, terminated, terminal = carry
    # To ensure that the terminal state is not masked by terminated
    terminated = jax.lax.select(terminated, jnp.array(1), terminal)
    action = policy(state, env.n_actions, key)
    next_state, reward, terminal = step(env, state, action)
    return (next_state, terminated, terminal), (
        jnp.array([state, action, next_state, terminal, reward]),
        terminated,
    )


def scan_episode(
    n_steps: jax.Array, key: jax.Array, env: TabularGymParameters, policy: Callable
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Runs a single fixed length episode of the provided environment. Data that
    correspond to steps performed after the end of the episode are masked out.
    """
    keys = random.split(key, env.max_steps + 1)
    state = reset(env, keys[-1])
    terminated = jnp.array(0)
    (state, terminated, _), (timesteps, terminated) = jax.lax.scan(
        partial(scan_timestep, env=env, policy=policy),
        (state, terminated, jnp.array(0)),
        keys[:-1],
    )
    not_terminated = jnp.logical_not(terminated)
    return n_steps + jnp.sum(not_terminated), (timesteps, not_terminated)


@partial(jax.jit, static_argnums=(0, 1, 3))
def generate_trajectory(
    n_episodes: int, env: TabularGymParameters, key: jax.Array, policy: Callable
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generates a trajectory consisting of `n_episodes` episodes.

    Provides the mask required to mask out the steps that occur in between an
    episodes termination and the next episode start.
    """
    keys = random.split(key, n_episodes + 1)
    n_steps, (episodic_data, not_terminated) = jax.lax.scan(
        partial(scan_episode, env=env, policy=policy),
        jnp.array(0),
        keys[:-1],
    )
    return n_steps, episodic_data, not_terminated, keys[-1]


def generate_samples(
    n_samples: int, env: TabularGymParameters, key: jax.Array, policy: Callable
) -> tuple[jax.Array, jax.Array]:
    """Generates a specified number of samples by generating a trajectory of
    episiodes and slicing the desired number of samples out of the non-terminated
    steps.
    """
    n_steps = 0
    data = []
    for e in range(n_samples):  # Worst case of one sample per trajectory (impossible)
        key, trajectory_key = random.split(key)
        n_steps, (episodic_data, not_terminated) = jax.jit(
            scan_episode, static_argnums=(2, 3)
        )(n_steps, trajectory_key, env, policy)
        data.append(episodic_data[not_terminated])
        if n_steps >= n_samples:
            break
    return jnp.concatenate(data)[:n_samples], key


def evaluate_policy(
    n_episodes: int, env: TabularGymParameters, key: jax.Array, policy: Callable
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates a policy over a specified number of episodes in the provided
    environment.

    Returns the total number of steps, rewards for non-terminated steps, and
    the updated PRNG key.
    """
    n_steps, data, not_terminated, key = generate_trajectory(
        n_episodes, env, key, policy
    )
    return n_steps, (data[..., -1] * not_terminated), key


def reset_parallel(
    env: TabularGymParameters, n_agents: int, key: jax.Array
) -> jax.Array:
    """Samples, JIT compatible, initial states for multiple agents in parallel."""
    states = random.choice(
        key, env.states, (n_agents,), p=env.initial_state_distribution
    )
    return states


def step_parallel(
    env: TabularGymParameters, states: jax.Array, actions: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs a parallel and JIT compatible step for multiple agents."""
    next_states, rewards, terminals = env.transition_dynamic[states, actions].T
    return next_states, rewards, terminals.astype(jnp.uint32)


def scan_timestep_parallel(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    keys: jax.Array,
    env: TabularGymParameters,
    policy: Callable,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Performs a single timestep for multiple agents in parallel, ensuring
    that already terminated trajectories remain masked out.
    """
    states, terminated, terminals = carry
    # To ensure that the terminal states are not masked by terminated
    terminated = jax.lax.select(
        terminated, jnp.ones(terminated.shape, dtype=jnp.uint32), terminals
    )
    actions = policy(states, env.n_actions, keys)
    next_states, rewards, terminals = step_parallel(env, states, actions)
    return (next_states, terminated, terminals), (
        jnp.array([states, actions, next_states, terminals, rewards]),
        terminated,
    )


def scan_episode_parallel(
    n_steps: jax.Array,
    key: jax.Array,
    n_agents: int,
    env: TabularGymParameters,
    policy: Callable,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Runs fixed-length episodes for multiple agents in parallel.

    Returns the updated step count and a tuple containing the episode data and
    a mask indicating which steps are not after termination.
    """
    keys = random.split(key, env.max_steps + 1)
    states = reset_parallel(env, n_agents, keys[-1])
    terminated = jnp.zeros(states.shape[0], dtype=jnp.uint32)
    (states, terminated, _), (timesteps, terminated) = jax.lax.scan(
        partial(scan_timestep_parallel, env=env, policy=policy),
        (states, terminated, jnp.zeros(states.shape[0], dtype=jnp.uint32)),
        keys[:-1],
    )
    not_terminated = jnp.logical_not(terminated)
    return n_steps + jnp.sum(not_terminated), (timesteps, not_terminated)


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def generate_trajectory_parallel(
    n_episodes: int,
    n_agents: int,
    env: TabularGymParameters,
    key: jax.Array,
    policy: Callable,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generates `n_episodes` trajectories for multiple agents in parallel.

    Returns the trajectory data, a mask for non-terminated steps, and the
    updated PRNG key.
    """
    keys = random.split(key, n_episodes + 1)
    _, (episodic_data, not_terminated) = jax.lax.scan(
        partial(scan_episode_parallel, n_agents=n_agents, env=env, policy=policy),
        jnp.array(0),
        keys[:-1],
    )
    return episodic_data, not_terminated, keys[-1]


def evaluate_policy_parallel(
    n_episodes: int,
    n_agents: int,
    env: TabularGymParameters,
    key: jax.Array,
    policy: Callable,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates a policy in parallel for multiple agents and episodes.

    Returns the rewards for non-terminated steps for each agent and episode.
    """
    data, not_terminated, key = generate_trajectory_parallel(
        n_episodes, n_agents, env, key, policy
    )
    return (data[..., -1, :] * not_terminated), key
