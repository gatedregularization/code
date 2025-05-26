from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax import random
import jax.numpy as jnp

from scs.tabular import env_gym

if TYPE_CHECKING:
    from scs.tabular.env_gym import TabularGymParameters
    from scs.utils import AgentConfig


def update_actor(
    state: jax.Array,
    action: jax.Array,
    policy_logits: jax.Array,
    policy: jax.Array,
    advantage: jax.Array,
    terminated: jax.Array,
    config: AgentConfig,
) -> jax.Array:
    """Update the actor policy logits via policy gradientusing the advantage
    function. Masks out the update for terminated samples.
    """
    alpha = (1 - terminated) * config.alpha
    n_actions = policy.shape[-1]
    action_one_hot = jax.nn.one_hot(action, num_classes=n_actions)
    policy_gradient = (action_one_hot - policy[state]) * advantage
    policy_logits = policy_logits.at[state].add(
        config.alpha_scaling * alpha * policy_gradient
    )
    return policy_logits


def update_critic(
    state: jax.Array,
    action: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
    reward: jax.Array,
    policy: jax.Array,
    q_values: jax.Array,
    terminated: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array]:
    """Update the critic q-values using the TD error.
    Masks out the update for terminated (which are not 'terminal') samples.
    """
    alpha = (1 - terminated) * config.alpha
    next_state_value = jnp.dot(policy[next_state], q_values[next_state])
    next_state_value = jax.lax.select(terminal, 0.0, next_state_value)
    td_target = reward + config.gamma * next_state_value
    td_error = td_target - q_values[state, action]
    q_values = q_values.at[state, action].add(alpha * td_error)
    return q_values, td_error


def update_step(
    state: jax.Array,
    action: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
    reward: jax.Array,
    policy_logits: jax.Array,
    q_values: jax.Array,
    terminated: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Updates the critic first, then the actor policy logits using the advantage
    function based on the updated q-values.
    """
    policy = jax.nn.softmax(policy_logits)
    q_values, td_error = update_critic(
        state,
        action,
        next_state,
        terminal,
        reward,
        policy,
        q_values,
        terminated,
        config,
    )
    state_value = jnp.dot(policy[state], q_values[state])
    advantage = q_values[state, action] - state_value
    policy_logits = update_actor(
        state,
        action,
        policy_logits,
        policy,
        advantage,
        terminated,
        config,
    )
    return policy_logits, q_values, td_error


def scan_timestep(
    carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    key: jax.Array,
    env: TabularGymParameters,
    config: AgentConfig,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    """Wraps a full update step for a single timstep into a `jax.lax.scan` function.

    Performs a single step in the environment, based on the current policy, and
    then updates critic and actor based on this timestep.

    Tracks the termination state of the environment.
    """
    state, q_values, policy_logits, epsilon, terminated = carry
    policy = jax.nn.softmax(policy_logits[state])
    action = random.choice(key, jnp.arange(env.n_actions), p=policy)
    next_state, reward, terminal = env_gym.step(env, state, action)
    policy_logits, q_values, td_error = update_step(
        state,
        action,
        next_state,
        terminal,
        reward,
        policy_logits,
        q_values,
        terminated,
        config,
    )
    terminated = jax.lax.select(terminated, jnp.array(1), terminal)
    return (next_state, q_values, policy_logits, epsilon, terminated), (
        reward,
        td_error,
        terminated,
    )


def scan_episode(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    key: jax.Array,
    env: TabularGymParameters,
    config: AgentConfig,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array],]:
    """Wraps a full episode into a `jax.lax.scan` function.
    Since episode length is fixed, due to JIT, a terminated mask keeps track of
    which timesteps were performed after the episode ended.
    """
    q_values, policy_logits, epsilon = carry
    episode_keys = random.split(key, env.max_steps + 1)
    state = env_gym.reset(env, episode_keys[-1])
    terminated = jnp.array(0)
    (state, q_values, policy_logits, epsilon, terminated), (
        rewards,
        errors,
        terminal,
    ) = jax.lax.scan(
        partial(scan_timestep, env=env, config=config),
        (state, q_values, policy_logits, epsilon, terminated),
        episode_keys[:-1],
    )
    epsilon *= 0.99
    not_terminal = jnp.logical_not(terminal)
    return (q_values, policy_logits, epsilon), (
        jnp.sum(rewards * not_terminal),
        jnp.sum(errors**2 * not_terminal) / jnp.sum(not_terminal),
    )


@partial(jax.jit, static_argnums=(0, 3))
def train(
    episodes: int,
    policy_logits: jax.Array,
    q_values: jax.Array,
    env: TabularGymParameters,
    config: AgentConfig,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Trains the policy and q-values for 'episodes' episodes using an A2C
    algorithm.
    """
    epsilon = jnp.array(0.2)
    keys = random.split(key, episodes + 1)
    training_keys, key = keys[:-1], keys[-1]
    (policy_logits, q_values, epsilon), (rewards, mstds) = jax.lax.scan(
        partial(scan_episode, env=env, config=config),
        (q_values, policy_logits, epsilon),
        training_keys,
    )
    return (
        policy_logits,
        q_values,
        rewards,
        mstds,
        key,
    )
