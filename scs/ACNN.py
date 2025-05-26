from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from scs.utils import BehavioralData

if TYPE_CHECKING:
    from flax import nnx
    import gymnasium as gym

    from scs.cartpole.agent_a2c_online import A2Cc
    from scs.lander.agent_a2c_online import A2Cll
    from scs.utils import AgentConfig


def generate_trajectory(
    steps: int, model: A2Cc | A2Cll, env: gym.Env, rngs: nnx.Rngs
) -> BehavioralData:
    """Generates a single trajectory of behavioral data using the A2Cc | A2Cll
    model and returns the collected data as a `BehavioralData` object.
    """
    state = env.reset()[0]
    states_data = np.zeros((steps, state.shape[0]), dtype=np.float32)
    rewards_data = np.zeros((steps), dtype=np.float32)
    actions_data = np.zeros((steps), dtype=np.uint32)
    # Simplifies handling of truncated vs terminal states but increases memory usage.
    next_states_data = np.zeros((steps, state.shape[0]), dtype=np.float32)
    terminals_data = np.zeros((steps), dtype=np.bool_)

    for ts in range(steps):
        action_logits, _values = model(jnp.asarray(state, dtype=jnp.float32))
        action = jax.random.categorical(rngs.action_select(), action_logits)
        next_state, reward, terminal, truncated, _info = env.step(int(action))
        states_data[ts] = state
        actions_data[ts] = action
        next_states_data[ts] = next_state
        rewards_data[ts] = reward
        terminals_data[ts] = terminal
        if terminal or truncated:
            print(f"Reset environment at timstep {ts + 1}")
            state = env.reset()[0]
        else:
            state = next_state
    next_state_logits, _next_state_values = model(jnp.asarray(next_states_data))
    return BehavioralData(
        jnp.asarray(states_data, dtype=jnp.float32),
        jnp.asarray(actions_data, dtype=jnp.uint32),
        jnp.asarray(rewards_data, dtype=jnp.float32),
        jnp.asarray(next_states_data, dtype=jnp.float32),
        jnp.asarray(terminals_data, dtype=jnp.uint32),
        next_state_logits,
        steps,
        jnp.arange(1, dtype=jnp.uint32),
    )


def generate_trajectories(
    steps: int,
    agents: int,
    model: A2Cc | A2Cll,
    envs: gym.vector.SyncVectorEnv,
    rngs: nnx.Rngs,
) -> BehavioralData:
    """Generates multiple trajectories in parallel using the A2Cc | A2Cll model.

    A vectorized version of generate_trajectory that runs multiple environment
    instances in parallel for more efficient data collection.

    Args:
        steps: Number of timesteps to generate for each trajectory.
        agents: Number of parallel agents/environments.
        model: The A2Cc | A2Cll model used for action selection.
        envs: Vectorized environment interface for parallel simulation.
        rngs: Random number generators for action selection.

    Returns:
        BehavioralData object containing the collected trajectories information
        with batch dimension for multiple agents on axis 0.
        Data has shape [steps, agents, ...].
    """
    states_data = np.zeros((steps, agents, 4), dtype=np.float32)
    rewards_data = np.zeros((steps, agents), dtype=np.float32)
    actions_data = np.zeros((steps, agents), dtype=np.uint32)
    # Required to handle truncated vs terminal states
    next_states_data = np.zeros((steps, agents, 4), dtype=np.float32)
    terminals_data = np.zeros((steps, agents), dtype=np.bool_)

    states = envs.reset()[0]  # type: ignore[var-annotated]
    for ts in range(steps):
        action_logits, _values = model(jnp.asarray(states, dtype=jnp.float32))
        actions = jax.random.categorical(rngs.action_select(), action_logits)
        next_states, rewards, terminals, truncated, _info = envs.step(
            np.asarray(actions)
        )
        states_data[ts] = states
        actions_data[ts] = actions
        rewards_data[ts] = rewards
        next_states_data[ts] = next_states
        terminals_data[ts] = terminals
        reset_env = np.logical_or(terminals, truncated)
        if reset_env.any():
            print(f"Reset environment {reset_env} at timstep {ts + 1}")
            states = envs.reset(options={"reset_mask": reset_env})[0]
        else:
            states = next_states
    next_state_logits, _next_state_values = model(jnp.asarray(next_states_data))
    return BehavioralData(
        jnp.asarray(states_data, dtype=jnp.float32),
        jnp.asarray(actions_data, dtype=jnp.uint32),
        jnp.asarray(rewards_data, dtype=jnp.float32),
        jnp.asarray(next_states_data, dtype=jnp.float32),
        jnp.asarray(terminals_data, dtype=jnp.uint32),
        next_state_logits,
        steps,
        jnp.arange(1, dtype=jnp.uint32),
    )


def evaluate_ACNN_model(
    model: A2Cc | A2Cll,
    envs: gym.vector.SyncVectorEnv,
    config: AgentConfig,
    rngs: nnx.Rngs,
    episodes: int = 100,
) -> tuple[list[float], nnx.Rngs]:
    """Evaluates a trained A2Cc | A2Cll model on the environment by running the
    model for a specified number of episodes and collects performance metrics.
    """
    evaluation_rewards = []
    episode_rewards = np.zeros(envs.num_envs)
    completed_episodes = 0
    max_steps = config.max_steps * episodes  # Upper bound on required steps

    states = envs.reset()[0]  # type: ignore[var-annotated]
    for ts in range(max_steps):
        policy_logits, _value = model(jnp.asarray(states, dtype=jnp.float32))
        actions = jax.random.categorical(rngs.action_select(), policy_logits)
        next_states, rewards, terminals, truncated, _info = envs.step(
            np.asarray(actions)
        )
        episode_rewards += rewards
        reset_env = np.logical_or(terminals, truncated)
        if reset_env.any():
            evaluation_rewards.extend(episode_rewards[reset_env].tolist())
            completed_episodes += np.sum(reset_env)
            if completed_episodes >= episodes:
                break
            episode_rewards[reset_env] = 0.0
            states = envs.reset(options={"reset_mask": reset_env})[0]
        else:
            states = next_states
    return evaluation_rewards[:episodes], rngs
