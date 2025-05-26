from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
)

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from flax import nnx
    import gymnasium as gym

    from scs.cartpole.agents_value_offline import QNNc
    from scs.lander.agents_value_offline import QNNll
    from scs.utils import AgentConfig


def evaluate_QNN_model(
    model: QNNc | QNNll,
    eval_policy: Callable,
    envs: gym.vector.SyncVectorEnv,
    config: AgentConfig,
    rngs: nnx.Rngs,
    episodes: int = 100,
) -> tuple[list[float], nnx.Rngs]:
    """Evaluate a trained Q-network model in gymnasium environment.

    The passed policy function can be chosen to be either a greedy policy or based
    on the regularized or gated maximizing policy..
    """
    evaluation_rewards = []
    episode_rewards = np.zeros(envs.num_envs)
    completed_episodes = 0
    max_steps = config.max_steps * episodes  # Upper bound on required steps

    states = envs.reset()[0]  # type: ignore[var-annotated]
    for ts in range(max_steps):
        q_values = model(jnp.asarray(states, dtype=jnp.float32))
        actions, rngs = eval_policy(states, q_values, rngs)
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
