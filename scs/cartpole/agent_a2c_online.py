from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from scs.utils import get_expected_return

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.utils import AgentConfig


class A2Cc(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=4,
            out_features=128,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.actor: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=2,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.citic: nnx.Linear = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nnx.relu(self.linear_1(x))
        return self.actor(x), self.citic(x)


def run_episode(
    model: A2Cc,
    env: gym.Env,
    max_steps: int,
    rng: nnx.Rngs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Runs a single episode with the given model and environment, collecting
    trajectory data for subsequent training.
    """
    states = np.zeros((max_steps + 1, 4), dtype=np.float32)
    rewards = np.zeros((max_steps), dtype=np.float32)
    actions = np.zeros((max_steps), dtype=np.uint32)
    terminals = np.zeros((max_steps), dtype=np.bool_)
    episode_mask = np.zeros((max_steps), dtype=np.float32)

    state = env.reset()[0]
    for ts in range(max_steps):
        action_logits, _value = model(jnp.asarray(state, dtype=jnp.float32))
        action = jax.random.categorical(rng.action_select(), action_logits)
        next_state, reward, terminal, _truncated, _info = env.step(int(action))
        states[ts] = state
        actions[ts] = action
        rewards[ts] = reward
        terminals[ts] = terminal
        episode_mask[ts] = 1.0
        state = next_state
        if terminal:
            states[ts + 1] = next_state
            break
    return (
        jnp.asarray(states, dtype=jnp.float32),
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.asarray(actions, dtype=jnp.uint32),
        jnp.asarray(terminals, dtype=jnp.uint32),
        jnp.asarray(episode_mask, dtype=jnp.float32),
    )


def compute_loss(
    model: A2Cc,
    states: jax.Array,
    rewards: jax.Array,
    actions: jax.Array,
    terminals: jax.Array,
    mask: jax.Array,
    returns: jax.Array,
    gamma: float,
) -> jax.Array:
    """Computes the combined actor-critic loss for the A2Cc model.

    Calculates the actor (policy) loss using advantages and the critic (value)
    loss using Huber loss between predicted values and returns.
    """
    actions_logits, values = model(states)
    next_values = jax.lax.stop_gradient(values[1:, 0] * (1 - terminals))
    values = values[:-1, 0]
    _td_targets = rewards + gamma * next_values
    advantages = jax.lax.stop_gradient(returns - values)
    log_policy = jax.nn.log_softmax(actions_logits[:-1])
    log_action_probs = log_policy[jnp.arange(actions.shape[0]), actions]
    actor_loss = -jnp.sum(log_action_probs * advantages * mask)
    critic_loss = jnp.sum(optax.huber_loss(values, returns) * mask)
    return actor_loss + critic_loss


@nnx.jit(static_argnums=(8,))
def train_step(
    model: A2Cc,
    optimizer: nnx.Optimizer,
    states: jax.Array,
    rewards: jax.Array,
    actions: jax.Array,
    terminals: jax.Array,
    episode_mask: jax.Array,
    returns: jax.Array,
    gamma: float,
) -> jax.Array:
    """Performs a single training step using the A2Cc algorithm.

    Computes gradients of the loss with respect to model parameters and applies
    an optimization update.
    """
    grad_fn = nnx.value_and_grad(compute_loss)
    loss, grads = grad_fn(
        model,
        states,
        rewards,
        actions,
        terminals,
        episode_mask,
        returns,
        gamma,
    )
    optimizer.update(grads)
    return loss


def training_loop(
    model: A2Cc,
    optimizer: nnx.Optimizer,
    env: gym.Env,
    config: AgentConfig,
    rngs: nnx.Rngs,
    stop_on_threshold: bool = True,
) -> tuple[A2Cc, nnx.Optimizer, nnx.Rngs, dict[str, list[float]]]:
    """Trains the A2Cc model on the provided environment for a specified number of
    episodes, with early stopping based on a reward threshold.
    """
    metric_history: dict[str, list[float]] = {
        "loss": [],
        "reward": [],
        "rolling_mean": [],
    }
    rolling_window: deque[jax.Array] = deque(maxlen=config.min_episodes_criterions)

    for e in tqdm(range(config.episodes), desc="Training", unit="episode"):
        states, rewards, actions, terminals, episode_mask = run_episode(
            model, env, config.max_steps, rngs
        )
        returns = get_expected_return(
            rewards, episode_mask, config.gamma, standardize=True
        )
        loss = train_step(
            model,
            optimizer,
            states,
            rewards,
            actions,
            terminals,
            episode_mask,
            returns,
            config.gamma,
        )
        metric_history["loss"].append(float(loss))
        metric_history["reward"].append(float(rewards.sum()))
        rolling_window.append(rewards.sum())
        rolling_mean = np.mean(rolling_window)
        metric_history["rolling_mean"].append(float(rolling_mean))
        tqdm.write(
            f"Episode {e}, Loss: {loss:.4f}, Reward: {metric_history['reward'][-1]} "
            f"Rolling Mean: {rolling_mean:.4f}"
        )
        if (
            stop_on_threshold
            and rolling_mean > config.reward_threshold
            and e > config.min_episodes_criterions
        ):
            print(f"Solved at episode {e}, Rolling Mean: {rolling_mean}")
            break

    return model, optimizer, rngs, metric_history
