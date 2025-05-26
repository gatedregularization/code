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
    from scs.utils_nn import NNTrainingState


class A2Cll(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=8,
            out_features=512,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=512,
            rngs=rngs,
        )
        self.linear_2: nnx.Linear = nnx.Linear(
            in_features=512,
            out_features=512,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=512,
            rngs=rngs,
        )
        self.actor: nnx.Linear = nnx.Linear(
            in_features=512,
            out_features=4,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.citic: nnx.Linear = nnx.Linear(
            in_features=512,
            out_features=1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nnx.relu(self.layernorm_1(self.linear_1(x)))
        x = nnx.relu(self.layernorm_2(self.linear_2(x)))
        return self.actor(x), self.citic(x)


def run_episode(
    model: A2Cll,
    envs: gym.vector.SyncVectorEnv,
    max_steps: int,
    rng: nnx.Rngs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Runs a single episode with the given model and environment, collecting
    trajectory data for subsequent training.
    """
    n_envs = envs.num_envs
    states = np.zeros((max_steps, n_envs, 8), dtype=np.float32)
    rewards = np.zeros((max_steps, n_envs), dtype=np.float32)
    actions = np.zeros((max_steps, n_envs), dtype=np.uint32)
    next_states = np.zeros((max_steps, n_envs, 8), dtype=np.float32)
    terminals = np.zeros((max_steps, n_envs), dtype=np.bool_)
    episode_mask = np.ones((max_steps, n_envs), dtype=np.float32)

    state = envs.reset()[0]
    terminated = np.zeros(n_envs, dtype=np.bool_)
    for ts in range(max_steps):
        action_logits, _value = model(jnp.asarray(state, dtype=jnp.float32))
        action = jax.random.categorical(rng.action_select(), action_logits)
        next_state, reward, terminal, truncated, _info = envs.step(np.asarray(action))
        states[ts] = state
        next_states[ts] = next_state
        actions[ts] = action
        rewards[ts] = reward
        terminals[ts] = terminal
        reset_env = np.logical_or(terminal, truncated)
        if reset_env.any():
            state = envs.reset(options={"reset_mask": reset_env})[0]
        else:
            state = next_state
        terminated = np.logical_or(terminated, reset_env)
        episode_mask[ts] = terminated
        if terminated.all():
            break
    return (
        jnp.asarray(states, dtype=jnp.float32),
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.asarray(actions, dtype=jnp.uint32),
        jnp.asarray(next_states, dtype=jnp.float32),
        jnp.asarray(terminals, dtype=jnp.uint32),
        jnp.asarray(np.logical_not(episode_mask), dtype=jnp.float32),
    )


def compute_loss(
    model: A2Cll,
    target_model: A2Cll,
    states: jax.Array,
    rewards: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    terminals: jax.Array,
    mask: jax.Array,
    returns: jax.Array,
    gamma: float,
) -> jax.Array:
    """Computes the combined actor-critic loss for the A2Cll model.

    Calculates the actor (policy) loss using advantages and the critic (value)
    loss using Huber loss between predicted values and returns.
    """
    actions_logits, values = model(states)
    values = values[..., 0]
    _actions_logits, target_values = target_model(states)
    target_values = target_values[..., 0]
    # _next_action_logits, next_values = target_model(next_states)
    # next_values = jax.lax.stop_gradient(next_values[..., 0] * (1 - terminals))
    # td_targets = rewards + gamma * next_values
    advantages = jax.lax.stop_gradient(returns - target_values)
    log_policy = jax.nn.log_softmax(actions_logits, axis=-1)
    # Slice [steps, n_envs, n_actions] array with [steps, n_envs] actions array
    log_action_probs = log_policy[
        jnp.arange(actions.shape[0])[:, jnp.newaxis],
        jnp.arange(actions.shape[1])[jnp.newaxis, :],
        actions,
    ]
    actor_loss = -jnp.sum(log_action_probs * advantages * mask)
    critic_loss = jnp.sum(optax.huber_loss(values, returns) * mask)
    return 0.1 * actor_loss + critic_loss


@nnx.jit(static_argnums=(8,))
def train_step(
    train_state: NNTrainingState,
    states: jax.Array,
    rewards: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    terminals: jax.Array,
    episode_mask: jax.Array,
    returns: jax.Array,
    gamma: float,
) -> tuple[NNTrainingState, jax.Array]:
    """Performs a single training step using the A2Cll algorithm.

    Computes gradients of the loss with respect to model parameters and applies
    an optimization update.
    """
    grad_fn = nnx.value_and_grad(compute_loss)
    loss, grads = grad_fn(
        nnx.merge(train_state.model_def, train_state.model_state),
        nnx.merge(train_state.model_def, train_state.target_model_state),
        states,
        rewards,
        actions,
        next_states,
        terminals,
        episode_mask,
        returns,
        gamma,
    )
    return train_state.apply_gradients(grads), loss


def training_loop(
    train_state: NNTrainingState,
    envs: gym.vector.SyncVectorEnv,
    config: AgentConfig,
    rngs: nnx.Rngs,
    stop_on_threshold: bool = True,
) -> tuple[NNTrainingState, nnx.Rngs, dict[str, list[float]]]:
    """Trains the A2Cll model on the provided environment for a specified number of
    episodes, with early stopping based on a reward threshold.
    """
    metric_history: dict[str, list[float]] = {
        "loss": [],
        "reward": [],
        "rolling_mean": [],
    }
    rolling_window: deque[jax.Array] = deque(maxlen=config.min_episodes_criterions)

    for e in tqdm(range(config.episodes), desc="Training", unit="episode"):
        states, rewards, actions, next_states, terminals, episode_mask = run_episode(
            nnx.merge(train_state.model_def, train_state.model_state),
            envs,
            config.max_steps,
            rngs,
        )
        returns = get_expected_return(
            rewards, episode_mask, config.gamma, standardize=True
        )
        train_state, loss = train_step(
            train_state,
            states,
            rewards,
            actions,
            next_states,
            terminals,
            episode_mask,
            returns,
            config.gamma,
        )
        metric_history["loss"].append(float(loss))
        mean_episodes_rewards = np.sum(rewards, axis=0).mean()
        metric_history["reward"].append(float(mean_episodes_rewards))
        rolling_window.append(mean_episodes_rewards)
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

    return train_state, rngs, metric_history
