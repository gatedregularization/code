from __future__ import annotations

from collections import defaultdict
import os

from flax import nnx
import gymnasium as gym
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import polars as pl

from scs import ACNN
from scs.cartpole import (
    agent_a2c_online,
    agents_value_offline,
)
from scs.utils import (
    AgentConfig,
    BehavioralData,
    get_train_batch_indices,
)
from scs.utils_nn import (
    NNTrainingState,
    evaluate_agents,
    train_value_agents,
)

############################################################################
# Hyperparameters
############################################################################
learning_rate_online: float = 0.01
learning_rate_offine: float = 1e-6
gamma: float = 0.99
beta: float = 100
tau: float = 0.01
batchsize: int = 32
max_training_episodes: int = 5000
behavioral_samples: int = 1000000
offline_training_steps: int = 10000
max_steps: int = 500
min_episodes_criterions: int = 100
reward_threshold: int = 175
seed: int = 0
n_seeds: int = 5
evaluation_episodes: int = 1000
n_environemnts: int = 1000
load_model: bool = False
save_model: bool = False
load_b_data: bool = False
############################################################################

# Setup Experiment
config = AgentConfig(
    alpha=learning_rate_online,
    gamma=gamma,
    tau=tau,
    episodes=max_training_episodes,
    max_steps=max_steps,
    min_episodes_criterions=min_episodes_criterions,
    reward_threshold=reward_threshold,
)
rngs = nnx.Rngs(seed, config=seed + 1, action_select=seed + 2, sample=seed + 3)
a2c_model = agent_a2c_online.A2Cc(rngs)
a2c_lr_decay_schedule = optax.exponential_decay(
    learning_rate_online, config.episodes, 0.9999
)
optimizer = nnx.Optimizer(a2c_model, optax.adam(a2c_lr_decay_schedule))


checkpointer = ocp.StandardCheckpointer()
checkpoint_path = (
    f"{os.getcwd()}/data/saved_models/cartpole/a2c_agent_behavioral_{reward_threshold}"
)
if load_model:
    graphdef, state = nnx.split(a2c_model)
    restored_state = checkpointer.restore(checkpoint_path, state)
    a2c_model = nnx.merge(graphdef, restored_state, restored_state)
    print(f"Model loaded from {checkpoint_path}")
else:
    # Train the A2C agent
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    a2c_model, optimizer, rngs, metric_history = agent_a2c_online.training_loop(
        a2c_model, optimizer, env, config, rngs
    )
    if save_model:
        _graphdef, state = nnx.split(a2c_model)
        checkpointer.save(checkpoint_path, state)
        print(f"Model saved to {checkpoint_path}")

if load_b_data:
    data_path = f"data/cartpole/b_data_{reward_threshold}_{behavioral_samples}.pkl"
    behavioral_data = BehavioralData.load(data_path)
    print(f"Behavioral data loaded from {data_path}")
else:
    behavioral_data = ACNN.generate_trajectory(behavioral_samples, a2c_model, env, rngs)
    behavioral_data.save(
        f"data/cartpole/b_data_{reward_threshold}_{behavioral_samples}.pkl"
    )
# Setup vectorized environment
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(n_environemnts)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
)
envs.reset(seed=seed)

# Evaluate behavioral policy
b_eval, rngs = ACNN.evaluate_ACNN_model(
    a2c_model,
    envs,
    config,
    rngs,
    episodes=evaluation_episodes,
)
b_mean = np.mean(b_eval)
b_std = np.std(b_eval)
b_se = b_std / np.sqrt(n_seeds * evaluation_episodes)

results: dict[str, list[int | float]] = defaultdict(list)
for offline_training_steps in [
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000,
    85000,
    90000,
    95000,
    100000,
]:
    seed_values = defaultdict(list)  # Store results for each seed
    for seed in range(n_seeds):  # Iterate over seeds to mean over randomness
        rngs = nnx.Rngs(seed, config=seed + 1, action_select=seed + 2, sample=seed + 3)
        envs.reset(seed=seed)

        # Draw random training batches to generate offline training data
        offline_train_batches = get_train_batch_indices(
            offline_training_steps,
            batchsize,
            behavioral_data.n_steps,
            rngs,
        )
        offline_train_data = behavioral_data.get_batch_data(offline_train_batches)

        # Setup baseline model to ensure all offline agents have the same initialization
        base_model = agents_value_offline.QNNc(rngs)
        lr_decay_schedule = optax.exponential_decay(
            learning_rate_offine, offline_train_data.n_steps, 0.99
        )
        base_train_state = NNTrainingState.create(
            model_def=nnx.graphdef(base_model),
            model_state=nnx.state(base_model),
            target_model_state=nnx.state(base_model),
            tau=tau,
            optimizer=optax.adam(lr_decay_schedule),
        )

        # Setup agent parameters
        agent_params = AgentConfig(
            alpha=learning_rate_offine,
            gamma=gamma,
            beta=beta,
            max_steps=max_steps,
            batchsize=batchsize,
        )
        learning_methods = {
            "q": agents_value_offline.train_on_data_greedy,
            "r": agents_value_offline.train_on_data_regularized,
            "g": agents_value_offline.train_on_data_gated,
        }

        training_results = train_value_agents(
            learning_methods,
            base_train_state,
            agent_params,
            offline_train_data,
        )

        print(
            f"offline training steps: {offline_training_steps}, "
            f"beta: {beta}, batchsize: {batchsize}, and seed: {seed}"
        )

        evaluation_results, rngs = evaluate_agents(
            training_results,
            agent_params,
            envs,
            evaluation_episodes,
            rngs,
        )
        for method_name, rewards in evaluation_results.items():
            seed_values[method_name].extend(rewards)
        jax.clear_caches()

    results["steps"].append(offline_training_steps)
    for method_name, rewards in seed_values.items():
        results[f"{method_name}_mean"].append(float(np.mean(rewards)))
        results[f"{method_name}_std"].append(float(np.std(rewards)))
        results[f"{method_name}_se"].append(
            float(np.std(rewards) / np.sqrt(n_seeds * evaluation_episodes))
        )
    results["b_mean"].append(float(b_mean))
    results["b_std"].append(float(b_std))
    results["b_se"].append(float(b_se))

# Save results to CSV
df = pl.DataFrame(results)
df.write_csv(
    f"data/cartpole_results_{reward_threshold}_{batchsize}"
    f"_{beta}_{learning_rate_offine}.csv"
)
