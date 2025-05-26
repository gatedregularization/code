from __future__ import annotations

import os

from flax import nnx
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from scs import ACNN
from scs.lander import (
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
learning_rate_online: float = 0.0002
learning_rate_offine: float = 1e-6
gamma: float = 0.99
beta: float = 100
tau: float = 0.01
batchsize: int = 32
max_training_episodes: int = 2500
behavioral_samples: int = 1000000
offline_training_steps: int = 50000
max_steps: int = 250
min_episodes_criterions: int = 100
reward_threshold: int = 100
seed: int = 0
evaluation_episodes: int = 1000
n_environemnts: int = 1000
load_model: bool = False
save_model: bool = False
load_b_data: bool = False
############################################################################

# Setup Experiment
config = AgentConfig(
    alpha=learning_rate_online,
    alpha_scaling=0.1,
    gamma=gamma,
    tau=tau,
    episodes=max_training_episodes,
    max_steps=max_steps,
    min_episodes_criterions=min_episodes_criterions,
    reward_threshold=reward_threshold,
)
rngs = nnx.Rngs(seed, config=seed + 1, action_select=seed + 2, sample=seed + 3)
a2c_model = agent_a2c_online.A2Cll(rngs)
a2c_lr_decay_schedule = optax.exponential_decay(
    learning_rate_online, config.episodes, 0.999
)
a2c_train_state = NNTrainingState.create(
    model_def=nnx.graphdef(a2c_model),
    model_state=nnx.state(a2c_model),
    target_model_state=nnx.state(a2c_model),
    tau=tau,
    optimizer=optax.adam(a2c_lr_decay_schedule),
)


checkpointer = ocp.StandardCheckpointer()
checkpoint_path = (
    f"{os.getcwd()}/data/saved_models/llander/a2c_agent_behavioral_{reward_threshold}"
)
if load_model:
    restored_state = checkpointer.restore(checkpoint_path, a2c_train_state.model_state)
    a2c_model = nnx.merge(a2c_train_state.model_def, restored_state, restored_state)
    print(f"Model loaded from {checkpoint_path}")
else:
    # Setup vectorized training environment and train A2C agent
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("LunarLander-v3") for _ in range(10)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    envs.reset(seed=seed)
    a2c_train_state, rngs, metric_history = agent_a2c_online.training_loop(
        a2c_train_state, envs, config, rngs
    )
    if save_model:
        checkpointer.save(checkpoint_path, a2c_train_state.model_state)
        print(f"Model saved to {checkpoint_path}")

if load_b_data:
    data_path = f"data/llander/b_data_{reward_threshold}_{behavioral_samples}.pkl"
    behavioral_data = BehavioralData.load(data_path)
    print(f"Behavioral data loaded from {data_path}")
else:
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    behavioral_data = ACNN.generate_trajectory(behavioral_samples, a2c_model, env, rngs)
    behavioral_data.save(
        f"data/llander/b_data_{reward_threshold}_{behavioral_samples}.pkl"
    )

# Draw random training batches to generate offline training data
offline_train_batches = get_train_batch_indices(
    offline_training_steps,
    batchsize,
    behavioral_data.n_steps,
    rngs,
)
offline_train_data = behavioral_data.get_batch_data(offline_train_batches)

# Setup baseline model to ensure all offline agents have the same initialization
base_model = agents_value_offline.QNNll(rngs)
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
training_results["b"] = (a2c_model, jnp.array([0]))

# Setup vectorized evaluation environment
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("LunarLander-v3") for _ in range(n_environemnts)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
)
envs.reset(seed=seed)

evaluation_results, rngs = evaluate_agents(
    training_results,
    agent_params,
    envs,
    evaluation_episodes,
    rngs,
)

for method_name, eval_results in evaluation_results.items():
    print(
        f"{method_name} Agent Evaluation: "
        f"{np.mean(eval_results):.2f} ± {np.std(eval_results):.2f}"
    )
