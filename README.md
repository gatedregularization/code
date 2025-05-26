# Gated Regularization for Offline Reinforcement Learning

This repository contains experimental code for evaluating gated regularization methods in offline reinforcement learning across multiple environments.

The experiments were conducted on a workstation with an AMD Ryzen 9 5950X (16 cores), 128GB RAM, and an RTX 3090Ti, running Ubuntu 22.04 and using Python 3.11 with package versions as specified in `requirements.txt`.

## Running the Experiments

### Tabular Experiments
```bash
python cliff_experiment.py    # CliffWalker environment
python taxi_experiment.py     # Taxi environment
```
These print results to the console.

### Neural Network Experiments
```bash
python cartpole_iterate.py     # CartPole environment
python llander_iterate.py      # LunarLander environment
```
These iterate through multiple seeds for each offline training dataset size and save results to CSV files.

## Data Management System

### Model Checkpoints
- **Location**: `data/saved_models/{environment}/`
- **Purpose**: Stores trained A2C behavioral policies
- **Control**: Set `load_model = False` to retrain from scratch
- **Control**: Set `save_model = True` to train and save a new model

### Behavioral Datasets
- **Location**: `data/{environment}/`
- **Purpose**: Stores large trajectory datasets, with size specified in the hyperparameter setup section, generated from behavioral policies
- **Control**: Set `load_b_data = False` to regenerate datasets
- **Files**: `b_data_{reward_threshold}_{behavioral_samples}.pkl`

### Workflow
1. **First run**: Models and data are generated and saved
2. **Subsequent runs**: Everything loads from disk
3. **Clean slate**: Set `load_model = False` and `load_b_data = False` to start fresh

## Configuration

Hyperparameters are defined at the top of each experiment file and have self-explanatory names.
Key flags for data management:

```python
load_model: bool = True     # Load saved behavioral policy
save_model: bool = False    # Save newly trained models
load_b_data: bool = True    # Load saved behavioral data
```
