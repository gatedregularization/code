# Gated Regularization for Offline Reinforcement Learning

This repository contains experimental code for evaluating gated regularization methods in offline reinforcement learning on the tabular Taxi and CliffWalking environments.

The experiments were conducted on a workstation with an AMD Ryzen 9 5950X (16 cores), 128GB RAM, and an RTX 3090Ti, running Ubuntu 24.04 and using Python 3.12 with package versions as specified in `requirements.txt`. The code also runs on CPU-only systems.

## Run with Docker

### Setup and Build

```bash
docker compose build
```

Compose defines two services:
- `dev`: development container (kept alive with `sleep infinity`)
- `run`: script runner container (entrypoint is `python`)

If you use a non-default UID/GID on your host:

```bash
UID=$(id -u) GID=$(id -g) docker compose build
```

### Run Scripts

Run a script directly with the `run` service:

```bash
docker compose run --rm run cliff_experiment.py
docker compose run --rm run taxi_experiment.py
```

### Dev Container Workflow

To start it:

```bash
docker compose up -d dev
```

Attach from a terminal:

```bash
docker exec -it gr-dev zsh
```

Detach by exiting the shell (`exit` or `Ctrl+D`).

To stop it:

```bash
docker compose stop dev
```

This container can also be used as a VS Code devcontainer by attaching VS Code to `gr-dev`.


## Run with a Virtual Environment

### Using: uv

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --frozen
```

### Using: pip + venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run Scripts

The random seeds set in the experiment files are those used to generate the published results.

```bash
python cliff_experiment.py
python taxi_experiment.py
```

## Data Management

### Tabular Experiment Results

Taxi and CliffWalking write each completed invocation to a unique directory:

```text
data/results/<environment>/<run_id>/
├── hyperparameters.json
├── evaluation_summary.csv
├── evaluation_agents.csv
└── state_value_comparison.csv
```

`hyperparameters.json` contains the schema version and the effective configuration
used by the run, including resolved perturbation and behavioral-policy choices.
`evaluation_summary.csv` contains pooled reward statistics,
while `evaluation_agents.csv` retains the same statistics for every individual
agent. `state_value_comparison.csv` stores the gated-versus-regularized condition
as typed boolean, count, and fraction columns for each agent.

Run IDs combine a UTC timestamp with a unique suffix. Repeating an identical
configuration therefore preserves both runs rather than appending to or replacing
an existing result. A run directory only becomes visible after all four files have
been written successfully.


## Configuration

Hyperparameters are defined at the top of each experiment file and have self-explanatory names.
