# DDPG for Pendulum-v1

This repository contains an implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm applied to the `Pendulum-v1` environment from OpenAI's Gymnasium. The aim is to train an agent to balance the pendulum using the actor-critic algorithm.

## Project Structure

The project is organized as follows:

- **`Train Expert (DDPG).ipynb`**: The Jupyter notebook that demonstrates the training of the agent and tests its performance.
- **`algorithms/ddpg.py`**: The implementation of the DDPG algorithm, including training and testing functions.
- **`networks/actor_critic.py`**: Defines the neural networks for the Actor and Critic models used in DDPG.
- **`utils/normalize_env.py`**: A utility to normalize the action space of the environment.
- **`utils/ou_noise.py`**: Implements the Ornstein-Uhlenbeck noise process for exploration.
- **`utils/replay.py`**: Implements a ReplayBuffer to store and sample experiences during training.

## Requirements

To run the project, you need to install the following dependencies:

```bash
pip install gymnasium matplotlib numpy torch imageio tqdm
```

## Results
After training the agent, the following files are generated:

- Model Checkpoints: The trained actor and critic networks are saved in the ./models/Expert directory.
- Training Plot: A plot showing the performance of the agent over episodes is displayed and optionally saved as a PNG image (if specified).
- Performance GIF: A GIF showing the agent's performance during testing is saved (if render_save_path is provided).

## Acknowledgments
- The DDPG implementation is based on the work of Lillicrap et al. (2015).
- The Pendulum-v1 environment is part of OpenAI's Gymnasium.
