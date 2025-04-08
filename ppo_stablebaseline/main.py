from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import gym
import os

# Create a directory to save the model
models_dir = "models/PPO_CartPole"
os.makedirs(models_dir, exist_ok=True)

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment (CartPole)
# `make_vec_env` runs multiple copies in parallel for faster training
env = make_vec_env("CartPole-v1", n_envs=4)
env = VecMonitor(env, log_dir)  # adds reward logging

# Create the PPO model with a Multi-Layer Perceptron (MLP) policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 100,000 timesteps
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS)

# Save the trained model
model.save(f"{models_dir}/ppo_cartpole")
print("✅ Model trained and saved.")
