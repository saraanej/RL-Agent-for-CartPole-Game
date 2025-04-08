from stable_baselines3 import PPO
import gym
import os
from gym.wrappers.record_video import RecordVideo

# Setup video folder
video_dir = "videos/"
os.makedirs(video_dir, exist_ok=True)

# Load the environment with recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True)
env.reset()

# Load the trained model
model = PPO.load("models/PPO_CartPole/ppo_cartpole")

# Run one episode
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
print("✅ Video saved to:", video_dir)
