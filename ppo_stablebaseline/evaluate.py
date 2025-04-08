from stable_baselines3 import PPO
import gym

# Load the environment
env = gym.make("CartPole-v1", render_mode="human")

# Load the trained model
model = PPO.load("models/PPO_CartPole/ppo_cartpole")

# Run the agent for a few episodes
episodes = 5

for ep in range(episodes):
    obs = env.reset()[0]  # .reset() now returns a tuple: (obs, info)
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Episode {ep + 1}: Reward = {total_reward}")

env.close()
