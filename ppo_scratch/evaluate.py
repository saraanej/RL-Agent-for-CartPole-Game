import torch
import gym
from model import ActorCritic

# Setup
env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Load model
model = ActorCritic(obs_dim, n_actions)
model.load_state_dict(torch.load("ppo_scratch_cartpole_final.pth"))
model.eval()

# Evaluate
episodes = 5
for ep in range(episodes):
    obs = env.reset()[0]
    done = False
    ep_reward = 0

    while not done:
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward

    print(f"Episode {ep + 1} reward: {ep_reward}")

env.close()
