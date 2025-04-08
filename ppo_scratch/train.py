import torch
import gym
import numpy as np
from model import ActorCritic
from buffer import RolloutBuffer
from utils import plot_rewards

def main():
    # ======= Hyperparameters =======
    ENV_ID = "CartPole-v1"
    TOTAL_TIMESTEPS = 100_000
    ROLLOUT_LENGTH = 2048
    BATCH_SIZE = 64
    EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    POLICY_LR = 2.5e-4
    VALUE_LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======= Environment Setup =======
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # ======= Model + Optimizers =======
    model = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = torch.optim.Adam([
        {'params': model.policy_head.parameters(), 'lr': POLICY_LR},
        {'params': model.value_head.parameters(), 'lr': VALUE_LR}
    ])

    # ======= Buffer =======
    buffer = RolloutBuffer(ROLLOUT_LENGTH, obs_dim, DEVICE)

    # ======= Training Loop =======
    obs = env.reset()[0] 
    episode_rewards = []
    ep_reward = 0

    for timestep in range(1, TOTAL_TIMESTEPS + 1):
        state = torch.tensor(obs, dtype=torch.float32).to(DEVICE)

        # Get action, log prob, entropy, and value from policy
        with torch.no_grad():
            action, log_prob, _, value = model.get_action(state)

        # Step the env
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        ep_reward += reward

        # Store in buffer
        buffer.add(state, action, reward, done, log_prob, value)

        # Reset if done
        if done:
            obs = env.reset()[0]
            episode_rewards.append(ep_reward)
            ep_reward = 0
        else:
            obs = next_obs

        # When buffer is full, update the model
        if buffer.ptr == ROLLOUT_LENGTH:
            buffer.compute_returns_and_advantages(gamma=GAMMA, lam=GAE_LAMBDA)

            # Train for multiple epochs
            for _ in range(EPOCHS):
                for states, actions, old_log_probs, advantages, returns in buffer.get_batches(BATCH_SIZE):
                    # Re-evaluate policy and value
                    logits, values = model(states)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy()

                    # PPO objective: clipped surrogate
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                    # Value function loss (MSE)
                    value_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)

                    # Entropy bonus (optional for exploration)
                    entropy_bonus = entropy.mean()

                    # Total loss
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                    # Gradient step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f"Timestep: {timestep} | Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
            buffer.clear()

        if timestep >= TOTAL_TIMESTEPS:
            break


    # Save model
    torch.save(model.state_dict(), "ppo_scratch_cartpole_final.pth")

    # Plot rewards
    plot_rewards(episode_rewards, save_path="reward_plot.png")
    print("✅ Model saved and reward plot generated.")

if __name__ == "__main__":
    main()