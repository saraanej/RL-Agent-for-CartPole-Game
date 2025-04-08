import matplotlib.pyplot as plt

def plot_rewards(rewards, save_path="reward_plot.png", window=10):
    """
    Plot the episode rewards over time.

    Args:
        rewards (list): List of episode rewards.
        save_path (str): File path to save the plot.
        window (int): Smoothing window size for moving average.
    """
    if len(rewards) < window:
        window = 1
    smoothed = [sum(rewards[max(0, i - window):i + 1]) / (i - max(0, i - window) + 1) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.plot(smoothed, label=f"{window}-Episode Moving Avg", linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
