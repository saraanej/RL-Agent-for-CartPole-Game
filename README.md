# PPO Reinforcement Learning — CartPole Agent

This repository contains two implementations of **Proximal Policy Optimization (PPO)** to solve the classic `CartPole-v1` environment:

1. **ppo_stablebaseline/** — PPO using Stable-Baselines3 (high-level framework)
2. **ppo_scratch/** — PPO built from scratch in PyTorch (for full understanding)

The goal is to both *quickly prototype an agent* using a battle-tested library, and *deeply understand* how PPO works by building it from the ground up.

---

## Project Structure

```
RL-Agent-for-CartPole-Game/
├── ppo_stablebaseline/    # Fast implementation using Stable-Baselines3
├── ppo_scratch/           # Custom PPO implementation using PyTorch
├── videos/                # Saved gameplay recordings
├── models/                # Trained model checkpoints
└── logs/                  # Training logs for plotting and monitoring
```

---

## Environments Used

- `CartPole-v1` from OpenAI Gym — a simple control environment to test PPO

---

## Setup

Depending on your use, you can install dependencies separately in each folder:

```bash
# Example: install Stable-Baselines version dependencies
cd ppo_stablebaseline
pip install -r requirements.txt

# Example: install PPO scratch version dependencies
cd ppo_scratch
pip install -r requirements.txt
```

---

## Example Output

You can see the trained agent in action by running:

```bash
cd ppo_stablebaseline
python record_video.py
```

A video will be saved in the `videos/` folder.

---

## Why Two Implementations?

| Folder              | Purpose                                            |
|---------------------|----------------------------------------------------|
| `ppo_stablebaseline` | Quick deployment, clean interface, production-ready |
| `ppo_scratch`        | Learn PPO internals, understand every math step     |



---

## References

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

