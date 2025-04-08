import torch
import numpy as np

class RolloutBuffer:
    """
    Rollout Buffer for PPO.

    Stores transitions collected during interaction with the environment,
    and computes advantages and returns at the end of each rollout.
    """

    def __init__(self, buffer_size, state_dim, device):
        """
        Initialize the buffer.

        Args:
            buffer_size (int): Number of transitions to store per rollout.
            state_dim (int): Dimension of the input state.
            device (torch.device): Device to store data on (CPU/GPU).
        """
        self.device = device
        self.buffer_size = buffer_size
        self.ptr = 0  # pointer to current insert index

        # Allocate space for rollout data
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)

        # Return: Total future reward from a time step onward. If I take this action right now, how much total reward will I get afterward?
        #         This is used to train the value network (critic). The critic tries to learn: "How good is this state?" -> V(s).
        # Advantage: How much better was this action than expected? Was this action actually better than average, or just average?
        #         This is used to train the policy network (actor). The actor tries to learn: "How good is this action?" -> A(s,a) to push probabilities of good actions.
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)

    def add(self, state, action, reward, done, log_prob, value):
        """
        Add a single transition to the buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        """
        Compute discounted returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            gamma (float): Discount factor.
            lam (float): GAE lambda parameter.
        """
        adv = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = 0
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            adv = delta + gamma * lam * (1 - next_done) * adv
            self.advantages[t] = adv

        self.returns = self.advantages + self.values

        # Normalize advantages (helps training)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        """
        Yield mini-batches of data for PPO training.

        Args:
            batch_size (int): Size of each mini-batch.
        """
        indices = np.random.permutation(self.buffer_size) # Shuffle indices
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx]
            )

    def clear(self):
        """
        Reset the buffer for the next rollout.
        """
        self.ptr = 0
