import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic neural network for PPO.

    This model has:
    - A shared base network to process the input state
    - A policy head that outputs action logits (actor)
    - A value head that outputs the estimated state value (critic)
    """

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        """
        Initialize the ActorCritic model.

        Args:
            input_dim (int): Dimension of the input (observation space).
            action_dim (int): Number of possible actions (action space).
            hidden_dim (int): Size of hidden layers. Default: 64.
        """
        super().__init__()

        # Shared base layers for feature extraction
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head: outputs logits for each action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head: outputs a single scalar representing V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass through both policy and value networks.

        Args:
            x (Tensor): The input state.

        Returns:
            logits (Tensor): Raw scores for each action.
            value (Tensor): Estimated value of the input state.
        """
        base_out = self.base(x)
        logits = self.policy_head(base_out)
        value = self.value_head(base_out)
        return logits, value

    def get_action(self, obs):
        """
        Sample an action from the policy for the given observation.

        Args:
            obs (Tensor): A batch of observations.

        Returns:
            action (Tensor): Sampled action from the policy.
            log_prob (Tensor): Log probability of the sampled action.
            entropy (Tensor): Entropy of the action distribution (used for exploration bonus).
            value (Tensor): Estimated value of the state.
        """
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        dist = torch.distributions.Categorical(probs)  # Create a categorical distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
