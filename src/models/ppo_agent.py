import torch
import torch.nn as nn
from torch.distributions import Categorical


STATE_DIM = 111
ACTION_DIM = 9


class PPOActor(nn.Module):
    """
    PPO policy network.

    Input:
        111-D ItaNet state

    Output:
        Probability distribution over 9 discrete discount actions.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return logits

    def get_distribution(
        self,
        state: torch.Tensor,
    ) -> Categorical:
        logits = self.forward(state)
        return Categorical(logits=logits)

    def get_action(
        self,
        state: torch.Tensor,
    ):
        distribution = self.get_distribution(state)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        return action, log_probability


class PPOCritic(nn.Module):
    """
    PPO value network.

    Input:
        111-D ItaNet state

    Output:
        Scalar state-value estimate V(s).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOAgent:
    """
    PPO actor-critic wrapper for the dynamic pricing environment.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        actor_lr: float = 8e-4,
        critic_lr: float = 8e-4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = PPOActor(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.critic = PPOCritic(
            state_dim=state_dim,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

    def select_action(self, state):
        """
        Select a discrete pricing action.

        Returns:
            action
            log probability
            state value
        """

        if not torch.is_tensor(state):
            state = torch.tensor(
                state,
                dtype=torch.float32,
            )

        if state.ndim == 1:
            state = state.unsqueeze(0)

        distribution = self.actor.get_distribution(state)

        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        value = self.critic(state)

        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Evaluate states/actions during PPO optimization.
        """

        distribution = self.actor.get_distribution(states)

        log_probabilities = distribution.log_prob(actions)

        entropy = distribution.entropy()

        values = self.critic(states)

        return (
            log_probabilities,
            values,
            entropy,
        )