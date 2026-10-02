import numpy as np
import torch

from src.models.ppo_agent import (
    STATE_DIM,
    ACTION_DIM,
    PPOActor,
    PPOCritic,
    PPOAgent,
)


def test_dimensions():
    assert STATE_DIM == 111
    assert ACTION_DIM == 9


def test_actor_output_shape():
    actor = PPOActor()

    states = torch.randn(4, STATE_DIM)

    logits = actor(states)

    assert logits.shape == (4, ACTION_DIM)
    assert torch.isfinite(logits).all()


def test_critic_output_shape():
    critic = PPOCritic()

    states = torch.randn(4, STATE_DIM)

    values = critic(states)

    assert values.shape == (4,)
    assert torch.isfinite(values).all()


def test_actor_distribution():
    actor = PPOActor()

    states = torch.randn(4, STATE_DIM)

    distribution = actor.get_distribution(states)

    probabilities = distribution.probs

    assert probabilities.shape == (4, ACTION_DIM)
    assert torch.isfinite(probabilities).all()

    # Each state's action probabilities should sum to 1.
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )


def test_actor_action_selection():
    actor = PPOActor()

    state = torch.randn(STATE_DIM)

    action, log_probability = actor.get_action(state)

    assert action.shape == torch.Size([])
    assert log_probability.shape == torch.Size([])

    assert 0 <= action.item() < ACTION_DIM
    assert torch.isfinite(log_probability)


def test_agent_initialization():
    agent = PPOAgent()

    assert agent.state_dim == STATE_DIM
    assert agent.action_dim == ACTION_DIM

    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.actor_optimizer is not None
    assert agent.critic_optimizer is not None


def test_agent_select_action_numpy():
    agent = PPOAgent()

    state = np.random.default_rng(42).normal(
        size=STATE_DIM
    ).astype(np.float32)

    action, log_probability, value = agent.select_action(state)

    assert 0 <= action < ACTION_DIM
    assert np.isfinite(log_probability)
    assert np.isfinite(value)


def test_agent_evaluate_actions():
    agent = PPOAgent()

    states = torch.randn(8, STATE_DIM)
    actions = torch.randint(
        low=0,
        high=ACTION_DIM,
        size=(8,),
    )

    log_probabilities, values, entropy = agent.evaluate_actions(
        states,
        actions,
    )

    assert log_probabilities.shape == (8,)
    assert values.shape == (8,)
    assert entropy.shape == (8,)

    assert torch.isfinite(log_probabilities).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(entropy).all()


def test_learning_rates():
    agent = PPOAgent()

    actor_lr = agent.actor_optimizer.param_groups[0]["lr"]
    critic_lr = agent.critic_optimizer.param_groups[0]["lr"]

    assert actor_lr == 8e-4
    assert critic_lr == 8e-4