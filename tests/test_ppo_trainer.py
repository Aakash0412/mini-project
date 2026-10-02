import numpy as np
import torch

from src.models.ppo_agent import PPOAgent
from src.models.ppo_trainer import (
    PPOConfig,
    PPORolloutBuffer,
    PPOTrainer,
)


def create_dummy_rollout(agent, length=16):
    rng = np.random.default_rng(42)

    buffer = PPORolloutBuffer()

    for i in range(length):
        state = rng.normal(
            size=agent.state_dim
        ).astype(np.float32)

        action, log_probability, value = agent.select_action(state)

        reward = float(rng.normal())

        done = i == length - 1

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            log_probability=log_probability,
            value=value,
            done=done,
        )

    return buffer


def test_default_ppo_config():
    config = PPOConfig()

    assert config.gamma == 0.9
    assert config.gae_lambda == 0.8
    assert config.clip_epsilon == 0.2
    assert config.batch_size == 64
    assert config.update_epochs == 4


def test_rollout_buffer():
    buffer = PPORolloutBuffer()

    state = np.zeros(111, dtype=np.float32)

    buffer.add(
        state=state,
        action=2,
        reward=1.0,
        log_probability=-0.5,
        value=0.2,
        done=False,
    )

    assert len(buffer) == 1

    assert buffer.states[0].shape == (111,)
    assert buffer.actions[0] == 2
    assert buffer.rewards[0] == 1.0
    assert buffer.dones[0] is False

    buffer.clear()

    assert len(buffer) == 0


def test_gae_output_shape():
    agent = PPOAgent()

    trainer = PPOTrainer(agent)

    rewards = np.array(
        [1.0, 0.5, -0.2, 0.8],
        dtype=np.float32,
    )

    values = np.array(
        [0.2, 0.3, 0.1, 0.4],
        dtype=np.float32,
    )

    dones = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    advantages, returns = trainer.compute_gae(
        rewards,
        values,
        dones,
        next_value=0.0,
    )

    assert advantages.shape == (4,)
    assert returns.shape == (4,)

    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()


def test_gae_terminal_state():
    agent = PPOAgent()

    trainer = PPOTrainer(agent)

    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    dones = np.array([1.0], dtype=np.float32)

    advantages, returns = trainer.compute_gae(
        rewards,
        values,
        dones,
        next_value=100.0,
    )

    # Terminal state must not bootstrap from next_value.
    assert np.isclose(
        advantages[0],
        0.5,
    )

    assert np.isclose(
        returns[0],
        1.0,
    )


def test_empty_buffer_rejected():
    agent = PPOAgent()

    trainer = PPOTrainer(agent)

    buffer = PPORolloutBuffer()

    try:
        trainer.update(buffer)
        assert False
    except ValueError:
        assert True


def test_ppo_update():
    torch.manual_seed(42)

    agent = PPOAgent()

    trainer = PPOTrainer(
        agent=agent,
        config=PPOConfig(
            update_epochs=1,
            batch_size=8,
        ),
    )

    buffer = create_dummy_rollout(
        agent,
        length=16,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.actor.named_parameters()
    }

    metrics = trainer.update(
        buffer,
        next_value=0.0,
    )

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "entropy" in metrics
    assert "mean_advantage" in metrics
    assert "mean_return" in metrics
    assert "samples" in metrics

    assert metrics["samples"] == 16

    for key, value in metrics.items():
        assert np.isfinite(value)

    changed = False

    for name, parameter in agent.actor.named_parameters():
        if not torch.equal(before[name], parameter.detach()):
            changed = True
            break

    assert changed


def test_ppo_update_rejects_empty_rollout():
    agent = PPOAgent()

    trainer = PPOTrainer(agent)

    buffer = PPORolloutBuffer()

    try:
        trainer.update(buffer)
        assert False
    except ValueError:
        assert True