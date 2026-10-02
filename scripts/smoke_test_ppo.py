import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.environment.pricing_environment import DynamicPricingEnvironment
from src.models.ppo_agent import PPOAgent
from src.models.ppo_trainer import PPOConfig, PPOTrainer, PPORolloutBuffer


STATE_PATH = "data/processed/state/state.npy"

SMOKE_EPISODES = 5
STEPS_PER_EPISODE = 8


def main():
    print("=" * 60)
    print("PHASE 11C - PPO END-TO-END SMOKE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load real ItaNet states
    # ---------------------------------------------------------
    print("\n[1] Loading real ItaNet states...")

    states = np.load(STATE_PATH)

    print("State matrix shape:", states.shape)

    assert states.ndim == 2
    assert states.shape[1] == 111
    assert np.isfinite(states).all()

    print("OK -- 111-D states loaded")

    # ---------------------------------------------------------
    # 2. Create real pricing environment
    # ---------------------------------------------------------
    print("\n[2] Creating Dynamic Pricing Environment...")

    environment = DynamicPricingEnvironment()

    print("State dimension :", environment.state_dim)
    print("Action dimension:", environment.action_dim)

    assert environment.state_dim == 111
    assert environment.action_dim == 9

    print("OK -- environment initialized")

    # ---------------------------------------------------------
    # 3. Create PPO agent
    # ---------------------------------------------------------
    print("\n[3] Creating PPO Agent...")

    agent = PPOAgent()

    print("Actor state dimension :", agent.state_dim)
    print("Actor action dimension:", agent.action_dim)

    # ---------------------------------------------------------
    # 4. Create PPO trainer
    # ---------------------------------------------------------
    print("\n[4] Creating PPO Trainer...")

    config = PPOConfig(
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        batch_size=64,
        update_epochs=2,
    )

    trainer = PPOTrainer(
        agent=agent,
        config=config,
        device="cpu",
    )

    print("Gamma       :", config.gamma)
    print("GAE lambda  :", config.gae_lambda)
    print("Clip epsilon:", config.clip_epsilon)

    # ---------------------------------------------------------
    # 5. Run small end-to-end rollout
    # ---------------------------------------------------------
    print("\n[5] Running smoke-training episodes...")

    rng = np.random.default_rng(42)

    all_rewards = []

    for episode in range(SMOKE_EPISODES):

        # Pick a real state from the dataset.
        state_index = int(
            rng.integers(0, len(states))
        )

        state = states[state_index]

        environment.reset(state)

        buffer = PPORolloutBuffer()

        episode_reward = 0.0

        for step in range(STEPS_PER_EPISODE):

            action, log_probability, value = agent.select_action(
                state
            )

            result = environment.step(
                action=action
            )

            next_state = result.state

            reward = result.reward

            done = (
                step == STEPS_PER_EPISODE - 1
            )

            buffer.add(
                state=state,
                action=action,
                reward=reward,
                log_probability=log_probability,
                value=value,
                done=done,
            )

            episode_reward += reward

            state = next_state

        # -----------------------------------------------------
        # PPO update
        # -----------------------------------------------------
        metrics = trainer.update(
            buffer=buffer,
            next_value=0.0,
        )

        all_rewards.append(episode_reward)

        print(
            f"Episode {episode + 1}/{SMOKE_EPISODES} | "
            f"Reward: {episode_reward:.6f} | "
            f"Actor Loss: {metrics['actor_loss']:.6f} | "
            f"Critic Loss: {metrics['critic_loss']:.6f} | "
            f"Entropy: {metrics['entropy']:.6f}"
        )

    # ---------------------------------------------------------
    # 6. Final validation
    # ---------------------------------------------------------
    print("\n[6] Final validation...")

    rewards = np.asarray(
        all_rewards,
        dtype=np.float32,
    )

    assert len(rewards) == SMOKE_EPISODES
    assert np.isfinite(rewards).all()

    # Verify PPO can still produce a valid action.
    test_state = states[0]

    action, log_probability, value = agent.select_action(
        test_state
    )

    assert 0 <= action < 9
    assert np.isfinite(log_probability)
    assert np.isfinite(value)

    print("Final action           :", action)
    print("Final discount         :", environment.action_to_discount(action))
    print("Final log probability  :", log_probability)
    print("Final state value      :", value)

    print("\n" + "=" * 60)
    print("PHASE 11C SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()