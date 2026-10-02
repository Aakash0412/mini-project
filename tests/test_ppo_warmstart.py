import numpy as np
import torch

from src.transfer.ppo_warmstart import (
    WarmStartActor,
    gaussian_discount_distribution,
)


def test_warm_start_actor_shape():
    actor = WarmStartActor(
        state_dim=111,
        action_dim=8,
    )

    states = torch.randn(
        16,
        111,
    )

    logits = actor(
        states
    )

    assert logits.shape == (
        16,
        8,
    )


def test_gaussian_distribution_shape():
    mean = torch.zeros(
        32,
        2,
    )

    std = torch.ones(
        32,
        2,
    ) * 0.02

    distribution = gaussian_discount_distribution(
        mean,
        std,
    )

    assert distribution.shape == (
        32,
        8,
    )

    assert torch.all(
        torch.isfinite(distribution)
    )

    assert torch.all(
        distribution >= 0
    )

    assert torch.allclose(
        distribution.sum(dim=1),
        torch.ones(32),
        atol=1e-5,
    )


def test_transfer_distribution_is_not_hard_label():
    mean = torch.tensor(
        [
            [0.1, 0.177],
            [0.1, 0.15],
        ],
        dtype=torch.float32,
    )

    std = torch.tensor(
        [
            [0.15, 0.02],
            [0.15, 0.02],
        ],
        dtype=torch.float32,
    )

    distribution = gaussian_discount_distribution(
        mean,
        std,
    )

    # The Gaussian mapping should produce a valid soft distribution.
    assert np.all(
        distribution.detach().numpy() > 0
    )