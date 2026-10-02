import torch

from src.transfer.soft_policy_distillation import (
    SoftPolicyDistillationConfig,
    SupportedTargetPricingPolicy,
    supported_discount_from_index,
    supported_discount_index,
)


def test_policy_shape():
    config = SoftPolicyDistillationConfig()
    model = SupportedTargetPricingPolicy(config)
    x = torch.randn(8, 111)
    assert model(x).shape == (8, 8)


def test_probabilities_sum_to_one():
    config = SoftPolicyDistillationConfig()
    model = SupportedTargetPricingPolicy(config)
    x = torch.randn(8, 111)
    p = model.probabilities(x)
    assert torch.allclose(
        p.sum(dim=1),
        torch.ones(8),
        atol=1e-6,
    )


def test_supported_discount_mapping():
    discounts = torch.tensor([5., 10., 20., 40.])
    idx = supported_discount_index(discounts)
    recovered = supported_discount_from_index(idx)
    assert torch.equal(
        idx,
        torch.tensor([0, 1, 3, 7]),
    )
    assert torch.equal(
        recovered,
        discounts,
    )
