import torch

from src.transfer.policy_distillation import (
    PolicyDistillationConfig,
    TargetPricingPolicy,
    discount_from_index,
    discount_index,
)


def test_target_policy_shape():
    cfg = PolicyDistillationConfig()
    model = TargetPricingPolicy(cfg)
    x = torch.randn(8, 111)
    logits = model(x)
    assert logits.shape == (8, 9)


def test_probabilities_sum_to_one():
    model = TargetPricingPolicy(PolicyDistillationConfig())
    p = model.probabilities(torch.randn(4, 111))
    assert p.shape == (4, 9)
    assert torch.allclose(p.sum(dim=1), torch.ones(4), atol=1e-6)


def test_discount_mapping():
    discounts = torch.tensor([0., 5., 10., 20., 40.])
    indices = discount_index(discounts)
    assert indices.tolist() == [0, 1, 2, 4, 8]
    assert torch.equal(discount_from_index(indices), discounts)
