import torch

from src.transfer.soft_policy_distillation_v2 import (
    SoftPolicyDistillationV2Config,
    SupportedTargetPricingPolicyV2,
    continuous_to_soft_bins,
    discount_from_index,
    discount_index,
)


def test_policy_shape():
    model = SupportedTargetPricingPolicyV2(
        SoftPolicyDistillationV2Config()
    )
    x = torch.randn(8, 111)
    assert model(x).shape == (8, 8)


def test_soft_bins_are_per_sample_and_normalized():
    discounts = torch.tensor([10., 20., 30.])
    target = continuous_to_soft_bins(discounts)

    assert target.shape == (3, 8)
    assert torch.allclose(
        target.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )

    assert target[0, 1] > target[0, 0]
    assert target[0, 1] > target[0, 2]


def test_discount_mapping():
    discounts = torch.tensor([5., 10., 20., 40.])
    indices = discount_index(discounts)
    recovered = discount_from_index(indices)

    assert torch.equal(
        indices,
        torch.tensor([0, 1, 3, 7]),
    )
    assert torch.equal(recovered, discounts)


def test_soft_bins_do_not_create_artificial_endpoints():
    discounts = torch.tensor([20.])
    target = continuous_to_soft_bins(
        discounts,
        bandwidth=3.0,
    )

    # 20% is index 3; nearby 15% and 25% should receive more
    # mass than the distant 5% endpoint.
    assert target[0, 3] == target.max()
    assert target[0, 0] < target[0, 3]
    assert target[0, 7] < target[0, 3]
