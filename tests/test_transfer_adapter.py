import torch

from src.transfer.transfer_adapter import TransferAdapter, TransferAdapterConfig


def test_transfer_adapter_shape():
    config = TransferAdapterConfig()
    source_min = torch.zeros(4)
    source_max = torch.tensor([168.0, 3.2142856, 51.14402, 6.0])
    model = TransferAdapter(config, source_min, source_max)

    x = torch.randn(8, 111)
    y = model(x)

    assert y.shape == (8, 4)
    assert torch.isfinite(y).all()
    assert torch.all(y >= source_min)
    assert torch.all(y <= source_max)


def test_transfer_adapter_rejects_wrong_dimension():
    model = TransferAdapter()
    bad = torch.randn(4, 110)

    try:
        model(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for incorrect target state dimension")
