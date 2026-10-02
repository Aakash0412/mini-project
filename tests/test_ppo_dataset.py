import numpy as np

from src.environment.ppo_dataset import load_ppo_dataset


def test_load_ppo_dataset():
    dataset = load_ppo_dataset()

    assert dataset.states.ndim == 2
    assert dataset.states.shape[1] == 111

    assert len(dataset.states) == len(dataset.products)
    assert len(dataset.states) == len(dataset.source_row_indices)

    assert len(dataset.states) == 32523

    assert np.isfinite(dataset.states).all()


def test_product_context_values_are_valid():
    dataset = load_ppo_dataset()

    product = dataset.products[0]

    assert product.observed_price > 0
    assert 0 <= product.observed_discount <= 100
    assert product.last_monthly_sales >= 0
    assert product.gross_margin_rate is not None
    assert np.isfinite(product.gross_margin_rate)


def test_source_indices_are_unique():
    dataset = load_ppo_dataset()

    assert len(
        np.unique(dataset.source_row_indices)
    ) == len(dataset.source_row_indices)