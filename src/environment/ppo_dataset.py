from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.pricing_environment import ProductContext


@dataclass
class PPODataset:
    states: np.ndarray
    products: list[ProductContext]
    source_row_indices: np.ndarray


def load_ppo_dataset(
    state_path: str | Path = "data/processed/state/state.npy",
    index_path: str | Path = "data/processed/state/state_row_indices.npy",
    products_path: str | Path = "data/processed/products_clean.parquet",
) -> PPODataset:
    """
    Load states and align them with product-level economic metadata.

    Only rows with a valid Gross Margin Rate are retained because the
    paper-style reward requires delta_margin.
    """

    states = np.load(state_path)
    source_indices = np.load(index_path)

    df = pd.read_parquet(products_path)

    if len(states) != len(source_indices):
        raise ValueError(
            "State matrix and state-row-index array have different lengths."
        )

    if states.ndim != 2 or states.shape[1] != 111:
        raise ValueError(
            f"Expected state matrix shape (N, 111), got {states.shape}"
        )

    if not np.all(np.isfinite(states)):
        raise ValueError("State matrix contains NaN or Inf.")

    if np.any(source_indices < 0) or np.any(source_indices >= len(df)):
        raise ValueError("State row indices contain invalid dataset rows.")

    aligned = df.iloc[source_indices].copy()
    aligned["_state_position"] = np.arange(len(aligned))

    required_columns = [
        "Buybox Price (USD)",
        "Discount",
        "Last Monthly Sales",
        "Gross Margin Rate",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in aligned.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required product columns: {missing_columns}"
        )

    # Paper-style reward requires margin.
    valid_margin = aligned["Gross Margin Rate"].notna()

    aligned = aligned.loc[valid_margin].copy()

    retained_positions = aligned["_state_position"].to_numpy(
        dtype=np.int64
    )

    filtered_states = states[retained_positions]

    if len(filtered_states) != len(aligned):
        raise RuntimeError(
            "State/product alignment failed after margin filtering."
        )

    products = []

    for _, row in aligned.iterrows():
        products.append(
            ProductContext(
                observed_price=float(row["Buybox Price (USD)"]),
                observed_discount=float(row["Discount"]),
                last_monthly_sales=float(row["Last Monthly Sales"]),
                gross_margin_rate=float(row["Gross Margin Rate"]),
            )
        )

    return PPODataset(
        states=filtered_states.astype(np.float32),
        products=products,
        source_row_indices=source_indices[retained_positions],
    )