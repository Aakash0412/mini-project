"""
src/environment/counterfactual_simulator.py — Phase 8

Counterfactual Sales Simulator
===============================

Wraps the FROZEN 112-D sales predictor to answer:

    "Given a product state, what would the predicted monthly sales
     growth rate be at each possible discount level?"

This is the foundation for counterfactual reasoning in the PPO
pricing environment (Phase 10).

Architecture alignment
----------------------
Paper: "A Multimodal Deep Reinforcement Learning Framework for Dynamic
        Pricing Optimisation in E-Commerce" (Liu et al., IEEE Access 2026)

The paper's RL agent queries a sales-response model to evaluate candidate
prices without interacting with a live market.  This module implements that
query interface using the trained MLP predictor as the surrogate sales model.

State representation
--------------------
ItaNet-111 (this project's available implementation):
    Image features      :  32-D  (ResNet-50 + JL projection, seed=42)
    Text features       :  32-D  (sentence-transformer + JL projection)
    Attribute features  :  47-D  (one-hot / ordinal encoding)
    ─────────────────────────────
    Total state dim     : 111-D

Paper target (unavailable — review text not in dataset):
    Sentiment features  :  10-D  (would add to 121-D)

Predictor input
---------------
state (111-D)  ‖  discount (1-D)  =  112-D  →  MLP  →  sales growth rate

Discount action space
---------------------
DISCOUNT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40]  (percentage)

IMPORTANT — 0% discount:
    0% is included because it is part of the paper-defined action space.
    However, 0% discounts were NOT observed in the training data
    (observed range: 5–40%).  Predictions at 0% are extrapolations
    beyond the training support and should be treated with caution.
    This is documented here rather than excluded, to match the paper's
    action-space definition.

Usage
-----
    from src.environment.counterfactual_simulator import CounterfactualSimulator

    sim = CounterfactualSimulator()                  # loads frozen predictor
    state = np.load("data/processed/state/state_train.npy")[0]  # (111,)

    # Single discount query
    pred = sim.predict(state, discount=20)           # float

    # All 9 discounts at once
    preds = sim.predict_all_discounts(state)         # dict[int, float]
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve project root so this module is importable from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.sales_predictor import SalesPredictorMLP, load_frozen_predictor  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM: int = 111
"""Expected ItaNet-111 state dimension."""

INPUT_DIM: int = 112
"""Predictor input dimension = STATE_DIM + 1 (discount)."""

DISCOUNT_BINS: List[int] = [0, 5, 10, 15, 20, 25, 30, 35, 40]
"""
Paper-defined discrete discount action space (%).

NOTE: 0% is out-of-training-support (observed discounts: 5–40%).
Predictions at 0% are extrapolations and should be treated with caution.
"""

DEFAULT_CHECKPOINT_DIR: str = os.path.join(_PROJECT_ROOT, "models", "sales_predictor")

# ---------------------------------------------------------------------------
# CounterfactualSimulator
# ---------------------------------------------------------------------------


class CounterfactualSimulator:
    """
    Counterfactual sales simulator backed by the frozen 112-D MLP predictor.

    The simulator is read-only: it never updates the predictor weights.
    It is safe to call from PPO training without gradient pollution.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Directory containing ``best_model.pth`` and ``config.json``.
        Defaults to ``models/sales_predictor/`` relative to the project root.
    device : torch.device, optional
        Inference device.  Defaults to CPU (required: no CUDA assumed).

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or required files are missing.
    ValueError
        If the loaded predictor's ``input_dim`` does not equal 112.
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            # CPU-first; Windows/Linux with no CUDA
            device = torch.device("cpu")
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self._model: SalesPredictorMLP = self._load_and_validate(checkpoint_dir, device)

        # Cache config for inspection
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, encoding="utf-8") as f:
            self.config: Dict = json.load(f)

        logger.info(
            "CounterfactualSimulator ready — predictor input_dim=%d, device=%s",
            self.config.get("input_dim"),
            device,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_validate(checkpoint_dir: str, device: torch.device) -> SalesPredictorMLP:
        """Load predictor through the project's canonical loading mechanism and validate."""
        model = load_frozen_predictor(checkpoint_dir=checkpoint_dir, device=device)

        # Confirm the first layer weight shape matches INPUT_DIM
        first_weight = next(
            (p for name, p in model.named_parameters() if "weight" in name), None
        )
        if first_weight is None:
            raise RuntimeError("Predictor has no named weight parameters.")

        actual_input_dim = first_weight.shape[1]
        if actual_input_dim != INPUT_DIM:
            raise ValueError(
                f"Predictor input_dim={actual_input_dim} but expected {INPUT_DIM}. "
                "Has the predictor been retrained with a different state dimension?"
            )

        return model

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        """Validate and reshape state to (STATE_DIM,)."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"state must have shape ({STATE_DIM},), got {state.shape}. "
                "Ensure the ItaNet-111 state pipeline has been run."
            )
        if not np.all(np.isfinite(state)):
            raise ValueError(
                "state contains NaN or Inf values. "
                "Check that this row is in state_row_indices.npy (13 rows are excluded)."
            )
        return state

    @staticmethod
    def _validate_discount(discount: Union[int, float]) -> float:
        """Validate that discount is one of the nine paper-defined bins."""
        discount = float(discount)
        if discount not in [float(b) for b in DISCOUNT_BINS]:
            raise ValueError(
                f"discount={discount} is not in DISCOUNT_BINS={DISCOUNT_BINS}. "
                "Only paper-defined discrete discounts are supported."
            )
        return discount

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        state: np.ndarray,
        discount: Union[int, float],
    ) -> float:
        """
        Predict the Estimated Monthly Sales Growth Rate for a single
        (state, discount) pair.

        Parameters
        ----------
        state : np.ndarray, shape (111,)
            ItaNet-111 feature vector for one product observation.
        discount : int or float
            Discount percentage.  Must be one of DISCOUNT_BINS.

        Returns
        -------
        float
            Predicted monthly sales growth rate (raw, not log-transformed).
            The predictor was trained on the raw target; no inverse transform
            is required.

        Notes
        -----
        0% discount is valid to query but is outside observed training support
        (5–40% observed).  Treat extrapolated predictions with caution.
        """
        state = self._validate_state(state)
        discount = self._validate_discount(discount)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 111)
        action_t = torch.tensor([[discount]], dtype=torch.float32, device=self.device)        # (1, 1)

        with torch.no_grad():
            pred = self._model.predict(state_t, action_t)  # (1,)

        return float(pred.item())

    def predict_all_discounts(self, state: np.ndarray) -> Dict[int, float]:
        """
        Predict the sales growth rate at every paper-defined discount level
        for a single product state.

        Parameters
        ----------
        state : np.ndarray, shape (111,)
            ItaNet-111 feature vector for one product observation.

        Returns
        -------
        dict
            Mapping ``{discount_pct: predicted_sales_growth_rate}`` for all
            9 discount levels in DISCOUNT_BINS.

            Key   : int  — discount percentage
            Value : float — predicted monthly sales growth rate

        Notes
        -----
        0% discount is included because it is part of the paper-defined action
        space, but it is outside the observed training support (5–40%).
        """
        state = self._validate_state(state)

        n = len(DISCOUNT_BINS)
        states_t = torch.tensor(
            np.tile(state, (n, 1)), dtype=torch.float32, device=self.device
        )  # (9, 111)
        discounts_t = torch.tensor(
            [[float(d)] for d in DISCOUNT_BINS], dtype=torch.float32, device=self.device
        )  # (9, 1)

        with torch.no_grad():
            preds = self._model.predict(states_t, discounts_t)  # (9,)

        return {int(d): float(p) for d, p in zip(DISCOUNT_BINS, preds.cpu().numpy())}

    def predict_batch(
        self,
        states: np.ndarray,
        discounts: Union[np.ndarray, List[Union[int, float]]],
    ) -> np.ndarray:
        """
        Predict sales growth rates for a batch of (state, discount) pairs.

        Parameters
        ----------
        states : np.ndarray, shape (B, 111)
            Batch of ItaNet-111 state vectors.
        discounts : array-like, shape (B,)
            Discount percentages for each state.  Each must be in DISCOUNT_BINS.

        Returns
        -------
        np.ndarray, shape (B,)
            Predicted sales growth rates.
        """
        states = np.asarray(states, dtype=np.float32)
        discounts = np.asarray(discounts, dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(
                f"states must have shape (B, {STATE_DIM}), got {states.shape}."
            )
        if discounts.ndim != 1 or len(discounts) != len(states):
            raise ValueError(
                f"discounts must have shape (B,) matching states ({len(states)}), "
                f"got {discounts.shape}."
            )

        invalid = [d for d in discounts if float(d) not in [float(b) for b in DISCOUNT_BINS]]
        if invalid:
            raise ValueError(f"Invalid discount values: {invalid}. Must be in {DISCOUNT_BINS}.")

        if not np.all(np.isfinite(states)):
            raise ValueError("states batch contains NaN or Inf values.")

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        discounts_t = torch.tensor(discounts, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            preds = self._model.predict(states_t, discounts_t)

        return preds.cpu().numpy()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        """Predictor input dimension (always 112 = 111 state + 1 discount)."""
        return INPUT_DIM

    @property
    def state_dim(self) -> int:
        """State dimension (always 111 for ItaNet-111)."""
        return STATE_DIM

    @property
    def discount_bins(self) -> List[int]:
        """Paper-defined discrete action space."""
        return list(DISCOUNT_BINS)

    def summary(self) -> str:
        """Return a human-readable summary of the simulator configuration."""
        lines = [
            "CounterfactualSimulator",
            f"  state_dim    : {STATE_DIM}  (ItaNet-111: Image 32 + Text 32 + Attrs 47)",
            f"  input_dim    : {INPUT_DIM}  (state + discount)",
            f"  discount_bins: {DISCOUNT_BINS}",
            f"  device       : {self.device}",
            f"  checkpoint   : {self.checkpoint_dir}",
            f"  architecture : {self.config.get('architecture', 'N/A')}",
            f"  train_samples: {self.config.get('train_samples', 'N/A')}",
            f"  val_samples  : {self.config.get('val_samples', 'N/A')}",
            "",
            "  NOTE: 0% discount is out-of-training-support (observed: 5-40%).",
            "        Predictions at 0% are extrapolations — treat with caution.",
        ]
        return "\n".join(lines)
