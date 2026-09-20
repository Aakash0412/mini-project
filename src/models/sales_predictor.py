"""
Sales Predictor Module — Phase 5

Provides:
  SalesPredictorMLP   — the trained MLP architecture
  load_frozen_predictor() — loads weights and freezes the model for PPO use

After Phase 5, this module is the *only* interface the RL environment
should use to query sales response. The predictor's weights must not be
updated during PPO training.
"""

import os
import json
import torch
import torch.nn as nn


class SalesPredictorMLP(nn.Module):
    """
    MLP Sales Predictor.

    IMPLEMENTATION DECISION: Architecture not fully specified in paper.
    Architecture: 80 → 256 → BN → 128 → BN → 64 → 1
    See docs/SALES_PREDICTOR_SPECIFICATION.md for justification.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for RL environment usage.

        Args:
            state:  (B, state_dim) float tensor
            action: (B,) or (B, 1) float tensor — discount percentage

        Returns:
            predicted sales growth: (B,) float tensor
        """
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1)
        return self.forward(x)


def load_frozen_predictor(
    checkpoint_dir: str = "models/sales_predictor",
    device: torch.device = None,
) -> SalesPredictorMLP:
    """
    Load the trained sales predictor and freeze all parameters.

    This is the interface Phase 6 (RL environment) should call.
    The model will be in eval mode with requires_grad=False.

    Args:
        checkpoint_dir: directory containing best_model.pth and config.json
        device: target device (defaults to MPS if available, else CPU)

    Returns:
        Frozen SalesPredictorMLP
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run scripts/train_sales_predictor.py first."
        )

    with open(config_path) as f:
        config = json.load(f)

    model = SalesPredictorMLP(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    # FREEZE — critical: PPO must not update the predictor
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
