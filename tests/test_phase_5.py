"""Phase 5 tests — Sales Predictor"""

import sys
import os
import json
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_weights_exist():
    assert os.path.exists("models/sales_predictor/best_model.pth"), \
        "Model weights missing — run train_sales_predictor.py"


def test_model_config_exists():
    assert os.path.exists("models/sales_predictor/config.json")
    with open("models/sales_predictor/config.json") as f:
        cfg = json.load(f)
    assert "input_dim" in cfg
    assert cfg["input_dim"] == 112   # 111 state + 1 action
    assert cfg.get("state_dim") == 111


def test_model_metrics_exist():
    assert os.path.exists("models/sales_predictor/metrics.json")
    with open("models/sales_predictor/metrics.json") as f:
        m = json.load(f)
    assert "test_MAE" in m
    assert "test_RMSE" in m
    assert "test_R2" in m
    assert m["test_MAE"] > 0
    assert m["test_RMSE"] > 0


def test_load_frozen_predictor():
    """Model loads and is properly frozen."""
    from src.models.sales_predictor import load_frozen_predictor
    model = load_frozen_predictor(device=torch.device("cpu"))
    assert not any(p.requires_grad for p in model.parameters()), \
        "Model must be frozen (all requires_grad=False)"


def test_predictor_forward_shape():
    """Forward pass returns correct output shape."""
    from src.models.sales_predictor import load_frozen_predictor
    model = load_frozen_predictor(device=torch.device("cpu"))
    model.eval()
    x = torch.randn(8, 112)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_predictor_no_nan():
    """Predictions are finite."""
    from src.models.sales_predictor import load_frozen_predictor
    model = load_frozen_predictor(device=torch.device("cpu"))
    model.eval()
    x = torch.randn(32, 112)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_predict_method():
    """Convenience predict() method works correctly."""
    from src.models.sales_predictor import load_frozen_predictor
    model = load_frozen_predictor(device=torch.device("cpu"))
    model.eval()
    state  = torch.randn(4, 111)
    action = torch.tensor([0.0, 10.0, 20.0, 40.0])
    with torch.no_grad():
        out = model.predict(state, action)
    assert out.shape == (4,)


def test_sanity_check_report_exists():
    assert os.path.exists("reports/sales_response_curves/sanity_check_issues.json")
    with open("reports/sales_response_curves/sanity_check_issues.json") as f:
        report = json.load(f)
    assert "issues" in report
    # Sanity check should have found no critical issues
    assert len(report["issues"]) == 0, f"Sanity issues: {report['issues']}"
