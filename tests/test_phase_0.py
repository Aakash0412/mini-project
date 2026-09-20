import os
import yaml
import pytest
import torch

def test_imports():
    import torch
    import torchvision
    import transformers
    import sentence_transformers
    import pandas
    import numpy
    import gymnasium
    import stable_baselines3
    
def test_configuration_loading():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/base.yaml")
    assert os.path.exists(config_path), "Configuration file missing"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert "hardware" in config
    assert "feature_extraction" in config
    
def test_device_detection():
    # At minimum CPU should be available
    assert torch.device("cpu").type == "cpu"
    
def test_mps_tensor_operation():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(2, 2).to(device)
        y = x * 2
        z = y.to("cpu")
        assert torch.all(z == 2)
    else:
        pytest.skip("MPS not available")

def test_path_configuration():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    assert paths.get("raw_data") == "data/raw"
    assert paths.get("interim_data") == "data/interim"
    assert paths.get("processed_data") == "data/processed"

def test_no_openai_env_required():
    assert os.environ.get("OPENAI_API_KEY") is None, "OPENAI_API_KEY should not be present in the environment."

def test_configuration_validity():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert config["hardware"]["max_ram_gb"] <= 8, "Configuration must respect 8GB RAM limit"
