import numpy as np
import pandas as pd
import os

def validate_features():
    print("Validating features...")
    
    assert os.path.exists("data/processed/features/feature_index.parquet")
    assert os.path.exists("data/processed/features/attribute_features.npy")
    assert os.path.exists("data/processed/features/text_features.npy")
    
    index = pd.read_parquet("data/processed/features/feature_index.parquet")
    N = len(index)
    
    attrs = np.load("data/processed/features/attribute_features.npy")
    text = np.load("data/processed/features/text_features.npy")
    
    assert attrs.shape == (N, 47), f"Expected {(N, 47)}, got {attrs.shape}"
    assert text.shape == (N, 32), f"Expected {(N, 32)}, got {text.shape}"
    
    assert not np.isnan(attrs).any(), "NaN found in attributes"
    assert not np.isinf(attrs).any(), "Inf found in attributes"
    
    assert not np.isnan(text).any(), "NaN found in text features"
    assert not np.isinf(text).any(), "Inf found in text features"
    
    print("All feature validations passed!")

if __name__ == "__main__":
    validate_features()
