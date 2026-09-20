import numpy as np
import pandas as pd
import json

def validate_state():
    print("Validating state...")
    
    state = np.load("data/processed/state/state.npy")
    
    with open("reports/state_validation.json", "r") as f:
        stats = json.load(f)
        
    assert state.shape[1] == 79, f"Expected 79-D state due to missing Image and Sentiment, got {state.shape[1]}"
    assert not np.isnan(state).any(), "NaN found in state"
    assert not np.isinf(state).any(), "Inf found in state"
    
    index = pd.read_parquet("data/processed/features/feature_index.parquet")
    assert state.shape[0] == len(index), "State length mismatch with index"
    
    print("State validation passed!")

if __name__ == "__main__":
    validate_state()
