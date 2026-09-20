import numpy as np
import pandas as pd
import os
import json

def build_state():
    print("Building state...")
    os.makedirs("data/processed/state", exist_ok=True)
    
    attrs = np.load("data/processed/features/attribute_features.npy")
    text = np.load("data/processed/features/text_features.npy")
    
    # Concatenate features: Text (32) + Attributes (47) = 79
    state = np.concatenate([text, attrs], axis=1)
    
    np.save("data/processed/state/state.npy", state)
    
    # Save statistics
    stats = {
        "num_samples": state.shape[0],
        "state_dimension": state.shape[1],
        "modalities_included": ["Text", "Attributes"],
        "modalities_missing": ["Image", "Sentiment"],
        "has_nan": bool(np.isnan(state).any()),
        "has_inf": bool(np.isinf(state).any())
    }
    
    with open("reports/state_validation.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    with open("reports/state_statistics.md", "w") as f:
        f.write("# State Statistics\n\n")
        f.write(f"- **Total Samples:** {stats['num_samples']}\n")
        f.write(f"- **Final State Dimension:** {stats['state_dimension']} (Expected 79 due to missing modalities)\n")
        f.write(f"- **Included Modalities:** {', '.join(stats['modalities_included'])}\n")
        f.write(f"- **Missing Modalities:** {', '.join(stats['modalities_missing'])}\n")
        f.write(f"- **Contains NaN:** {stats['has_nan']}\n")
        f.write(f"- **Contains Inf:** {stats['has_inf']}\n")
        
    print(f"State constructed with shape {state.shape}")

if __name__ == "__main__":
    build_state()
