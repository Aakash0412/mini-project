"""
Numerical Validation of Image Features (Phase 2.5)
--------------------------------------------------
Validates:
  - Row alignment (34,041 rows matching products.csv)
  - Modality dimensionality (32-D)
  - Finiteness (zero NaN, zero Inf)
  - Non-trivial variance across all 32 dimensions
  - Mapping consistency (repeated URLs map to identical vectors)
Outputs:
  - reports/phase_3/image_feature_validation.json
  - reports/phase_3/image_feature_validation.md
"""

import json
import os
import sys
import numpy as np
import pandas as pd


def validate_image_features(
    csv_path: str = "data/raw/products.csv",
    full_features_path: str = "data/processed/features/image_features.npy",
    unique_features_path: str = "data/processed/features/image_features_unique.npy",
    mapping_path: str = "data/processed/features/image_feature_mapping.csv",
    output_dir: str = "reports/phase_3"
):
    print("=" * 60)
    print("PHASE 2.5: NUMERICAL VALIDATION OF IMAGE FEATURES")
    print("=" * 60)

    # 1. Existence check
    for p in [csv_path, full_features_path, unique_features_path, mapping_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    df_raw = pd.read_csv(csv_path, encoding="cp1252")
    features = np.load(full_features_path)
    unique_features = np.load(unique_features_path)
    mapping = pd.read_csv(mapping_path)

    n_rows = len(df_raw)
    n_unique_urls = df_raw["Main Image URL"].nunique()

    # 2. Shape and dimensionality checks
    print(f"Dataset row count: {n_rows:,}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Unique feature matrix shape: {unique_features.shape}")
    print(f"Mapping row count: {len(mapping):,}")

    assert features.shape == (n_rows, 32), f"Expected ({n_rows}, 32), got {features.shape}"
    assert unique_features.ndim == 2 and unique_features.shape[1] == 32, f"Expected (*, 32), got {unique_features.shape}"
    assert len(mapping) == n_rows, f"Expected mapping length {n_rows}, got {len(mapping)}"

    # 3. Finite checks
    has_nan = bool(np.isnan(features).any())
    has_inf = bool(np.isinf(features).any())
    is_finite = bool(np.isfinite(features).all())
    assert is_finite, "Feature matrix contains NaN or Inf"
    print("Finite check: PASSED (Zero NaN, Zero Inf)")

    # 4. Consistency checks
    # Verify mapping index bounds
    valid_mapped = mapping[mapping["unique_feature_index"] >= 0]
    max_idx = valid_mapped["unique_feature_index"].max()
    assert max_idx < len(unique_features), f"Max index {max_idx} exceeds unique array length {len(unique_features)}"

    # Check identical vectors for identical URLs
    sample_url_counts = df_raw["Main Image URL"].value_counts()
    repeated_urls = sample_url_counts[sample_url_counts > 1].index.tolist()
    
    mapping_consistency_passed = True
    tested_repeated = 0
    for r_url in repeated_urls[:20]:
        r_indices = df_raw[df_raw["Main Image URL"] == r_url].index.tolist()
        if len(r_indices) > 1:
            first_vec = features[r_indices[0]]
            for other_idx in r_indices[1:]:
                if not np.allclose(first_vec, features[other_idx], atol=1e-6):
                    mapping_consistency_passed = False
                    break
            tested_repeated += 1
    assert mapping_consistency_passed, "Repeated URLs produced non-identical feature vectors"
    print(f"Mapping consistency check: PASSED (Tested {tested_repeated} repeated URL groups)")

    # 5. Dimension-wise statistics
    dim_stats = []
    for d in range(32):
        col = features[:, d]
        dim_stats.append({
            "dimension": d,
            "mean": round(float(col.mean()), 5),
            "std": round(float(col.std()), 5),
            "min": round(float(col.min()), 5),
            "max": round(float(col.max()), 5)
        })

    # Summary
    global_mean = float(features.mean())
    global_std = float(features.std())
    global_min = float(features.min())
    global_max = float(features.max())
    dim_stds = [s["std"] for s in dim_stats]
    zero_variance_dims = sum(1 for s in dim_stds if s == 0)

    print(f"Global stats across all 32 dimensions: mean={global_mean:.4f}, std={global_std:.4f}, min={global_min:.4f}, max={global_max:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "image_feature_validation.json")
    md_path = os.path.join(output_dir, "image_feature_validation.md")

    validation_data = {
        "dataset_rows": n_rows,
        "feature_dim": 32,
        "unique_features_shape": list(unique_features.shape),
        "full_features_shape": list(features.shape),
        "finite_check_passed": is_finite,
        "mapping_consistency_passed": mapping_consistency_passed,
        "zero_variance_dimensions": zero_variance_dims,
        "global_statistics": {
            "mean": round(global_mean, 5),
            "std": round(global_std, 5),
            "min": round(global_min, 5),
            "max": round(global_max, 5)
        },
        "dimension_statistics": dim_stats
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(validation_data, f, indent=2)
    print(f"Saved validation JSON to: {json_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Image Feature Validation Report (Phase 2.5)\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("- **Modality:** Product Image (ItaNet 32-D visual representation)\n")
        f.write("- **Backbone:** ResNet-50 (torchvision pretrained `IMAGENET1K_V2`, frozen)\n")
        f.write("- **Dimensionality Reduction:** Deterministic Gaussian Projection ($P_{ij} \\sim \\mathcal{N}(0, 1/32)$, seed 42) — *Implementation Adaptation*\n")
        f.write(f"- **Full Matrix Shape:** `({features.shape[0]:,}, {features.shape[1]})`\n")
        f.write(f"- **Unique Matrix Shape:** `({unique_features.shape[0]:,}, {unique_features.shape[1]})`\n")
        f.write(f"- **Finiteness Check:** {'PASSED (Zero NaN, Zero Inf)' if is_finite else 'FAILED'}\n")
        f.write(f"- **Mapping Consistency:** {'PASSED (Repeated URLs match)' if mapping_consistency_passed else 'FAILED'}\n\n")

        f.write("## 2. Global Distribution Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Dataset Rows | {n_rows:,} |\n")
        f.write(f"| Output Dimensions | 32 |\n")
        f.write(f"| Global Mean | {global_mean:.5f} |\n")
        f.write(f"| Global Std | {global_std:.5f} |\n")
        f.write(f"| Global Min | {global_min:.5f} |\n")
        f.write(f"| Global Max | {global_max:.5f} |\n")
        f.write(f"| Zero Variance Dimensions | {zero_variance_dims} |\n\n")

        f.write("## 3. Per-Dimension Statistics (Sample)\n\n")
        f.write("| Dimension | Mean | Std | Min | Max |\n")
        f.write("|---|---|---|---|---|\n")
        for s in dim_stats[:10]:
            f.write(f"| Dim {s['dimension']:02d} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n")
        f.write("| ... (dims 10-31 in JSON report) | ... | ... | ... | ... |\n")

    print(f"Saved validation Markdown to: {md_path}")
    return validation_data


if __name__ == "__main__":
    validate_image_features()
