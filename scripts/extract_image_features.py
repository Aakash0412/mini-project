"""
Resumable ResNet-50 + JL Projection Feature Extraction Pipeline (Phase 2.5)
--------------------------------------------------------------------------
Extracts 32-D visual features for all unique product images using:
  ResNet-50 (2048-D) -> Deterministic JL Projection P_ij ~ N(0, 1/32) -> 32-D

Features are extracted ONCE per unique image, then mapped back to all 34,041 observations.
Independent resumability via:
  - Checkpoint array: data/processed/features/image_features_unique_checkpoint.npy
  - Progress metadata: data/processed/features/extraction_progress.json

Outputs:
  - data/processed/features/image_features_unique.npy      (N_unique, 32)
  - data/processed/features/image_features.npy             (34041, 32)
  - data/processed/features/image_feature_mapping.csv      Mapping metadata
"""

import io
import json
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.image_encoder import ImageEncoder


BATCH_SIZE = 16
CHECKPOINT_INTERVAL = 500


def run_feature_extraction(
    csv_path: str = "data/raw/products.csv",
    images_dir: str = "data/processed/images",
    features_dir: str = "data/processed/features",
    device: str = "cpu",
    batch_size: int = BATCH_SIZE
):
    print("=" * 60)
    print("PHASE 2.5: RESUMABLE IMAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 60)

    os.makedirs(features_dir, exist_ok=True)
    unique_manifest_path = os.path.join(images_dir, "unique_image_manifest.csv")
    checkpoint_npy_path = os.path.join(features_dir, "image_features_unique_checkpoint.npy")
    progress_json_path = os.path.join(features_dir, "extraction_progress.json")
    final_unique_npy_path = os.path.join(features_dir, "image_features_unique.npy")
    final_full_npy_path = os.path.join(features_dir, "image_features.npy")
    mapping_csv_path = os.path.join(features_dir, "image_feature_mapping.csv")

    if not os.path.exists(unique_manifest_path):
        raise FileNotFoundError(
            f"Unique image manifest not found at {unique_manifest_path}. "
            "Run scripts/download_images.py first."
        )

    print(f"Loading unique image manifest from: {unique_manifest_path}")
    unique_df = pd.read_csv(unique_manifest_path)
    
    # Sort deterministically by image_id
    unique_df = unique_df.sort_values(by="image_id").reset_index(drop=True)
    total_unique = len(unique_df)
    print(f"Total unique images in manifest: {total_unique:,}")

    # Check for valid/cached images
    valid_mask = unique_df["download_status"].isin(["cached", "downloaded"]) & unique_df["local_path"].notnull()
    valid_df = unique_df[valid_mask].copy().reset_index(drop=True)
    print(f"Valid images ready for feature extraction: {len(valid_df):,} / {total_unique:,}")

    if len(valid_df) == 0:
        raise ValueError("No valid images available to extract features from.")

    # Initialize ImageEncoder
    print(f"\nInitializing ImageEncoder (device={device}, seed=42)...")
    encoder = ImageEncoder(device=device, seed=42)

    # Initialize or load checkpoint
    num_valid = len(valid_df)
    completed_indices = set()
    features_32 = np.zeros((num_valid, 32), dtype=np.float32)

    if os.path.exists(checkpoint_npy_path) and os.path.exists(progress_json_path):
        try:
            with open(progress_json_path, "r") as f:
                progress = json.load(f)
            completed_indices = set(progress.get("completed_indices", []))
            loaded_checkpoint = np.load(checkpoint_npy_path)
            if loaded_checkpoint.shape == (num_valid, 32):
                features_32 = loaded_checkpoint
                print(f"Resumed from checkpoint: {len(completed_indices):,} / {num_valid:,} features already extracted.")
            else:
                print("Warning: Checkpoint shape mismatch. Restarting extraction from scratch.")
                completed_indices = set()
        except Exception as ex:
            print(f"Warning: Could not load checkpoint ({ex}). Restarting.")
            completed_indices = set()

    # Identify pending indices
    pending_indices = [i for i in range(num_valid) if i not in completed_indices]
    print(f"Pending features to extract: {len(pending_indices):,}")

    if pending_indices:
        t0 = time.time()
        batch_images = []
        batch_indices = []
        extracted_this_run = 0

        for idx in pending_indices:
            row = valid_df.iloc[idx]
            img_path = str(row["local_path"])
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch_images.append(img)
                batch_indices.append(idx)
            except Exception as e:
                print(f"Warning: Corrupt local file at {img_path} ({e})")
                continue

            # When batch is full or at end
            if len(batch_images) >= batch_size or idx == pending_indices[-1]:
                _, b_feat_32 = encoder.encode_pil_images(batch_images)
                for b_idx, feat_vec in zip(batch_indices, b_feat_32):
                    features_32[b_idx] = feat_vec
                    completed_indices.add(b_idx)

                extracted_this_run += len(batch_images)
                batch_images = []
                batch_indices = []

                # Periodic checkpointing
                if extracted_this_run % CHECKPOINT_INTERVAL == 0 or len(completed_indices) == num_valid:
                    elapsed = time.time() - t0
                    speed = extracted_this_run / max(elapsed, 0.001)
                    print(
                        f"[{len(completed_indices):,}/{num_valid:,}] "
                        f"Extracted: {extracted_this_run:,} | Speed: {speed:.1f} img/s | "
                        f"Elapsed: {elapsed:.0f}s"
                    )
                    # Save checkpoint
                    np.save(checkpoint_npy_path, features_32)
                    with open(progress_json_path, "w") as f:
                        json.dump({
                            "completed_indices": list(completed_indices),
                            "total": num_valid,
                            "timestamp": time.time()
                        }, f)

    # Save final unique features array
    np.save(final_unique_npy_path, features_32)
    print(f"\nSaved final unique feature array ({features_32.shape}) to: {final_unique_npy_path}")

    # Build unique URL to feature index lookup table
    valid_df["unique_feature_index"] = np.arange(len(valid_df))
    url_to_idx = dict(zip(valid_df["image_url"], valid_df["unique_feature_index"]))
    id_to_idx = dict(zip(valid_df["image_id"], valid_df["unique_feature_index"]))

    # Map back to full dataset rows
    print(f"Loading raw dataset from {csv_path} for full row alignment...")
    raw_df = pd.read_csv(csv_path, encoding="cp1252")
    total_raw_rows = len(raw_df)
    
    full_features = np.zeros((total_raw_rows, 32), dtype=np.float32)
    mapping_rows = []
    missing_features_count = 0

    for row_id, (_, row) in enumerate(raw_df.iterrows()):
        asin = row.get("ASIN", "")
        url = str(row.get("Main Image URL", ""))
        feat_idx = url_to_idx.get(url)

        if feat_idx is not None:
            full_features[row_id] = features_32[feat_idx]
            mapping_rows.append({
                "row_id": row_id,
                "ASIN": asin,
                "image_url": url,
                "image_id": valid_df.iloc[feat_idx]["image_id"],
                "unique_feature_index": feat_idx
            })
        else:
            missing_features_count += 1
            mapping_rows.append({
                "row_id": row_id,
                "ASIN": asin,
                "image_url": url,
                "image_id": None,
                "unique_feature_index": -1
            })

    # Save full aligned features array and mapping CSV
    np.save(final_full_npy_path, full_features)
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_csv_path, index=False)

    print(f"Saved full row-aligned image features array ({full_features.shape}) to: {final_full_npy_path}")
    print(f"Saved feature mapping metadata ({len(mapping_df):,} rows) to: {mapping_csv_path}")

    # Integrity verification
    assert full_features.shape == (total_raw_rows, 32), f"Expected shape ({total_raw_rows}, 32), got {full_features.shape}"
    assert np.isfinite(full_features).all(), "Full feature matrix contains NaN or Inf"
    print("\nFeature Extraction Complete & Verified:")
    print(f"  Full Matrix Shape: {full_features.shape}")
    print(f"  Missing Features: {missing_features_count}")
    print(f"  Global Mean: {full_features.mean():.4f}")
    print(f"  Global Std:  {full_features.std():.4f}")
    return full_features, mapping_df


if __name__ == "__main__":
    run_feature_extraction()
