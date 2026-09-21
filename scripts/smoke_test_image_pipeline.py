"""
Smoke Test: End-to-End Image Modality Pipeline
----------------------------------------------
Tests 10 unique images through:
  download/validate -> ResNet-50 -> 2048-D -> JL projection -> 32-D
Verifies dimensions, finite values, and non-zero variance.
"""

import io
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.image_encoder import ImageEncoder


def run_smoke_test(sample_size: int = 10, random_seed: int = 42):
    print("=" * 60)
    print("PHASE 2.5 PRE-FLIGHT SMOKE TEST: 10 UNIQUE IMAGES")
    print("=" * 60)

    csv_path = "data/raw/products.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, encoding="cp1252")
    unique_urls = df["Main Image URL"].dropna().unique()
    print(f"Total unique URLs in dataset: {len(unique_urls):,}")

    # Sample deterministically
    np.random.seed(random_seed)
    sampled_urls = np.random.choice(unique_urls, size=sample_size, replace=False)

    print(f"\nDownloading and validating {sample_size} unique sample images...")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    images = []
    for idx, url in enumerate(sampled_urls, start=1):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            print(f"  [{idx}/{sample_size}] Loaded image ({img.format}, {img.width}x{img.height}) from {url[:50]}...")

    assert len(images) == sample_size, f"Expected {sample_size} images, got {len(images)}"

    print("\nInitializing ImageEncoder (ResNet-50 + JL projection P_ij ~ N(0, 1/32))...")
    encoder = ImageEncoder(device="cpu", seed=42)

    print(f"Encoding {sample_size} images through ImageEncoder...")
    feat_2048, feat_32 = encoder.encode_pil_images(images)

    print("\nVerifying outputs:")
    print(f"  feat_2048 shape: {feat_2048.shape} (Expected: ({sample_size}, 2048))")
    print(f"  feat_32   shape: {feat_32.shape} (Expected: ({sample_size}, 32))")

    assert feat_2048.shape == (sample_size, 2048), f"Unexpected 2048-D shape: {feat_2048.shape}"
    assert feat_32.shape == (sample_size, 32), f"Unexpected 32-D shape: {feat_32.shape}"

    # Check finite values
    assert np.isfinite(feat_2048).all(), "feat_2048 contains NaN or Inf"
    assert np.isfinite(feat_32).all(), "feat_32 contains NaN or Inf"
    print("  Finite check: PASSED (Zero NaN, Zero Inf)")

    # Check variance / standard deviation
    dim_stds = np.std(feat_32, axis=0)
    assert (dim_stds > 0).all(), "feat_32 has zero-variance dimensions"
    print(f"  Variance check: PASSED (Min dim std: {dim_stds.min():.4f}, Max dim std: {dim_stds.max():.4f})")

    print("\n32-D Feature Matrix Statistics:")
    print(f"  Mean:   {feat_32.mean():.4f}")
    print(f"  Std:    {feat_32.std():.4f}")
    print(f"  Min:    {feat_32.min():.4f}")
    print(f"  Max:    {feat_32.max():.4f}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED SUCCESSFULLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_smoke_test()
