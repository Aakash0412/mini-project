from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.models.itanet import ItaNetStateBuilder

FEATURE_DIR = Path("data/processed/features")
CORRECTED_DIR = Path("data/processed/features_train_only")
OUTPUT_DIR = Path("data/processed/state_train_only")
REPORT_DIR = Path("reports")

TEXT_PATH = FEATURE_DIR / "text_features.npy"
IMAGE_PATH = FEATURE_DIR / "image_features.npy"
ATTRIBUTE_PATH = CORRECTED_DIR / "attribute_features_train_only.npy"
INDEX_PATH = CORRECTED_DIR / "feature_index.parquet"
DATA_PATH = Path("data/processed/products_clean.parquet")
SPLIT_PATH = Path("configs/data_split.yaml")


def main():
    # Refuse to overwrite a previous corrected run.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Choose a new output directory before proceeding."
        )

    text = np.load(TEXT_PATH, mmap_mode="r")
    image = np.load(IMAGE_PATH, mmap_mode="r")
    attributes = np.load(ATTRIBUTE_PATH, mmap_mode="r")
    index = pd.read_parquet(INDEX_PATH).reset_index(drop=True)
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    n = len(df)
    for name, array in [
        ("text", text),
        ("image", image),
        ("attributes", attributes),
    ]:
        if len(array) != n:
            raise ValueError(f"{name} has {len(array)} rows; expected {n}.")

    if len(index) != n:
        raise ValueError("Feature index row count does not match dataset.")
    if not index["ASIN"].equals(df["ASIN"]):
        raise ValueError("ASIN order mismatch between index and dataset.")
    if text.shape[1] != 32 or image.shape[1] != 32:
        raise ValueError("Expected 32-D image and text features.")
    if attributes.shape[1] != 47:
        raise ValueError("Expected 47-D corrected attributes.")

    with SPLIT_PATH.open("r", encoding="utf-8-sig") as f:
        split = yaml.safe_load(f)["split"]

    # Keep only rows with finite image features, matching the existing
    # project's complete-state policy. Record original row indices.
    image_valid = np.isfinite(image).all(axis=1)
    valid_indices = np.flatnonzero(image_valid)

    if not np.isfinite(text).all():
        raise ValueError("Text features contain non-finite values.")
    if not np.isfinite(attributes).all():
        raise ValueError("Corrected attributes contain non-finite values.")

    builder = ItaNetStateBuilder(
        image_features=np.asarray(image[image_valid]),
        text_features=np.asarray(text[image_valid]),
        attribute_features=np.asarray(attributes[image_valid]),
        sentiment_features=None,
    )
    state = builder.build()

    if state.shape != (len(valid_indices), 111):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("Constructed state contains NaN or Inf.")

    months = df["MonthNum"].to_numpy()
    train_months = split["train_months"]
    val_months = split["val_months"]
    test_months = split["test_months"]

    state_months = months[valid_indices]
    train_mask = np.isin(state_months, train_months)
    val_mask = np.isin(state_months, val_months)
    test_mask = np.isin(state_months, test_months)

    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask):
        raise ValueError("Overlapping chronological splits.")
    if np.any(val_mask & test_mask):
        raise ValueError("Validation and test splits overlap.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "state.npy", state)
    np.save(OUTPUT_DIR / "state_row_indices.npy", valid_indices)
    np.save(OUTPUT_DIR / "state_train.npy", state[train_mask])
    np.save(OUTPUT_DIR / "state_val.npy", state[val_mask])
    np.save(OUTPUT_DIR / "state_test.npy", state[test_mask])

    validity = pd.DataFrame({
        "row_index": np.arange(n),
        "ASIN": df["ASIN"],
        "image_features_finite": image_valid,
        "split": np.select(
            [
                np.isin(months, train_months),
                np.isin(months, val_months),
                np.isin(months, test_months),
            ],
            ["train", "validation", "test"],
            default="unknown",
        ),
    })
    validity.to_csv(OUTPUT_DIR / "state_validity.csv", index=False)

    report = {
        "variant": "train-only attribute preprocessing",
        "state_dimension": int(state.shape[1]),
        "paper_state_dimension": 121,
        "missing_modality": "10-D review sentiment",
        "source_rows": int(n),
        "valid_image_rows": int(image_valid.sum()),
        "excluded_nonfinite_image_rows": int((~image_valid).sum()),
        "train_months": train_months,
        "validation_months": val_months,
        "test_months": test_months,
        "train_state_rows": int(train_mask.sum()),
        "validation_state_rows": int(val_mask.sum()),
        "test_state_rows": int(test_mask.sum()),
        "state_finite": bool(np.isfinite(state).all()),
        "image_embedding_caveat": (
            "448 rows are marked failed in the existing image extraction "
            "status report despite finite vectors; those vectors were "
            "retained provisionally and not independently verified."
        ),
        "existing_artifacts_modified": False,
        "output_directory": str(OUTPUT_DIR),
    }

    report_path = REPORT_DIR / "state_train_only_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nCorrected state saved separately. Original artifacts untouched.")


if __name__ == "__main__":
    main()

