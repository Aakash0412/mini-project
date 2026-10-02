import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = "data/processed/images/image_manifest.csv"
OUTPUT_DIR = "data/processed/features"
FEATURE_PATH = os.path.join(OUTPUT_DIR, "image_features.npy")
STATUS_PATH = os.path.join(OUTPUT_DIR, "image_feature_status.csv")

BATCH_SIZE = 32
NUM_WORKERS = 0

IMAGE_DIM = 2048
TARGET_DIM = 32

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("RESNET-50 IMAGE FEATURE EXTRACTION")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Output dimension: {TARGET_DIM}")


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST_PATH)

available_mask = df["download_status"].isin(["downloaded", "cached"])

available_df = df[available_mask].copy()
missing_df = df[~available_mask].copy()

print()
print(f"Total dataset rows : {len(df)}")
print(f"Available images   : {len(available_df)}")
print(f"Missing images     : {len(missing_df)}")


# ============================================================
# ResNet-50
# ============================================================

print()
print("Loading ResNet-50 pretrained weights...")

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Remove final classification layer.
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

print("ResNet-50 loaded.")
print(f"Feature dimension before projection: {IMAGE_DIM}")


# ============================================================
# Deterministic 2048 -> 32 projection
# ============================================================

print()
print("Creating deterministic 2048 -> 32 projection...")

rng = np.random.RandomState(SEED)

projection = rng.normal(
    loc=0.0,
    scale=1.0 / np.sqrt(TARGET_DIM),
    size=(IMAGE_DIM, TARGET_DIM),
).astype(np.float32)

projection_tensor = torch.from_numpy(projection).to(DEVICE)

print("Projection created.")


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Dataset
# ============================================================

class AmazonImageDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        path = str(row["local_path"])

        # Normalize Windows path separators.
        path = path.replace("\\", os.sep)

        try:
            image = Image.open(path).convert("RGB")
            image = transform(image)

            return (
                image,
                int(row["row_id"]),
                True,
                "",
            )

        except Exception as exc:
            return (
                torch.zeros(3, 224, 224),
                int(row["row_id"]),
                False,
                str(exc),
            )


dataset = AmazonImageDataset(available_df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# Allocate full dataset feature matrix
# ============================================================

# NaN means no valid image feature exists for that dataset row.
all_features = np.full(
    (len(df), TARGET_DIM),
    np.nan,
    dtype=np.float32,
)

status_records = []

processed = 0
successful = 0
failed = 0


# ============================================================
# Extraction
# ============================================================

print()
print("Starting feature extraction...")
print()

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        images, row_ids, valid_flags, errors = batch

        valid_flags = valid_flags.bool()

        if valid_flags.any():

            valid_images = images[valid_flags].to(DEVICE)

            # ResNet-50 2048-D representation.
            features_2048 = model(valid_images)

            # Deterministic 2048 -> 32 projection.
            features_32 = features_2048 @ projection_tensor

            features_32 = features_32.cpu().numpy()

            valid_row_ids = row_ids[valid_flags].numpy()

            for i, row_id in enumerate(valid_row_ids):
                all_features[row_id] = features_32[i]

                status_records.append({
                    "row_id": int(row_id),
                    "status": "success",
                    "error": "",
                })

                successful += 1

        # Record failures.
        for i in range(len(row_ids)):

            if not bool(valid_flags[i]):

                status_records.append({
                    "row_id": int(row_ids[i]),
                    "status": "failed",
                    "error": str(errors[i]),
                })

                failed += 1

        processed += len(row_ids)

        if batch_idx % 10 == 0 or processed == len(dataset):
            print(
                f"Processed {processed}/{len(dataset)} "
                f"| Successful: {successful} "
                f"| Failed: {failed}"
            )


# ============================================================
# Add explicit missing-image rows
# ============================================================

for _, row in missing_df.iterrows():

    status_records.append({
        "row_id": int(row["row_id"]),
        "status": "missing_image",
        "error": str(row.get("error_message", "")),
    })


# ============================================================
# Save
# ============================================================

print()
print("Saving feature matrix...")

np.save(FEATURE_PATH, all_features)

status_df = pd.DataFrame(status_records)
status_df = status_df.sort_values("row_id")

status_df.to_csv(
    STATUS_PATH,
    index=False,
)


# ============================================================
# Validation
# ============================================================

valid_feature_rows = np.isfinite(all_features).all(axis=1)

print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Dataset rows              : {len(df)}")
print(f"Available image rows      : {len(available_df)}")
print(f"Successful feature rows   : {valid_feature_rows.sum()}")
print(f"Missing/failed rows       : {(~valid_feature_rows).sum()}")

print(f"Feature matrix shape      : {all_features.shape}")
print(f"Expected shape            : ({len(df)}, {TARGET_DIM})")

print(
    f"NaN values                : "
    f"{np.isnan(all_features).sum()}"
)

print(
    f"Inf values                : "
    f"{np.isinf(all_features).sum()}"
)

print()
print(f"Feature file: {FEATURE_PATH}")
print(f"Status file : {STATUS_PATH}")

# Release resources.
del model
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()