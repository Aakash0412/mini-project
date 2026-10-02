import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = "data/processed/images/image_manifest.csv"
FEATURE_PATH = "data/processed/features/image_features.npy"

BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

TARGET_DIM = 32
IMAGE_DIM = 2048

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("IMAGE FEATURE REPAIR")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# Load manifest and existing features
# ============================================================

manifest = pd.read_csv(MANIFEST_PATH)
features = np.load(FEATURE_PATH)

print("Existing feature matrix:", features.shape)


# ============================================================
# Identify rows requiring repair
# ============================================================

available = manifest["download_status"].isin(
    ["downloaded", "cached"]
)

feature_missing = ~np.isfinite(features).all(axis=1)

repair_mask = available & feature_missing

repair_df = manifest[repair_mask].copy()

print()
print("Available images:", available.sum())
print("Existing valid features:", (~feature_missing).sum())
print("Rows requiring repair:", len(repair_df))


if len(repair_df) == 0:
    print("Nothing to repair.")
    raise SystemExit(0)


# ============================================================
# Preprocessing
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

class RepairDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = str(row["local_path"]).replace("\\", os.sep)

        image = Image.open(path).convert("RGB")
        image = transform(image)

        return image, int(row["row_id"])


dataset = RepairDataset(repair_df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# Load ResNet-50
# ============================================================

print()
print("Loading ResNet-50...")

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Remove classification layer.
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

print("ResNet-50 ready.")


# ============================================================
# Recreate EXACT projection used during original extraction
# ============================================================

rng = np.random.RandomState(SEED)

projection = rng.normal(
    loc=0.0,
    scale=1.0 / np.sqrt(TARGET_DIM),
    size=(IMAGE_DIM, TARGET_DIM),
).astype(np.float32)

projection_tensor = torch.from_numpy(
    projection
).to(DEVICE)


# ============================================================
# Extract only missing features
# ============================================================

processed = 0

print()
print("Starting repair...")
print()

with torch.no_grad():

    for batch_idx, (images, row_ids) in enumerate(loader):

        images = images.to(DEVICE)

        # 2048-D ResNet representation.
        features_2048 = model(images)

        # Same deterministic 2048 -> 32 projection.
        features_32 = features_2048 @ projection_tensor

        features_32 = features_32.cpu().numpy()
        row_ids = row_ids.numpy()

        for i, row_id in enumerate(row_ids):
            features[row_id] = features_32[i]

        processed += len(row_ids)

        print(
            f"Processed {processed}/{len(repair_df)}"
        )


# ============================================================
# Save repaired feature matrix
# ============================================================

np.save(FEATURE_PATH, features)


# ============================================================
# Validation
# ============================================================

valid_rows = np.isfinite(features).all(axis=1)
missing_rows = ~valid_rows

actual_missing_downloads = (
    manifest["download_status"] == "failed"
).sum()

print()
print("=" * 60)
print("REPAIR COMPLETE")
print("=" * 60)

print("Feature matrix:", features.shape)
print("Valid feature rows:", valid_rows.sum())
print("Missing feature rows:", missing_rows.sum())
print("Expected actual missing downloads:", actual_missing_downloads)
print("NaN values:", np.isnan(features).sum())
print("Inf values:", np.isinf(features).sum())

assert features.shape == (len(manifest), TARGET_DIM)
assert valid_rows.sum() == len(manifest) - actual_missing_downloads
assert np.isinf(features).sum() == 0

print()
print("VALIDATION PASSED")
print("Feature matrix saved:", FEATURE_PATH)