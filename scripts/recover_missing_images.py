import os
import time
import hashlib
from io import BytesIO

import pandas as pd
import requests
from PIL import Image


MANIFEST_PATH = "data/processed/images/image_manifest.csv"
IMAGE_DIR = "data/processed/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# Amazon-compatible browser-style headers.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.amazon.com/",
}

# ------------------------------------------------------------
# Load manifest
# ------------------------------------------------------------

df = pd.read_csv(MANIFEST_PATH)

failed = df[df["download_status"] == "failed"].copy()

print("=" * 60)
print("TARGETED MISSING IMAGE RECOVERY")
print("=" * 60)
print("Failed rows:", len(failed))

# Only unique URLs need to be tested.
unique_urls = failed["image_url"].dropna().unique()

print("Unique failed URLs:", len(unique_urls))
print()

session = requests.Session()
session.headers.update(HEADERS)

recovered = 0
still_failed = 0

recovered_ids = set()

for i, url in enumerate(unique_urls, start=1):

    rows = failed[failed["image_url"] == url]
    image_id = str(rows.iloc[0]["image_id"])
    output_path = os.path.join(IMAGE_DIR, image_id + ".jpg")

    print(f"[{i}/{len(unique_urls)}]")
    print("URL:", url)
    print("Image ID:", image_id)

    success = False

    # Try a few request variants.
    attempts = [
        {"params": None},
        {"params": {"raw": "1"}},
    ]

    for attempt_no, attempt in enumerate(attempts, start=1):

        try:
            response = session.get(
                url,
                timeout=(10, 30),
                allow_redirects=True,
                params=attempt["params"],
            )

            print(
                f"  Attempt {attempt_no}: "
                f"HTTP {response.status_code}, "
                f"{len(response.content):,} bytes"
            )

            if response.status_code != 200:
                continue

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            # Validate actual image content rather than trusting
            # the Content-Type header.
            image = Image.open(BytesIO(response.content))
            image.load()

            image = image.convert("RGB")

            # Save as JPEG using the existing image_id convention.
            image.save(
                output_path,
                format="JPEG",
                quality=95,
            )

            # Verify saved file.
            with Image.open(output_path) as check:
                check.verify()

            print(
                f"  RECOVERED: {output_path}"
            )

            recovered += len(rows)
            recovered_ids.add(image_id)
            success = True
            break

        except Exception as exc:
            print(
                f"  Attempt {attempt_no} error: {exc}"
            )

        time.sleep(1)

    if not success:
        print("  Could not recover.")

        still_failed += len(rows)

    print()

# ------------------------------------------------------------
# Update manifest
# ------------------------------------------------------------

for image_id in recovered_ids:

    mask = (
        (df["image_id"].astype(str) == image_id)
        & (df["download_status"] == "failed")
    )

    path = os.path.join(
        IMAGE_DIR,
        image_id + ".jpg"
    )

    df.loc[mask, "local_path"] = path
    df.loc[mask, "download_status"] = "downloaded"
    df.loc[mask, "http_status"] = 200
    df.loc[mask, "content_type"] = "image/jpeg"

    # Record file metadata.
    if os.path.exists(path):

        with open(path, "rb") as f:
            data = f.read()

        df.loc[mask, "file_size_bytes"] = len(data)
        df.loc[mask, "sha256"] = hashlib.sha256(data).hexdigest()

        try:
            with Image.open(path) as img:
                df.loc[mask, "width"] = img.width
                df.loc[mask, "height"] = img.height
        except Exception:
            pass

    df.loc[mask, "error_message"] = ""

# Save updated manifest.
df.to_csv(
    MANIFEST_PATH,
    index=False,
)

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 60)
print("RECOVERY COMPLETE")
print("=" * 60)

print("Rows recovered       :", recovered)
print("Rows still failed    :", still_failed)

print()
print("Current manifest:")
print(df["download_status"].value_counts())

available = df["download_status"].isin(
    ["downloaded", "cached"]
).sum()

failed_count = (
    df["download_status"] == "failed"
).sum()

print()
print("Available images:", available)
print("Failed images   :", failed_count)
print("Manifest rows   :", len(df))
print()
print("Manifest updated:", MANIFEST_PATH)