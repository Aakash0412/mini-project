import os
import time
import hashlib
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

MANIFEST = "data/processed/images/image_manifest.csv"
IMAGE_DIR = "data/processed/images"

df = pd.read_csv(MANIFEST)

failed = df[df["download_status"] == "failed"].copy()

print(f"Failed images to retry: {len(failed)}")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Connection": "keep-alive",
})

recovered = 0
still_failed = 0

for n, (idx, row) in enumerate(failed.iterrows(), 1):
    url = row["image_url"]
    image_id = row["image_id"]
    output = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

    print(f"[{n}/{len(failed)}] {url}")

    success = False
    last_error = ""

    for attempt in range(1, 4):
        try:
            r = session.get(
                url,
                timeout=(15, 30),
                allow_redirects=True
            )

            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
                print(f"  Attempt {attempt}: {last_error}")
                if r.status_code in (400, 404):
                    break
                time.sleep(2 * attempt)
                continue

            content_type = r.headers.get("Content-Type", "").lower()

            if not content_type.startswith("image/"):
                last_error = f"Invalid content type: {content_type}"
                print(f"  Attempt {attempt}: {last_error}")
                time.sleep(2 * attempt)
                continue

            # Validate that the response is actually an image.
            image = Image.open(BytesIO(r.content))
            image.verify()

            # Re-open for dimensions.
            image = Image.open(BytesIO(r.content))
            width, height = image.size

            with open(output, "wb") as f:
                f.write(r.content)

            sha256 = hashlib.sha256(r.content).hexdigest()

            df.loc[idx, "download_status"] = "downloaded"
            df.loc[idx, "http_status"] = r.status_code
            df.loc[idx, "content_type"] = content_type
            df.loc[idx, "width"] = width
            df.loc[idx, "height"] = height
            df.loc[idx, "file_size_bytes"] = len(r.content)
            df.loc[idx, "sha256"] = sha256
            df.loc[idx, "error_message"] = ""

            recovered += 1
            success = True

            print(f"  SUCCESS ({width}x{height}, {len(r.content):,} bytes)")
            break

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"  Attempt {attempt}: {last_error}")
            time.sleep(2 * attempt)

    if not success:
        df.loc[idx, "error_message"] = last_error
        still_failed += 1
        print("  FAILED")

# Save updated manifest.
df.to_csv(MANIFEST, index=False)

print()
print("=" * 60)
print("TARGETED RETRY COMPLETE")
print("=" * 60)
print(f"Originally failed : {len(failed)}")
print(f"Recovered         : {recovered}")
print(f"Still failed      : {still_failed}")
print(f"Total downloaded  : {(df['download_status'] == 'downloaded').sum()}")
print(f"Total cached      : {(df['download_status'] == 'cached').sum()}")
print()
print(f"Manifest updated: {MANIFEST}")