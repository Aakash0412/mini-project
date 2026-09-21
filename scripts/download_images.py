"""
Safe, Resumable Image Downloader & Manifest Generator (Phase 2.5)
-----------------------------------------------------------------
Downloads all unique product images directly from Amazon CDN URLs in data/raw/products.csv.
- Deduplicates 34,041 dataset rows to 22,260 unique image URLs.
- Names files deterministically: data/processed/images/<url_sha256>.jpg.
- Concurrency: 8 workers max (polite, robust).
- Resumable: Validates existing files on disk, skips already cached images.
- Validates every image with Pillow (no corrupt images).
- Generates data/processed/images/image_manifest.csv (all 34,041 rows mapped).
"""

import io
import os
import sys
import time
import hashlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple
import pandas as pd
from PIL import Image


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_WORKERS = 8
TIMEOUT_SECONDS = 15
MAX_RETRIES = 3


def get_url_hash(url: str) -> str:
    """Deterministic hash of normalized URL for filename."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:32]


def validate_local_image(file_path: str) -> Tuple[bool, int, int, str]:
    """Validate that a local image file exists and can be decoded by Pillow."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False, 0, 0, ""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        sha256_hash = hashlib.sha256(data).hexdigest()
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        return True, img.width, img.height, sha256_hash
    except Exception:
        return False, 0, 0, ""


def download_single_image(url: str, output_path: str) -> Dict[str, Any]:
    """Download and validate a single image with retries."""
    url_hash = get_url_hash(url)
    
    # 1. Check if already cached and valid
    is_valid, width, height, sha256_hash = validate_local_image(output_path)
    if is_valid:
        return {
            "image_id": url_hash,
            "image_url": url,
            "local_path": output_path,
            "download_status": "cached",
            "http_status": 200,
            "content_type": "image/jpeg",
            "width": width,
            "height": height,
            "file_size_bytes": os.path.getsize(output_path),
            "sha256": sha256_hash,
            "error_message": None
        }

    # 2. Download with retry and exponential backoff
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "image/jpeg")
                body = resp.read()

                # Validate image integrity before saving
                img = Image.open(io.BytesIO(body))
                img.verify()
                img = Image.open(io.BytesIO(body))
                w, h = img.width, img.height
                sha256_val = hashlib.sha256(body).hexdigest()

                # Write to disk atomically
                temp_path = output_path + ".tmp"
                with open(temp_path, "wb") as f:
                    f.write(body)
                os.replace(temp_path, output_path)

                return {
                    "image_id": url_hash,
                    "image_url": url,
                    "local_path": output_path,
                    "download_status": "downloaded",
                    "http_status": status,
                    "content_type": content_type,
                    "width": w,
                    "height": h,
                    "file_size_bytes": len(body),
                    "sha256": sha256_val,
                    "error_message": None
                }
        except urllib.error.HTTPError as he:
            last_error = f"HTTPError_{he.code}: {he.reason}"
            if he.code in (404, 410):
                break  # Do not retry permanent not-found errors
        except urllib.error.URLError as ue:
            last_error = f"URLError: {ue.reason}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"

        if attempt < MAX_RETRIES:
            time.sleep(0.5 * (2 ** (attempt - 1)))

    return {
        "image_id": url_hash,
        "image_url": url,
        "local_path": output_path if os.path.exists(output_path) else None,
        "download_status": "failed",
        "http_status": None,
        "content_type": None,
        "width": None,
        "height": None,
        "file_size_bytes": None,
        "sha256": None,
        "error_message": last_error
    }


def run_downloader(
    csv_path: str = "data/raw/products.csv",
    images_dir: str = "data/processed/images",
    limit: int = None
):
    print("=" * 60)
    print("PHASE 2.5: RESUMABLE IMAGE DOWNLOADER & MANIFEST GENERATOR")
    print("=" * 60)

    os.makedirs(images_dir, exist_ok=True)
    unique_manifest_path = os.path.join(images_dir, "unique_image_manifest.csv")
    full_manifest_path = os.path.join(images_dir, "image_manifest.csv")

    print(f"Reading dataset: {csv_path} (encoding=cp1252)...")
    df = pd.read_csv(csv_path, encoding="cp1252")
    total_rows = len(df)
    unique_urls = df["Main Image URL"].dropna().unique().tolist()
    
    if limit is not None:
        unique_urls = unique_urls[:limit]
        print(f"Limiting download run to first {limit} unique URLs.")

    total_unique = len(unique_urls)
    print(f"Total dataset rows: {total_rows:,}")
    print(f"Unique image URLs to verify/download: {total_unique:,}")

    # Load existing unique manifest if present to resume
    cached_records = {}
    if os.path.exists(unique_manifest_path):
        try:
            prev_df = pd.read_csv(unique_manifest_path)
            for _, r in prev_df.iterrows():
                if r.get("download_status") in ("cached", "downloaded") and pd.notnull(r.get("local_path")):
                    if os.path.exists(str(r["local_path"])):
                        cached_records[str(r["image_url"])] = r.to_dict()
            print(f"Loaded {len(cached_records):,} previously recorded unique images from manifest.")
        except Exception as ex:
            print(f"Warning: Could not read existing manifest ({ex}). Rescanning directory.")

    results_dict = {}
    urls_to_download = []

    for url in unique_urls:
        if url in cached_records:
            results_dict[url] = cached_records[url]
        else:
            url_hash = get_url_hash(url)
            local_file = os.path.join(images_dir, f"{url_hash}.jpg")
            is_valid, w, h, sha256_val = validate_local_image(local_file)
            if is_valid:
                results_dict[url] = {
                    "image_id": url_hash,
                    "image_url": url,
                    "local_path": local_file,
                    "download_status": "cached",
                    "http_status": 200,
                    "content_type": "image/jpeg",
                    "width": w,
                    "height": h,
                    "file_size_bytes": os.path.getsize(local_file),
                    "sha256": sha256_val,
                    "error_message": None
                }
            else:
                urls_to_download.append((url, local_file))

    print(f"Already valid and cached locally: {len(results_dict):,}")
    print(f"Queued for download: {len(urls_to_download):,}")

    if urls_to_download:
        t0 = time.time()
        completed_count = 0
        newly_downloaded = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_single_image, url, path): url
                for url, path in urls_to_download
            }

            for future in as_completed(futures):
                res = future.result()
                results_dict[res["image_url"]] = res
                completed_count += 1

                if res["download_status"] == "downloaded":
                    newly_downloaded += 1
                elif res["download_status"] == "failed":
                    failed_count += 1

                # Periodic status log and manifest checkpoint every 250 images
                if completed_count % 250 == 0 or completed_count == len(urls_to_download):
                    elapsed = time.time() - t0
                    speed = completed_count / max(elapsed, 0.001)
                    print(
                        f"[{completed_count:,}/{len(urls_to_download):,}] "
                        f"Downloaded: {newly_downloaded:,} | Failed: {failed_count:,} | "
                        f"Speed: {speed:.1f} img/s | Elapsed: {elapsed:.0f}s"
                    )
                    # Checkpoint unique manifest
                    temp_df = pd.DataFrame(list(results_dict.values()))
                    temp_df.to_csv(unique_manifest_path, index=False)

    # Final unique manifest save
    unique_df = pd.DataFrame(list(results_dict.values()))
    unique_df.to_csv(unique_manifest_path, index=False)
    print(f"\nSaved unique image manifest ({len(unique_df):,} entries) to: {unique_manifest_path}")

    # Build full dataset row-aligned manifest
    print("Building full 34,041-row image manifest...")
    manifest_rows = []
    for row_id, (_, row) in enumerate(df.iterrows()):
        asin = row.get("ASIN", "")
        url = str(row.get("Main Image URL", ""))
        info = results_dict.get(url, {})
        manifest_rows.append({
            "row_id": row_id,
            "ASIN": asin,
            "image_url": url,
            "image_id": info.get("image_id", get_url_hash(url)),
            "local_path": info.get("local_path"),
            "download_status": info.get("download_status", "missing"),
            "http_status": info.get("http_status"),
            "content_type": info.get("content_type"),
            "width": info.get("width"),
            "height": info.get("height"),
            "file_size_bytes": info.get("file_size_bytes"),
            "sha256": info.get("sha256"),
            "error_message": info.get("error_message")
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(full_manifest_path, index=False)
    print(f"Saved full image manifest ({len(manifest_df):,} rows) to: {full_manifest_path}")

    # Summary
    success_count = (manifest_df["download_status"].isin(["cached", "downloaded"])).sum()
    print("\nDownload Summary:")
    print(f"  Total observations: {total_rows:,}")
    print(f"  Successfully available images: {success_count:,} ({(success_count/total_rows)*100:.2f}%)")
    print(f"  Failed images: {total_rows - success_count:,}")
    return manifest_df


if __name__ == "__main__":
    run_downloader()
