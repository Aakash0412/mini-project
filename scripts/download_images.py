"""
Phase 2.5 — Cache-First Resumable Image Downloader

Dataset:
    data/raw/products.csv

Images:
    data/processed/images/<sha256(url)[:32]>.jpg

Behavior:
    - Detects existing images by deterministic filename.
    - DOES NOT validate existing cached images during startup.
    - Downloads only missing images.
    - Validates newly downloaded images with Pillow.
    - Supports --smoke-test and --limit.
    - Limited runs NEVER overwrite production manifests.
    - Full runs rebuild:
        unique_image_manifest.csv
        image_manifest.csv
"""

import argparse
import hashlib
import io
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_WORKERS = 8
TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
CHECKPOINT_EVERY = 250

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_url_hash(url: str) -> str:
    """Return deterministic 32-character hash for an image URL."""
    return hashlib.sha256(
        url.strip().encode("utf-8")
    ).hexdigest()[:32]


def get_output_path(url: str, images_dir: str) -> str:
    """Return deterministic local path for a URL."""
    image_id = get_url_hash(url)
    return os.path.join(images_dir, f"{image_id}.jpg")


def lightweight_cached_file_exists(path: str) -> bool:
    """
    Fast cache check.

    IMPORTANT:
    This deliberately does NOT open the image with Pillow.
    It only checks that the expected file exists and is non-empty.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def validate_downloaded_image(
    body: bytes,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """Validate newly downloaded image bytes with Pillow."""
    try:
        with Image.open(io.BytesIO(body)) as img:
            img.verify()

        with Image.open(io.BytesIO(body)) as img:
            width, height = img.size

        return True, width, height

    except Exception:
        return False, None, None


def download_single_image(
    url: str,
    output_path: str,
) -> Dict[str, Any]:
    """Download one missing image with retries and validation."""

    image_id = get_url_hash(url)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )

            with urllib.request.urlopen(
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:

                status = response.status
                content_type = (
                    response.headers.get(
                        "Content-Type",
                        "image/jpeg",
                    )
                )

                body = response.read()

            # Validate newly downloaded bytes.
            valid, width, height = validate_downloaded_image(body)

            if not valid:
                raise ValueError(
                    "Downloaded response is not a valid image"
                )

            # Atomic write.
            temp_path = output_path + ".tmp"

            with open(temp_path, "wb") as f:
                f.write(body)

            os.replace(temp_path, output_path)

            sha256_hash = hashlib.sha256(body).hexdigest()

            return {
                "image_id": image_id,
                "image_url": url,
                "local_path": output_path,
                "download_status": "downloaded",
                "http_status": status,
                "content_type": content_type,
                "width": width,
                "height": height,
                "file_size_bytes": len(body),
                "sha256": sha256_hash,
                "error_message": None,
            }

        except urllib.error.HTTPError as error:

            last_error = (
                f"HTTPError_{error.code}: {error.reason}"
            )

            # Permanent missing resources.
            if error.code in (404, 410):
                break

        except urllib.error.URLError as error:

            last_error = (
                f"URLError: {error.reason}"
            )

        except Exception as error:

            last_error = (
                f"{type(error).__name__}: {error}"
            )

        # Exponential backoff.
        if attempt < MAX_RETRIES:
            time.sleep(0.5 * (2 ** (attempt - 1)))

    return {
        "image_id": image_id,
        "image_url": url,
        "local_path": (
            output_path
            if lightweight_cached_file_exists(output_path)
            else None
        ),
        "download_status": "failed",
        "http_status": None,
        "content_type": None,
        "width": None,
        "height": None,
        "file_size_bytes": None,
        "sha256": None,
        "error_message": last_error,
    }


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def make_cached_record(
    url: str,
    images_dir: str,
) -> Dict[str, Any]:
    """
    Create a manifest record for an existing cached image.

    We intentionally do not inspect the image with Pillow here.
    """

    image_id = get_url_hash(url)
    output_path = get_output_path(url, images_dir)

    return {
        "image_id": image_id,
        "image_url": url,
        "local_path": output_path,
        "download_status": "cached",
        "http_status": 200,
        "content_type": "image/jpeg",
        "width": None,
        "height": None,
        "file_size_bytes": os.path.getsize(output_path),
        "sha256": None,
        "error_message": None,
    }


def save_unique_manifest(
    records: Dict[str, Dict[str, Any]],
    path: str,
) -> None:
    """Save unique image manifest."""

    if not records:
        return

    df = pd.DataFrame(list(records.values()))

    # Stable column order.
    columns = [
        "image_id",
        "image_url",
        "local_path",
        "download_status",
        "http_status",
        "content_type",
        "width",
        "height",
        "file_size_bytes",
        "sha256",
        "error_message",
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = None

    df = df[columns]

    temp_path = path + ".tmp"

    df.to_csv(
        temp_path,
        index=False,
    )

    os.replace(
        temp_path,
        path,
    )


def build_full_manifest(
    df: pd.DataFrame,
    results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """Build the complete row-aligned 34,041-row manifest."""

    print()
    print("Building full dataset image manifest...")

    manifest_rows: List[Dict[str, Any]] = []

    for row_id, (_, row) in enumerate(
        df.iterrows()
    ):

        raw_url = row.get(
            "Main Image URL",
            "",
        )

        if pd.isna(raw_url):
            url = ""
        else:
            url = str(raw_url).strip()

        info = results.get(url, {})

        image_id = (
            info.get("image_id")
            if info
            else (
                get_url_hash(url)
                if url
                else None
            )
        )

        manifest_rows.append(
            {
                "row_id": row_id,
                "ASIN": row.get("ASIN", ""),
                "image_url": url,
                "image_id": image_id,
                "local_path": info.get("local_path"),
                "download_status": info.get(
                    "download_status",
                    "missing",
                ),
                "http_status": info.get(
                    "http_status"
                ),
                "content_type": info.get(
                    "content_type"
                ),
                "width": info.get(
                    "width"
                ),
                "height": info.get(
                    "height"
                ),
                "file_size_bytes": info.get(
                    "file_size_bytes"
                ),
                "sha256": info.get(
                    "sha256"
                ),
                "error_message": info.get(
                    "error_message"
                ),
            }
        )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    temp_path = output_path + ".tmp"

    manifest_df.to_csv(
        temp_path,
        index=False,
    )

    os.replace(
        temp_path,
        output_path,
    )

    print(
        f"Saved full manifest: "
        f"{output_path}"
    )
    print(
        f"Rows in manifest: "
        f"{len(manifest_df):,}"
    )


# ---------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------

def run_downloader(
    csv_path: str = "data/raw/products.csv",
    images_dir: str = "data/processed/images",
    limit: Optional[int] = None,
) -> None:

    print("=" * 70)
    print("PHASE 2.5: CACHE-FIRST IMAGE DOWNLOADER")
    print("=" * 70)

    os.makedirs(
        images_dir,
        exist_ok=True,
    )

    unique_manifest_path = os.path.join(
        images_dir,
        "unique_image_manifest.csv",
    )

    full_manifest_path = os.path.join(
        images_dir,
        "image_manifest.csv",
    )

    # -------------------------------------------------------------
    # Read dataset
    # -------------------------------------------------------------

    print(
        f"Reading dataset: {csv_path} "
        "(encoding=cp1252)..."
    )

    df = pd.read_csv(
        csv_path,
        encoding="cp1252",
    )

    total_rows = len(df)

    unique_urls = (
        df["Main Image URL"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    total_unique_urls = len(unique_urls)

    print(
        f"Total dataset rows: "
        f"{total_rows:,}"
    )

    print(
        f"Unique image URLs: "
        f"{total_unique_urls:,}"
    )

    # -------------------------------------------------------------
    # Limited test mode
    # -------------------------------------------------------------

    is_limited_run = limit is not None

    if is_limited_run:

        unique_urls = unique_urls[:limit]

        print()
        print(
            f"TEST MODE: processing only "
            f"{len(unique_urls):,} URLs."
        )

        print(
            "Production manifests will NOT "
            "be modified by this run."
        )

    # -------------------------------------------------------------
    # Build filesystem cache index
    # -------------------------------------------------------------

    print()
    print(
        "Scanning image directory "
        "(filename check only)..."
    )

    existing_files = set()

    try:

        for filename in os.listdir(images_dir):

            if filename.lower().endswith(
                ".jpg"
            ):

                stem = os.path.splitext(
                    filename
                )[0]

                existing_files.add(stem)

    except OSError as error:

        print(
            f"Warning: could not scan "
            f"image directory: {error}"
        )

    print(
        f"Existing JPG files found: "
        f"{len(existing_files):,}"
    )

    # -------------------------------------------------------------
    # Determine cached vs missing
    # -------------------------------------------------------------

    results: Dict[
        str,
        Dict[str, Any]
    ] = {}

    urls_to_download: List[
        Tuple[str, str]
    ] = []

    cached_count = 0

    for url in unique_urls:

        image_id = get_url_hash(url)

        output_path = os.path.join(
            images_dir,
            f"{image_id}.jpg",
        )

        # IMPORTANT:
        # First use filename/hash cache.
        if (
            image_id in existing_files
            and lightweight_cached_file_exists(
                output_path
            )
        ):

            results[url] = (
                make_cached_record(
                    url,
                    images_dir,
                )
            )

            cached_count += 1

        else:

            urls_to_download.append(
                (
                    url,
                    output_path,
                )
            )

    print()
    print(
        f"Already cached: "
        f"{cached_count:,}"
    )

    print(
        f"Queued for download: "
        f"{len(urls_to_download):,}"
    )

    # -------------------------------------------------------------
    # Nothing to download
    # -------------------------------------------------------------

    if not urls_to_download:

        print()
        print(
            "Nothing to download."
        )

        if is_limited_run:

            print(
                "Smoke/limited test completed."
            )

            return

        # Full run: build manifests.
        save_unique_manifest(
            results,
            unique_manifest_path,
        )

        build_full_manifest(
            df,
            results,
            full_manifest_path,
        )

        return

    # -------------------------------------------------------------
    # Download missing images
    # -------------------------------------------------------------

    print()
    print(
        "Starting download of missing images..."
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print(
        f"Timeout: {TIMEOUT_SECONDS}s"
    )

    print(
        f"Retries per image: {MAX_RETRIES}"
    )

    print()

    start_time = time.time()

    completed = 0
    downloaded = 0
    failed = 0

    total_to_download = len(
        urls_to_download
    )

    try:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    download_single_image,
                    url,
                    output_path,
                ): url
                for url, output_path
                in urls_to_download
            }

            for future in as_completed(
                future_map
            ):

                url = future_map[future]

                try:

                    result = future.result()

                except Exception as error:

                    result = {
                        "image_id": get_url_hash(
                            url
                        ),
                        "image_url": url,
                        "local_path": None,
                        "download_status": "failed",
                        "http_status": None,
                        "content_type": None,
                        "width": None,
                        "height": None,
                        "file_size_bytes": None,
                        "sha256": None,
                        "error_message": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }

                results[url] = result

                completed += 1

                if (
                    result["download_status"]
                    == "downloaded"
                ):

                    downloaded += 1

                elif (
                    result["download_status"]
                    == "failed"
                ):

                    failed += 1

                # Always show progress every 25 images.
                if (
                    completed % 25 == 0
                    or completed
                    == total_to_download
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    speed = (
                        completed
                        / max(
                            elapsed,
                            0.001,
                        )
                    )

                    print(
                        f"[{completed:,}/"
                        f"{total_to_download:,}] "
                        f"Downloaded: "
                        f"{downloaded:,} | "
                        f"Failed: "
                        f"{failed:,} | "
                        f"Speed: "
                        f"{speed:.2f} img/s | "
                        f"Elapsed: "
                        f"{elapsed:.0f}s",
                        flush=True,
                    )

                # Checkpoint only on full runs.
                if (
                    not is_limited_run
                    and (
                        completed
                        % CHECKPOINT_EVERY
                        == 0
                    )
                ):

                    save_unique_manifest(
                        results,
                        unique_manifest_path,
                    )

    except KeyboardInterrupt:

        print()
        print(
            "DOWNLOAD INTERRUPTED "
            "BY USER."
        )

        print(
            "Already downloaded files "
            "remain safely on disk."
        )

        if not is_limited_run:

            print(
                "The next run will automatically "
                "recognize them by filename."
            )

        return

    # -------------------------------------------------------------
    # Limited run ends here
    # -------------------------------------------------------------

    if is_limited_run:

        print()
        print("=" * 70)
        print("LIMITED TEST SUMMARY")
        print("=" * 70)

        print(
            f"URLs tested: "
            f"{len(unique_urls):,}"
        )

        print(
            f"Already cached: "
            f"{cached_count:,}"
        )

        print(
            f"Downloaded: "
            f"{downloaded:,}"
        )

        print(
            f"Failed: "
            f"{failed:,}"
        )

        print()
        print(
            "Production manifests were NOT "
            "modified."
        )

        return

    # -------------------------------------------------------------
    # Final full manifests
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINALIZING IMAGE MANIFESTS")
    print("=" * 70)

    save_unique_manifest(
        results,
        unique_manifest_path,
    )

    build_full_manifest(
        df,
        results,
        full_manifest_path,
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    final_cached = sum(
        1
        for r in results.values()
        if r.get("download_status")
        == "cached"
    )

    final_downloaded = sum(
        1
        for r in results.values()
        if r.get("download_status")
        == "downloaded"
    )

    final_failed = sum(
        1
        for r in results.values()
        if r.get("download_status")
        == "failed"
    )

    available = (
        final_cached
        + final_downloaded
    )

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Dataset observations: "
        f"{total_rows:,}"
    )

    print(
        f"Unique URLs: "
        f"{total_unique_urls:,}"
    )

    print(
        f"Previously cached: "
        f"{final_cached:,}"
    )

    print(
        f"Downloaded this run: "
        f"{final_downloaded:,}"
    )

    print(
        f"Available images: "
        f"{available:,}"
    )

    print(
        f"Missing/failed: "
        f"{final_failed:,}"
    )

    print()
    print(
        f"Unique manifest: "
        f"{unique_manifest_path}"
    )

    print(
        f"Full manifest: "
        f"{full_manifest_path}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Cache-first resumable "
            "Amazon image downloader"
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N "
            "unique image URLs. "
            "Does not modify production manifests."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Process only the first 10 "
            "unique URLs."
        ),
    )

    args = parser.parse_args()

    limit = args.limit

    if args.smoke_test:

        limit = 10

    run_downloader(
        limit=limit
    )


if __name__ == "__main__":
    main()