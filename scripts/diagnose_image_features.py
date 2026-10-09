from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("=" * 65)
    print("IMAGE FEATURE DIAGNOSTICS")
    print("=" * 65)

    matches = [
        p for p in ROOT.rglob("image_features.npy")
        if ".venv" not in p.parts
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
    ]

    if not matches:
        print("[FAIL] No image_features.npy found.")
        return

    index_matches = [
        p for p in ROOT.rglob("state_row_indices.npy")
        if ".venv" not in p.parts
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
    ]
    index_path = index_matches[0] if index_matches else None

    if index_path is None:
        print("[INFO] state_row_indices.npy was not found.")

    for path in matches:
        print(f"\nFile: {path.relative_to(ROOT)}")

        features = np.load(path, mmap_mode="r", allow_pickle=False)
        finite_mask = np.isfinite(features)

        print(f"Shape: {features.shape}")
        print(f"Dtype: {features.dtype}")
        print(f"Total values: {features.size:,}")
        print(f"Finite values: {finite_mask.sum():,}")
        print(f"Non-finite values: {(~finite_mask).sum():,}")

        if features.dtype.kind not in "fiu":
            print("[WARNING] Unexpected feature dtype.")
            continue

        if finite_mask.all():
            print("[PASS] All image features are finite.")
            continue

        bad_rows = np.where(~finite_mask.all(axis=1))[0]
        print(f"Rows containing non-finite values: {len(bad_rows):,}")
        print(f"Affected row indices: {bad_rows.tolist()}")
        print(
            "Rows entirely non-finite:",
            int((~finite_mask).all(axis=1).sum())
        )
        print(f"NaN count: {int(np.isnan(features).sum()):,}")
        print(f"+Infinity count: {int(np.isposinf(features).sum()):,}")
        print(f"-Infinity count: {int(np.isneginf(features).sum()):,}")

        if index_path is not None:
            state_indices = np.load(
                index_path, mmap_mode="r", allow_pickle=False
            )

            valid_indices = state_indices[
                (state_indices >= 0)
                & (state_indices < len(features))
            ]

            affected_state_rows = np.intersect1d(
                bad_rows, valid_indices
            )

            print(f"Index file: {index_path.relative_to(ROOT)}")
            print(
                "Affected image rows referenced by state indices:",
                len(affected_state_rows),
            )

            if len(affected_state_rows):
                print(
                    "Affected referenced indices:",
                    affected_state_rows.tolist(),
                )

            print(
                "State indices outside image-feature array bounds:",
                len(state_indices) - len(valid_indices),
            )

    print("\nDiagnostics complete. No files were modified.")


if __name__ == "__main__":
    main()