from pathlib import Path
import json
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHAPES = {
    "attribute_features.npy": (34041, 47),
    "text_features.npy": (34041, 32),
    "image_features.npy": (34041, 32),
    "state.npy": (34028, 111),
    "state_row_indices.npy": (34028,),
}

REQUIRED_FILES = [
    "models/sales_predictor/config.json",
    "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth",
    "models/ppo_transfer/transfer_supported_8action_policy_evaluation.csv",
    "reports/ablation/ablation_results.csv",
    "reports/ablation/ablation_metadata.json",
]

OPTIONAL_REPORTS = [
    "reports/robustness/robustness_results.csv",
    "reports/robustness/robustness_summary.json",
]

errors = []
warnings = []
passed = 0


def report(ok, message, optional=False):
    global passed

    if ok:
        passed += 1
        print(f"[PASS] {message}")
    elif optional:
        warnings.append(message)
        print(f"[PENDING] {message}")
    else:
        errors.append(message)
        print(f"[FAIL] {message}")


def find_file(filename):
    """Find a file by name anywhere inside the project."""
    matches = [
        path for path in ROOT.rglob(filename)
        if ".venv" not in path.parts
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    ]
    return matches[0] if matches else None


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 68)
    print("DYNAMIC PRICING PROJECT - END-TO-END ARTIFACT VALIDATION")
    print("=" * 68)
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print()

    # 1. Check expected files.
    print("--- 1. Required project artifacts ---")
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        report(path.is_file(), f"Found {relative_path}")

    # 2. Validate feature arrays and dimensions.
    
    # 2. Validate feature arrays and dimensions.
    print("\n--- 2. Feature and state arrays ---")
    arrays = {}

    for filename, expected_shape in EXPECTED_SHAPES.items():
        path = find_file(filename)

        if path is None:
            report(False, f"Could not find {filename}")
            continue

        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[filename] = array

            shape_ok = array.shape == expected_shape
            report(
                shape_ok,
                f"{filename}: shape={array.shape}, expected={expected_shape}"
            )

            if array.dtype.kind not in "biufc":
                report(False, f"{filename} has a non-numeric dtype: {array.dtype}")
                continue

            # Image embeddings may contain NaNs for products with unavailable
            # images. Validate only rows referenced by the final state mapping.
            if filename == "image_features.npy":
                index_path = find_file("state_row_indices.npy")

                if index_path is None:
                    report(
                        False,
                        "Cannot validate image features: state row indices missing"
                    )
                    continue

                indices = np.load(
                    index_path, mmap_mode="r", allow_pickle=False
                )

                indices_valid = (
                    indices.ndim == 1
                    and np.issubdtype(indices.dtype, np.integer)
                    and len(indices) == expected_shape[0] - 13
                    and len(indices) > 0
                    and indices.min() >= 0
                    and indices.max() < len(array)
                    and len(np.unique(indices)) == len(indices)
                )

                if not indices_valid:
                    report(
                        False,
                        "State row mapping failed image-feature validation"
                    )
                    continue

                selected = array[indices]
                report(
                    bool(np.isfinite(selected).all()),
                    "Image features referenced by final state are finite"
                )
            else:
                report(
                    bool(np.isfinite(array).all()),
                    f"{filename} contains only finite numeric values"
                )

        except Exception as exc:
            report(False, f"Could not load {filename}: {exc}")


    # 3. Validate row-index mapping.
    print("\n--- 3. State row-index integrity ---")
    indices = arrays.get("state_row_indices.npy")

    if indices is not None and indices.ndim == 1:
        integer_indices = np.issubdtype(indices.dtype, np.integer)
        report(integer_indices, "State row indices use an integer dtype")

        if integer_indices and len(indices) > 0:
            in_range = bool(
                indices.min() >= 0 and indices.max() < 34041
            )
            report(in_range, "State row indices are within the 34,041-row source")

            unique = len(np.unique(indices)) == len(indices)
            report(unique, "State row indices contain no duplicates")

    # 4. Check predictor configuration.
    print("\n--- 4. Predictor configuration ---")
    config_path = ROOT / "models/sales_predictor/config.json"

    if config_path.is_file():
        try:
            config = load_json(config_path)

            state_dim = config.get("state_dim")
            input_dim = config.get("input_dim")

            report(
                state_dim == 111,
                f"Predictor state_dim={state_dim}; expected 111"
            )
            report(
                input_dim == 112,
                f"Predictor input_dim={input_dim}; expected 112"
            )
        except Exception as exc:
            report(False, f"Could not read predictor config: {exc}")

    # Locate a predictor checkpoint without assuming its filename.
    predictor_dir = ROOT / "models/sales_predictor"
    predictor_checkpoints = []

    if predictor_dir.is_dir():
        predictor_checkpoints = [
            p for p in predictor_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".pth", ".pt", ".ckpt", ".joblib"}
        ]

    report(
        bool(predictor_checkpoints),
        "Sales predictor checkpoint exists"
    )

    # 5. Check the latest transfer PPO checkpoint.
    print("\n--- 5. Transfer PPO checkpoint ---")
    ppo_path = (
        ROOT / "models/ppo_transfer/"
        "ppo_transfer_supported_8action_1000_updates.pth"
    )
    report(ppo_path.is_file(), "Latest supported 8-action PPO checkpoint exists")

    # 6. Validate evaluation and ablation reports.
    print("\n--- 6. Evaluation reports ---")
    evaluation_path = (
        ROOT / "models/ppo_transfer/"
        "transfer_supported_8action_policy_evaluation.csv"
    )
    ablation_path = ROOT / "reports/ablation/ablation_results.csv"

    for path in [evaluation_path, ablation_path]:
        if not path.is_file():
            report(False, f"Missing report: {path.relative_to(ROOT)}")
            continue

        try:
            import pandas as pd

            df = pd.read_csv(path)
            has_rows = len(df) > 0
            report(has_rows, f"{path.relative_to(ROOT)} has {len(df)} rows")

            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                finite = bool(np.isfinite(numeric.to_numpy()).all())
                report(
                    finite,
                    f"Numeric values in {path.relative_to(ROOT)} are finite"
                )
        except Exception as exc:
            report(False, f"Could not validate {path.name}: {exc}")

    # 7. Robustness results are tracked separately.
    print("\n--- 7. Robustness evaluation ---")
    for relative_path in OPTIONAL_REPORTS:
        path = ROOT / relative_path
        report(
            path.is_file(),
            f"Robustness artifact exists: {relative_path}",
            optional=True,
        )

    # 8. Summary.
    print("\n" + "=" * 68)
    print("VALIDATION SUMMARY")
    print("=" * 68)
    print(f"Passed checks: {passed}")
    print(f"Failed checks: {len(errors)}")
    print(f"Pending/optional checks: {len(warnings)}")

    if errors:
        print("\nFailures to investigate:")
        for item in errors:
            print(f"  - {item}")

    if warnings:
        print("\nPending items:")
        for item in warnings:
            print(f"  - {item}")

    summary = {
        "project_root": str(ROOT),
        "python_version": sys.version.split()[0],
        "passed_checks": passed,
        "failed_checks": len(errors),
        "pending_checks": len(warnings),
        "failures": errors,
        "pending": warnings,
        "overall_status": (
            "FAIL" if errors else
            "PASS_WITH_PENDING_ITEMS" if warnings else
            "PASS"
        ),
    }

    output_dir = ROOT / "reports" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "end_to_end_validation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nValidation summary saved to: {output_file.relative_to(ROOT)}")

    if errors:
        return 1
    if warnings:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        sys.exit(130)
    except Exception:
        print("\n[ERROR] Unexpected validation-script failure:")
        traceback.print_exc()
        sys.exit(1)