
from pathlib import Path
import ast
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = Path.cwd()
OUTPUT = ROOT / "PROJECT_INSPECTION_REPORT.md"

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    ".pytest_cache", "node_modules", "build", "dist"
}

KEY_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "docs/PHASE_1_PAPER_SPECIFICATION.md",
    "src/models/itanet.py",
    "src/models/sales_predictor.py",
    "src/environment/pricing_environment.py",
    "src/environment/reward.py",
    "src/environment/ppo_dataset.py",
    "scripts/build_state.py",
    "scripts/train_sales_predictor.py",
    "scripts/train_ppo_warmstart.py",
    "scripts/train_ppo_transfer.py",
    "scripts/evaluate_ppo_reproduction.py",
    "scripts/evaluate_ppo_transfer.py",
    "scripts/evaluate_ppo_rankings.py",
    "scripts/audit_paper_evaluation.py",
    "scripts/evaluate_ppo_economic_robustness.py",
    "scripts/evaluate_uncertainty_aware_pricing.py",
]

IMPORTANT_REPORTS = [
    "reports/base_paper_reproduction/ppo_test_summary.json",
    "reports/base_paper_reproduction/pricing_baseline_comparison.json",
    "reports/base_paper_reproduction/ppo_vs_fixed20_paired_bootstrap.json",
    "reports/evaluation_audit/paper_evaluation_audit.json",
    "reports/algorithm_comparison/paper_aligned_metrics/growth_summary.csv",
    "reports/base_paper_reproduction/ppo_test_action_distribution.csv",
    "reports/base_paper_reproduction/ppo_test_predictions.csv",
    "reports/base_paper_reproduction/all_action_counterfactuals.csv",
]

def run_command(command):
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return result.returncode, output.strip()
    except Exception as exc:
        return 1, str(exc)

def code_block(text, language="text"):
    return f"```{language}\n{text.rstrip()}\n```"

def relative(path):
    return path.relative_to(ROOT).as_posix()

def inspect_python_file(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    try:
        tree = ast.parse(text)
        imports = []
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(
                    f"from {node.module or ''} import "
                    + ", ".join(alias.name for alias in node.names)
                )
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        summary = [
            f"**Lines:** {len(lines)}",
            f"**Syntax parse:** PASS",
            f"**Imports:** {', '.join(sorted(set(imports))) or 'None detected'}",
            f"**Classes:** {', '.join(sorted(set(classes))) or 'None detected'}",
            f"**Functions:** {', '.join(sorted(set(functions))) or 'None detected'}",
        ]
    except SyntaxError as exc:
        summary = [
            f"**Lines:** {len(lines)}",
            f"**Syntax parse:** FAILED — line {exc.lineno}: {exc.msg}",
        ]

    # Include source content for the files most relevant to methodology.
    limit = 12000
    source = text[:limit]
    if len(text) > limit:
        source += "\n\n[TRUNCATED: inspect the original file for remaining content.]"

    return "\n".join(summary) + "\n\n" + code_block(source, "python")

def summarize_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if len(text) > 10000:
            text = text[:10000] + "\n[TRUNCATED]"
        return code_block(text, "json")
    except Exception as exc:
        return f"Could not parse JSON: `{exc}`"

def summarize_csv(path):
    try:
        with path.open(
            "r", encoding="utf-8-sig",
            errors="replace", newline=""
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            row_count = 0
            sample = []

            for row in reader:
                row_count += 1
                if len(sample) < 5:
                    sample.append(row)

        sample_text = "\n".join(
            [",".join(header)] + [",".join(row) for row in sample]
        )

        return (
            f"- **Data rows:** {row_count}\n"
            f"- **Columns:** {len(header)}\n"
            f"- **Column names:** {', '.join(header)}\n"
            f"- **Sample:**\n\n"
            f"{code_block(sample_text, 'csv')}"
        )
    except Exception as exc:
        return f"Could not parse CSV: `{exc}`"

def main():
    if not (ROOT / "src").exists() or not (ROOT / "scripts").exists():
        print(
            "ERROR: Run this script from the project root "
            "containing both src and scripts."
        )
        return 1

    sections = [
        "# Dynamic Pricing for Ecommerce — Project Inspection Report",
        (
            f"> Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"> Project root: `{ROOT}`\n"
            "> Inspection mode: read-only; no training or tests executed."
        ),
        """## 1. Scope and safety

This report inventories the current project, selected source code,
Git status, and existing evaluation artifacts.

This script does not train models, modify source code, alter checkpoints,
delete files, or execute project tests. It writes only this report.

Existing metrics are evidence to inspect, not proof that the implementation
matches the base paper.""",
    ]

    return_code, git_status = run_command(
        ["git", "status", "--short", "--branch"]
    )
    sections.append(
        "## 2. Git status\n" +
        (code_block(git_status or "(empty)") if return_code == 0
         else f"Git unavailable: `{git_status}`")
    )

    return_code, git_log = run_command(
        ["git", "log", "-5", "--oneline", "--decorate"]
    )
    sections.append(
        "## 3. Recent commits\n" +
        (code_block(git_log or "(no commits)") if return_code == 0
         else f"Git log unavailable: `{git_log}`")
    )

    files = []
    for folder, directories, filenames in os.walk(ROOT):
        directories[:] = sorted(
            name for name in directories if name not in EXCLUDED_DIRS
        )
        for filename in filenames:
            path = Path(folder) / filename
            if path != OUTPUT and path.name != OUTPUT.name:
                files.append(path)

    files.sort(key=lambda path: relative(path).lower())

    sections.append(
        f"## 4. File inventory\n\n"
        f"**Files found:** {len(files)}\n\n" +
        "\n".join(
            f"- `{relative(path)}` — {path.stat().st_size:,} bytes"
            for path in files
        )
    )

    sections.append(
        "## 5. Python runtime\n\n" +
        code_block(
            f"Executable: {sys.executable}\n"
            f"Version: {sys.version}"
        )
    )

    for name in [
        "requirements.txt", "pyproject.toml",
        "environment.yml", "setup.cfg"
    ]:
        path = ROOT / name
        if path.exists():
            content = path.read_text(
                encoding="utf-8", errors="replace"
            )
            sections.append(
                f"### `{name}`\n\n" +
                code_block(content[:10000])
            )

    sections.append("## 6. Key source-code inspection")

    for name in KEY_FILES:
        path = ROOT / name

        if not path.exists():
            sections.append(f"### `{name}`\n\n**Not found.**")
            continue

        if path.suffix == ".py":
            details = inspect_python_file(path)
        else:
            details = (
                f"**Size:** {path.stat().st_size:,} bytes\n\n"
                + code_block(
                    path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:10000]
                )
            )

        sections.append(f"### `{name}`\n\n{details}")

    sections.append("## 7. Existing evaluation artifacts")

    for name in IMPORTANT_REPORTS:
        path = ROOT / name

        if not path.exists():
            sections.append(f"### `{name}`\n\n**Not found.**")
            continue

        sections.append(
            f"### `{name}`\n\n"
            f"**File size:** {path.stat().st_size:,} bytes\n\n"
        )

        if path.suffix.lower() == ".json":
            sections.append(summarize_json(path))
        elif path.suffix.lower() == ".csv":
            sections.append(summarize_csv(path))
        else:
            sections.append("Present; format not summarized.")

    test_files = [
        path for path in files
        if path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or "tests" in path.parts
    ]

    sections.append(
        "## 8. Test-file discovery\n\n"
        "**Tests executed:** No\n\n" +
        ("\n".join(f"- `{relative(path)}`" for path in test_files)
         if test_files else "No test files found by filename/path scan.")
    )

    sections.append("""## 9. Methodology audit checklist

The following items require comparison with the actual base-paper PDF.

- [ ] Verify the paper's equations and experimental protocol.
- [ ] Compare state modalities and feature dimensions.
- [ ] Document unavailable review-sentiment data without fabricating features.
- [ ] Verify action spaces for each trained checkpoint.
- [ ] Verify reward equations, normalization, and coefficients.
- [ ] Verify sales-predictor architecture and training protocol.
- [ ] Verify PPO training settings and policy-evaluation procedure.
- [ ] Verify temporal/product splits and leakage prevention.
- [ ] Map every reported metric to the paper's equation, table, or figure.
- [ ] Check that baseline comparisons use compatible conditions.
- [ ] Separate paper-reproduction results from the proposed research extension.
- [ ] Implement only experiments supported by available data and evidence.
- [ ] Record unavailable data and irreproducible details as limitations.

## 10. Interpretation warning

Different result files may come from different checkpoints, action spaces,
reward definitions, or evaluation scripts. Their numbers must not be combined
until methodological comparability has been verified.

This report is an inventory, not the final scientific assessment.""")

    OUTPUT.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8"
    )

    print(f"\nInspection report created:\n{OUTPUT}")
    print(f"Files inventoried: {len(files)}")
    print("No models trained. No tests executed.")
    print("No existing source files or checkpoints modified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
