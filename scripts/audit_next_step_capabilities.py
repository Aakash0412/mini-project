from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "evaluation_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = REPORT_DIR / "next_step_capability_audit.md"
OUT_JSON = REPORT_DIR / "next_step_capability_audit.json"

TARGETS = [
    "src/environment/reward.py",
    "scripts/evaluate_ppo_transfer.py",
    "scripts/evaluate_ppo.py",
    "scripts/run_robustness.py",
    "src/environment/pricing_environment.py",
    "src/environment/environment.py",
]

def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="cp1252")
        except Exception:
            return None
    except Exception:
        return None

def inspect_python(path: Path, source: str) -> dict:
    result = {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "line_count": len(source.splitlines()),
        "classes": [],
        "functions": [],
        "cli_flags": sorted(set(re.findall(r'add_argument\(\s*["\'](--[A-Za-z0-9_-]+)', source))),
        "reward_weight_mentions": sorted(set(re.findall(
            r'.{0,70}(?:alpha|beta|SALES_REWARD|BALANCED_REWARD|PROFIT_REWARD|reward_config|reward_type).{0,100}',
            source, flags=re.IGNORECASE
        ))),
        "action_space_mentions": sorted(set(re.findall(
            r'.{0,60}(?:action_dim|n_actions|discount_bins|discounts|action_space).{0,100}',
            source, flags=re.IGNORECASE
        ))),
    }
    try:
        tree = ast.parse(source)
        result["classes"] = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        result["functions"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError as exc:
        result["parse_error"] = str(exc)
    return result

files = {}
for rel in TARGETS:
    path = ROOT / rel
    if not path.exists():
        files[rel] = {"path": rel, "exists": False}
        continue
    source = read_text(path)
    if source is None:
        files[rel] = {"path": rel, "exists": True, "read_error": "Could not decode source"}
    elif path.suffix == ".py":
        files[rel] = inspect_python(path, source)
    else:
        files[rel] = {"path": rel, "exists": True, "line_count": len(source.splitlines())}

# Summarize known evaluation artifacts without modifying them.
artifacts = {}
for rel in [
    "models/ppo_transfer/transfer_supported_8action_policy_evaluation.csv",
    "reports/ablation/ablation_results.csv",
    "reports/robustness/robustness_results.csv",
    "models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth",
]:
    p = ROOT / rel
    artifacts[rel] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None}

payload = {
    "read_only": True,
    "training_performed": False,
    "checkpoint_modified": False,
    "purpose": "Determine whether current scripts support reward-weight evaluation and what baseline/environment code exists before adding or running experiments.",
    "files": files,
    "artifacts": artifacts,
    "interpretation": [
        "This is a source-capability inventory only; it does not run model inference, training, or pricing experiments.",
        "A reward configuration being defined does not prove that evaluation under it has been run.",
        "Do not launch training until the evaluation entry points and reward override behavior are verified."
    ]
}
OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# Next-Step Capability Audit",
    "",
    "Read-only source inventory. No training, inference, or checkpoint changes.",
    "",
    "## Existing evaluation artifacts",
]
for rel, info in artifacts.items():
    lines.append(f"- **{rel}:** {'FOUND' if info['exists'] else 'MISSING'}" +
                 (f" ({info['size_bytes']} bytes)" if info["size_bytes"] is not None else ""))

lines += ["", "## Source files and capabilities"]
for rel, info in files.items():
    lines.append(f"### `{rel}`")
    if not info.get("exists"):
        lines.append("- Missing.")
        continue
    if info.get("read_error"):
        lines.append(f"- Read error: {info['read_error']}")
        continue
    lines.append(f"- Lines: {info.get('line_count', 'unknown')}")
    if info.get("classes"):
        lines.append("- Classes: " + ", ".join(info["classes"]))
    if info.get("functions"):
        lines.append("- Top-level functions: " + ", ".join(info["functions"]))
    if info.get("cli_flags"):
        lines.append("- CLI flags: " + ", ".join(info["cli_flags"]))
    if info.get("reward_weight_mentions"):
        lines.append("- Reward-related source excerpts:")
        for excerpt in info["reward_weight_mentions"][:20]:
            lines.append("  - `" + " ".join(excerpt.strip().split()) + "`")
    if info.get("action_space_mentions"):
        lines.append("- Action-space source excerpts:")
        for excerpt in info["action_space_mentions"][:12]:
            lines.append("  - `" + " ".join(excerpt.strip().split()) + "`")
    if info.get("parse_error"):
        lines.append("- AST parse issue: " + info["parse_error"])

lines += [
    "",
    "## Safety notes",
    "- This script only reads source and artifact metadata.",
    "- It does not execute project scripts, run inference, train models, or alter checkpoints.",
    "- Use this report to decide the smallest safe next experiment."
]
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Capability audit complete: {REPORT_DIR}")
print("Created next_step_capability_audit.md and next_step_capability_audit.json")
print("Read-only: no inference/training performed; checkpoints unchanged.")
