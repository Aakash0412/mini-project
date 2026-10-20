from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "evaluation_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = OUT_DIR / "reward_evaluation_source_review.md"
OUT_JSON = OUT_DIR / "reward_evaluation_source_review.json"

TARGETS = {
    "scripts/evaluate_ppo_transfer.py": [
        r"^def main\b", r"ArgumentParser", r"add_argument", r"parse_args",
        r"DynamicPricingEnvironment", r"get_reward_config", r"SALES_REWARD",
        r"RewardConfig", r"reward_config", r"alpha\s*=", r"beta\s*=",
        r"to_csv", r"OUTPUT", r"output_path", r"checkpoint_path",
    ],
    "src/environment/reward.py": [
        r"^class RewardConfig", r"^def get_reward_config", r"^def calculate_reward",
        r"^def calculate_reward_from_config", r"^def reward_breakdown",
        r"SALES_REWARD\s*=", r"BALANCED_REWARD\s*=", r"PROFIT_REWARD\s*=",
        r"REWARD_CONFIGS\s*=",
    ],
    "src/environment/pricing_environment.py": [
        r"^class DynamicPricingEnvironment", r"def __init__", r"reward_config",
        r"def step", r"def evaluate", r"def .*candidate", r"DISCOUNT_BINS",
        r"def reset",
    ],
}
results = {}
md = [
    "# Reward Evaluation Source Review",
    "",
    "Read-only source excerpt report. This script does not import or execute project modules, run inference, train, or modify checkpoints.",
]
for rel, patterns in TARGETS.items():
    p = ROOT / rel
    entry = {"exists": p.exists(), "path": rel, "matches": []}
    md += ["", f"## `{rel}`"]
    if not p.exists():
        md.append("- File not found.")
        results[rel] = entry
        continue
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = p.read_text(encoding="cp1252").splitlines()
    selected = set()
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [i for i, line in enumerate(lines) if rx.search(line)]
        # For repeated broad matches, keep only first 12 hits per pattern.
        for i in hits[:12]:
            for j in range(max(0, i - 3), min(len(lines), i + 9)):
                selected.add(j)
    ranges = []
    for idx in sorted(selected):
        if not ranges or idx > ranges[-1][-1] + 1:
            ranges.append([idx])
        else:
            ranges[-1].append(idx)
    for group in ranges:
        # cap large sections to avoid overwhelming report
        group = group[:100]
        block = {"start_line": group[0] + 1, "end_line": group[-1] + 1,
                 "text": "\n".join(f"{i+1}: {lines[i]}" for i in group)}
        entry["matches"].append(block)
        md += ["", f"### Lines {block['start_line']}–{block['end_line']}", "```python", block["text"], "```"]
    results[rel] = entry

md += [
    "",
    "## Review checklist",
    "- Confirm the evaluator's reward weights come from the environment's selected `RewardConfig`.",
    "- Confirm whether the reward configuration can be changed at evaluation time without retraining.",
    "- Confirm the reward function used for action selection is not accidentally confused with the metric used for scoring.",
    "- Confirm output CSV names/paths will not overwrite existing evaluation artifacts.",
    "- Only after these are verified should a separate read-only evaluation command be run.",
]
OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
OUT_JSON.write_text(json.dumps({"read_only": True, "execution_performed": False,
                                "training_performed": False, "checkpoint_modified": False,
                                "sources": results}, indent=2), encoding="utf-8")
print(f"Source review created: {OUT_DIR}")
print("Files: reward_evaluation_source_review.md and reward_evaluation_source_review.json")
print("No project code was imported or executed. No inference/training performed; checkpoints unchanged.")
