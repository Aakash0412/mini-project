# Base Paper vs. Implementation — Research Comparison

## 1. Reference paper

Liu et al., "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce," IEEE Access, 2026.

This report distinguishes the paper specification from the implemented system. A difference is not automatically a reproduction failure when the paper leaves a detail unspecified or the implementation documents a data limitation.

## 2. Methodology comparison

| Component | Paper specification | Implemented system | Assessment |
|---|---|---|---|
| Dataset | Amazon discounted products, Jan–Oct 2023 | 34,041 source rows; 27 categories | Dataset scope aligned |
| Image features | ResNet-50, 32-D output | ResNet-50 features projected to 32-D | Architecture aligned; projection is an implementation choice |
| Title features | OpenAI ada-002, 32-D state contribution | Local all-MiniLM-L6-v2 embeddings projected to 32-D | Engineering substitution |
| Review sentiment | GPT-4o-based sentiment, 10-D | Not included in final state | Missing modality; not a full multimodal reproduction |
| Structured attributes | 47-D | 47-D | Dimensionality aligned |
| Total state | 121-D | 111-D | Ten dimensions absent because sentiment is unavailable |
| Action space | 0%, 5%, ..., 40% (nine tiers) | 5%, 10%, ..., 40% (eight tiers) | 0% action excluded from final supported policy |
| Sales predictor | Predicts monthly sales growth; frozen before policy optimization | Frozen predictor used in the counterfactual environment | Functional design aligned; target provenance remains a limitation |
| Reward | alpha × profit-margin growth + beta × sales growth | Weighted relative changes in quantity and a gross-margin proxy | General weighted form aligned; economic quantities are not identical |
| PPO | Primary DRL algorithm | Transfer-initialized PPO checkpoint, 1,000 updates | PPO used; transfer initialization is an extension |
| Data split | Exact split not established by the current paper specification | Chronological: train months 1–6, validation 7–8, test 9–10 | Documented implementation choice |
| Test set | Paper reports its own evaluation population | 7,139 held-out products | Results are not directly comparable without matching evaluation definitions |

## 3. Held-out baseline comparison

All six policies were evaluated on the same 7,139 held-out products. Reward uses the evaluation configuration alpha = 0.2 and beta = 0.8.

| Policy | Mean reward | Median reward | Mean discount | Estimated sales | Estimated revenue | Margin proxy |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 0.384784 | 0.209864 | 21.04% | 3,296.10 | 105,996.27 | 69,800.18 |
| Observed discount, nearest grid | 0.340311 | 0.160053 | 21.88% | 3,221.65 | 98,408.89 | 64,903.25 |
| Fixed 10% | 0.333709 | 0.132166 | 10.00% | 3,087.88 | 112,235.15 | 73,932.22 |
| Fixed 20% | 0.360722 | 0.178051 | 20.00% | 3,231.95 | 104,753.23 | 69,091.52 |
| Fixed 30% | 0.348241 | 0.198548 | 30.00% | 3,305.60 | 93,977.37 | 62,017.96 |
| Offline oracle, 5–40% | 0.414999 | 0.239181 | 19.69% | 3,359.38 | 108,242.65 | 71,268.14 |

### Interpretation

- PPO has the highest mean reward among the tested practical policies.
- PPO's mean reward is approximately 13.07% higher than the observed-discount baseline and 6.67% higher than the fixed 20% baseline.
- Fixed 10% yields the highest estimated margin proxy, while fixed 30% yields the highest estimated sales. These are different objectives from the weighted reward.
- PPO reaches 92.72% of the oracle's mean reward, using the ratio of aggregate mean rewards.
- The oracle selects the highest-reward action from the supported discount grid using the same simulator. It is an offline model-based comparator, not a deployable policy or ground truth.

All economic outcomes are simulator estimates. The margin metric is a gross-margin proxy, not a complete accounting-profit calculation.

## 4. Robustness evidence

The transfer-PPO state-perturbation experiment reported:

| Relative input-noise level | Action flip rate | Mean absolute discount change | Mean total-variation distance |
|---|---:|---:|---:|
| 0% | 0.0000% | 0.0000 percentage points | 0.000000 |
| 1% | 0.6023% | 0.0630 percentage points | 0.006415 |
| 5% | 3.2918% | 0.3432 percentage points | 0.032174 |
| 10% | 6.3174% | 0.6549 percentage points | 0.063529 |

In the associated model-based economic evaluation, mean reward at 10% input noise was 0.384286 versus 0.384784 at baseline, approximately 99.87% retention.

These experiments support a limited claim: the evaluated policy was relatively stable under the specified synthetic perturbations. They do not establish robustness to every real-world data error, distribution shift, or market change. The older nine-action PPO robustness experiment is a separate experiment and should not be mixed with these transfer-PPO results.

## 5. Reproduction limitations

1. The final state omits the paper's 10-D sentiment modality.
2. The final transfer policy uses eight supported actions rather than the paper specification's nine.
3. The implemented margin calculation is a proxy and does not reproduce every cost and sales decomposition in the paper's economic equations.
4. The checkpoint does not independently verify its training reward weights. The 0.2/0.8 weights are verified for the evaluation configuration; do not claim they are proven training weights.
5. The supplied sales-growth target's exact derivation has not been established. Avoid describing it as verified ground truth.
6. Results are based on a counterfactual simulator, not a live pricing trial.
7. No statistical significance claim should be made without suitable repeated runs or uncertainty analysis.

## 6. Defensible contribution statement

This implementation evaluates a transfer-initialized PPO pricing policy against observed-discount and fixed-discount baselines, and measures recommendation stability and simulated reward retention under controlled state perturbations on a chronological held-out set.

The experiments provide evidence about simulator-level baseline performance and sensitivity to the tested perturbations. They do not, by themselves, establish real-world profit improvement or prove that the method is novel relative to all prior literature.

## 7. Recommended final checks

- Verify target provenance before treating sales-growth predictions as ground truth.
- Keep the final transfer-PPO robustness results separate from the legacy nine-action PPO results.
- Record the exact evaluation command, checkpoint hash, package versions and random seeds.
- If claiming a statistically reliable advantage, run multiple independent seeds or a justified uncertainty analysis without overwriting the current baseline.
- Ensure the final paper clearly labels reproduced components, engineering substitutions, omissions and research extensions.

