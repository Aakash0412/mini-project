# Final Experimental Results
## Dynamic Pricing for E-commerce

## 1. Evaluation setup

- Final policy: transfer-initialized PPO.
- Checkpoint: `models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth`.
- PPO updates recorded in checkpoint metadata: 1,000.
- State dimension: 111.
- Supported discount actions: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%.
- Held-out test population: 7,139 products.
- Chronological split: training months 1–6, validation months 7–8, test months 9–10.
- Evaluation reward configuration: alpha = 0.2 for margin proxy and beta = 0.8 for quantity.
- Important qualification: the checkpoint metadata does not independently verify the original training reward weights.

## 2. Pricing-policy comparison

All six policies were evaluated on the same held-out products.

| Policy | Mean reward | Median reward | Mean discount | Estimated sales | Estimated revenue | Margin proxy |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 0.384784 | 0.209864 | 21.04% | 3296.10 | 105996.27 | 69800.18 |
| Observed discount, nearest grid | 0.340311 | 0.160053 | 21.88% | 3221.65 | 98408.89 | 64903.25 |
| Fixed 10% | 0.333709 | 0.132166 | 10.00% | 3087.88 | 112235.15 | 73932.22 |
| Fixed 20% | 0.360722 | 0.178051 | 20.00% | 3231.95 | 104753.23 | 69091.52 |
| Fixed 30% | 0.348241 | 0.198548 | 30.00% | 3305.60 | 93977.37 | 62017.96 |
| Offline oracle, 5–40% | 0.414999 | 0.239181 | 19.69% | 3359.38 | 108242.65 | 71268.14 |

### Main findings

- PPO achieved the highest mean reward among the tested practical policies.
- Its mean reward was approximately 13.07% higher than observed-discount pricing and 6.67% higher than fixed 20% pricing.
- Fixed 10% achieved the highest estimated margin proxy, while fixed 30% achieved the highest estimated sales. These outcomes differ because the configured reward combines margin and quantity changes.
- PPO achieved 92.72% of the offline oracle's mean reward. The oracle is a model-based comparator selecting the best action from the supported grid; it is not ground truth or a deployable policy.

Estimated sales, revenue and margin are counterfactual simulator outputs. Margin is a proxy, not full accounting profit.

## 3. Transfer-PPO state sensitivity

The experiment adds seeded Gaussian perturbations to test states. Noise magnitude is defined relative to each feature's standard deviation in the training months. The same noise realization is reused across noise levels.

| Noise relative to training standard deviation | Test products | Action flip rate | Mean absolute discount change | Mean total variation distance |
|---|---:|---:|---:|---:|
| 0% | 7,139 | 0.000% | 0.0000 pp | 0.000000 |
| 1% | 7,139 | 0.602% | 0.0630 pp | 0.006415 |
| 5% | 7,139 | 3.292% | 0.3432 pp | 0.032174 |
| 10% | 7,139 | 6.317% | 0.6549 pp | 0.063529 |

At 10% noise, the 95th-percentile absolute discount change was 10 percentage points. Thus, aggregate recommendations were fairly stable, but individual products could experience a larger change.

## 4. Economic sensitivity

The economic robustness evaluation selects actions using perturbed states but computes economic outcomes using the original, unperturbed state and product data.

| Noise | Mean reward | Reward retention | Mean estimated sales | Mean margin proxy |
|---|---:|---:|---:|---:|
| 0% | 0.384784 | 100.00% | 3296.10 | 69800.18 |
| 1% | 0.384750 | 99.99% | 3295.98 | 69790.07 |
| 5% | 0.384510 | 99.93% | 3294.88 | 69811.90 |
| 10% | 0.384286 | 99.87% | 3294.05 | 69818.10 |

At 10% noise, mean reward retention was approximately 99.87%. The small aggregate change is evidence of simulator-level sensitivity under this specific perturbation setup; it is not proof of robustness to real-world errors or market shifts.

## 5. Multi-seed noise analysis

The separate multi-seed experiment uses 10 noise seeds (101–110), the same fixed policy checkpoint, and 7,139 test products per seed. The policy was not retrained.

| Noise | Mean reward | Between-noise-seed SD | 95% confidence interval for mean reward | Mean reward retention |
|---|---:|---:|---:|---:|
| 0% | 0.384784 | 0.000000 | [0.384784, 0.384784] | 100.000% |
| 5% | 0.384522 | 0.000374 | [0.384254, 0.384790] | 99.932% |
| 10% | 0.384260 | 0.000749 | [0.383725, 0.384796] | 99.864% |
| 15% | 0.383998 | 0.001123 | [0.383195, 0.384802] | 99.796% |

These t-based confidence intervals quantify variation across synthetic noise realizations for a fixed policy and fixed test set. They do not quantify variation across independently trained policies, datasets, or real deployments. The 0% interval is degenerate because the same deterministic baseline is repeated.

## 6. Base-paper deviations and limitations

1. The paper specification describes a 121-D state: image 32-D, text 32-D, sentiment 10-D and attributes 47-D. The implemented state is 111-D because sentiment is absent.
2. The paper specification includes a 0% action, but the final transfer policy supports only 5–40% actions.
3. The implementation uses local MiniLM title embeddings instead of the paper's ada-002 embeddings.
4. The implemented reward uses relative quantity change and a gross-margin proxy. It is not a complete reproduction of every economic term in the paper's profit equations.
5. The exact derivation of the dataset's supplied sales-growth target remains unresolved; it should not be described as verified ground truth.
6. The oracle is computed within the same learned simulator, so its comparison can inherit simulator bias.
7. Synthetic Gaussian noise has not been calibrated to real measurement error.
8. The baseline and robustness experiments do not establish causal real-world gains.
9. No claim of statistical superiority over the base paper is warranted because the evaluation setups and metrics are not directly matched.

## 7. Defensible conclusion

On 7,139 chronological held-out products, the transfer-initialized PPO policy achieved a higher mean simulator reward than the observed-discount baseline and the tested fixed-discount strategies. Controlled state perturbations produced relatively small changes in aggregate simulated reward, including approximately 99.80% retention at 15% noise in the separate 10-noise-seed experiment.

The evidence supports a focused claim about baseline performance and recommendation sensitivity in the implemented simulator. It does not establish real-world profit improvement, optimality, or robustness to all forms of distribution shift.

## 8. Source result files

- Baseline comparison: `reports/evaluation_audit/final_baseline_comparison/summary.csv`
- Baseline metadata: `reports/evaluation_audit/final_baseline_comparison/metadata.json`
- State sensitivity: `reports/robustness/ppo_transfer_state_sensitivity/summary.csv`
- State sensitivity metadata: `reports/robustness/ppo_transfer_state_sensitivity/metadata.json`
- Economic robustness: `reports/robustness/ppo_transfer_economic_robustness/summary.csv`
- Economic robustness metadata: `reports/robustness/ppo_transfer_economic_robustness/metadata.json`
- Multi-seed aggregates: `reports/robustness/multiseed_aggregate.csv`
- Multi-seed metadata: `reports/robustness/multiseed_metadata.json`
- Paper-to-implementation comparison: `reports/evaluation_audit/final_baseline_comparison/paper_vs_implementation.md`
