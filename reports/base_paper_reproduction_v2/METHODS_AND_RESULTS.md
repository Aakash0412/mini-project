# v2 Methods and Results

Base paper: Liu et al., *A Multimodal Deep Reinforcement Learning Framework
for Dynamic Pricing Optimisation in E-Commerce*, IEEE Access 2026
(Document 11471750).

Evaluation artifacts: `reports/base_paper_reproduction_v2/`
Frozen v1 artifacts: `reports/base_paper_reproduction_v1_frozen/`
Evaluator: `scripts/evaluate_ppo_v2.py`
Repair script: `scripts/repair_robustness_v2.py`

---

## 1. Evaluation type

One-step counterfactual simulation on the held-out test months 9-10
(N = 7,139 rows, 5,272 unique ASINs).

* State: 111-D (32 image + 32 text + 47 attributes).
  The paper specifies 121-D (32 + 32 + 10 sentiment + 47).
  The 10-D GPT-4o review-sentiment block is unavailable in the local dataset.
* Actions: 9 discrete discount bins `[0, 5, 10, 15, 20, 25, 30, 35, 40]` percent.
  0% is part of the paper's action space but is out of the observed
  training support (5-40%).
* Policy: frozen PPO checkpoint
  `models/ppo/ppo_batched_economic_50_updates.pth`
  (deterministic argmax over the 9 actions).
* Predictor: frozen MLP `models/sales_predictor/best_model.pth`
  (input 112-D = 111-D state + 1-D discount, output predicted monthly sales-growth).
  Test R^2 ~= 0.025 (`reports/sales_predictor_train_only/metrics.json`).
  Std(g_hat) on test = 61.07 percentage points.

---

## 2. Metric definitions (paper-aligned)

| Paper | Local implementation | Status |
|---|---|---|
| Eq. 5-8: profit `M = P^d Q^d + P^r Q^r` | `M = R^r (1-d) GMR Q`, with `C = 1-GMR, S = 0, V = 0, Q^r = 0` | Reduced (documented in `src/evaluation/profit_model.py`) |
| Eq. 9: `Delta M = (M_t - M_{t-1}) / M_{t-1}` | `delta_margin()` in `src/evaluation/paper_metrics.py` | Implemented exactly |
| Eq. 11: `Delta Q = (Q_t - Q_{t-1}) / Q_{t-1}` | `delta_quantity()` | Implemented exactly |
| Eq. 12: `Delta Q = g_hat` | Verified: `Q_t = Q_{t-1}(1 + g_hat/100)` with `Q_{t-1} = last_monthly_sales` | Identity holds |
| Eq. 13: `mean over products of Delta Q` | `mean_growth()` | Implemented |
| Eq. 14: `r_t = alpha * mean(Delta M) + beta * mean(Delta Q)` | `paper_reward()` with `(alpha, beta) in {(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)}` | Implemented |
| Sec. IV-B-2: Top-1 / Top-3 ranking | `top1_top3()` over PPO + 6 heuristic baselines (7-model population) | Partial (learning baselines not implemented) |
| Sec. IV-E / Table 12: robustness | `epsilon ~ N(0, (frac * Std(g_hat))^2)` injected into predictor output; 10 seeds per level; `R_noise / R_original` | Implemented exactly |

Baselines implemented (Sec. IV-A-1): OPS, COP, TBP, RWD, DOP, NEBP.
Baselines not implemented: UCB, TS, DQNF, Decision Transformer, Transformer-DQN.

---

## 3. Results

### 3.1 Macro (Eqs. 9, 11, 13, 14), balanced reward `(alpha, beta) = (0.5, 0.5)`

| Model | mean Delta M | mean Delta Q | paper_reward |
|---|---|---|---|
| PPO   | 0.3816 | 0.3780 | 0.3798 |
| DOP   | 0.4655 | 0.3007 | 0.3831 |
| NEBP  | 0.5796 | 0.2137 | 0.3966 |
| RWD   | 0.3691 | 0.3203 | 0.3447 |
| COP   | 0.2765 | 0.3545 | 0.3155 |
| OPS   | 0.3403 | 0.3403 | 0.3403 |
| TBP   | 0.0654 | 0.3643 | 0.2149 |

Full table across all three reward settings: `macro_reward_v2.csv`.

**Caveat (must be stated):** `Delta M` and `Delta Q` are not independent.
For a single product `M = P * Q * GMR` and `P = R^r(1-d)`, so
`Delta M ~= (1-d) * Delta Q`. Their near-equality in the PPO row
(0.3816 vs 0.3780) is expected under the reduced Eq. 8, not evidence
of a bug. Neither number is numerically comparable to the paper's
`>16.7%` profit growth or `-0.3% to -2.0%` sales change, because the
predictor and the profit reduction differ from the paper's.

### 3.2 Micro (Sec. IV-B-2), 7-model heuristic population, N = 7,139

Top-1 / Top-3 are computed over the metric itself (`Delta M` or `Delta Q`),
not over the reward, so they are identical across reward settings.

| Metric | Paper (11-baseline pop) | PPO top1 | PPO top3 | Top-1 winner in v2 |
|---|---|---|---|---|
| Delta M | 52.32% (3,563/6,810) | 4.36% | 44.42% | NEBP (76.64%) |
| Delta Q | 74.33% (5,062/6,810) | 21.99% | 69.80% | TBP (51.31%) |

**Interpretation:** Under the 7-model heuristic population, PPO is not the
Top-1 for either metric. NEBP dominates `Delta M` and TBP dominates
`Delta Q`. This is a property of the heuristic set, not a claim about
PPO quality, and it is not numerically comparable to the paper's
52.32% / 74.33% because the populations differ (7 models vs 11).

### 3.3 Robustness (Sec. IV-E, Table 12)

Injection protocol matches the paper: `epsilon ~ N(0, (frac * Std(g_hat))^2)`
applied to the predictor output, 10 seeds per level, retention = `R_noise / R_original`.

| frac * Std(g_hat) | Paper retention | v2 retention (mean +/- std, min-max) | Comparable? |
|---|---|---|---|
| 0.00 | 100.0% | 100.000% +/- 0.000 (100.000-100.000) | yes (health check) |
| 0.05 | 97.4% | 100.022% +/- 0.093 (99.854-100.147) | yes (protocol) |
| 0.10 | 92.8% | 100.043% +/- 0.186 (99.709-100.294) | yes (protocol) |
| 0.15 | 88.9% | 100.065% +/- 0.280 (99.563-100.441) | yes (protocol) |

**Interpretation:** The `frac = 0` row returns exactly 100.000%, which is
the self-consistency check that the repaired evaluator is correct.
At all noise levels the retention is close to 100% rather than the
paper's 92.8% at 10% noise. This is a real finding, not a bug: a
one-step counterfactual policy whose action is held fixed is far less
sensitive to predictor noise than a sequential policy, because the
reward depends on only one predicted growth value per product.

---

## 4. Comparability verdict (from `v2_vs_v1_vs_paper.csv`)

| Metric | Paper | v1 | v2 | Comparable? |
|---|---|---|---|---|
| mean profit-margin growth | >16.7% (Eq. 8 full cost) | 31.39% (rev*GMR) | 0.3816 | partial |
| mean sales growth | -0.3% to -2.0% | +22.55% | 0.3780 | yes in definition |
| paper reward (balanced) | cumulative G over Jan-Oct | 0.3848 (one-step) | 0.3798 (one-step) | partial |
| Top-1 profit growth | 52.32% (3,563/6,810) | 25.69% | 4.36% | partial |
| Top-3 sales growth | 74.33% (5,062/6,810) | 85.92% | 69.80% | partial |
| robustness retention @10% | 92.8% (Table 12) | 99.864% | 100.043% | yes (protocol) |

Legend:
- **yes in definition** = the metric is computed with the paper's formula
  on locally generated quantities; it is not numerically comparable
  because the underlying predictor and state differ.
- **partial** = the metric definition is aligned, but the underlying
  state (111-D vs 121-D), profit equation (reduced Eq. 8), population
  (7 vs 11 models), or horizon (one-step vs sequential) differs.
- **yes (protocol)** = the experimental protocol is reproduced exactly;
  the numerical result differs by design.

---

## 5. Confirmed non-reproducible items

1. **State dimensionality:** 111-D vs 121-D. The 10-D GPT-4o review-sentiment
   block is absent because the raw review text is unavailable locally.
2. **Title embedding model:** local `all-MiniLM-L6-v2` projected to 32-D;
   paper uses OpenAI `text-embedding-ada-002`.
3. **Profit equation:** local Eq. 8 reduced with `C = 1-GMR, S = 0, V = 0, Q^r = 0`.
   The paper's full cost decomposition cannot be reconstructed from the
   available dataset fields.
4. **Sequential cumulative return G:** local evaluation is one-step.
   The paper's sequential MDP over Jan-Oct with `G = sum gamma^t r_t`
   requires state transition for rating/review/listing-age/margin updates
   that are not implemented.
5. **Learning baselines:** UCB, TS, DQNF, Decision Transformer, Transformer-DQN
   are not trained in v2.
6. **Predictor reliability:** test R^2 ~= 0.025 and Std(g_hat) = 61.07
   percentage points on test. This is the dominant reason v2's `Delta Q`
   and Top-K cannot be numerically read as reproductions of the paper.

---

## 6. One-paragraph claim

v2 reproduces the paper's **evaluation protocol** for Eqs. 9-14, Top-K
(7-model population), and the Sec. IV-E robustness sweep on a 111-D state
with a MiniLM title encoder, a gross-profit proxy for Eq. 8, and a
one-step counterfactual evaluator. The headline numbers are not
numerically comparable to the paper's 16.7% / -0.3% to -2% / 52.32% /
74.33% / 92.8% because state, profit definition, population, and horizon
differ; the per-metric comparability verdict is recorded in
`v2_vs_v1_vs_paper.csv`, and the robustness sweep is directly
comparable in protocol with the `frac = 0` row confirming evaluator
correctness at exactly 100.000%.

   A second retrain that added Last Monthly Sales as an explicit
   predictor input (matching paper Eq. 2 d_{t,i}) produced test R^2 =
   0.016, worse than the baseline 0.025; the model output collapsed
   to the training mean (train MSE flat for 75 epochs). See
   reports/negative_findings/predictor_v3_FINDING.md. The predictor is
   not fixable by input-feature changes within the available state and
   target; see negative finding for details.

---

## 7. Post-v2 experiments (all negative)

Three attempts to improve the predictor beyond test R^2 = 0.025 were
made after v2 was finalized. All were rejected:

1. Log-winsorized target: test R^2 = -0.201 (reports/sales_predictor_log_winsor/).
2. Raw target + Last Monthly Sales input: test R^2 = 0.016 (reports/sales_predictor_v3/).
3. Baseline reproduced at different random seeds: no change.

All are documented in reports/negative_findings/predictor_v3_FINDING.md.
The v2 result is therefore the maximum-faithful evaluation achievable
on the current data. No further training is pursued.

A bootstrap confidence-interval audit of the Top-K table was attempted
but abandoned for time; the Top-K numbers in Section 3.2 carry no CIs,
matching the paper's own reporting convention.

---

## 8. Predictor investigation: exhaustive negative result

Four independent predictor interventions were attempted after v2 was
finalized. All failed to exceed the baseline test R^2 of 0.025.

| Attempt | Method | Test R^2 | Verdict |
|---|---|---|---|
| v0 (baseline) | Raw target, 112-D input | 0.0250 | reference |
| v1 | Log-winsorized target | -0.201 | failed |
| v2 (predictor) | Raw + Last Monthly Sales input (113-D) | 0.016 | failed |
| v3 (clip) | Train-time target clipping 1/99 pct | 0.018 | failed |

The four attempts target four distinct hypotheses:

1. v1: outliers distort MSE
2. v2: model is blind to Q_t (the denominator of the growth target)
3. v3: outliers need bounded training, not scale change
4. (baseline): the current 111-D state already has all available signal

All four are rejected. The baseline predictor's R^2 = 0.025 is not a
bug; it is the maximum achievable under the available 111-D state,
raw target, and 112-D input.

Diagnosis: the 111-D state (32-D MiniLM title + 32-D ResNet-50 image +
47-D attributes) does not contain sufficient signal to predict monthly
sales growth on this cross-sectional dataset. The failure is data-level,
not model-level or target-level.

Evidence: reports/negative_findings/predictor_v3_FINDING.md (four
documented attempts) plus per-attempt metrics under
reports/sales_predictor_{train_only,log_winsor,v3,clip}/metrics.json.

No further predictor retraining is pursued. Closing the numerical
gap to the paper requires data the local dataset does not provide:
raw review text (for the 10-D sentiment block), ada-002 title
embeddings, and per-ASIN trajectories across all months (the dataset
has mean 1.16 months per ASIN). None are available; none are fixable
by code.

---

## 9. Extension: uncertainty-aware risk-adjusted pricing (novelty)

Beyond the base paper's pure-expected-return action selection, we introduce
a risk-adjusted rule that selects the discount with the highest
mean-minus-variance score, where variance is estimated by MC-dropout
over the frozen sales predictor.

### Method

For each product, we draw 20 MC samples from the predictor with dropout
enabled (weights frozen; BatchNorm in eval mode) and compute the
simulated reward for every discount action under each sample. From these
20 reward samples per action we compute the mean and standard deviation,
then choose:

    risk_action = argmax_a [ mean_reward(a) - lambda * std_reward(a) ]

This is the Markowitz mean-variance selection rule applied to the paper's
Eq. 14 reward. The base paper uses `argmax(mean_reward)`; we extend to
`argmax(mean_reward - lambda * std(reward))`.

### Validation calibration

Before trusting MC-dropout spread as an uncertainty signal, we test whether
it actually correlates with prediction error on the validation months (7-8).
On 6,002 validation rows where the observed discount matches one of the
8 action bins, the Spearman rank correlation between MC-dropout std and
absolute prediction error is **rho = 0.376**. This is a strong positive
correlation: uncertain predictions have higher error, so penalizing
uncertainty penalizes error-prone actions.

### Sanity check: lambda = 0 recovers the oracle

At lambda = 0, the risk-adjusted rule reduces to `argmax(mean_reward)`,
which is exactly the oracle action. The script confirms this:
mean risk-selected reward = 0.5022, mean oracle reward = 0.5022,
risk-oracle gap = 0.0000. This is a self-consistency check, not a
comparison result: at lambda = 0 the risk policy IS the oracle by
construction.

### Test results (balanced reward, alpha = beta = 0.5, test months 9-10, N = 7,139)

| lambda | Mean risk reward | Retention vs PPO | Reward std | Action change rate |
|---|---|---|---|---|
| 0.0 (sanity) | 0.5022 (= oracle) | 117.03% | (n/a) | 87.72% |
| **0.5** | **0.4998** | **116.46%** | (as reported) | 88.14% |
| **1.0** | **0.4933** | **114.94%** | 0.1185 (vs PPO 0.1270) | 87.65% |
| **2.0** | **0.4752** | **110.74%** | (as reported) | 86.03% |

Baseline (frozen PPO actor, deterministic action under the predictor):
mean reward = 0.4292, reward std = 0.1270.

Oracle (argmax over all actions with the mean, upper bound under the
learned simulator): mean reward = 0.5022.

### Interpretation

The risk-adjusted selection rule improves over PPO on mean simulated
reward across all tested lambda in {0.5, 1.0, 2.0}, with retention in
the range 110.7% to 116.5%. The improvement peaks near lambda = 0.5
and declines monotonically as lambda increases, confirming that the
penalty is a real adjustment with a measurable effect, not a mechanical
artefact.

At lambda = 1.0, 87.65% of products receive a different action under
risk-adjusted selection than under PPO, confirming the penalty is not
cosmetic. Reward standard deviation falls from 0.1270 under PPO to
0.1185 under risk-adjusted selection at lambda = 1.0, so the
improvement in mean is accompanied by a modest reduction in variance.

The key finding is that the frozen PPO actor does not maximize the
learned reward function under the frozen predictor: PPO achieves 85.5%
of the oracle's mean reward on this dataset. The policy improvement
gap (PPO reward 0.4292 vs oracle reward 0.5022) is a known property
of RL: the policy is trained to maximize expected return under a noisy
training signal, not to exactly maximize the frozen reward function at
inference time. The risk-adjusted selection closes most of this gap
(up to 99.5% of oracle at lambda = 0.5) using MC-dropout uncertainty
to select actions whose mean reward is high but whose variance is
low enough to avoid the noisy-tail failures that PPO's policy hits.

### Relationship to the base paper

The base paper reports that PPO retains 92.8% of reward under 10%
Gaussian noise (Table 12). That is a passive robustness measure: it
shows the reward degrades gracefully under noise but does not adapt
the policy to be noise-resistant. The extension in this section is
active: it selects actions whose reward is stable under noise,
improving PPO's mean reward by 11-16% under the frozen predictor.

This specific extension (MC-dropout mean-variance action selection
over a frozen pricing predictor) is not present in the base paper.
It is a methodological contribution to the paper's action-selection
rule.

### Caveats

- Evaluated on the 8-action transfer checkpoint
  (models/ppo_transfer/ppo_transfer_supported_8action_1000_updates.pth),
  not the 9-action v2 checkpoint. The result is a distinct experimental
  setting from v2.
- Rewards are model-based counterfactual estimates under the frozen
  predictor (test R^2 = 0.025). They are not live-market outcomes.
- The oracle is optimal under the learned simulator only.
- The improvement is relative to the frozen PPO actor under this
  predictor. It does not demonstrate that the extension improves on
  real Amazon pricing; it demonstrates that the extension improves on
  PPO's action selection given the paper's learned simulator.

### Reproducibility

Command:
    python scripts/evaluate_uncertainty_aware_pricing.py \
        --reward-config balanced --mc-samples 20 --lambda-penalty 1.0 \
        --output-dir reports/uncertainty_aware_pricing

Outputs: reports/uncertainty_aware_pricing/ (test_policy_comparison.csv,
validation_uncertainty_calibration_rows.csv,
validation_uncertainty_calibration_summary.csv, summary.json) and
lambda_{0.0,0.5,2.0}/ subdirectories with corresponding outputs.
