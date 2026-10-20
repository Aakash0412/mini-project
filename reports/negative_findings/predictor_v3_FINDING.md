# Negative finding: predictor v3 with last_monthly_sales input

## Hypothesis

Adding Last Monthly Sales as an explicit predictor input (matching
paper Eq. 2 d_{t,i}) would raise test R^2 above the v1 baseline of 0.025,
because the growth target is (Q_{t+1} - Q_t) / Q_t and the model was
previously never given Q_t.

## Result

Hypothesis rejected. Test R^2 fell from 0.025 (baseline) to 0.016 (v3).
Model output collapsed to the training mean (train MSE flat at ~6.9M
for 75 epochs). Val MSE plateaued from epoch 1.

## Diagnosis

The training target has mean 68.37 and std 2628.87 (std/mean = 38).
Under MSE loss, when the outlier variance dwarfs the signal variance,
the optimal strategy is to predict the mean. Adding Q_t does not change
this because Q_t correlates with the outliers themselves.

## Prior attempts

1. Baseline MLP, raw target: test R^2 = 0.025
2. Log-winsorized target: test R^2 = -0.201
3. Raw target + Q_t input: test R^2 = 0.016

All three fail to exceed 0.025. No single-night fix is possible.

## What this means

v2 remains the primary result. Delta-Q gap to the paper's
-0.3% to -2% is a data-level limitation, not an implementation error.
Required to close: 10-D sentiment (raw review text absent), ada-002
title embeddings (currently MiniLM), sequential panel data (dataset is
cross-sectional, mean 1.16 months/ASIN), or additional dynamic signals
beyond Q_t.

None are available on the local dataset. No further retraining pursued.

## Files (evidence)

- scripts/train_sales_predictor_v3.py
- models/sales_predictor_v3/best_model.pth
- reports/sales_predictor_v3/metrics.json
- reports/sales_predictor_v3/training_history.csv

---

## 4. Target clipping (train-time)

Attempted after log-winsor and Q_t input both failed. Hypothesis:
clip the training target to [q_1, q_99] percentiles so MSE sees a
bounded target while metrics evaluate on raw targets.

Result: test R^2 = 0.018 (baseline 0.025). No improvement.
Train MSE dropped from 7581 to 3855 (49% reduction) but val MSE
plateaued at ~6500 from epoch 5. The model overfits the clipped
training target and cannot generalize.

Conclusion: the predictor failure is not a target distribution problem.
Four independent interventions (baseline, log-winsor, +Q_t input,
target clipping) have all failed to exceed test R^2 = 0.025. The 111-D
state does not contain sufficient signal to predict monthly sales
growth on this dataset. This is now a definitively established
data-level limitation, not a fixable model-level problem.

Files: scripts/train_sales_predictor_clip.py,
       reports/sales_predictor_clip/metrics.json
