# Phase 5 Completion Report — Sales Predictor

## 1. Target Definition
| Item | Value |
|---|---|
| Target variable | `Estimated Monthly Sales Growth Rate` |
| Units | Percentage (%) |
| Mean | 66.55 |
| Std | **2104.95** (very high — extreme outliers present) |
| Median | 0.75 |
| Min / Max | −100.00 / 245,100.00 |

> **Note:** The extreme standard deviation (2104.95) and max value (245,100%) reveal
> heavy-tailed outliers in the target. This is the dominant driver of the low R² and
> high RMSE. The dataset is not cleaned for such outliers by the original repository.

## 2. Predictor Input
| Component | Dimension | Source |
|---|---|---|
| ItaNet State | 79 | `data/processed/state/state.npy` |
| Discount action | 1 | `Discount` column |
| **Total input** | **80** | Concatenated |

> No future information is included. Target is from the same observation, not a future period.

## 3. Architecture (IMPLEMENTATION DECISION)
The paper does not specify exact MLP layer sizes for the sales predictor.

```
Input  80
 → Linear(80, 256) → BatchNorm1d → ReLU → Dropout(0.2)
 → Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.1)
 → Linear(128, 64)  → ReLU
 → Linear(64, 1)
Total parameters: 62,721
```

## 4. Training Configuration (IMPLEMENTATION DECISION)
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Batch size | 256 |
| Max epochs | 60 |
| Early stopping patience | 8 |
| Loss | MSELoss |
| Random seed | 42 |
| Device | MPS (Apple Silicon) |

## 5. Training Data Split
| Split | Months | Samples |
|---|---|---|
| Train | 1–6 (Jan–Jun) | 20,187 |
| Val | 7–8 (Jul–Aug) | 6,513 |
| Test | 9–10 (Sep–Oct) | 7,341 |

Early stopping triggered at **epoch 25**.

## 6. Test Set Metrics
| Metric | Value |
|---|---|
| MAE | 107.01 |
| RMSE | 1200.44 |
| R² | 0.052 |

## 7. Per-Discount-Level Performance
| Discount % | N | MAE | RMSE |
|---|---|---|---|
| 5 | 510 | 58.3 | 172.5 |
| 10 | 824 | 172.5 | 2333.7 |
| 15 | 1347 | 115.6 | 609.7 |
| 20 | 1625 | 88.0 | 552.9 |
| 25 | 898 | 101.8 | 757.1 |
| 30 | 773 | 161.6 | 2391.7 |
| 35 | 685 | 67.2 | 267.7 |
| 40 | 679 | 77.5 | 370.1 |

## 8. Error Analysis & Known Limitations

> **Low R² (0.052) is expected and honest, not a bug.**  
> Key reasons:
> 1. **Extreme outlier targets** — target std is 2,104.95% (max 245,100%). A predictor that captures mean behaviour will still have poor R² against fat-tailed targets.
> 2. **Missing modalities** — Image (32-D) and Sentiment (10-D) are absent. The paper's full 121-D state would provide substantially richer context.
> 3. **Single-observation products** — 19,516 ASINs appear in only one month, limiting temporal signal for sales response learning.
> 4. **No cross-product generalisation signal** — the dataset has 22,514 unique ASINs but only 34,041 observations (avg 1.5 obs/product).

## 9. Counterfactual Sanity Check
- **10 products** checked against all 9 discount bins (0–40%).
- Result: **PASSED** — no NaN, no constant predictions, no extreme extrapolation.
- Curves saved to: `reports/sales_response_curves/sample_response_curves.png`

## 10. Model Checkpoint
| File | Description |
|---|---|
| `models/sales_predictor/best_model.pth` | Trained weights (best val loss) |
| `models/sales_predictor/config.json` | Architecture & training config |
| `models/sales_predictor/metrics.json` | Test metrics |
| `src/models/sales_predictor.py` | Importable module for Phase 6 |

## 11. Model Freeze
After training, `requires_grad=False` was set on all parameters.  
The PPO environment (Phase 6) must use `src.models.sales_predictor.load_frozen_predictor()` exclusively.

## 12. Recommendation for Phase 6
- Apply **log1p transformation** to the target before training to reduce the impact of extreme outliers.  
  This is an engineering improvement, not a paper deviation — it should be documented as an IMPLEMENTATION DECISION.
- Consider a **per-product normalisation** strategy if product-level sales scale differs by orders of magnitude.
