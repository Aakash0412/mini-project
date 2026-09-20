# Phase 4 Completion Report — ItaNet State Construction

## State Formula
Paper: `s_t,i = [c_i || d_t,i]`  
Implemented via: `ItaNetStateBuilder` in [`src/models/itanet.py`](../src/models/itanet.py)

## Modality Dimensions

| Modality | Paper Dim | Status | Note |
|---|---|---|---|
| Image | 32 | **UNAVAILABLE** | No image files in dataset — only URLs |
| Text (Title) | 32 | ✅ Available | all-MiniLM-L6-v2 → random projection |
| Sentiment (Reviews) | 10 | **UNAVAILABLE** | No review text in dataset |
| Attributes | 47 | ✅ Available | Numeric + one-hot structured fields |
| **TOTAL (Paper)** | **121** | — | — |
| **TOTAL (This impl)** | **79** | — | Text + Attrs only |

## Fusion Mechanism
Direct concatenation in paper-defined order: `[Image, Text, Sentiment, Attributes]`.  
No cross-modal attention, CLIP, or novel fusion layers were introduced.

## Static / Dynamic Split
- **Static context** (`c_i`): Title embedding, Category one-hot, Brand-related features
- **Dynamic signals** (`d_t,i`): Month, Price, BSR Ranking, Review counts, Rating score

## Sample Counts

| Split | Months | Samples |
|---|---|---|
| Train | 1–6 (Jan–Jun) | 20,187 |
| Val | 7–8 (Jul–Aug) | 6,513 |
| Test | 9–10 (Sep–Oct) | 7,341 |
| **Total** | | **34,041** |

## State Statistics
- Mean: 0.0256 | Std: 0.5035
- No NaN, No Inf
- Leakage check: **PASSED**

## Validation Results
All 8 checks in `scripts/validate_state.py` passed:
1. ✅ Correct dimension (79)
2. ✅ Row alignment with feature index
3. ✅ No NaN
4. ✅ No Inf
5. ✅ Split sizes sum correctly
6. ✅ No temporal leakage
7. ✅ Text sub-block matches `text_features.npy`
8. ✅ Attribute sub-block matches `attribute_features.npy`

## Implementation Deviations
- **State dim is 79, not 121.** Image (32) and Sentiment (10) modalities are missing from the dataset.  
  No synthetic data was introduced. This is documented in [`docs/STATE_CONSTRUCTION.md`](STATE_CONSTRUCTION.md).

## Files Produced
- `data/processed/state/state.npy` — full state (34041, 79)
- `data/processed/state/state_train.npy` — (20187, 79)
- `data/processed/state/state_val.npy` — (6513, 79)
- `data/processed/state/state_test.npy` — (7341, 79)
- `reports/state_validation.json`
- `reports/state_statistics.md`
- `src/models/itanet.py`
