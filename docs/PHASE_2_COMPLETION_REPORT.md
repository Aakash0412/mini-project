# Phase 2: Completion Report

## 1. Dataset Source
https://github.com/Larry-Liu02/Dynamic-Pricing-Discount (Downloaded `products.csv` raw).

## 2. File Size
~15 MB raw CSV file.

## 3. Row Count
34,041 total observations.

## 4. Column Count
38 columns.

## 5. Product Count
22,514 unique ASINs.

## 6. Category Count
27 categories identified via `Main Category BSR Category`.

## 7. Date Range
January 2023 - October 2023.

## 8. Missingness
Most columns have 0% missingness. Variables like `Weight`, `Volume`, and `BSR Change` have up to 20% missingness.

## 9. Paper Feature Availability
- ASIN, Title, Category: Yes.
- Ratings & BSR: Yes.
- Sales & Margins (`Last Monthly Sales`, `Gross Margin Rate`): Yes.
- Sales Growth (`Estimated Monthly Sales Growth Rate`): Yes (can act as target).

## 10. Missing Modalities
- **Images:** Missing entirely from the repository. Only `Main Image URL` is provided. Cannot extract ResNet-50 features locally without downloading 22K+ images via scraping.
- **Customer Reviews:** Missing text entirely. The dataset only provides aggregate review/rating counts. Cannot extract sentiment features.

## 11. Selected Preprocessing
- Stripped `%` and `,` from numeric/percentage columns (e.g., `Buybox Price`, `Gross Margin Rate`, `Estimated Monthly Sales Growth Rate`).
- Removed rows where `Buybox Price (USD)` <= 0.
- Assured `Discount` ranges between 0 and 100.
- Normalized `Deal Month` to integers (1-12).

## 12. Train/Validation/Test Split
- The paper did not specify chronological split boundaries.
- **Implementation Decision (Chronological Split):**
  - Train: Months 1-6 (Jan-Jun)
  - Validation: Months 7-8 (Jul-Aug)
  - Test: Months 9-10 (Sep-Oct)

## 13. Leakage Checks
Chronological split prevents validation/test periods from appearing in the training period. Verified via `tests/test_phase_2.py`.

## 14. Tests
All tests passed for data availability, missing bounds, schema matching, and proper chronological alignment without leakage.

## 15. Known Limitations
Because Images and Customer Reviews are unavailable, the 121-D ItaNet representation cannot be fully constructed. Phase 3 and Phase 4 will only incorporate Title text and structured attributes.
