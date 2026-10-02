# ItaNet-111 State Statistics

The current implementation uses Image + Text + Attributes. The 10-D review sentiment modality is not included because the required review data is not available in the released dataset.

| Metric | Value |
|---|---|
| implementation | ItaNet-111 |
| num_source_samples | 34041 |
| num_complete_state_samples | 34028 |
| num_incomplete_samples | 13 |
| state_dimension | 111 |
| paper_state_dimension | 121 |
| available_state_dimension | 111 |
| missing_dimensions | 10 |
| modalities_included | ['Image (32)', 'Text (32)', 'Attributes (47)'] |
| modalities_missing | ['Sentiment (10)'] |
| train_samples_complete | 20179 |
| val_samples_complete | 6512 |
| test_samples_complete | 7337 |
| train_samples_source | 20187 |
| val_samples_source | 6513 |
| test_samples_source | 7341 |
| has_nan | False |
| has_inf | False |
| leakage_check | PASSED |
| state_mean | 0.10307531254734748 |
| state_std | 1.3428523290679453 |
| state_min | -67.75262645868216 |
| state_max | 129.71403099090617 |
