# Phase 3: Completion Report

## Models Used & Local Substitutions
- **Image:** ResNet-50 was specified, but due to the total absence of image files in the dataset (only URLs were present, and web scraping is prohibited), this modality is completely **UNAVAILABLE**.
- **Product Title:** Extracted using `all-MiniLM-L6-v2` locally (engineering adaptation, no OpenAI API). The native 384-D output is deterministically projected to 32-D using a fixed random projection matrix.
- **Customer Reviews:** `cardiffnlp/twitter-roberta-base-sentiment-latest` was specified, but review text is entirely **UNAVAILABLE** in the dataset (only aggregate rating counts). This modality cannot be extracted.
- **Structured Attributes:** 47-D representation extracted using StandardScaler for numerics and OneHotEncoder for categoricals.

## Feature Dimensions
- `attribute_features.npy`: (34041, 47)
- `text_features.npy`: (34041, 32)
- Total valid products processed: 34,041.

## Unavailable Modalities
- Images (Expected 32-D)
- Review Sentiment (Expected 10-D)

## Extraction Hardware & Metrics
- **Device Used:** MPS (Apple Silicon).
- **Batch Size:** 16 for Text.
- **Memory Management:** Models loaded, executed, and cleaned up efficiently to respect 8 GB RAM limits.
- **Feature Caching:** All features cached as NumPy arrays in `data/processed/features/`.

## Validation
`scripts/validate_features.py` confirms shapes, alignment, and lack of NaN/Inf values.
