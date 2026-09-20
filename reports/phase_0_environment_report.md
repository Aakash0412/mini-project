# Phase 0: Environment Report

## Repository Status
- A new, clean repository has been scaffolded at `dynamic-pricing-ecommerce`.
- Target structure includes separate directories for `data`, `models`, `configs`, `scripts`, `src`, `tests`, `docs`, `reports`, `checkpoints`, and `logs`.

## Dependencies & Configuration
- **Python Version:** To be confirmed by script.
- **Virtual Environment:** Set up locally in `venv/`.
- **Dependencies Installed:** `torch`, `torchvision`, `transformers`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `gymnasium`, `stable-baselines3`, `joblib`, `pydantic`, `psutil`.
- **API Dependencies:** Removed. `.env.example` explicitly clarifies no API keys are required.

## Memory-Safe Configuration
- Initial hardware-aware configuration generated at `configs/base.yaml`.
- Includes safe batch sizes (e.g., `image_batch_size: 8`, `text_batch_size: 16`, `sentiment_batch_size: 4`) suitable for an 8 GB Apple Silicon MacBook.
- Targets MPS (Metal Performance Shaders) by default, falling back to CPU.

## Tests
- Added baseline tests for device checking and configuration (`scripts/check_environment.py`).
- Added basic `test_env.py` to assert configurations are valid and models can run. (Will add shortly).
