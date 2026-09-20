# Phase 1: Specification Report

## Base Paper Identified
- "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce" (Liu et al., 2026).

## Specifications Extracted
- **Dataset:** Amazon marketplace "Today's Deal", Jan-Oct 2023, 34,041 samples, 27 categories.
- **Multimodal Architecture:** ItaNet (Image + Text + Attributes Fusion Network).
- **121-D State:** Confirmed composition (Image: 32, Title: 32, Reviews: 10, Attributes: 47).
- **Reward Function:** Confirmed `r_t = alpha * Delta M_t + beta * Delta Q_t`.
- **Action Space:** Discrete (0% to 40% with 5% increments).
- **PPO Specification:** 121 state dim, 9-bin discrete actions, gamma 0.9, batch size 64, 500 episodes, reward sample proportion 0.5.
- **Evaluation Methodology:** Cumulative reward, profit-margin growth, sales growth, ablation, and robustness metrics extracted.

## Local API-Free Substitutions Documented
- Replaced `text-embedding-ada-002` with local `all-MiniLM-L6-v2`.
- Replaced GPT-4o with local `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- These are officially classified as **Engineering Adaptations** and do not constitute research novelties.

## Paper Ambiguities Identified
Documented missing details regarding:
- PPO network architecture and specific learning rates.
- Internal architecture of the sales predictor.
- Exact train/validation/test chronological splitting.
- Exact mechanism of sentiment aggregation for the 10-dimensional vector.
- Discrete implementation choice vs continuous methodology description.
