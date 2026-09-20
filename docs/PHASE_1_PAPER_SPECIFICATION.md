# PHASE 1 — PAPER SPECIFICATION
**Source Paper:** "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce" (Liu et al., IEEE Access 2026, DOI: 10.1109/ACCESS.2026.3680140)

This document is the implementation source of truth. It details the exact methodology extracted from the base paper.

## 1. Dataset Specification
- **Data Source:** Amazon marketplace
- **Collection Period:** January–October 2023 (November and December excluded due to major seasonal effects).
- **Product Count:** 34,041 distinct discounted product samples.
- **Category Count:** 27 Amazon product categories.
- **Collection Process:** Monthly collection of "Today's Deal" product listings at the beginning of the month, followed by re-crawling at the end of the month to track same ASIN changes (e.g., Best Sellers Rank (BSR)).
- **Data Points:** Product identifiers, discount information, price information, review information, images, titles, BSR, sales estimation, gross margin estimation, historical information.
- **AMBIGUITY — REQUIRES IMPLEMENTATION DECISION:** The exact train/validation/test split mechanism is not explicitly specified in the paper summary. Will need to decide on an appropriate chronological or random split that respects temporal boundaries.

## 2. Data Collection Pipeline
- Promotion/discount bins documented in the paper are exactly:
  **0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%**.

## 3. Multimodal Feature Specification
The state represents a combination of numerical/categorical data and embeddings from multiple modalities.

### A. Image Modality
- **Original Paper Model:** ResNet-50
- **Implementation:** ResNet-50 (Local, PyTorch/torchvision)
- **Output Dimensionality:** 32 dimensions.

### B. Product Title Modality
- **Original Paper Model:** OpenAI `text-embedding-ada-002` (offline API processing).
- **Implementation (ENGINEERING ADAPTATION):** Local sentence-transformer `all-MiniLM-L6-v2`. We will project the embeddings to the required 32 dimensions.
- **Output Dimensionality:** 32 dimensions.

### C. Customer Reviews Modality
- **Original Paper Model:** GPT-4o (sentiment extraction, top-10 review usage, sentiment normalization).
- **Implementation (ENGINEERING ADAPTATION):** Local RoBERTa sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Output Dimensionality:** 10 dimensions.

## 4. 121-D State Specification
The static/dynamic state formulation is: `s_t,i = [c_i || d_t,i]`.

| Component | Original model | Local model | Output dims |
|-----------|----------------|-------------|-------------|
| Image | ResNet-50 | ResNet-50 | 32 |
| Title | ada-002 | MiniLM | 32 |
| Reviews | GPT-4o | RoBERTa sentiment | 10 |
| Attributes | tabular | tabular | 47 |
| TOTAL | — | — | 121 |

The total state dimension is explicitly **121**.

## 5. Feature Fusion
- ItaNet is the proposed Image + Text + Attributes Fusion Network.
- **AMBIGUITY — REQUIRES IMPLEMENTATION DECISION:** The exact mechanism of concatenating or projecting before/after the individual feature extraction (e.g., whether there are learnable fusion layers like MLPs) is not fully described. Standard concatenation `[Image, Title, Reviews, Attributes]` matching the 121-D output will be assumed unless further paper sections dictate otherwise. We will **NOT** introduce novel cross-attention or multimodal transformers unless explicitly stated.

## 6. Sales Predictor
- **Input:** State/context + action.
- **Output:** Predicted monthly sales growth.
- **Purpose:** Acts as a counterfactual simulation / transition engine.
- **Principle:** The predictor must be trained first and then frozen prior to DRL policy optimization to prevent information leakage.
- **AMBIGUITY — REQUIRES IMPLEMENTATION DECISION:** The exact neural architecture (layers, hidden sizes) of the sales predictor is missing. Will require a standard MLP default.

## 7. Action Space
- **Formulation:** Operational discrete discount tiers.
- **Range:** 0% to 40%.
- **Bins:** 0, 5, 10, 15, 20, 25, 30, 35, 40%.
- **AMBIGUITY — REQUIRES IMPLEMENTATION DECISION:** The paper mentions continuous-action algorithms, yet states that the framework adopts a discrete action space for robustness and operational feasibility. Our default implementation direction is **DISCRETE DISCOUNT ACTIONS**.

## 8. Reward Function
- **Formula:** `r_t = alpha * Delta M_t + beta * Delta Q_t`
  - `Delta M`: profit-margin growth
  - `Delta Q`: sales growth
  - Constraint: `alpha + beta = 1`
- **Representative Settings:**
  - Sales-oriented: `alpha = 0.2, beta = 0.8`
  - Balanced: `alpha = 0.5, beta = 0.5`
  - Profit-oriented: `alpha = 0.8, beta = 0.2`

## 9. DRL Algorithms
- **Primary Algorithm:** Proximal Policy Optimization (PPO).
- **Secondary Algorithms (Evaluated but NOT primary implementation):** DDPG, SAC, DDQN.

## 10. PPO Specification
- **State Dimension:** 121
- **Action Representation:** Discrete (9 bins)
- **Gamma:** 0.9
- **Batch Size:** 64
- **Training Episodes:** 500
- **Reward Aggregation (Sample Proportion):** 0.5 (evaluated 0.25, 0.5, 1.0)
- **Convergence Criterion:** Based on change in cumulative reward with `epsilon = 1e-3`.
- **AMBIGUITY — REQUIRES IMPLEMENTATION DECISION:** Values for learning rate, GAE lambda, clipping parameter, PPO network architecture, and optimizer are not explicitly specified in the prompt summary. These will be marked as "NOT SPECIFIED IN PAPER" and require hardware-safe defaults.

## 11. Evaluation Specification
Reported metrics to reproduce:
- Cumulative reward
- Profit-margin growth
- Sales growth
- Cumulative reward curves
- Product-level performance
- Top-1 / Top-3 comparisons (where applicable)
- Ablation metrics
- Robustness metrics

## 12. Baselines
Baseline families mentioned for future evaluation (NOT to be implemented in Phase 0/1):
1. DRL baselines (DDPG, SAC, DDQN)
2. Heuristic/strategy baselines
3. Bandit approaches
4. Transformer-based approaches
5. Other model baselines

## 13. Ablation Specification
Evaluates progressive modality additions:
1. Attributes
2. Attributes + Text
3. Attributes + Text + Image
4. Attributes + Text + Image + Sentiment

## 14. Robustness Specification
Experiment adding noise to predicted sales growth:
- Noise type, sigma levels, evaluation metric, comparison methodology (Not to be implemented yet).

## 15. Transfer Learning (OPTIONAL ADVANCED REPRODUCTION STAGE)
Methodology involving the NeoRL Sales Promotion benchmark, source state, target state, pseudo-state adapter, policy distillation, and fine-tuning.

## 16. Known Ambiguities / Reproduction Risks
1. **Discrete vs Continuous Action Spaces:** The paper describes discrete bins but also continuous DRL methods. *Decision:* Use discrete 9-bin action space as explicitly motivated by operational feasibility limitations.
2. **Train/Validation/Test Split:** Exact splitting logic over the Jan-Oct 2023 dataset is unspecified.
3. **Sales Predictor Architecture:** Layers, activation functions, and optimizer for the sales predictor are missing.
4. **Text Embedding Dimensionality Projection:** ADA-002 outputs 1536 dims, the state requires 32. The exact projection mechanism (e.g., linear layer or PCA) is not specified.
5. **Sentiment Aggregation Mechanism:** GPT-4o extracts sentiment from top-10 reviews. The method of pooling this into a 10-dimensional vector is unspecified.
6. **Numerical/Categorical Features:** Exact list of the 47 tabular features is missing.
7. **PPO Network Architecture and Hyperparameters:** Missing exact values for LR, clip ratio, GAE lambda, and MLP layer sizes.
8. **Random Seeds:** Unspecified.
