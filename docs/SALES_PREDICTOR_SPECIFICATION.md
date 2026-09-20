# Sales Predictor Specification

## Purpose
The Sales Predictor serves as the environment simulator for the DRL agent. It predicts the sales response given a product's state context and a candidate discount action.

## Target Definition
- **Target Variable:** `Estimated Monthly Sales Growth Rate`
- **Derivation:** Provided natively in the dataset. Values are preprocessed to float percentages.
- **Validity:** This represents the exact concept (monthly sales growth $\Delta Q_t$) required for the paper's reward function.

## Predictor Input
- **Features:** 79-D ItaNet State + 1-D Action (Candidate Discount) = 80-D Input.
- **Data Leakage Prevention:** The input strictly relies on the state constructed from the given month. Future metrics are NOT included.
- **Action Range:** [0, 40] representing discount percentages.

## Data Splits
- **Training Period:** Months 1-6
- **Validation Period:** Months 7-8
- **Test Period:** Months 9-10

## Model Architecture
- **IMPLEMENTATION DECISION:** The paper states a predictor is used but omits the exact layer sizes. We implement a lightweight Multi-Layer Perceptron (MLP) for feasibility and memory safety on the target hardware.
- **Architecture:** 80-D Input $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 1.
- **Activation:** ReLU
- **Loss:** MSELoss
- **Optimizer:** Adam (LR=1e-3)
- **Early Stopping:** Monitored on validation loss.

## Usage in RL
Once trained and validated, this model is **FROZEN**. During PPO training, the RL agent queries this model with various candidate actions (0, 5, ..., 40) to estimate the resultant sales growth and compute the reward. The Predictor's weights are not updated during RL.
