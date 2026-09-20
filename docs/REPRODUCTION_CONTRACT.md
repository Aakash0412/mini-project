# REPRODUCTION CONTRACT

## A. MUST MATCH PAPER
The following components must exactly match the conceptual architecture and reported logic of the base paper:
- **ItaNet Concept:** The Image + Text + Attributes Fusion Network.
- **121-D State:** The state representation must be exactly 121 dimensions (Image: 32, Title: 32, Reviews: 10, Attributes: 47).
- **Reward Formulation:** `r_t = alpha * Delta M_t + beta * Delta Q_t` where `alpha + beta = 1`.
- **0–40% Discount Range:** The action space consists of 9 discrete discount tiers (0, 5, 10, 15, 20, 25, 30, 35, 40%).
- **Multimodal Inputs:** The model must process image, text (title), review (sentiment), and numerical/categorical attributes.
- **Sales Predictor:** Must act as a frozen transition engine trained before DRL policy optimization.
- **Primary Algorithm:** Proximal Policy Optimization (PPO).
- **Evaluation Logic:** Must evaluate Cumulative reward, Profit-margin growth, Sales growth, etc., matching the paper's metrics.

## B. ENGINEERING ADAPTATIONS
The following are deliberate technical adaptations made to satisfy the local, API-independent, and hardware-constrained implementation requirements:
- **Product Title Feature Extraction:** Replaced OpenAI's `text-embedding-ada-002` API with a local sentence-transformer model (`all-MiniLM-L6-v2`), projected to 32 dimensions.
- **Customer Review Sentiment:** Replaced GPT-4o with a local RoBERTa sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`), projecting or aggregating to the 10-dimensional representation.
- **Hardware Target:** Designed for Apple Silicon using `mps` (Metal Performance Shaders) rather than NVIDIA CUDA.
- **Memory Configuration:** Safe, low batch sizes configured as default rather than the paper's potentially larger batch sizes to accommodate the 8 GB unified memory limit.
- **Local Model Caching:** Models and embeddings will be cached locally without needing continuous API inference.

## C. NOT IN SCOPE
The following items are explicitly excluded from this implementation to prevent scope creep and maintain the focus on accurate reproduction:
- **Research Novelty:** No new DRL algorithms, reward functions, or fusion architectures will be introduced.
- **Cloud Deployment / Scalability:** Real-time production optimization, commercial deployment, or heavy cloud-based data ingestion is out of scope.
- **Real-Time LLM Calls:** No external API dependency (OpenAI, ChatGPT, etc.).
- **Extensive Hyperparameter Search:** No large-scale HPO beyond the reported baseline configurations and memory-safe overrides.
