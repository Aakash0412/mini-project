# Multimodal Deep Reinforcement Learning for Dynamic Pricing Optimisation in E-Commerce

## Base Paper
"A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce" (Liu et al., IEEE Access 2026, DOI: 10.1109/ACCESS.2026.3680140)

## Project Objective
The core objective is to reproduce the base paper methodology accurately, preserving its conceptual architecture and experimental logic. This reproduction avoids introducing new research novelty and ensures all code can execute efficiently on an 8 GB Apple Silicon MacBook without dependency on external, paid LLM APIs.

## Current Implementation Status
- **PHASE 0 — COMPLETE:** Project setup, environment audit, memory-safe configuration, hardware testing.
- **PHASE 1 — COMPLETE:** Base paper analysis, strict implementation specification, reproduction contract formulation.
- **PHASE 2 — NOT STARTED:** ML/DRL implementation.

## Hardware Target
- 8 GB Apple Silicon MacBook. Memory efficiency and MPS (Metal Performance Shaders) execution are prioritized.

## API Policy
**No external OpenAI/LLM API required.** This codebase does NOT use API keys.

## Current Local Model Plan
External models used in the paper have been replaced with the following local, equivalent engineering adaptations:
- **Image:** ResNet-50
- **Product Title:** `all-MiniLM-L6-v2` (Replaces `text-embedding-ada-002`)
- **Customer Reviews:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (Replaces GPT-4o)

> **Important:** The system is currently in the initial scaffolding stage (Phase 0/1). The ML pipelines, predictors, and RL agents are not yet implemented.
