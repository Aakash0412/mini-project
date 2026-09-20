"""
ItaNet: Image + Text + Attributes Fusion Network
-------------------------------------------------
Paper: "A Multimodal Deep Reinforcement Learning Framework for Dynamic
Pricing Optimisation in E-Commerce" (Liu et al., IEEE Access 2026)

This module implements the ItaNet state representation.

Paper architecture:
    Image  (32-D)  |
    Title  (32-D)  | ---> concat ---> 121-D state s_t,i = [c_i || d_t,i]
    Reviews(10-D)  |
    Attrs  (47-D)  |

IMPLEMENTATION DEVIATION:
    Because the dataset does not contain product images or review text,
    the Image (32-D) and Sentiment (10-D) modalities are unavailable.

    Final state dimension used in this implementation: 79-D
        Text (32) + Attributes (47) = 79

    This deviation is documented in docs/STATE_CONSTRUCTION.md and
    docs/PHASE_4_COMPLETION_REPORT.md.
    No synthetic data has been introduced.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# Paper-specified state dimensions
IMAGE_DIM = 32       # ResNet-50 projection (UNAVAILABLE — no images in dataset)
TEXT_DIM = 32        # Title embedding projection (all-MiniLM-L6-v2 → 32)
SENTIMENT_DIM = 10   # Review sentiment (UNAVAILABLE — no review text in dataset)
ATTRIBUTE_DIM = 47   # Structured numerical/categorical attributes

# Paper-specified full state
PAPER_STATE_DIM = IMAGE_DIM + TEXT_DIM + SENTIMENT_DIM + ATTRIBUTE_DIM  # 121

# This implementation's actual state (missing image + sentiment)
AVAILABLE_STATE_DIM = TEXT_DIM + ATTRIBUTE_DIM  # 79


class ItaNetStateBuilder:
    """
    Constructs the ItaNet multimodal state vector by concatenating
    available modality feature arrays.

    Fusion mechanism: Direct concatenation as implied by the paper.
    No cross-modal attention or novel fusion layers are introduced.

    Paper formulation:
        s_t,i = [c_i || d_t,i]
        where c_i = static context, d_t,i = dynamic signals
    """

    def __init__(
        self,
        text_features: Optional[np.ndarray] = None,
        attribute_features: Optional[np.ndarray] = None,
        image_features: Optional[np.ndarray] = None,
        sentiment_features: Optional[np.ndarray] = None,
    ):
        """
        Args:
            text_features:       (N, 32)  product title embeddings
            attribute_features:  (N, 47)  structured attributes
            image_features:      (N, 32)  ResNet-50 image features  [OPTIONAL — unavailable]
            sentiment_features:  (N, 10)  review sentiment           [OPTIONAL — unavailable]
        """
        self.text_features = text_features
        self.attribute_features = attribute_features
        self.image_features = image_features         # UNAVAILABLE in this dataset
        self.sentiment_features = sentiment_features  # UNAVAILABLE in this dataset

        self._validate_inputs()

    def _validate_inputs(self):
        sizes = []
        if self.text_features is not None:
            assert self.text_features.ndim == 2 and self.text_features.shape[1] == TEXT_DIM, \
                f"text_features must be (N, {TEXT_DIM}), got {self.text_features.shape}"
            sizes.append(len(self.text_features))

        if self.attribute_features is not None:
            assert self.attribute_features.ndim == 2 and self.attribute_features.shape[1] == ATTRIBUTE_DIM, \
                f"attribute_features must be (N, {ATTRIBUTE_DIM}), got {self.attribute_features.shape}"
            sizes.append(len(self.attribute_features))

        if self.image_features is not None:
            assert self.image_features.ndim == 2 and self.image_features.shape[1] == IMAGE_DIM, \
                f"image_features must be (N, {IMAGE_DIM}), got {self.image_features.shape}"
            sizes.append(len(self.image_features))

        if self.sentiment_features is not None:
            assert self.sentiment_features.ndim == 2 and self.sentiment_features.shape[1] == SENTIMENT_DIM, \
                f"sentiment_features must be (N, {SENTIMENT_DIM}), got {self.sentiment_features.shape}"
            sizes.append(len(self.sentiment_features))

        if len(set(sizes)) > 1:
            raise ValueError(f"All feature arrays must have the same N. Got sizes: {sizes}")

    def build(self) -> np.ndarray:
        """
        Concatenate all available modalities in the paper-defined order:
            [Image, Text, Sentiment, Attributes]

        For unavailable modalities, they are simply omitted (NOT zero-padded)
        to preserve representational integrity.

        Returns:
            state: (N, D) numpy array where D = sum of available dims
        """
        parts = []

        # Paper ordering: Image → Text → Sentiment → Attributes
        if self.image_features is not None:
            parts.append(self.image_features)          # 32-D

        if self.text_features is not None:
            parts.append(self.text_features)           # 32-D

        if self.sentiment_features is not None:
            parts.append(self.sentiment_features)      # 10-D

        if self.attribute_features is not None:
            parts.append(self.attribute_features)      # 47-D

        if not parts:
            raise ValueError("No feature modalities provided. Cannot build state.")

        state = np.concatenate(parts, axis=1)

        # Integrity checks
        assert not np.isnan(state).any(), "State contains NaN values"
        assert not np.isinf(state).any(), "State contains Inf values"

        return state

    @property
    def state_dim(self) -> int:
        """Returns the expected state dimension given available modalities."""
        dim = 0
        if self.image_features is not None:
            dim += IMAGE_DIM
        if self.text_features is not None:
            dim += TEXT_DIM
        if self.sentiment_features is not None:
            dim += SENTIMENT_DIM
        if self.attribute_features is not None:
            dim += ATTRIBUTE_DIM
        return dim

    def modality_report(self) -> dict:
        """Returns a report of which modalities are available."""
        return {
            "image":     self.image_features is not None,
            "text":      self.text_features is not None,
            "sentiment": self.sentiment_features is not None,
            "attributes": self.attribute_features is not None,
            "state_dim": self.state_dim,
            "paper_state_dim": PAPER_STATE_DIM,
            "missing_dims": PAPER_STATE_DIM - self.state_dim,
        }
