"""
ItaNet: Image + Text + Attributes Fusion Network
-------------------------------------------------
Paper:
"A Multimodal Deep Reinforcement Learning Framework for Dynamic
Pricing Optimisation in E-Commerce"
Liu et al., IEEE Access 2026

This module implements the ItaNet state representation.

Paper architecture:
    Image  (32-D)  |
    Title  (32-D)  | ---> concat ---> 121-D state
    Reviews(10-D)  |
    Attrs  (47-D)  |

Current local implementation:
    Image  (32-D)  |
    Title  (32-D)  | ---> concat ---> 111-D state
    Attrs  (47-D)  |

The 10-D review-sentiment modality is not included because the
publicly released dataset does not provide the review text/database
required to reproduce that component.

Therefore:

    Paper state       = 32 + 32 + 10 + 47 = 121-D
    Current state     = 32 + 32 + 47      = 111-D

This is an API-independent local adaptation of the paper's
multimodal representation. No synthetic sentiment features are
introduced.

Fusion mechanism:
    Direct concatenation in the paper-defined modality order:
        [Image || Text || Sentiment || Attributes]

For the current implementation:
        [Image || Text || Attributes]
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Paper-specified modality dimensions
# ─────────────────────────────────────────────────────────────

IMAGE_DIM = 32
TEXT_DIM = 32
SENTIMENT_DIM = 10
ATTRIBUTE_DIM = 47

# Full state defined by the paper:
# 32 + 32 + 10 + 47 = 121
PAPER_STATE_DIM = (
    IMAGE_DIM
    + TEXT_DIM
    + SENTIMENT_DIM
    + ATTRIBUTE_DIM
)

# Current reproducible local state:
# 32 + 32 + 47 = 111
#
# Sentiment is omitted because the required review text/database
# is not available in the released dataset.
AVAILABLE_STATE_DIM = (
    IMAGE_DIM
    + TEXT_DIM
    + ATTRIBUTE_DIM
)


class ItaNetStateBuilder:
    """
    Constructs the ItaNet state vector by concatenating the
    available modality feature arrays.

    Paper formulation:
        s_t,i = [c_i || d_t,i]

    Fusion:
        Direct concatenation; no cross-modal attention or
        additional fusion layer is introduced.

    Current target configuration:
        Image (32) + Text (32) + Attributes (47) = 111-D
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
            text_features:
                Shape (N, 32). Product-title embeddings.

            attribute_features:
                Shape (N, 47). Structured numerical/categorical
                product attributes.

            image_features:
                Shape (N, 32). ResNet-50 image features projected
                to 32-D.

            sentiment_features:
                Shape (N, 10). Review sentiment representation.
                Not available in the current implementation.
        """
        self.text_features = text_features
        self.attribute_features = attribute_features
        self.image_features = image_features
        self.sentiment_features = sentiment_features

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate feature dimensions and row alignment."""

        sizes = []

        if self.image_features is not None:
            if (
                self.image_features.ndim != 2
                or self.image_features.shape[1] != IMAGE_DIM
            ):
                raise ValueError(
                    f"image_features must be "
                    f"(N, {IMAGE_DIM}), "
                    f"got {self.image_features.shape}"
                )

            sizes.append(len(self.image_features))

        if self.text_features is not None:
            if (
                self.text_features.ndim != 2
                or self.text_features.shape[1] != TEXT_DIM
            ):
                raise ValueError(
                    f"text_features must be "
                    f"(N, {TEXT_DIM}), "
                    f"got {self.text_features.shape}"
                )

            sizes.append(len(self.text_features))

        if self.sentiment_features is not None:
            if (
                self.sentiment_features.ndim != 2
                or self.sentiment_features.shape[1] != SENTIMENT_DIM
            ):
                raise ValueError(
                    f"sentiment_features must be "
                    f"(N, {SENTIMENT_DIM}), "
                    f"got {self.sentiment_features.shape}"
                )

            sizes.append(len(self.sentiment_features))

        if self.attribute_features is not None:
            if (
                self.attribute_features.ndim != 2
                or self.attribute_features.shape[1] != ATTRIBUTE_DIM
            ):
                raise ValueError(
                    f"attribute_features must be "
                    f"(N, {ATTRIBUTE_DIM}), "
                    f"got {self.attribute_features.shape}"
                )

            sizes.append(len(self.attribute_features))

        if len(set(sizes)) > 1:
            raise ValueError(
                f"All feature arrays must have the same N. "
                f"Got sizes: {sizes}"
            )

    def build(self) -> np.ndarray:
        """
        Concatenate available modalities in paper-defined order:

            Image → Text → Sentiment → Attributes

        Current intended configuration:

            Image → Text → Attributes

        Unavailable modalities are omitted rather than zero-padded.

        Returns:
            np.ndarray:
                State matrix of shape (N, D), where D is the sum
                of the included modality dimensions.
        """

        parts = []

        # Paper ordering
        if self.image_features is not None:
            parts.append(self.image_features)

        if self.text_features is not None:
            parts.append(self.text_features)

        if self.sentiment_features is not None:
            parts.append(self.sentiment_features)

        if self.attribute_features is not None:
            parts.append(self.attribute_features)

        if not parts:
            raise ValueError(
                "No feature modalities provided. "
                "Cannot build state."
            )

        state = np.concatenate(parts, axis=1)

        # Integrity checks.
        # Missing image rows are handled by build_state.py before
        # calling build(). Therefore the builder itself expects
        # complete feature rows.
        if np.isnan(state).any():
            raise ValueError(
                "State contains NaN values. "
                "Incomplete feature rows must be filtered before "
                "state construction."
            )

        if np.isinf(state).any():
            raise ValueError(
                "State contains Inf values."
            )

        return state

    @property
    def state_dim(self) -> int:
        """Return the state dimension for the supplied modalities."""

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
        """
        Return a structured report describing the current
        modality configuration.
        """

        return {
            "image": self.image_features is not None,
            "text": self.text_features is not None,
            "sentiment": self.sentiment_features is not None,
            "attributes": self.attribute_features is not None,
            "state_dim": self.state_dim,
            "paper_state_dim": PAPER_STATE_DIM,
            "available_state_dim": AVAILABLE_STATE_DIM,
            "missing_dims": PAPER_STATE_DIM - self.state_dim,
        }