"""
Image Encoder Module
--------------------
Paper: "A Multimodal Deep Reinforcement Learning Framework for Dynamic Pricing Optimisation in E-Commerce"
       (Liu et al., IEEE Access 2026, DOI: 10.1109/ACCESS.2026.3680140)

RESEARCH SPECIFICATION & IMPLEMENTATION ADAPTATION:
The base paper specifies that product images are processed using ResNet-50 to obtain a 32-D
representation for the ItaNet multimodal state. Standard ResNet-50 produces a 2048-D penultimate
global-average-pooled visual representation. The paper does not specify how this 2048-D vector is
reduced to 32-D (e.g., fine-tuned bottleneck, linear projection, or unsupervised mapping).

IMPLEMENTATION ADAPTATION (Documented):
To maintain complete reproducibility without fabricating arbitrary trained weights or violating
the frozen-encoder assumption, this module extracts the 2048-D penultimate features from a frozen
torchvision pretrained ResNet-50 and projects them to 32 dimensions via a deterministic Gaussian
projection matrix:
    P in R^(2048 x 32)
    P_ij ~ N(0, 1/32)
constructed with fixed random seed 42.

This preserves the Euclidean geometric distances of the visual embedding space according to the
Johnson-Lindenstrauss lemma and mirrors the deterministic projection approach adopted in Phase 3
for text embeddings.
"""

import math
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ImageEncoder(nn.Module):
    """
    Extracts 2048-D penultimate visual features using pretrained ResNet-50
    and deterministically projects them to 32-D visual features for ItaNet.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu", seed: int = 42):
        super().__init__()
        self.device = torch.device(device)
        self.seed = seed

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # Replace classification head with Identity to extract 2048-D penultimate features
        self.backbone.fc = nn.Identity()

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.backbone.to(self.device)

        # Construct deterministic Gaussian projection matrix: P in R^(2048 x 32), P_ij ~ N(0, 1/32)
        # Using fixed seed for exact reproducibility across runs and platforms
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        std = 1.0 / math.sqrt(32.0)
        proj = torch.randn(2048, 32, generator=gen, dtype=torch.float32) * std
        self.register_buffer("projection_matrix", proj.to(self.device))

        # Standard ImageNet preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, 3, 224, 224) normalized input image batch
        Returns:
            features_2048: (B, 2048) penultimate features
            features_32:   (B, 32) projected features
        """
        x = x.to(self.device)
        feat_2048 = self.backbone(x)  # (B, 2048)
        feat_32 = torch.matmul(feat_2048, self.projection_matrix)  # (B, 32)
        return feat_2048, feat_32

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Ensure 3-channel RGB and apply ImageNet transform."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    @torch.inference_mode()
    def encode_pil_images(
        self, images: List[Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a list of PIL Images to numpy feature arrays.
        Returns:
            (features_2048, features_32) as float32 numpy arrays.
        """
        if not images:
            return np.empty((0, 2048), dtype=np.float32), np.empty((0, 32), dtype=np.float32)

        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.stack(tensors, dim=0)
        feat_2048, feat_32 = self.forward(batch)
        return feat_2048.cpu().numpy(), feat_32.cpu().numpy()
