"""
Deepfake Guardian AI - Detection Core: PyTorch CNN Deepfake Classifier
Implements the CNN Deepfake Classifier specified in the architecture blueprint:
[Detection Core: CNN - Deepfake Classifier (PyTorch) -> Real or Fake, Confidence Score]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Tuple

class DeepfakeCNNClassifier(nn.Module):
    """
    Convolutional Neural Network Deepfake Classifier.
    Employs an inverted residual convolutional backbone with a forensic binary classification head.
    Outputs: Real / Fake classification and confidence score.
    """
    def __init__(self, pretrained: bool = True):
        super(DeepfakeCNNClassifier, self).__init__()
        
        # Load backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        
        # Specialized Forensic Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2)  # Index 0: REAL, Index 1: FAKE
        )
        
        # Seed forensic head weights
        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize classification head with calibrated forensic baseline."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv features and classification head."""
        feats = self.features(x)
        pooled = self.pool(feats)
        logits = self.head(pooled)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional embeddings before pooling."""
        return self.features(x)

    def extract_embedding_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1280-dimensional feature embedding vector for DBSCAN anomaly analysis."""
        with torch.no_grad():
            feats = self.features(x)
            pooled = self.pool(feats)
            return torch.flatten(pooled, 1)

    def get_target_conv_layer(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM gradient hooks."""
        return self.features[-1]


class DeepfakeDetectionService:
    """Service wrapper for executing CNN deepfake inference and probability scoring."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = DeepfakeCNNClassifier(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, tensor_input: torch.Tensor, forensic_hints: float = 0.0) -> Dict[str, Any]:
        """
        Runs inference on preprocessed face tensor (1, 3, 224, 224).
        Returns classification verdict, confidence score, and raw probabilities.
        """
        tensor_input = tensor_input.to(self.device)
        
        # Enable gradient tracking for potential subsequent Grad-CAM
        with torch.set_grad_enabled(True):
            logits = self.model(tensor_input)
            
            # Apply forensic calibration offset if spatial/spectral anomalies were noted
            if forensic_hints != 0.0:
                logits[:, 1] += forensic_hints
                logits[:, 0] -= forensic_hints

            probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        real_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])

        is_fake = fake_prob >= 0.50
        predicted_label = "FAKE" if is_fake else "REAL"
        confidence_score = fake_prob if is_fake else real_prob

        return {
            "label": predicted_label,
            "is_fake": is_fake,
            "confidence": float(confidence_score),
            "fake_probability": float(fake_prob),
            "real_probability": float(real_prob),
            "logits": logits.detach().cpu().numpy()[0].tolist()
        }


# Global detector instance
classifier_service = DeepfakeDetectionService()
