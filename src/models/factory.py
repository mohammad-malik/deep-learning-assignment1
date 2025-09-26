from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    num_classes: int = 8
    use_landmarks: bool = True
    landmark_dim: int = 136
    landmark_hidden: int = 128
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False


class FacialAffectModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        backbone, feat_dim = _build_backbone(config.backbone, config.pretrained)
        self.backbone = backbone
        self.feature_dim = feat_dim

        total_dim = feat_dim
        if config.use_landmarks:
            self.landmark_encoder = nn.Sequential(
                nn.Linear(config.landmark_dim, config.landmark_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=config.dropout),
                nn.Linear(config.landmark_hidden, config.landmark_hidden),
                nn.ReLU(inplace=True),
            )
            total_dim += config.landmark_hidden
        else:
            self.landmark_encoder = None

        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(total_dim, config.num_classes)
        self.valence_head = nn.Linear(total_dim, 1)
        self.arousal_head = nn.Linear(total_dim, 1)

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, landmarks: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = features.view(features.size(0), -1)

        if self.landmark_encoder is not None:
            if landmarks is None:
                raise ValueError("Landmarks are required but were not provided.")
            landmark_input = landmarks.view(landmarks.size(0), -1)
            landmark_features = self.landmark_encoder(landmark_input)
            features = torch.cat([features, landmark_features], dim=1)

        features = self.dropout(features)

        logits = self.classifier(features)
        valence = self.valence_head(features).squeeze(-1)
        arousal = self.arousal_head(features).squeeze(-1)

        return {
            "logits": logits,
            "valence": valence,
            "arousal": arousal,
        }


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    name = name.lower()

    if name == "resnet18":
        model = _load_resnet18(pretrained)
        modules = list(model.children())[:-1]
        feature_dim = model.fc.in_features
        backbone = nn.Sequential(*modules)
        return backbone, feature_dim

    if name == "efficientnet_b0":
        model = _load_efficientnet_b0(pretrained)
        feature_dim = model.classifier[1].in_features
        backbone = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(start_dim=1),
        )
        return backbone, feature_dim

    if name == "mobilenet_v3_large":
        model = _load_mobilenet_v3_large(pretrained)
        feature_dim = model.classifier[0].in_features
        backbone = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        return backbone, feature_dim

    raise ValueError(f"Unsupported backbone '{name}'.")


def _load_resnet18(pretrained: bool):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:  # Older torchvision
        return models.resnet18(pretrained=pretrained)


def _load_efficientnet_b0(pretrained: bool):
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights)
    except AttributeError:
        return models.efficientnet_b0(pretrained=pretrained)


def _load_mobilenet_v3_large(pretrained: bool):
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v3_large(weights=weights)
    except AttributeError:
        return models.mobilenet_v3_large(pretrained=pretrained)


def create_model(config: ModelConfig) -> FacialAffectModel:
    return FacialAffectModel(config)
