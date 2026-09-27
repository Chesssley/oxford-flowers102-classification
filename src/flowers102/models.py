"""The course CNN baseline and ImageNet-pretrained ResNet18."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import CLASS_COUNT

MODEL_NAMES = ("simple-cnn", "resnet18")


class SimpleCNN(nn.Module):
    """Small convolutional baseline with input-size-independent pooling."""

    def __init__(self, num_classes: int = CLASS_COUNT) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(name: str, *, pretrained: bool = False) -> nn.Module:
    """Build a 102-output CNN; download weights only when requested for training."""
    if name == "simple-cnn":
        return SimpleCNN()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
        return model
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count total and currently trainable parameters."""
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
