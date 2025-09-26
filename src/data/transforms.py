from __future__ import annotations

from typing import Callable

import torch
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(stage: str, image_size: int = 224) -> Callable:
    """Return torchvision transform pipeline for the requested stage."""

    stage = stage.lower()
    base_ops = [transforms.Resize((image_size, image_size))]

    if stage == "train":
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.7,
            ),
            transforms.RandomRotation(degrees=15),
        ]
        base_ops = augmentation + base_ops
    elif stage not in {"val", "valid", "validation", "test"}:
        raise ValueError(f"Unsupported stage '{stage}'.")

    base_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(base_ops)


def build_landmark_transform(image_size: int = 224) -> Callable[[torch.Tensor], torch.Tensor]:
    """Normalise (x, y) landmark coordinates into [-1, 1]."""

    scale = torch.tensor([image_size, image_size], dtype=torch.float32)

    def _transform(landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.dim() != 2 or landmarks.size(-1) != 2:
            raise ValueError("Expected landmark tensor with shape (N, 2).")
        return (landmarks / scale.to(device=landmarks.device)) * 2.0 - 1.0

    return _transform


def denormalise_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)
    return tensor * std[:, None, None] + mean[:, None, None]
