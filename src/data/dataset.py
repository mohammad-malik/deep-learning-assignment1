from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .types import FaceSample


class FaceDataset(Dataset):
    """PyTorch dataset that returns multi-task labels for facial affect samples."""

    def __init__(
        self,
        samples: Sequence[FaceSample],
        transform: Optional[Callable] = None,
        landmark_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        include_landmarks: bool = True,
        return_metadata: bool = False,
        image_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.landmark_transform = landmark_transform
        self.include_landmarks = include_landmarks
        self.return_metadata = return_metadata
        self.image_dtype = image_dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(dtype=self.image_dtype)

        expression = torch.tensor(sample.expression, dtype=torch.long)
        valence = torch.tensor(sample.valence, dtype=torch.float32)
        arousal = torch.tensor(sample.arousal, dtype=torch.float32)

        example: Dict[str, torch.Tensor] = {
            "image": image,
            "expression": expression,
            "valence": valence,
            "arousal": arousal,
        }

        if self.include_landmarks and sample.landmark_path and sample.landmark_path.exists():
            landmarks = np.load(sample.landmark_path).astype(np.float32)
            if landmarks.ndim == 1:
                # Stored as flat vector (136,); reshape to (68, 2)
                landmarks = landmarks.reshape(-1, 2)
            landmarks_tensor = torch.from_numpy(landmarks)
            if self.landmark_transform is not None:
                landmarks_tensor = self.landmark_transform(landmarks_tensor)
            example["landmarks"] = landmarks_tensor

        if self.return_metadata:
            example["sample_id"] = sample.sample_id

        return example


def build_samples(dataset_root: Path, cache_file: Optional[Path] = None) -> List[FaceSample]:
    """Parse dataset directory into a list of :class:`FaceSample` objects."""

    dataset_root = dataset_root.expanduser().resolve()
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if cache_file and cache_file.exists():
        loaded = json.loads(cache_file.read_text())
        return [
            FaceSample(
                sample_id=entry["sample_id"],
                image_path=Path(entry["image_path"]),
                expression=int(entry["expression"]),
                valence=float(entry["valence"]),
                arousal=float(entry["arousal"]),
                landmark_path=Path(entry["landmark_path"]) if entry["landmark_path"] else None,
                face_bbox=tuple(entry["face_bbox"]) if entry["face_bbox"] else None,
            )
            for entry in loaded
        ]

    samples: List[FaceSample] = []
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for image_path in image_paths:
        sample_id = image_path.stem
        expression_path = annotations_dir / f"{sample_id}_exp.npy"
        valence_path = annotations_dir / f"{sample_id}_val.npy"
        arousal_path = annotations_dir / f"{sample_id}_aro.npy"
        landmark_path = annotations_dir / f"{sample_id}_lnd.npy"

        expression = int(np.load(expression_path).astype(np.int64).item())
        valence = float(np.load(valence_path).astype(np.float32).item())
        arousal = float(np.load(arousal_path).astype(np.float32).item())

        samples.append(
            FaceSample(
                sample_id=sample_id,
                image_path=image_path,
                expression=expression,
                valence=valence,
                arousal=arousal,
                landmark_path=landmark_path if landmark_path.exists() else None,
            )
        )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = [sample.to_dict() for sample in samples]
        cache_file.write_text(json.dumps(serializable))

    return samples
