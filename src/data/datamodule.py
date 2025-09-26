from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader

from src.config import DatasetConfig
from src.models import ModelConfig

from . import FaceDataset, build_image_transform, build_landmark_transform, build_samples, create_splits


def create_dataloaders(
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
) -> Tuple[Dict[str, FaceDataset], Dict[str, DataLoader]]:
    dataset_root = dataset_cfg.root.expanduser().resolve()

    cache_root = dataset_cfg.cache_dir
    if not cache_root.is_absolute():
        cache_root = dataset_root / cache_root

    cache_root.mkdir(parents=True, exist_ok=True)

    samples = build_samples(dataset_root, cache_file=cache_root / "samples.json")
    splits = create_splits(
        samples,
        val_ratio=dataset_cfg.val_ratio,
        test_ratio=dataset_cfg.test_ratio,
        seed=dataset_cfg.seed,
        cache_file=cache_root / f"splits_seed{dataset_cfg.seed}.json",
    )

    include_landmarks = dataset_cfg.include_landmarks and model_cfg.use_landmarks
    landmark_transform = build_landmark_transform(dataset_cfg.image_size) if include_landmarks else None

    datasets = {
        "train": FaceDataset(
            splits.train,
            transform=build_image_transform("train", dataset_cfg.image_size),
            landmark_transform=landmark_transform,
            include_landmarks=include_landmarks,
        ),
        "val": FaceDataset(
            splits.val,
            transform=build_image_transform("val", dataset_cfg.image_size),
            landmark_transform=landmark_transform,
            include_landmarks=include_landmarks,
        ),
        "test": FaceDataset(
            splits.test,
            transform=build_image_transform("test", dataset_cfg.image_size),
            landmark_transform=landmark_transform,
            include_landmarks=include_landmarks,
        ),
    }

    dataloaders = {
        name: DataLoader(
            ds,
            batch_size=dataset_cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=dataset_cfg.num_workers,
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=dataset_cfg.persistent_workers,
            drop_last=(name == "train"),
        )
        for name, ds in datasets.items()
        if len(ds) > 0
    }

    return datasets, dataloaders
