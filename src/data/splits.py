from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .types import FaceSample


@dataclass
class DatasetSplits:
    train: List[FaceSample]
    val: List[FaceSample]
    test: List[FaceSample]

    def as_dict(self) -> Dict[str, List[FaceSample]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def create_splits(
    samples: Sequence[FaceSample],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    cache_file: Path | None = None,
) -> DatasetSplits:
    """Create stratified splits for the facial expression dataset."""

    if cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        sample_lookup = {sample.sample_id: sample for sample in samples}
        return DatasetSplits(
            train=_load_subset(cached["train"], sample_lookup),
            val=_load_subset(cached["val"], sample_lookup),
            test=_load_subset(cached["test"], sample_lookup),
        )

    indices = np.arange(len(samples))
    labels = np.array([sample.expression for sample in samples])

    test_ratio = np.clip(test_ratio, 0.0, 1.0)
    val_ratio = np.clip(val_ratio, 0.0, 1.0)
    remaining_ratio = 1.0 - test_ratio

    train_val_idx, test_idx = _split_indices(
        indices,
        labels,
        test_size=test_ratio,
        random_state=seed,
    )

    effective_val_ratio = val_ratio / remaining_ratio if remaining_ratio > 0 else 0.0
    train_idx, val_idx = _split_indices(
        train_val_idx,
        labels[train_val_idx],
        test_size=effective_val_ratio,
        random_state=seed + 1,
    )

    splits = DatasetSplits(
        train=[samples[i] for i in sorted(train_idx)],
        val=[samples[i] for i in sorted(val_idx)],
        test=[samples[i] for i in sorted(test_idx)],
    )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": [sample.sample_id for sample in splits.train],
            "val": [sample.sample_id for sample in splits.val],
            "test": [sample.sample_id for sample in splits.test],
        }
        cache_file.write_text(json.dumps(payload, indent=2))

    return splits


def _split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if test_size <= 0.0 or len(indices) == 0:
        return indices, np.array([], dtype=int)

    try:
        stratify = labels if len(np.unique(labels)) > 1 else None
        left, right = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # Fallback to a random split when stratification fails (e.g. rare classes)
        left, right = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return left, right


def _load_subset(ids: Iterable[str], lookup: Dict[str, FaceSample]) -> List[FaceSample]:
    return [lookup[item_id] for item_id in ids]
