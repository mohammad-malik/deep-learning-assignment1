from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from src.models import ModelConfig


@dataclass
class DatasetConfig:
    root: Path = Path("Dataset")
    cache_dir: Path = Path("metadata")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    include_landmarks: bool = True
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class OptimizationConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    t_max: int = 10
    eta_min: float = 1e-6


@dataclass
class TrainerConfig:
    epochs: int = 20
    grad_clip: float | None = 1.0
    mixed_precision: bool = False
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "classification": 1.0,
            "valence": 0.5,
            "arousal": 0.5,
        }
    )
    checkpoint_dir: Path = Path("checkpoints")
    save_best_only: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    trainer: TrainerConfig = TrainerConfig()
