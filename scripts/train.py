from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.models import create_model
from src.training import Trainer
from src.utils import (
    get_default_device,
    get_logger,
    load_json,
    load_yaml,
    save_json,
    seed_everything,
    update_dataclass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train facial affect model.")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to dataset root.")
    parser.add_argument("--backbone", type=str, default=None, help="CNN backbone name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store outputs.")
    parser.add_argument("--no-landmarks", action="store_true", help="Disable landmark branch.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet weights.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP training.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def load_config(path: Path | None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config

    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = load_yaml(path)
    elif path.suffix.lower() == ".json":
        payload = load_json(path)
    else:
        raise ValueError("Unsupported config file format.")

    if not isinstance(payload, dict):
        raise ValueError("Config file must define a mapping at the root level.")

    update_dataclass(config, payload)
    return config


def build_scheduler(config, optimizer):
    name = config.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.t_max,
            gamma=0.1,
        )
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
        )
    raise ValueError(f"Unknown scheduler '{config.name}'.")


def to_serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.learning_rate is not None:
        config.optimization.lr = args.learning_rate
    if args.backbone is not None:
        config.model.backbone = args.backbone
    if args.freeze_backbone:
        config.model.freeze_backbone = True
    if args.no_landmarks:
        config.model.use_landmarks = False
        config.dataset.include_landmarks = False
    if args.no_pretrained:
        config.model.pretrained = False
    if args.mixed_precision:
        config.trainer.mixed_precision = True
    if args.seed is not None:
        config.dataset.seed = args.seed

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / f"{config.model.backbone}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config.trainer.checkpoint_dir = output_dir / "checkpoints"

    logger = get_logger("train")
    logger.info("Using output directory: %s", output_dir)

    seed_everything(config.dataset.seed)

    datasets, dataloaders = create_dataloaders(config.dataset, config.model)

    device = get_default_device()
    logger.info("Training on device: %s", device)

    model = create_model(config.model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimization.lr,
        betas=config.optimization.betas,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = build_scheduler(config.scheduler, optimizer)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weights=config.trainer.loss_weights,
        checkpoint_dir=config.trainer.checkpoint_dir,
        mixed_precision=config.trainer.mixed_precision,
        grad_clip=config.trainer.grad_clip,
        save_best_only=config.trainer.save_best_only,
    )

    history = trainer.fit(
        dataloaders["train"],
        dataloaders.get("val"),
        epochs=config.trainer.epochs,
    )

    best_checkpoint = trainer.best_checkpoint_path
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)

    eval_results: Dict[str, Dict[str, Any]] = {}
    for split_name in ["val", "test"]:
        loader = dataloaders.get(split_name)
        if loader is None:
            continue
        metrics = trainer.evaluate(loader)
        eval_results[split_name] = {
            "loss": metrics["total_loss"],
            "classification": metrics["classification"].to_dict(),
            "valence": metrics["valence"].to_dict(),
            "arousal": metrics["arousal"].to_dict(),
        }
        logger.info(
            "%s - loss: %.4f accuracy: %.4f valence CCC: %.4f arousal CCC: %.4f",
            split_name,
            metrics["total_loss"],
            metrics["classification"].accuracy,
            metrics["valence"].ccc,
            metrics["arousal"].ccc,
        )

    # Persist outputs
    save_json(history, output_dir / "history.json")
    save_json(eval_results, output_dir / "metrics.json")

    config_payload = {
        "dataset": to_serialisable(asdict(config.dataset)),
        "model": to_serialisable(asdict(config.model)),
        "optimization": to_serialisable(asdict(config.optimization)),
        "scheduler": to_serialisable(asdict(config.scheduler)),
        "trainer": to_serialisable(asdict(config.trainer)),
    }
    save_json(config_payload, output_dir / "config_used.json")

    logger.info("Training complete. Artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
