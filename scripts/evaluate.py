from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
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
    update_dataclass(config, payload)
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size

    logger = get_logger("evaluate")
    seed_everything(config.dataset.seed)

    datasets, dataloaders = create_dataloaders(config.dataset, config.model)
    loader = dataloaders.get(args.split)
    if loader is None:
        raise ValueError(f"Split '{args.split}' is empty or unavailable.")

    device = get_default_device()
    model = create_model(config.model)

    optimizer = optim.AdamW(model.parameters(), lr=config.optimization.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        loss_weights=config.trainer.loss_weights,
        checkpoint_dir=config.trainer.checkpoint_dir,
        mixed_precision=False,
    )

    checkpoint = trainer.load_checkpoint(args.checkpoint)
    logger.info("Loaded checkpoint from %s (epoch %s)", args.checkpoint, checkpoint.get("epoch"))

    metrics = trainer.evaluate(loader)
    logger.info(
        "%s - loss: %.4f accuracy: %.4f valence CCC: %.4f arousal CCC: %.4f",
        args.split,
        metrics["total_loss"],
        metrics["classification"].accuracy,
        metrics["valence"].ccc,
        metrics["arousal"].ccc,
    )

    results = {
        "loss": metrics["total_loss"],
        "classification": metrics["classification"].to_dict(),
        "valence": metrics["valence"].to_dict(),
        "arousal": metrics["arousal"].to_dict(),
    }
    save_json(results, args.checkpoint.with_suffix(".evaluation.json"))


if __name__ == "__main__":
    main()
