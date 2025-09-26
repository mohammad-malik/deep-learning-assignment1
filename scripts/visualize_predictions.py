from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.data.transforms import denormalise_image
from src.models import create_model
from src.training import Trainer
from src.utils import (
    get_default_device,
    get_logger,
    load_json,
    load_yaml,
    seed_everything,
    update_dataclass,
)

EXPRESSION_LABELS = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise qualitative predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--config", type=Path, default=None, help="Config file.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--num-images", type=int, default=12, help="Number of images to visualise.")
    parser.add_argument("--output", type=Path, default=Path("qualitative.png"), help="Output figure path.")
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


def gather_examples(
    dataloader: DataLoader,
    trainer: Trainer,
    max_total: int,
) -> Tuple[List[dict], List[dict]]:
    correct_examples: List[dict] = []
    incorrect_examples: List[dict] = []

    model = trainer.model
    device = trainer.device
    model.eval()

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device)
            landmarks = batch.get("landmarks")
            if landmarks is not None:
                landmarks = landmarks.to(device)

            outputs = model(images, landmarks=landmarks)
            probs = torch.softmax(outputs["logits"], dim=1)
            preds = probs.argmax(dim=1)

            for i in range(images.size(0)):
                sample = {
                    "image": images[i].cpu(),
                    "true": int(batch["expression"][i].item()),
                    "pred": int(preds[i].item()),
                    "prob": float(probs[i, preds[i]].item()),
                    "val_true": float(batch["valence"][i].item()),
                    "val_pred": float(outputs["valence"][i].item()),
                    "aro_true": float(batch["arousal"][i].item()),
                    "aro_pred": float(outputs["arousal"][i].item()),
                }
                if sample["true"] == sample["pred"]:
                    correct_examples.append(sample)
                else:
                    incorrect_examples.append(sample)

                if len(correct_examples) + len(incorrect_examples) >= max_total:
                    return correct_examples, incorrect_examples

    return correct_examples, incorrect_examples


def plot_examples(correct: List[dict], incorrect: List[dict], output: Path) -> None:
    total = len(correct) + len(incorrect)
    cols = 4
    rows = max(1, (total + cols - 1) // cols)

    plt.figure(figsize=(cols * 4, rows * 4))
    idx = 1
    for group, title in [(correct, "Correct"), (incorrect, "Incorrect")]:
        for sample in group:
            plt.subplot(rows, cols, idx)
            idx += 1
            image = denormalise_image(sample["image"]).permute(1, 2, 0).numpy()
            image = image.clip(0, 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(
                f"{title}\nT:{EXPRESSION_LABELS[sample['true']]} P:{EXPRESSION_LABELS[sample['pred']]}\n"
                f"V {sample['val_true']:.2f}/{sample['val_pred']:.2f} A {sample['aro_true']:.2f}/{sample['aro_pred']:.2f}"
            )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root

    logger = get_logger("viz")
    seed_everything(config.dataset.seed)

    config.dataset.batch_size = min(config.dataset.batch_size, args.num_images)

    datasets, dataloaders = create_dataloaders(config.dataset, config.model)
    loader = dataloaders.get(args.split)
    if loader is None:
        raise ValueError(f"Split '{args.split}' has no samples.")

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
    trainer.load_checkpoint(args.checkpoint)

    subset_loader = DataLoader(
        loader.dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    correct, incorrect = gather_examples(subset_loader, trainer, args.num_images)
    if not correct and not incorrect:
        raise RuntimeError("Failed to collect any samples for visualisation.")

    plot_examples(correct, incorrect, args.output)
    logger.info("Saved qualitative figure to %s", args.output)


if __name__ == "__main__":
    main()
