from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import (
    ClassificationReport,
    RegressionReport,
    compute_classification_metrics,
    compute_regression_metrics,
)
from src.utils import get_logger, move_to_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler | optim.lr_scheduler.ReduceLROnPlateau],
        device: torch.device,
        loss_weights: Dict[str, float],
        checkpoint_dir: Path,
        mixed_precision: bool = False,
        grad_clip: Optional[float] = None,
        save_best_only: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_weights = loss_weights
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=mixed_precision)
        self.save_best_only = save_best_only

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

        self.best_val_loss = math.inf
        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"

        self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["total_loss"])
                history["val_accuracy"].append(val_metrics["classification"].accuracy)

                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(epoch, val_metrics, best=True)
                elif not self.save_best_only:
                    self._save_checkpoint(epoch, val_metrics, best=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    value = val_metrics["total_loss"] if val_metrics is not None else train_loss
                    self.scheduler.step(value)
                else:
                    self.scheduler.step()

        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, object]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        cls_targets: List[int] = []
        cls_predictions: List[int] = []
        cls_probs: List[torch.Tensor] = []
        valence_targets: List[float] = []
        valence_preds: List[float] = []
        arousal_targets: List[float] = []
        arousal_preds: List[float] = []

        with torch.inference_mode():
            for batch in loader:
                batch = move_to_device(batch, self.device)
                outputs = self.model(
                    batch["image"],
                    landmarks=batch.get("landmarks"),
                )

                loss, components = self._compute_loss(outputs, batch)
                total_loss += loss.item() * batch["image"].size(0)
                total_samples += batch["image"].size(0)

                logits = outputs["logits"].softmax(dim=1)
                predictions = logits.argmax(dim=1)

                cls_targets.extend(batch["expression"].tolist())
                cls_predictions.extend(predictions.tolist())
                cls_probs.append(logits.cpu())

                valence_targets.extend(batch["valence"].tolist())
                valence_preds.extend(outputs["valence"].detach().cpu().tolist())

                arousal_targets.extend(batch["arousal"].tolist())
                arousal_preds.extend(outputs["arousal"].detach().cpu().tolist())

        cls_prob_tensor = torch.cat(cls_probs, dim=0).numpy() if cls_probs else None

        classification_report: ClassificationReport = compute_classification_metrics(
            cls_targets,
            cls_predictions,
            cls_prob_tensor,
        )
        valence_report: RegressionReport = compute_regression_metrics(
            valence_targets,
            valence_preds,
        )
        arousal_report: RegressionReport = compute_regression_metrics(
            arousal_targets,
            arousal_preds,
        )

        metrics = {
            "total_loss": total_loss / max(total_samples, 1),
            "classification": classification_report,
            "valence": valence_report,
            "arousal": arousal_report,
        }

        self.model.train()
        return metrics

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_to_device(batch, self.device)
            images = batch["image"]
            landmarks = batch.get("landmarks")

            self.optimizer.zero_grad(set_to_none=True)

            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images, landmarks=landmarks)
                    loss, components = self._compute_loss(outputs, batch)
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, landmarks=landmarks)
                loss, components = self._compute_loss(outputs, batch)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            progress.set_postfix({
                "loss": running_loss / total_samples,
                "cls": components["classification"].item(),
                "val": components["valence"].item(),
                "aro": components["arousal"].item(),
            })

        return running_loss / max(total_samples, 1)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        cls_loss = self.cls_loss_fn(outputs["logits"], batch["expression"])
        val_loss = self.reg_loss_fn(outputs["valence"], batch["valence"])
        aro_loss = self.reg_loss_fn(outputs["arousal"], batch["arousal"])

        total = (
            self.loss_weights.get("classification", 1.0) * cls_loss
            + self.loss_weights.get("valence", 1.0) * val_loss
            + self.loss_weights.get("arousal", 1.0) * aro_loss
        )

        return total, {
            "classification": cls_loss.detach(),
            "valence": val_loss.detach(),
            "arousal": aro_loss.detach(),
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, object], best: bool = True) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": {
                "loss": metrics["total_loss"],
                "classification": metrics["classification"].to_dict(),
                "valence": metrics["valence"].to_dict(),
                "arousal": metrics["arousal"].to_dict(),
            },
        }
        path = self.best_checkpoint_path if best else self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
