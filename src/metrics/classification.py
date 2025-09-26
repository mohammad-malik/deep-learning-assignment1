from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    accuracy: float
    f1_macro: float
    cohen_kappa: float
    krippendorff_alpha: float
    auroc_macro: Optional[float]
    aupr_macro: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "cohen_kappa": self.cohen_kappa,
            "krippendorff_alpha": self.krippendorff_alpha,
            "auroc_macro": self.auroc_macro,
            "aupr_macro": self.aupr_macro,
        }


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Optional[np.ndarray] = None,
) -> ClassificationReport:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro")
    cohen = cohen_kappa_score(y_true_arr, y_pred_arr)
    alpha = krippendorff_alpha_nominal(np.vstack([y_true_arr, y_pred_arr]))

    auroc_macro = None
    aupr_macro = None

    if y_prob is not None:
        try:
            auroc_macro = roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auroc_macro = None
        try:
            aupr_macro = average_precision_score(
                _to_one_hot(y_true_arr, y_prob.shape[1]),
                y_prob,
                average="macro",
            )
        except ValueError:
            aupr_macro = None

    return ClassificationReport(
        accuracy=accuracy,
        f1_macro=f1_macro,
        cohen_kappa=cohen,
        krippendorff_alpha=alpha,
        auroc_macro=auroc_macro,
        aupr_macro=aupr_macro,
    )


def krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    """Compute Krippendorff's alpha (nominal) for a 2D array (raters x items)."""

    if ratings.ndim != 2:
        raise ValueError("Expected ratings matrix with shape (raters, items).")

    raters, items = ratings.shape
    if items == 0:
        return float("nan")

    # Observed disagreement
    disagreements = 0
    pair_count = 0
    for item_idx in range(items):
        item_ratings = ratings[:, item_idx]
        for i in range(raters):
            for j in range(i + 1, raters):
                pair_count += 1
                disagreements += item_ratings[i] != item_ratings[j]
    if pair_count == 0:
        return 1.0
    D_o = disagreements / pair_count

    # Expected disagreement based on the distribution of labels
    flattened = ratings.flatten()
    categories, counts = np.unique(flattened, return_counts=True)
    total = counts.sum()
    if total <= 1:
        return 1.0
    numerator = sum(count * (total - count) for count in counts)
    D_e = numerator / (total * (total - 1))

    if D_e == 0:
        return 1.0
    return 1.0 - D_o / D_e


def _to_one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((targets.shape[0], num_classes), dtype=float)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return one_hot
