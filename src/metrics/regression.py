from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class RegressionReport:
    rmse: float
    pearson: float
    sagr: float
    ccc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "pearson": self.pearson,
            "sagr": self.sagr,
            "ccc": self.ccc,
        }


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> RegressionReport:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)

    rmse = float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))
    pearson = _pearsonr(y_true_arr, y_pred_arr)
    sagr = float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr)))
    ccc = _concordance_cc(y_true_arr, y_pred_arr)

    return RegressionReport(rmse=rmse, pearson=pearson, sagr=sagr, ccc=ccc)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(x.dot(y) / denom)


def _concordance_cc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = float(((x - mean_x) * (y - mean_y)).mean())

    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0:
        return 0.0
    return float((2 * cov) / denominator)
