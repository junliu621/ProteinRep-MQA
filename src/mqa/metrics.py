from __future__ import annotations

import math

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(truth: list[float] | np.ndarray, prediction: list[float] | np.ndarray) -> dict[str, float]:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    error = prediction_array - truth_array
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _correlation(truth_array, prediction_array),
        "spearman": _correlation(_rankdata(truth_array), _rankdata(prediction_array)),
        "n_residues": int(len(truth_array)),
    }

