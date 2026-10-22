from __future__ import annotations

import numpy as np


def masked_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and MAPE. Zeros in the target are treated as missing detector readings."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(target) & (np.abs(target) > 1e-3)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}
    error = prediction[mask] - target[mask]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = float(np.mean(np.abs(error) / np.abs(target[mask])) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "count": int(mask.sum())}


def metrics_by_horizon(prediction: np.ndarray, target: np.ndarray, horizons_min: list[int]) -> dict[str, dict]:
    """prediction/target shape: (samples, horizons, nodes)."""
    report = {}
    for index, minutes in enumerate(horizons_min):
        report[str(minutes)] = masked_metrics(prediction[:, index], target[:, index])
    return report
