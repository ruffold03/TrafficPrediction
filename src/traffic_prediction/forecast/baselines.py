from __future__ import annotations

import numpy as np


def persistence_predict(windows: np.ndarray, horizon_steps: list[int]) -> np.ndarray:
    """Repeat the last observed value. windows: (S, T, N)."""
    last = windows[:, -1, :]
    return np.stack([last for _ in horizon_steps], axis=1)


def historical_average_table(values: np.ndarray, timestamps_min: np.ndarray, slots: int = 7 * 24 * 12) -> np.ndarray:
    """Mean speed by minute-of-week slot. values: (T, N), timestamps in minutes from an epoch."""
    table = np.zeros((slots, values.shape[1]), dtype=np.float64)
    counts = np.zeros((slots, values.shape[1]), dtype=np.float64)
    slot = np.mod(timestamps_min.astype(np.int64), slots)
    for index, bucket in enumerate(slot):
        row = values[index]
        valid = np.isfinite(row) & (row > 1e-3)
        table[bucket, valid] += row[valid]
        counts[bucket, valid] += 1
    counts = np.maximum(counts, 1.0)
    return (table / counts).astype(np.float32)


def historical_average_predict(
    table: np.ndarray,
    target_timestamps_min: np.ndarray,
    horizon_steps: list[int],
    step_minutes: int,
    n_nodes: int,
) -> np.ndarray:
    """target_timestamps_min is the timestamp of the last input step, shape (S,)."""
    slots = table.shape[0]
    predictions = []
    for step in horizon_steps:
        future = target_timestamps_min + step * step_minutes
        bucket = np.mod(future.astype(np.int64), slots)
        predictions.append(table[bucket])
    return np.stack(predictions, axis=1)
