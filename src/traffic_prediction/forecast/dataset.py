from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from traffic_prediction.external.weather import zscore_apply, zscore_fit


@dataclass
class SplitData:
    values: np.ndarray
    timestamps_ns: np.ndarray
    adjacency: np.ndarray
    external: np.ndarray
    sensor_ids: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    split_at: int
    mean: float
    std: float
    weather_mean: np.ndarray
    weather_std: np.ndarray
    free_flow: np.ndarray


class WindowDataset(Dataset):
    def __init__(self, values_norm: np.ndarray, values_raw: np.ndarray, external: np.ndarray, indices: np.ndarray, history: int, horizons: list[int]):
        self.values_norm = values_norm
        self.values_raw = values_raw
        self.external = external
        self.indices = indices
        self.history = history
        self.horizons = horizons

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        t = int(self.indices[item])
        x = self.values_norm[t - self.history + 1 : t + 1]
        y = np.stack([self.values_norm[t + step] for step in self.horizons], axis=0)
        raw = np.stack([self.values_raw[t + step] for step in self.horizons], axis=0)
        mask = (raw > 1e-3).astype(np.float32)
        external = self.external[t]
        return (
            torch.from_numpy(x),
            torch.from_numpy(external),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.from_numpy(raw),
        )


def _subsample(indices: np.ndarray, stride: int, limit: int | None) -> np.ndarray:
    chosen = indices[:: max(stride, 1)]
    if limit is not None and len(chosen) > limit:
        chosen = chosen[:limit]
    return chosen


def load_split(path, cfg: dict) -> SplitData:
    forecast = cfg["forecast"]
    history = int(forecast["history"])
    horizons = [int(step) for step in forecast["horizons"]]
    with np.load(path, allow_pickle=False) as blob:
        values = blob["values"].astype(np.float32)
        timestamps = blob["timestamps_ns"].astype(np.int64)
        adjacency = blob["adjacency"].astype(np.float32)
        external = blob["external"].astype(np.float32)
        sensor_ids = blob["sensor_ids"]

    max_horizon = max(horizons)
    last_input = np.arange(history - 1, values.shape[0] - max_horizon)
    train_end = int(len(last_input) * float(forecast["train_ratio"]))
    val_end = int(len(last_input) * (float(forecast["train_ratio"]) + float(forecast["val_ratio"])))
    train_ids = last_input[:train_end]
    val_ids = last_input[train_end:val_end]
    test_ids = last_input[val_end:]
    split_at = int(train_ids[-1]) + 1
    observed = values[:split_at]
    valid = observed[observed > 1e-3]
    mean = float(valid.mean()) if len(valid) else 0.0
    std = float(valid.std()) if len(valid) else 1.0
    std = std if std > 1e-6 else 1.0

    weather = external[:split_at, 5:]
    weather_mean, weather_std = zscore_fit(weather)
    external = external.copy()
    external[:, 5:] = zscore_apply(external[:, 5:], weather_mean, weather_std)

    masked = np.where(values[:split_at] > 1e-3, values[:split_at], np.nan)
    free_flow = np.nanquantile(masked, 0.95, axis=0)
    free_flow = np.nan_to_num(free_flow, nan=mean).astype(np.float32)

    stride = int(forecast.get("train_stride", 1))
    return SplitData(
        values=values,
        timestamps_ns=timestamps,
        adjacency=adjacency,
        external=external,
        sensor_ids=sensor_ids,
        train_index=_subsample(train_ids, stride, int(forecast.get("max_train_windows") or 0) or None),
        val_index=_subsample(val_ids, stride, int(forecast.get("max_eval_windows") or 0) or None),
        test_index=_subsample(test_ids, max(stride // 2, 1), int(forecast.get("max_eval_windows") or 0) or None),
        split_at=split_at,
        mean=mean,
        std=std,
        weather_mean=weather_mean,
        weather_std=weather_std,
        free_flow=free_flow,
    )


def make_loader(split: SplitData, indices: np.ndarray, cfg: dict, shuffle: bool) -> DataLoader:
    forecast = cfg["forecast"]
    values_norm = ((split.values - split.mean) / split.std).astype(np.float32)
    dataset = WindowDataset(
        values_norm,
        split.values,
        split.external,
        indices,
        int(forecast["history"]),
        [int(step) for step in forecast["horizons"]],
    )
    return DataLoader(
        dataset,
        batch_size=int(forecast["batch_size"]),
        shuffle=shuffle,
        drop_last=shuffle,
    )
