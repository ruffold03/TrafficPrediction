from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from traffic_prediction.forecast.graph import congestion_level, normalize_adjacency, volume_to_speed
from traffic_prediction.forecast.model import GRUForecaster, STGCN


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).abs() * mask
    return error.sum() / mask.sum().clamp(min=1.0)


def build_model(name: str, fusion: str, cfg: dict, external_dim: int) -> torch.nn.Module:
    forecast = cfg["forecast"]
    horizons = len(forecast["horizons"])
    hidden = int(forecast["hidden"])
    dropout = float(forecast["dropout"])
    if name == "gru":
        return GRUForecaster(hidden, horizons, external_dim, fusion="none", dropout=dropout)
    return STGCN(hidden, horizons, external_dim, fusion=fusion, dropout=dropout)


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


class ForecastEngine:
    def __init__(self, checkpoint_path: str | Path, npz_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        blob = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.meta = blob["meta"]
        self.model = build_model(self.meta["model_name"], self.meta["fusion"], {"forecast": self.meta["forecast"]}, self.meta["external_dim"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        with np.load(npz_path, allow_pickle=False) as data:
            self.values = data["values"].astype(np.float32)
            self.timestamps_ns = data["timestamps_ns"].astype(np.int64)
            self.external = data["external"].astype(np.float32)
            self.sensor_ids = [str(item) for item in data["sensor_ids"]]
        weather = self.external[:, 5:]
        mean = np.asarray(self.meta["weather_mean"], dtype=np.float32)
        std = np.asarray(self.meta["weather_std"], dtype=np.float32)
        self.external = self.external.copy()
        self.external[:, 5:] = (weather - mean) / std
        adjacency = np.asarray(self.meta["adjacency"], dtype=np.float32)
        self.adjacency = torch.from_numpy(normalize_adjacency(adjacency)).to(self.device)
        self.mean = float(self.meta["mean"])
        self.std = float(self.meta["std"])
        self.free_flow = np.asarray(self.meta["free_flow"], dtype=np.float32)
        self.history = int(self.meta["forecast"]["history"])
        self.horizons = [int(step) for step in self.meta["forecast"]["horizons"]]
        self.step_minutes = int(self.meta["forecast"]["step_minutes"])

    def node_index(self, sensor_id: str) -> int:
        return self.sensor_ids.index(str(sensor_id))

    def predict_window(self, raw_window: np.ndarray, external: np.ndarray) -> np.ndarray:
        normalized = (raw_window - self.mean) / self.std
        x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        ext = torch.from_numpy(external.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x, self.adjacency, ext).cpu().numpy()[0]
        return denormalize(pred, self.mean, self.std)

    def latest_window(self) -> tuple[np.ndarray, np.ndarray, datetime]:
        raw = self.values[-self.history :].copy()
        external = self.external[-1].copy()
        issued = datetime.fromtimestamp(int(self.timestamps_ns[-1]) / 1e9, tz=timezone.utc).replace(tzinfo=None)
        return raw, external, issued

    def forecast_sensor(self, sensor_id: str, replacement_speeds: np.ndarray | None = None) -> list[dict]:
        node = self.node_index(sensor_id)
        raw, external, issued = self.latest_window()
        if replacement_speeds is not None:
            series = np.asarray(replacement_speeds, dtype=np.float32)
            if len(series) < self.history:
                pad = np.full(self.history - len(series), series[-1] if len(series) else self.free_flow[node])
                series = np.concatenate([pad, series])
            raw[:, node] = series[-self.history :]
        prediction = self.predict_window(raw, external)
        rows = []
        for index, step in enumerate(self.horizons):
            speed = float(max(prediction[index, node], 0.0))
            rows.append(
                {
                    "horizon_min": step * self.step_minutes,
                    "target_ts": issued + timedelta(minutes=step * self.step_minutes),
                    "issued_at": issued,
                    "predicted_speed": speed,
                    "congestion_level": congestion_level(
                        speed,
                        float(self.free_flow[node]),
                        float(self.meta["forecast"]["congestion_free"]),
                        float(self.meta["forecast"]["congestion_moderate"]),
                    ),
                    "model_name": f"{self.meta['model_name']}:{self.meta['fusion']}",
                }
            )
        return rows

    def bridge_volume(self, volumes: list[float], sensor_id: str, volume_cap: float | None = None) -> list[dict]:
        node = self.node_index(sensor_id)
        cap = volume_cap if volume_cap and volume_cap > 0 else max(volumes or [1.0])
        speeds = [volume_to_speed(value, cap, float(self.free_flow[node])) for value in volumes]
        return self.forecast_sensor(sensor_id, np.asarray(speeds, dtype=np.float32))


def time_inference(model: torch.nn.Module, batches: int = 20, nodes: int = 32, history: int = 12, external_dim: int = 9) -> dict:
    device = next(model.parameters()).device
    adjacency = torch.eye(nodes, device=device)
    model.eval()
    starter = torch.randn(8, history, nodes, device=device)
    external = torch.randn(8, external_dim, device=device)
    with torch.no_grad():
        model(starter, adjacency, external)
        started = time.perf_counter()
        for _ in range(batches):
            model(starter, adjacency, external)
        elapsed = time.perf_counter() - started
    return {
        "batches": batches,
        "batch_size": 8,
        "nodes": nodes,
        "seconds": elapsed,
        "ms_per_batch": elapsed / batches * 1000.0,
    }
