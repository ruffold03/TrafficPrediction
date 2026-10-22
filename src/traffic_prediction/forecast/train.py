from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_prediction.config import load_config, resolve_path
from traffic_prediction.forecast.baselines import (
    historical_average_predict,
    historical_average_table,
    persistence_predict,
)
from traffic_prediction.forecast.dataset import SplitData, load_split, make_loader
from traffic_prediction.forecast.graph import normalize_adjacency
from traffic_prediction.forecast.metrics import metrics_by_horizon
from traffic_prediction.forecast.model import GRUForecaster, STGCN
from traffic_prediction.forecast.predict import build_model, masked_mae, time_inference
from traffic_prediction.forecast.prepare import prepare


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader: DataLoader, adjacency: torch.Tensor, optimizer, device) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    steps = 0
    for batch in loader:
        x, external, y, mask, _raw = [item.to(device) for item in batch]
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(x, adjacency, external)
        loss = masked_mae(prediction, y, mask)
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


def _predict_loader(model, loader: DataLoader, adjacency: torch.Tensor, device, mean: float, std: float):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, external, _y, _mask, raw = [item.to(device) for item in batch]
            prediction = model(x, adjacency, external).cpu().numpy()
            preds.append(prediction * std + mean)
            targets.append(raw.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _collect_windows(split: SplitData, indices: np.ndarray, history: int) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack([split.values[t - history + 1 : t + 1] for t in indices], axis=0)
    stamps = split.timestamps_ns[indices] // 60_000_000_000
    return windows, stamps


def _targets(split: SplitData, indices: np.ndarray, horizons: list[int]) -> np.ndarray:
    return np.stack([np.stack([split.values[t + step] for step in horizons], axis=0) for t in indices], axis=0)


def train_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    npz_path = prepare(cfg)
    split = load_split(npz_path, cfg)
    device = _device()
    forecast = cfg["forecast"]
    horizons = [int(step) for step in forecast["horizons"]]
    horizon_min = [step * int(forecast["step_minutes"]) for step in horizons]
    adjacency = torch.from_numpy(normalize_adjacency(split.adjacency)).to(device)
    train_loader = make_loader(split, split.train_index, cfg, shuffle=True)
    val_loader = make_loader(split, split.val_index, cfg, shuffle=False)
    test_loader = make_loader(split, split.test_index, cfg, shuffle=False)
    external_dim = int(split.external.shape[1])

    specs = [
        ("gru", "none"),
        ("stgcn", "none"),
        ("stgcn", "concat"),
        ("stgcn", "gate"),
    ]
    checkpoints = resolve_path(cfg, cfg["paths"]["checkpoints_dir"])
    checkpoints.mkdir(parents=True, exist_ok=True)
    experiments = resolve_path(cfg, cfg["paths"]["experiments_dir"])
    experiments.mkdir(parents=True, exist_ok=True)

    neural_predictions = {}
    history = []
    for model_name, fusion in specs:
        model = build_model(model_name, fusion, cfg, external_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(forecast["lr"]))
        best_state = None
        best_val = float("inf")
        for epoch in range(int(forecast["epochs"])):
            train_loss = _run_epoch(model, train_loader, adjacency, optimizer, device)
            val_loss = _run_epoch(model, val_loader, adjacency, None, device)
            history.append({"model": model_name, "fusion": fusion, "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            print(f"{model_name}/{fusion} epoch {epoch + 1}: train {train_loss:.4f} val {val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        label = model_name if model_name == "gru" else f"{model_name}_{fusion}"
        prediction, target = _predict_loader(model, test_loader, adjacency, device, split.mean, split.std)
        neural_predictions[label] = prediction
        shared_target = target
        if label == "stgcn_gate":
            _save_checkpoint(checkpoints / "stgcn_gate.pt", model, cfg, split, external_dim, npz_path, model_name, fusion)
            gate_model = model
        else:
            _save_checkpoint(checkpoints / f"{label}.pt", model, cfg, split, external_dim, npz_path, model_name, fusion)

    windows, stamps = _collect_windows(split, split.test_index, int(forecast["history"]))
    persistence = persistence_predict(windows, horizons)
    train_slots = split.timestamps_ns[: split.split_at] // 60_000_000_000
    table = historical_average_table(split.values[: split.split_at], train_slots)
    average = historical_average_predict(table, stamps, horizons, int(forecast["step_minutes"]), split.values.shape[1])

    report = {
        "dataset": "METR-LA",
        "nodes": int(split.values.shape[1]),
        "timesteps": int(split.values.shape[0]),
        "epochs": int(forecast["epochs"]),
        "history_steps": int(forecast["history"]),
        "horizons_min": horizon_min,
        "train_windows": int(len(split.train_index)),
        "test_windows": int(len(split.test_index)),
        "device": str(device),
        "models": {
            "persistence": metrics_by_horizon(persistence, shared_target, horizon_min),
            "historical_average": metrics_by_horizon(average, shared_target, horizon_min),
        },
        "training": history,
    }
    for label, prediction in neural_predictions.items():
        report["models"][label] = metrics_by_horizon(prediction, shared_target, horizon_min)

    timing = time_inference(gate_model, nodes=split.values.shape[1], history=int(forecast["history"]), external_dim=external_dim)
    report["inference"] = timing
    (experiments / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metrics_csv(experiments / "metrics.csv", report)
    (experiments / "performance.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(report["models"], ensure_ascii=False, indent=2), flush=True)
    return report


def _save_checkpoint(path: Path, model, cfg, split: SplitData, external_dim: int, npz_path: Path, model_name: str, fusion: str) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "model_name": model_name,
                "fusion": fusion,
                "external_dim": external_dim,
                "forecast": cfg["forecast"],
                "mean": split.mean,
                "std": split.std,
                "weather_mean": split.weather_mean.tolist(),
                "weather_std": split.weather_std.tolist(),
                "free_flow": split.free_flow.tolist(),
                "adjacency": split.adjacency.tolist(),
                "sensor_ids": [str(item) for item in split.sensor_ids],
                "npz": str(npz_path),
            },
        },
        path,
    )


def _write_metrics_csv(path: Path, report: dict) -> None:
    lines = ["model,horizon_min,mae,rmse,mape,count"]
    for model, horizons in report["models"].items():
        for horizon, metrics in horizons.items():
            lines.append(
                f"{model},{horizon},{metrics['mae']:.6f},{metrics['rmse']:.6f},{metrics['mape']:.6f},{metrics['count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    train_all()
