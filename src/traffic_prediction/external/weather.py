from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def fetch_hourly_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        frame = pd.read_json(cache)
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame

    response = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "UTC",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "precipitation": hourly["precipitation"],
            "weather_code": hourly["weather_code"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_json(cache, date_format="iso", orient="records")
    return frame


def load_cached_weather(cache_path: str | Path) -> pd.DataFrame:
    frame = pd.read_json(cache_path)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def align_weather(timestamps: pd.DatetimeIndex, hourly: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill hourly weather onto the traffic timestamps."""
    weather = hourly.copy()
    weather["ts"] = pd.to_datetime(weather["ts"], utc=True)
    weather = weather.set_index("ts").sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    target = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).sort_values()
    combined = weather.reindex(weather.index.union(target)).sort_index()
    combined[["temperature", "precipitation", "weather_code", "wind_speed"]] = combined[
        ["temperature", "precipitation", "weather_code", "wind_speed"]
    ].ffill().bfill()
    aligned = combined.loc[target]
    aligned.index.name = "ts"
    return aligned


def clean_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in ("temperature", "precipitation", "wind_speed", "weather_code"):
        series = pd.to_numeric(cleaned[column], errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        series = series.fillna(median)
        if column != "weather_code":
            low, high = series.quantile(0.01), series.quantile(0.99)
            series = series.clip(low, high)
        cleaned[column] = series
    return cleaned


def zscore_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
