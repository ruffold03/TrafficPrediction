from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from traffic_prediction.external.calendar import FEATURE_NAMES, calendar_features


def build_external_matrix(timestamps: pd.DatetimeIndex, weather: pd.DataFrame | None = None) -> np.ndarray:
    rows = []
    weather_index = None
    if weather is not None and len(weather):
        weather_index = weather.copy()
        weather_index.index = pd.to_datetime(weather_index.index, utc=True)
        weather_index = weather_index[~weather_index.index.duplicated(keep="last")].sort_index()
    utc_stamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    positions = (
        weather_index.index.get_indexer(utc_stamps, method="pad")
        if weather_index is not None
        else np.full(len(utc_stamps), -1)
    )
    for ts, position in zip(utc_stamps, positions):
        local = ts.tz_convert("America/Los_Angeles")
        moment = local.to_pydatetime().replace(tzinfo=None)
        features = calendar_features(moment)
        temperature = precipitation = wind = code = 0.0
        if weather_index is not None and position >= 0:
            row = weather_index.iloc[int(position)]
            temperature = float(row["temperature"])
            precipitation = float(row["precipitation"])
            wind = float(row["wind_speed"])
            code = float(row["weather_code"])
        rows.append(
            [
                features["hour_sin"],
                features["hour_cos"],
                features["dow_sin"],
                features["dow_cos"],
                features["is_holiday"],
                temperature,
                precipitation,
                wind,
                code,
            ]
        )
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("External feature width does not match FEATURE_NAMES")
    return matrix


def external_record(moment: datetime, weather_row: dict | None = None) -> dict:
    features = calendar_features(moment)
    weather_row = weather_row or {}
    return {
        "temperature": weather_row.get("temperature"),
        "precipitation": weather_row.get("precipitation"),
        "weather_code": None if weather_row.get("weather_code") is None else int(weather_row["weather_code"]),
        "wind_speed": weather_row.get("wind_speed"),
        "hour": features["hour"],
        "day_of_week": features["day_of_week"],
        "is_holiday": bool(features["is_holiday"]),
    }
