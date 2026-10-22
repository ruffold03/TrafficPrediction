from __future__ import annotations

from pathlib import Path

import h5py
import httpx
import numpy as np
import pandas as pd

from traffic_prediction.config import load_config, resolve_path
from traffic_prediction.external.align import build_external_matrix
from traffic_prediction.external.weather import align_weather, clean_weather_frame, fetch_hourly_weather
from traffic_prediction.forecast.graph import connected_subset, gaussian_adjacency


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=600.0)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    temporary.replace(destination)
    return destination


def _decode_array(values) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.ndarray):
            value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
        decoded.append(str(value))
    return decoded


def read_speed_h5(path: Path) -> tuple[pd.DataFrame, list[str]]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["df/block0_values"][:], dtype=np.float32)
        axis0 = _decode_array(handle["df/axis0"][:])
        axis1 = handle["df/axis1"][:]
    if values.shape[1] == len(axis0):
        columns = axis0
        stamps = axis1
    elif values.shape[0] == len(axis0) and values.shape[1] == len(axis1):
        values = values.T
        columns = axis1 if len(axis1) == values.shape[1] else axis0
        stamps = axis0 if columns is axis1 else axis1
        columns = _decode_array(columns) if not isinstance(columns[0], str) else list(columns)
    else:
        raise RuntimeError(f"Unexpected METR-LA layout: values {values.shape}, axis0 {len(axis0)}, axis1 {len(axis1)}")
    index = pd.to_datetime(np.asarray(stamps).astype("int64"), unit="ns")
    frame = pd.DataFrame(values, index=index, columns=[str(col) for col in columns])
    frame = frame.sort_index()
    return frame, list(frame.columns)


def read_distances(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [col.strip().lower() for col in frame.columns]
    rename = {}
    for col in frame.columns:
        if col in {"from", "origin", "src"}:
            rename[col] = "src"
        elif col in {"to", "destination", "dst"}:
            rename[col] = "dst"
        elif col in {"cost", "distance"}:
            rename[col] = "distance"
    frame = frame.rename(columns=rename)
    frame["src"] = frame["src"].astype(str)
    frame["dst"] = frame["dst"].astype(str)
    return frame


def read_locations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [col.strip().lower() for col in frame.columns]
    if "sensor_id" not in frame.columns:
        for col in frame.columns:
            if "sensor" in col or col == "id":
                frame = frame.rename(columns={col: "sensor_id"})
                break
    frame["sensor_id"] = frame["sensor_id"].astype(str)
    return frame


def distance_matrix(sensor_ids: list[str], distances: pd.DataFrame) -> np.ndarray:
    index = {sensor_id: pos for pos, sensor_id in enumerate(sensor_ids)}
    matrix = np.full((len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float64)
    matched = 0
    for src, dst, distance in distances[["src", "dst", "distance"]].itertuples(index=False):
        if src in index and dst in index:
            i, j = index[src], index[dst]
            matrix[i, j] = min(matrix[i, j], float(distance))
            matrix[j, i] = min(matrix[j, i], float(distance))
            matched += 1
    if matched == 0:
        raise RuntimeError("Sensor ids from metr-la.h5 do not match distances_la_2012.csv")
    np.fill_diagonal(matrix, 0.0)
    finite = matrix[np.isfinite(matrix) & (matrix > 0)]
    fill = float(np.quantile(finite, 0.95)) if len(finite) else 1e6
    matrix[~np.isfinite(matrix)] = fill
    return matrix


def localize_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward")
    return index.tz_convert("America/Los_Angeles")


def prepare(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    forecast = cfg["forecast"]
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"]) / "metr_la"
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = download(forecast["h5_url"], data_dir / "metr-la.h5")
    distances_path = download(forecast["distances_url"], data_dir / "distances_la_2012.csv")
    locations_path = download(forecast["locations_url"], data_dir / "graph_sensor_locations.csv")

    speeds, sensor_ids = read_speed_h5(h5_path)
    distances = distance_matrix(sensor_ids, read_distances(distances_path))
    adjacency = gaussian_adjacency(
        distances,
        sigma_m=float(forecast["distance_sigma_m"]),
        epsilon=float(forecast["distance_epsilon"]),
    )
    chosen = connected_subset(adjacency, int(forecast["max_nodes"]))
    sensor_ids = [sensor_ids[i] for i in chosen]
    speeds = speeds.iloc[:, chosen]
    adjacency = adjacency[np.ix_(chosen, chosen)]
    distances = distances[np.ix_(chosen, chosen)]

    locations = read_locations(locations_path).set_index("sensor_id")
    latitude = np.array([float(locations.loc[sid, "latitude"]) if sid in locations.index else np.nan for sid in sensor_ids])
    longitude = np.array([float(locations.loc[sid, "longitude"]) if sid in locations.index else np.nan for sid in sensor_ids])

    local_index = localize_index(pd.DatetimeIndex(speeds.index))
    speeds = speeds.copy()
    speeds.index = local_index
    speeds = speeds[~speeds.index.isna()]

    start = speeds.index.min().tz_convert("UTC")
    end = speeds.index.max().tz_convert("UTC")
    hourly = fetch_hourly_weather(
        float(forecast["latitude"]),
        float(forecast["longitude"]),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        data_dir / "weather_la.json",
    )
    hourly = clean_weather_frame(hourly)
    aligned = align_weather(speeds.index.tz_convert("UTC"), hourly)
    external = build_external_matrix(speeds.index, aligned)

    output = data_dir / "processed.npz"
    np.savez_compressed(
        output,
        values=speeds.to_numpy(dtype=np.float32),
        timestamps_ns=speeds.index.tz_convert("UTC").asi8,
        adjacency=adjacency.astype(np.float32),
        sensor_ids=np.array(sensor_ids),
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        external=external.astype(np.float32),
    )
    return output


if __name__ == "__main__":
    path = prepare()
    print(path)
