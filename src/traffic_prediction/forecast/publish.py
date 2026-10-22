from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from traffic_prediction.storage.repository import Repository


def publish_benchmark(session, npz_path: str | Path, free_flow: list[float] | None = None, steps: int = 288) -> int:
    with np.load(npz_path, allow_pickle=False) as blob:
        values = blob["values"].astype(np.float32)
        timestamps = blob["timestamps_ns"].astype(np.int64)
        adjacency = blob["adjacency"].astype(np.float32)
        sensor_ids = [str(item) for item in blob["sensor_ids"]]
        latitude = blob["latitude"].astype(np.float32)
        longitude = blob["longitude"].astype(np.float32)
    repo = Repository(session)
    segments = []
    for index, sensor_id in enumerate(sensor_ids):
        flow = None if free_flow is None else float(free_flow[index])
        segments.append(
            repo.upsert_segment(
                external_key=f"metr-la:{sensor_id}",
                name=f"Датчик {sensor_id}",
                source="benchmark",
                free_flow=flow,
                latitude=None if np.isnan(latitude[index]) else float(latitude[index]),
                longitude=None if np.isnan(longitude[index]) else float(longitude[index]),
            )
        )
    for i, left in enumerate(segments):
        for j, right in enumerate(segments):
            weight = float(adjacency[i, j])
            if i != j and weight > 0:
                repo.add_edge(left.id, right.id, weight)
    start = max(values.shape[0] - steps, 0)
    written = 0
    for row in range(start, values.shape[0]):
        moment = datetime.fromtimestamp(int(timestamps[row]) / 1e9, tz=timezone.utc).replace(tzinfo=None)
        for index, segment in enumerate(segments):
            speed = float(values[row, index])
            repo.add_flow(segment.id, moment, source="benchmark", speed=speed)
            written += 1
    session.commit()
    return written
