from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


def load_polygon(path: str | Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        left, right = line.replace(";", ",").split(",")[:2]
        points.append((float(left), float(right)))
    if len(points) < 3:
        raise ValueError(f"ROI polygon needs at least 3 points, got {len(points)} in {path}")
    return points


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    sx = sum(p[0] for p in polygon)
    sy = sum(p[1] for p in polygon)
    n = len(polygon)
    return sx / n, sy / n


def angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def signed_turn_deg(
    entry: tuple[float, float],
    exit_point: tuple[float, float],
    center: tuple[float, float],
) -> float:
    vx1 = center[0] - entry[0]
    vy1 = center[1] - entry[1]
    vx2 = exit_point[0] - center[0]
    vy2 = exit_point[1] - center[1]
    cross = vx1 * vy2 - vy1 * vx2
    dot = vx1 * vx2 + vy1 * vy2
    if vx1 == 0 and vy1 == 0 or vx2 == 0 and vy2 == 0:
        return 0.0
    return math.degrees(math.atan2(cross, dot))


def movement_id(
    entry: tuple[float, float],
    exit_point: tuple[float, float],
    center: tuple[float, float],
    approaches: list[dict],
    turn_threshold_deg: float,
) -> int:
    entry_angle = angle_deg(entry[0] - center[0], entry[1] - center[1])
    best = min(approaches, key=lambda item: abs(wrap_deg(entry_angle - float(item["angle"]))))
    turn = signed_turn_deg(entry, exit_point, center)
    if turn > turn_threshold_deg:
        return int(best["right"])
    if turn < -turn_threshold_deg:
        return int(best["left"])
    return int(best["through"])


@dataclass
class CountEvent:
    frame: int
    movement_id: int
    class_id: int
    track_id: int


def coco_to_aic_class(coco_class: int) -> int:
    # AIC21 sample classes: 1 car, 2 truck. Bus and truck map to truck.
    if int(coco_class) in (5, 7):
        return 2
    return 1


def events_from_tracks(
    tracks: dict[int, list[dict]],
    polygon: list[tuple[float, float]],
    approaches: list[dict],
    min_inside_frames: int = 5,
    turn_threshold_deg: float = 35.0,
) -> list[CountEvent]:
    """Count a track when its centroid leaves the ROI after a long enough stay."""
    center_points = [
        (point["cx"], point["cy"])
        for points in tracks.values()
        for point in points
        if point_in_polygon(point["cx"], point["cy"], polygon)
    ]
    if center_points:
        center = (
            sum(point[0] for point in center_points) / len(center_points),
            sum(point[1] for point in center_points) / len(center_points),
        )
    else:
        center = polygon_centroid(polygon)
    events: list[CountEvent] = []
    for track_id, points in tracks.items():
        ordered = sorted(points, key=lambda item: item["frame"])
        inside_flags = [point_in_polygon(p["cx"], p["cy"], polygon) for p in ordered]
        inside_points = [p for p, flag in zip(ordered, inside_flags) if flag]
        if len(inside_points) < min_inside_frames:
            continue
        left = False
        exit_frame_index = None
        for index in range(1, len(ordered)):
            if inside_flags[index - 1] and not inside_flags[index]:
                left = True
                exit_frame_index = index - 1
                break
        if not left:
            continue
        entry = inside_points[0]
        exit_point = ordered[exit_frame_index]
        move = movement_id(
            (entry["cx"], entry["cy"]),
            (exit_point["cx"], exit_point["cy"]),
            center,
            approaches,
            turn_threshold_deg,
        )
        events.append(
            CountEvent(
                frame=int(exit_point["frame"]),
                movement_id=move,
                class_id=coco_to_aic_class(exit_point.get("coco_class", 2)),
                track_id=int(track_id),
            )
        )
    events.sort(key=lambda item: (item.frame, item.track_id))
    return events


def aggregate_minutes(events: list[CountEvent], fps: float, origin_frame: int = 0) -> list[dict]:
    bins: dict[int, dict] = {}
    for event in events:
        minute = int((event.frame - origin_frame) // max(fps * 60.0, 1.0))
        bucket = bins.setdefault(
            minute,
            {"minute": minute, "total": 0, "by_movement": {}, "by_class": {}},
        )
        bucket["total"] += 1
        bucket["by_movement"][event.movement_id] = bucket["by_movement"].get(event.movement_id, 0) + 1
        bucket["by_class"][event.class_id] = bucket["by_class"].get(event.class_id, 0) + 1
    return [bins[key] for key in sorted(bins)]
