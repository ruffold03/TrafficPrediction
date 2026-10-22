from __future__ import annotations

import csv
from pathlib import Path

from traffic_prediction.video.counter import CountEvent


def load_counting_gt(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame_id"]),
                    "movement_id": int(row["movement_id"]),
                    "class_id": int(row["vehicle_class_id"]),
                }
            )
    return rows


def counts_by_movement(rows: list[dict], movements: range | list[int] | None = None) -> dict[int, int]:
    keys = list(movements) if movements is not None else sorted({int(row["movement_id"]) for row in rows})
    totals = {int(key): 0 for key in keys}
    for row in rows:
        movement = int(row["movement_id"])
        totals[movement] = totals.get(movement, 0) + 1
    return totals


def counting_errors(predicted: list[CountEvent], ground_truth: list[dict]) -> dict:
    movements = list(range(1, 13))
    pred_rows = [{"movement_id": event.movement_id, "class_id": event.class_id} for event in predicted]
    pred = counts_by_movement(pred_rows, movements)
    truth = counts_by_movement(ground_truth, movements)
    per_movement = []
    abs_sum = 0
    sq_sum = 0
    for movement in movements:
        error = pred[movement] - truth[movement]
        abs_sum += abs(error)
        sq_sum += error * error
        per_movement.append(
            {
                "movement_id": movement,
                "predicted": pred[movement],
                "ground_truth": truth[movement],
                "error": error,
            }
        )
    n = len(movements)
    pred_total = sum(pred.values())
    truth_total = sum(truth.values())
    return {
        "per_movement": per_movement,
        "mae": abs_sum / n,
        "rmse": (sq_sum / n) ** 0.5,
        "predicted_total": pred_total,
        "ground_truth_total": truth_total,
        "total_absolute_error": abs(pred_total - truth_total),
    }
