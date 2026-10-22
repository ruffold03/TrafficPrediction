from __future__ import annotations

import json

from traffic_prediction.config import load_config, resolve_path
from traffic_prediction.video.counter import CountEvent
from traffic_prediction.video.evaluate import counting_errors, load_counting_gt
from traffic_prediction.video.service import process_video


def evaluate_sample(max_frames: int | None = None) -> dict:
    cfg = load_config()
    video_path = resolve_path(cfg, cfg["video"]["sample_video"])
    gt_path = resolve_path(cfg, cfg["video"]["sample_gt"])
    database = resolve_path(cfg, cfg["paths"]["data_dir"]) / "traffic.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    result = process_video(
        str(video_path),
        camera_name="cam_5",
        database_url=f"sqlite:///{database.as_posix()}",
        max_frames=max_frames,
        cfg=cfg,
    )
    truth = load_counting_gt(gt_path)
    events = [
        CountEvent(frame=item["frame"], movement_id=item["movement_id"], class_id=item["class_id"], track_id=item["track_id"])
        for item in result["events"]
    ]
    report = {
        "video": str(video_path),
        "frames": result["frames"],
        "fps": result["fps"],
        "tracks": result["tracks"],
        "counted_vehicles": result["counted_vehicles"],
        "segment_id": result["segment_id"],
        "errors": counting_errors(events, truth),
    }
    destination = resolve_path(cfg, cfg["paths"]["experiments_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "video_cam5_1min.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_sample(), ensure_ascii=False, indent=2))
