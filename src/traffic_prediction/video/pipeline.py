from __future__ import annotations

from pathlib import Path

import cv2

from traffic_prediction.video.counter import events_from_tracks


def track_video(
    video_path: str | Path,
    polygon: list[tuple[float, float]],
    approaches: list[dict],
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    iou: float = 0.5,
    classes: list[int] | None = None,
    min_inside_frames: int = 5,
    turn_threshold_deg: float = 35.0,
    imgsz: int = 640,
    max_frames: int | None = None,
) -> dict:
    from ultralytics import YOLO

    classes = classes or [2, 3, 5, 7]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
    model = YOLO(model_name)
    tracks: dict[int, list[dict]] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                iou=iou,
                classes=classes,
                imgsz=imgsz,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().tolist()
                xyxy = boxes.xyxy.tolist()
                clss = boxes.cls.int().tolist()
                for track_id, box, coco_class in zip(ids, xyxy, clss):
                    x1, y1, x2, y2 = box
                    tracks.setdefault(int(track_id), []).append(
                        {
                            "frame": frame_index,
                            "cx": (x1 + x2) / 2.0,
                            "cy": (y1 + y2) / 2.0,
                            "coco_class": int(coco_class),
                        }
                    )
            frame_index += 1
    finally:
        capture.release()

    events = events_from_tracks(
        tracks,
        polygon,
        approaches,
        min_inside_frames=min_inside_frames,
        turn_threshold_deg=turn_threshold_deg,
    )
    return {
        "fps": fps,
        "frames": frame_index,
        "tracks": len(tracks),
        "events": events,
    }
