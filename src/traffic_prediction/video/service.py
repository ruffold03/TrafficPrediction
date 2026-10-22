from __future__ import annotations

from datetime import datetime, timedelta

from traffic_prediction.config import load_config, resolve_path
from traffic_prediction.storage import open_database
from traffic_prediction.storage.repository import Repository
from traffic_prediction.video.counter import aggregate_minutes, load_polygon
from traffic_prediction.video.pipeline import track_video


def congestion_from_volume(volume: float, free_flow_volume: float) -> str:
    if free_flow_volume <= 0:
        return "unknown"
    ratio = volume / free_flow_volume
    if ratio < 0.4:
        return "free"
    if ratio < 0.7:
        return "moderate"
    return "congested"


def process_video(
    video_path: str,
    camera_name: str = "cam_5",
    database_url: str | None = None,
    max_frames: int | None = None,
    started_at: datetime | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_config()
    url = database_url or cfg["database_url"]
    camera_cfg = cfg["cameras"][camera_name]
    polygon = load_polygon(resolve_path(cfg, camera_cfg["roi"]))
    video_cfg = cfg["video"]
    engine, factory = open_database(url)
    session = factory()
    repo = Repository(session)
    try:
        camera = repo.upsert_camera(camera_name, roi_path=str(resolve_path(cfg, camera_cfg["roi"])))
        segment = repo.upsert_segment(
            external_key=f"video:{camera_name}",
            name=camera_name,
            source="video",
            camera_id=camera.id,
        )
        job = repo.start_video_job(camera.id, video_path)
        session.commit()
        try:
            result = track_video(
                video_path,
                polygon,
                camera_cfg["approaches"],
                model_name=video_cfg["model"],
                confidence=float(video_cfg["confidence"]),
                iou=float(video_cfg["iou"]),
                classes=list(video_cfg["classes"]),
                min_inside_frames=int(video_cfg["min_inside_frames"]),
                turn_threshold_deg=float(video_cfg["turn_threshold_deg"]),
                imgsz=int(video_cfg["imgsz"]),
                max_frames=max_frames,
            )
        except Exception as exc:
            repo.finish_video_job(job, "failed", error=str(exc))
            session.commit()
            raise
        minutes = aggregate_minutes(result["events"], result["fps"])
        origin = started_at or datetime(2018, 4, 30, 14, 29, 0)
        volumes = [bucket["total"] for bucket in minutes] or [0]
        free_flow = max(volumes) if volumes else 0.0
        segment.free_flow = float(free_flow)
        for bucket in minutes:
            ts = origin + timedelta(minutes=int(bucket["minute"]))
            repo.add_flow(
                segment.id,
                ts,
                source="video",
                volume=float(bucket["total"]),
                density=float(bucket["total"]),
            )
        repo.finish_video_job(
            job,
            "done",
            frames=result["frames"],
            counted_vehicles=len(result["events"]),
        )
        session.commit()
        return {
            "job_id": job.id,
            "segment_id": segment.id,
            "frames": result["frames"],
            "fps": result["fps"],
            "tracks": result["tracks"],
            "counted_vehicles": len(result["events"]),
            "minutes": minutes,
            "events": [
                {
                    "frame": event.frame,
                    "movement_id": event.movement_id,
                    "class_id": event.class_id,
                    "track_id": event.track_id,
                }
                for event in result["events"]
            ],
        }
    finally:
        session.close()
        engine.dispose()
