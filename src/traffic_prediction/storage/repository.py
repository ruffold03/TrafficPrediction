from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from traffic_prediction.storage.models import (
    Camera,
    Edge,
    ExternalObservation,
    FlowObservation,
    Forecast,
    Segment,
    VideoJob,
)


class Repository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_camera(self, name: str, roi_path: str | None = None, note: str | None = None) -> Camera:
        camera = self.session.scalar(select(Camera).where(Camera.name == name))
        if camera is None:
            camera = Camera(name=name, roi_path=roi_path, note=note)
            self.session.add(camera)
        else:
            camera.roi_path = roi_path or camera.roi_path
            camera.note = note or camera.note
        self.session.flush()
        return camera

    def upsert_segment(
        self,
        external_key: str,
        name: str,
        source: str,
        camera_id: int | None = None,
        free_flow: float | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> Segment:
        segment = self.session.scalar(select(Segment).where(Segment.external_key == external_key))
        if segment is None:
            segment = Segment(
                external_key=external_key,
                name=name,
                source=source,
                camera_id=camera_id,
                free_flow=free_flow,
                latitude=latitude,
                longitude=longitude,
            )
            self.session.add(segment)
        else:
            segment.name = name
            segment.source = source
            segment.camera_id = camera_id if camera_id is not None else segment.camera_id
            segment.free_flow = free_flow if free_flow is not None else segment.free_flow
            segment.latitude = latitude if latitude is not None else segment.latitude
            segment.longitude = longitude if longitude is not None else segment.longitude
        self.session.flush()
        return segment

    def add_edge(self, source_id: int, target_id: int, weight: float) -> None:
        existing = self.session.scalar(
            select(Edge).where(Edge.source_id == source_id, Edge.target_id == target_id)
        )
        if existing is None:
            self.session.add(Edge(source_id=source_id, target_id=target_id, weight=weight))
        else:
            existing.weight = weight

    def add_flow(
        self,
        segment_id: int,
        ts: datetime,
        source: str,
        volume: float | None = None,
        speed: float | None = None,
        density: float | None = None,
    ) -> None:
        row = self.session.scalar(
            select(FlowObservation).where(
                FlowObservation.segment_id == segment_id,
                FlowObservation.ts == ts,
                FlowObservation.source == source,
            )
        )
        if row is None:
            self.session.add(
                FlowObservation(
                    segment_id=segment_id,
                    ts=ts,
                    source=source,
                    volume=volume,
                    speed=speed,
                    density=density,
                )
            )
        else:
            row.volume = volume
            row.speed = speed
            row.density = density

    def add_external(self, ts: datetime, region: str, **fields) -> None:
        row = self.session.scalar(
            select(ExternalObservation).where(ExternalObservation.ts == ts, ExternalObservation.region == region)
        )
        if row is None:
            self.session.add(ExternalObservation(ts=ts, region=region, **fields))
        else:
            for key, value in fields.items():
                setattr(row, key, value)

    def add_forecast(self, **fields) -> Forecast:
        row = Forecast(**fields)
        self.session.add(row)
        self.session.flush()
        return row

    def start_video_job(self, camera_id: int | None, video_path: str) -> VideoJob:
        job = VideoJob(
            camera_id=camera_id,
            video_path=video_path,
            status="running",
            started_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(job)
        self.session.flush()
        return job

    def finish_video_job(
        self,
        job: VideoJob,
        status: str,
        frames: int | None = None,
        counted_vehicles: int | None = None,
        error: str | None = None,
    ) -> None:
        job.status = status
        job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.frames = frames
        job.counted_vehicles = counted_vehicles
        job.error = error

    def list_segments(self) -> list[Segment]:
        return list(self.session.scalars(select(Segment).order_by(Segment.id)))

    def get_segment(self, segment_id: int) -> Segment | None:
        return self.session.get(Segment, segment_id)

    def get_segment_by_key(self, external_key: str) -> Segment | None:
        return self.session.scalar(select(Segment).where(Segment.external_key == external_key))

    def recent_flows(self, segment_id: int, limit: int = 48) -> list[FlowObservation]:
        rows = self.session.scalars(
            select(FlowObservation)
            .where(FlowObservation.segment_id == segment_id)
            .order_by(FlowObservation.ts.desc())
            .limit(limit)
        )
        return list(reversed(list(rows)))

    def list_jobs(self, limit: int = 10) -> list[VideoJob]:
        return list(self.session.scalars(select(VideoJob).order_by(VideoJob.id.desc()).limit(limit)))
