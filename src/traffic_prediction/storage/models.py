from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    roi_path: Mapped[str | None] = mapped_column(String)
    note: Mapped[str | None] = mapped_column(String)
    segments: Mapped[list[Segment]] = relationship(back_populates="camera")


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id"))
    free_flow: Mapped[float | None] = mapped_column(Float)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    camera: Mapped[Camera | None] = relationship(back_populates="segments")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("source_id", "target_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)


class FlowObservation(Base):
    __tablename__ = "flow_observations"
    __table_args__ = (UniqueConstraint("segment_id", "ts", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    volume: Mapped[float | None] = mapped_column(Float)
    speed: Mapped[float | None] = mapped_column(Float)
    density: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String, nullable=False)


class ExternalObservation(Base):
    __tablename__ = "external_observations"
    __table_args__ = (UniqueConstraint("ts", "region"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    region: Mapped[str] = mapped_column(String, nullable=False)
    temperature: Mapped[float | None] = mapped_column(Float)
    precipitation: Mapped[float | None] = mapped_column(Float)
    weather_code: Mapped[int | None] = mapped_column(Integer)
    wind_speed: Mapped[float | None] = mapped_column(Float)
    hour: Mapped[int | None] = mapped_column(Integer)
    day_of_week: Mapped[int | None] = mapped_column(Integer)
    is_holiday: Mapped[bool | None] = mapped_column(Boolean)


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    target_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_min: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_volume: Mapped[float | None] = mapped_column(Float)
    predicted_speed: Mapped[float | None] = mapped_column(Float)
    congestion_level: Mapped[str | None] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String, nullable=False)


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id"))
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    frames: Mapped[int | None] = mapped_column(Integer)
    counted_vehicles: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(String)


def make_engine(url: str):
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    elif url.startswith("postgresql"):
        connect_args["connect_timeout"] = 3
    return create_engine(url, connect_args=connect_args, future=True, pool_pre_ping=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
