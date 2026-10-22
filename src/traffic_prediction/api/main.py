from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from traffic_prediction.config import load_config, resolve_path
from traffic_prediction.forecast.predict import ForecastEngine
from traffic_prediction.forecast.publish import publish_benchmark
from traffic_prediction.storage import open_database
from traffic_prediction.storage.repository import Repository
from traffic_prediction.video.service import process_video

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "ui" / "templates"))


def _sqlite_fallback(cfg: dict) -> str:
    path = resolve_path(cfg, cfg["paths"]["data_dir"]) / "traffic.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def open_with_fallback(url: str, cfg: dict):
    try:
        return open_database(url), url
    except Exception as exc:
        fallback = _sqlite_fallback(cfg)
        print(f"PostgreSQL недоступен ({exc!r}), используется {fallback}")
        return open_database(fallback), fallback


def _chart(series) -> str:
    numbers = []
    for row in series:
        value = row.speed if row.speed is not None else row.volume
        numbers.append(0.0 if value is None else float(value))
    if not numbers:
        return ""
    low, high = min(numbers), max(numbers)
    span = high - low or 1.0
    points = []
    last = max(len(numbers) - 1, 1)
    for index, value in enumerate(numbers):
        x = 8 + 624 * index / last
        y = 168 - 150 * (value - low) / span
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _checkpoint(cfg: dict) -> Path | None:
    path = resolve_path(cfg, cfg["paths"]["checkpoints_dir"]) / "stgcn_gate.pt"
    return path if path.exists() else None


def create_app(database_url: str | None = None, checkpoint_path: str | None = None, npz_path: str | None = None) -> FastAPI:
    cfg = load_config()
    requested = database_url or cfg["database_url"]
    (engine, factory), active_url = open_with_fallback(requested, cfg)
    app = FastAPI(title="Прогноз загруженности")
    app.state.factory = factory
    app.state.engine = engine
    app.state.database_url = active_url
    app.state.cfg = cfg
    app.state.forecast = None
    checkpoint = Path(checkpoint_path) if checkpoint_path else _checkpoint(cfg)
    if checkpoint and checkpoint.exists():
        import torch

        meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
        series = Path(npz_path or meta["npz"])
        if series.exists():
            app.state.forecast = ForecastEngine(checkpoint, series, device="cpu")
            session = factory()
            try:
                repo = Repository(session)
                if not any(segment.source == "benchmark" for segment in repo.list_segments()):
                    publish_benchmark(session, series, meta.get("free_flow"), steps=288)
            finally:
                session.close()

    @app.get("/health")
    def health():
        return {"status": "ok", "database": app.state.database_url, "model": app.state.forecast is not None}

    @app.get("/segments")
    def segments():
        session = factory()
        try:
            return [
                {
                    "id": segment.id,
                    "key": segment.external_key,
                    "name": segment.name,
                    "source": segment.source,
                    "free_flow": segment.free_flow,
                }
                for segment in Repository(session).list_segments()
            ]
        finally:
            session.close()

    @app.get("/observations/{segment_id}")
    def observations(segment_id: int, limit: int = 48):
        session = factory()
        try:
            rows = Repository(session).recent_flows(segment_id, limit=limit)
            return [
                {
                    "ts": row.ts.isoformat(sep=" "),
                    "volume": row.volume,
                    "speed": row.speed,
                    "density": row.density,
                    "source": row.source,
                }
                for row in rows
            ]
        finally:
            session.close()

    @app.post("/videos/process")
    def process(video_path: str = Form(...), camera_name: str = Form("cam_5"), max_frames: int = Form(0)):
        try:
            result = process_video(
                video_path,
                camera_name=camera_name,
                database_url=app.state.database_url,
                max_frames=max_frames or None,
                cfg=cfg,
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {key: value for key, value in result.items() if key != "events"} | {
            "events": len(result["events"]),
            "by_movement": result["minutes"],
        }

    @app.post("/forecast/{segment_id}")
    def forecast(segment_id: int):
        if app.state.forecast is None:
            return JSONResponse({"error": "Модель ещё не обучена"}, status_code=409)
        session = factory()
        try:
            repo = Repository(session)
            segment = repo.get_segment(segment_id)
            if segment is None:
                return JSONResponse({"error": "Участок не найден"}, status_code=404)
            engine_model: ForecastEngine = app.state.forecast
            if segment.source == "video":
                flows = repo.recent_flows(segment.id, limit=engine_model.history)
                volumes = [row.volume or 0.0 for row in flows]
                sensor_id = engine_model.sensor_ids[0]
                rows = engine_model.bridge_volume(volumes or [0.0], sensor_id, segment.free_flow)
                bridge = sensor_id
            else:
                sensor_id = segment.external_key.split(":", 1)[1]
                rows = engine_model.forecast_sensor(sensor_id)
                bridge = None
            saved = []
            for row in rows:
                repo.add_forecast(
                    segment_id=segment.id,
                    issued_at=row["issued_at"],
                    target_ts=row["target_ts"],
                    horizon_min=row["horizon_min"],
                    predicted_speed=row["predicted_speed"],
                    congestion_level=row["congestion_level"],
                    model_name=row["model_name"],
                )
                saved.append(
                    {
                        "horizon_min": row["horizon_min"],
                        "target_ts": row["target_ts"].strftime("%Y-%m-%d %H:%M"),
                        "predicted_speed": round(row["predicted_speed"], 2),
                        "congestion_level": row["congestion_level"],
                        "model_name": row["model_name"],
                        "bridged_sensor": bridge,
                    }
                )
            session.commit()
            return {"segment": segment.name, "forecasts": saved}
        finally:
            session.close()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        session = factory()
        try:
            repo = Repository(session)
            segments = repo.list_segments()
            selected = segments[0] if segments else None
            series = repo.recent_flows(selected.id, limit=72) if selected else []
            jobs = repo.list_jobs()
            return TEMPLATES.TemplateResponse(
                request,
                "index.html",
                {
                    "segments": segments,
                    "selected": selected,
                    "series": series,
                    "chart": _chart(series),
                    "jobs": jobs,
                    "model_ready": app.state.forecast is not None,
                    "database": app.state.database_url,
                    "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                },
            )
        finally:
            session.close()

    return app


app = create_app()
