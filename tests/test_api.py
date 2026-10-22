import numpy as np
import torch
from fastapi.testclient import TestClient

from traffic_prediction.api.main import create_app
from traffic_prediction.forecast.model import STGCN
from traffic_prediction.forecast.predict import ForecastEngine, time_inference
from traffic_prediction.storage import open_database
from traffic_prediction.storage.repository import Repository
from traffic_prediction.video.counter import events_from_tracks


def _bundle(path, nodes=4, steps=36):
    values = np.full((steps, nodes), 40, dtype=np.float32)
    values[:, 0] = np.linspace(20, 55, steps, dtype=np.float32)
    timestamps = (np.arange(steps) * 300_000_000_000 + 1_330_560_000_000_000_000).astype(np.int64)
    adjacency = np.eye(nodes, dtype=np.float32) * 0
    adjacency[0, 1] = adjacency[1, 0] = 1
    external = np.zeros((steps, 9), dtype=np.float32)
    external[:, 0] = 0.5
    np.savez(
        path,
        values=values,
        timestamps_ns=timestamps,
        adjacency=adjacency,
        sensor_ids=np.array([str(i) for i in range(nodes)]),
        latitude=np.zeros(nodes, dtype=np.float32),
        longitude=np.zeros(nodes, dtype=np.float32),
        external=external,
    )


def _checkpoint(path, nodes=4):
    model = STGCN(hidden=8, horizon_steps=3, external_dim=9, fusion="gate", dropout=0.0)
    meta = {
        "model_name": "stgcn",
        "fusion": "gate",
        "external_dim": 9,
        "forecast": {
            "history": 12,
            "horizons": [3, 6, 12],
            "step_minutes": 5,
            "hidden": 8,
            "dropout": 0.0,
            "congestion_free": 0.7,
            "congestion_moderate": 0.4,
        },
        "mean": 40.0,
        "std": 10.0,
        "weather_mean": [0, 0, 0, 0],
        "weather_std": [1, 1, 1, 1],
        "free_flow": [60.0] * nodes,
        "adjacency": np.eye(nodes).tolist(),
        "sensor_ids": [str(i) for i in range(nodes)],
        "npz": "",
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return model


def test_model_forward_and_inference_time():
    model = STGCN(8, 3, 9, "gate", 0.0)
    prediction = model(torch.randn(2, 12, 4), torch.eye(4), torch.randn(2, 9))
    assert prediction.shape == (2, 3, 4)
    timing = time_inference(model, batches=3, nodes=4, history=12)
    assert timing["ms_per_batch"] > 0


def test_video_stub_database_and_forecast(tmp_path):
    npz = tmp_path / "series.npz"
    checkpoint = tmp_path / "model.pt"
    _bundle(npz)
    _checkpoint(checkpoint)
    engine = ForecastEngine(checkpoint, npz, device="cpu")
    rows = engine.bridge_volume([2, 4, 9], "0", volume_cap=10)
    assert [row["horizon_min"] for row in rows] == [15, 30, 60]
    assert rows[0]["congestion_level"] in {"free", "moderate", "congested"}

    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    approaches = [
        {"angle": 180, "right": 1, "through": 2, "left": 3},
        {"angle": -90, "right": 4, "through": 5, "left": 6},
        {"angle": 0, "right": 7, "through": 8, "left": 9},
        {"angle": 90, "right": 10, "through": 11, "left": 12},
    ]
    tracks = {1: [{"frame": i, "cx": x, "cy": 5, "coco_class": 2} for i, x in enumerate([-1, 2, 4, 6, 8, 12])]}
    events = events_from_tracks(tracks, polygon, approaches, min_inside_frames=3)
    _db, factory = open_database(f"sqlite:///{tmp_path / 'flow.db'}")
    session = factory()
    repo = Repository(session)
    camera = repo.upsert_camera("cam_5")
    segment = repo.upsert_segment("video:cam_5", "cam_5", "video", camera_id=camera.id, free_flow=3)
    repo.add_flow(segment.id, rows[0]["issued_at"], "video", volume=float(len(events)))
    session.commit()
    assert repo.recent_flows(segment.id)[0].volume == 1
    session.close()


def test_api_page_and_forecast(tmp_path):
    npz = tmp_path / "series.npz"
    checkpoint = tmp_path / "model.pt"
    _bundle(npz)
    _checkpoint(checkpoint)
    app = create_app(
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        checkpoint_path=str(checkpoint),
        npz_path=str(npz),
    )
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Прогноз загруженности" in page.text
    assert "Датчик 0" in page.text
    segments = client.get("/segments").json()
    response = client.post(f"/forecast/{segments[0]['id']}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecasts"]) == 3
    assert body["forecasts"][0]["horizon_min"] == 15
    health = client.get("/health").json()
    assert health["model"] is True
