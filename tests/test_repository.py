from datetime import datetime

from traffic_prediction.storage import open_database
from traffic_prediction.storage.repository import Repository


def test_repository_roundtrip(tmp_path):
    _engine, factory = open_database(f"sqlite:///{tmp_path / 't.db'}")
    session = factory()
    repo = Repository(session)
    camera = repo.upsert_camera("cam_5", "roi.txt")
    segment = repo.upsert_segment("video:cam_5", "cam_5", "video", camera_id=camera.id, free_flow=12)
    moment = datetime(2018, 4, 30, 14, 29)
    repo.add_flow(segment.id, moment, "video", volume=8, density=8)
    repo.add_external(moment, "los-angeles", temperature=18.0, precipitation=0.0, weather_code=1, wind_speed=3.0, hour=14, day_of_week=0, is_holiday=False)
    job = repo.start_video_job(camera.id, "clip.mp4")
    repo.finish_video_job(job, "done", frames=10, counted_vehicles=2)
    repo.add_forecast(
        segment_id=segment.id,
        issued_at=moment,
        target_ts=moment,
        horizon_min=15,
        predicted_speed=40,
        congestion_level="moderate",
        model_name="stgcn:gate",
    )
    session.commit()
    again = factory()
    stored = Repository(again)
    assert stored.get_segment_by_key("video:cam_5").free_flow == 12
    assert stored.recent_flows(segment.id)[0].volume == 8
    assert stored.list_jobs()[0].status == "done"
    session.close()
    again.close()
