from __future__ import annotations

from traffic_prediction.storage.models import init_db, make_engine, make_session_factory


def open_database(url: str):
    engine = make_engine(url)
    init_db(engine)
    return engine, make_session_factory(engine)
