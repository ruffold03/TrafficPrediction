CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    roi_path TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id SERIAL PRIMARY KEY,
    external_key TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    camera_id INTEGER REFERENCES cameras (id),
    free_flow DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES segments (id),
    target_id INTEGER NOT NULL REFERENCES segments (id),
    weight DOUBLE PRECISION NOT NULL DEFAULT 1,
    UNIQUE (source_id, target_id)
);

CREATE TABLE IF NOT EXISTS flow_observations (
    id BIGSERIAL PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments (id),
    ts TIMESTAMPTZ NOT NULL,
    volume DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    density DOUBLE PRECISION,
    source TEXT NOT NULL,
    UNIQUE (segment_id, ts, source)
);

CREATE INDEX IF NOT EXISTS ix_flow_segment_ts ON flow_observations (segment_id, ts);

CREATE TABLE IF NOT EXISTS external_observations (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    region TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    weather_code INTEGER,
    wind_speed DOUBLE PRECISION,
    hour INTEGER,
    day_of_week INTEGER,
    is_holiday BOOLEAN,
    UNIQUE (ts, region)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id BIGSERIAL PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments (id),
    issued_at TIMESTAMPTZ NOT NULL,
    target_ts TIMESTAMPTZ NOT NULL,
    horizon_min INTEGER NOT NULL,
    predicted_volume DOUBLE PRECISION,
    predicted_speed DOUBLE PRECISION,
    congestion_level TEXT,
    model_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_jobs (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras (id),
    video_path TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    frames INTEGER,
    counted_vehicles INTEGER,
    error TEXT
);
