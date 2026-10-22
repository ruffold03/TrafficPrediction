import numpy as np
import pandas as pd

from traffic_prediction.external.align import build_external_matrix
from traffic_prediction.external.weather import align_weather, clean_weather_frame
from traffic_prediction.forecast.baselines import persistence_predict
from traffic_prediction.forecast.graph import congestion_level, connected_subset, gaussian_adjacency, volume_to_speed
from traffic_prediction.forecast.metrics import masked_metrics


def test_gaussian_adjacency_drops_weak_and_diagonal():
    distance = np.array([[0, 100, 5000], [100, 0, 5000], [5000, 100, 0]], dtype=float)
    weights = gaussian_adjacency(distance, sigma_m=200, epsilon=0.1)
    assert weights[0, 0] == 0
    assert weights[0, 1] > 0
    assert weights[0, 2] == 0


def test_connected_subset_keeps_neighbours():
    adjacency = np.array([[0, 1, 0], [1, 0, 0.2], [0, 0.2, 0]], dtype=float)
    chosen = connected_subset(adjacency, 2)
    assert len(chosen) == 2
    assert set(chosen.tolist()) == {0, 1}


def test_weather_alignment_and_external_width():
    hourly = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2012-03-01 00:00", "2012-03-01 01:00"], utc=True),
            "temperature": [10.0, 11.0],
            "precipitation": [0.0, 5.0],
            "weather_code": [0, 61],
            "wind_speed": [1.0, 100.0],
        }
    )
    cleaned = clean_weather_frame(hourly)
    stamps = pd.date_range("2012-03-01 00:10", periods=3, freq="5min", tz="UTC")
    aligned = align_weather(stamps, cleaned)
    matrix = build_external_matrix(stamps, aligned)
    assert matrix.shape == (3, 9)
    assert np.isclose(matrix[0, 5], aligned["temperature"].iloc[0])


def test_masked_metrics_ignore_zeros():
    report = masked_metrics(np.array([10.0, 99.0]), np.array([12.0, 0.0]))
    assert report["count"] == 1
    assert report["mae"] == 2


def test_persistence_repeats_last_value():
    windows = np.arange(24, dtype=float).reshape(2, 3, 4)
    pred = persistence_predict(windows, [3, 6])
    assert pred.shape == (2, 2, 4)
    assert np.allclose(pred[:, 0], windows[:, -1])


def test_congestion_and_volume_bridge():
    assert congestion_level(50, 60) == "free"
    assert congestion_level(30, 60) == "moderate"
    assert congestion_level(10, 60) == "congested"
    assert volume_to_speed(0, 10, 60) == 60
    assert volume_to_speed(10, 10, 60) == 0
