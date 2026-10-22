from datetime import datetime

from traffic_prediction.external.calendar import calendar_features, is_holiday


def test_memorial_day_2012_is_a_holiday():
    assert is_holiday(datetime(2012, 5, 28, 8, 0))
    assert not is_holiday(datetime(2012, 5, 29, 8, 0))


def test_calendar_features_are_bounded():
    features = calendar_features(datetime(2012, 3, 5, 15, 30))
    assert -1.0 <= features["hour_sin"] <= 1.0
    assert features["day_of_week"] == 0
    assert features["is_holiday"] == 0.0
