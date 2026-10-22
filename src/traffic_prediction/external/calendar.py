from __future__ import annotations

import math
from datetime import date, datetime, timedelta


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _observe(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def us_federal_holidays(year: int) -> set[date]:
    holidays = {
        _observe(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observe(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observe(date(year, 11, 11)),
        _nth_weekday(year, 11, 3, 4),
        _observe(date(year, 12, 25)),
    }
    if year >= 2021:
        holidays.add(_observe(date(year, 6, 19)))
    return holidays


def is_holiday(moment: datetime, region: str = "US") -> bool:
    if region != "US":
        raise ValueError(f"Unsupported holiday region: {region}")
    return moment.date() in us_federal_holidays(moment.year)


def calendar_features(moment: datetime, region: str = "US") -> dict:
    hour = moment.hour + moment.minute / 60.0
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * moment.weekday() / 7.0),
        "dow_cos": math.cos(2 * math.pi * moment.weekday() / 7.0),
        "is_holiday": 1.0 if is_holiday(moment, region) else 0.0,
        "hour": moment.hour,
        "day_of_week": moment.weekday(),
    }


FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "temperature",
    "precipitation",
    "wind_speed",
    "weather_code",
)
