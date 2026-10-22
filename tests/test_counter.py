from traffic_prediction.video.counter import events_from_tracks, movement_id, point_in_polygon
from traffic_prediction.video.evaluate import counting_errors
from traffic_prediction.video.counter import CountEvent


APPROACHES = [
    {"angle": 180, "right": 1, "through": 2, "left": 3},
    {"angle": -90, "right": 4, "through": 5, "left": 6},
    {"angle": 0, "right": 7, "through": 8, "left": 9},
    {"angle": 90, "right": 10, "through": 11, "left": 12},
]
POLYGON = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon():
    assert point_in_polygon(5, 5, POLYGON)
    assert not point_in_polygon(-1, 5, POLYGON)


def test_through_movement_from_the_left():
    move = movement_id((1, 5), (9, 5), (5, 5), APPROACHES, 35)
    assert move == 2


def test_track_is_counted_once_on_exit():
    tracks = {
        7: [{"frame": frame, "cx": x, "cy": 5, "coco_class": 2} for frame, x in enumerate([-2, 1, 3, 5, 7, 9, 12])]
    }
    events = events_from_tracks(tracks, POLYGON, APPROACHES, min_inside_frames=3)
    assert len(events) == 1
    assert events[0].movement_id == 2
    assert events[0].class_id == 1
    assert events[0].track_id == 7


def test_counting_mae_against_manual_rows():
    predicted = [CountEvent(1, 2, 1, 1), CountEvent(2, 2, 1, 2)]
    truth = [{"movement_id": 2, "class_id": 1}, {"movement_id": 1, "class_id": 1}]
    report = counting_errors(predicted, truth)
    assert report["predicted_total"] == 2
    assert report["ground_truth_total"] == 2
    assert report["per_movement"][1]["error"] == 1
