import pytest

from openadobe.timebase import Rate, parse_time

R30 = Rate(num=30, den=1)
NTSC = Rate(num=30000, den=1001)


@pytest.mark.parametrize("value,frames", [
    (45, 45), ("45", 45), ("1.5s", 45), ("500ms", 15), ("00:00:01:15", 45), ("00:01.500", 45), ("12f", 12),
])
def test_parse_time(value, frames):
    assert parse_time(value, R30) == frames


def test_parse_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_time("soon", R30)
    with pytest.raises(ValueError):
        parse_time(1.5, R30)


def test_rate_parse_and_ntsc():
    assert Rate.parse("29.97") == NTSC
    assert Rate.parse("30000/1001") == NTSC
    assert Rate.parse("25") == Rate(num=25, den=1)
    assert NTSC.to_frames(10.0) == 300 - 0  # round(10 * 29.97) = 300
    assert NTSC.timecode(30) == "00:00:01:00"
    assert R30.clock(45) == "00:00:01.500"
