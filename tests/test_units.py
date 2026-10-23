import pytest

from usub import UsubError
from usub.units import minutes, parse_mem, parse_time

@pytest.mark.parametrize("value, mb", [
    ("4G", 4096), ("4GB", 4096), ("4GiB", 4096), ("500M", 500), ("500", 500), (500, 500),
    ("1.5g", 1536), ("2048K", 2), ("1T", 1024 * 1024), ("100K", 1),
])
def test_parse_mem(value, mb):
    assert parse_mem(value) == mb

@pytest.mark.parametrize("value", ["", "4X", "G", "-1G", "4 G B", 0, -1, True, None, [4], "0", "0G", "0.0M"])
def test_parse_mem_invalid(value):
    with pytest.raises(UsubError):
        parse_mem(value)

@pytest.mark.parametrize("value, seconds", [
    # Slurm --time formats
    ("90", 5400), (" 90 ", 5400), ("1:30", 90), ("90:00", 5400), ("1:30:15", 5415),
    ("36:00:00", 129600), ("2-12", 216000), ("2-12:30", 217800), ("2-12:00:00", 216000),
    ("0-1:00:00", 3600),
    # unit strings
    ("2h", 7200), ("1h30m", 5400), ("1h 30m", 5400), ("1H30M", 5400), ("1d", 86400),
    ("1d 12h", 129600), ("45s", 45), ("1.5h", 5400), ("0.5m", 30),
    # numbers are minutes
    (90, 5400), (1.5, 90),
])
def test_parse_time(value, seconds):
    assert parse_time(value) == seconds

@pytest.mark.parametrize("value", [
    "", "abc", "1h30", "1:2:3:4", "x-1:00", "2hours", "2-", "-5", "1:90", "1:2", "1:60:00",
    "2-24", "2-12:60", "1-2h", "30m1h", "1.5:00", "h", -1, True, None,
    0, "0", "0:00", "0s", "0-0", "0h",  # zero means "no limit" to some schedulers, so is refused
])
def test_parse_time_invalid(value):
    with pytest.raises(UsubError):
        parse_time(value)

def test_formatting():
    assert minutes(30) == 1
    assert minutes(5415) == 91
