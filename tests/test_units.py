import pytest

from usub import UsubError
from usub.units import minutes, parse_mem, parse_time, slurm_time


@pytest.mark.parametrize("value, mb", [
    ("4G", 4096), ("4GB", 4096), ("4GiB", 4096), ("500M", 500), ("500", 500), (500, 500),
    ("1.5g", 1536), ("2048K", 2), ("1T", 1024 * 1024), ("100K", 1),
])
def test_parse_mem(value, mb):
    assert parse_mem(value) == mb


@pytest.mark.parametrize("value", ["", "4X", "G", "-1G", "4 G B"])
def test_parse_mem_invalid(value):
    with pytest.raises(UsubError):
        parse_mem(value)


@pytest.mark.parametrize("value, seconds", [
    ("90", 5400), (90, 5400), ("2h", 7200), ("1h30m", 5400), ("1d", 86400), ("45s", 45),
    ("1:30", 5400), ("1:30:15", 5415), ("2-12:00:00", 216000), ("1.5h", 5400), ("2-", 172800),
])
def test_parse_time(value, seconds):
    assert parse_time(value) == seconds


@pytest.mark.parametrize("value", ["", "abc", "1h30", "1:2:3:4", "x-1:00", "2hours"])
def test_parse_time_invalid(value):
    with pytest.raises(UsubError):
        parse_time(value)


def test_formatting():
    assert minutes(30) == 1
    assert minutes(5415) == 91
    assert slurm_time(5415) == "01:30:15"
    assert slurm_time(216000) == "2-12:00:00"
