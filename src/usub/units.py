"""Parsing and formatting of memory and time values."""

from __future__ import annotations

import math
import re

from .errors import UsubError

_MEM_FACTORS_MB = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
_TIME_UNITS_S = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_TIME_UNITS_RE = re.compile(r"(\d+(?:\.\d+)?)([dhms])", re.IGNORECASE)


def parse_mem(value: str | int | float) -> int:
    """Parse a memory amount into whole megabytes (rounded up).

    Bare numbers are megabytes. Units K, M, G, T (optionally followed by B or iB)
    are binary multiples, e.g. ``"4G"`` -> 4096.
    """
    if isinstance(value, (int, float)):
        return math.ceil(value)
    m = _MEM_RE.match(value)
    if not m:
        raise UsubError(f"invalid memory value {value!r} (expected e.g. 500M, 4G)")
    return math.ceil(float(m.group(1)) * _MEM_FACTORS_MB[m.group(2).upper()])


def parse_time(value: str | int) -> int:
    """Parse a duration into seconds.

    Accepts bare minutes (``90``), unit strings (``2h``, ``1h30m``, ``1d``),
    ``H:MM``, ``H:MM:SS`` and Slurm-style ``D-HH:MM:SS``.
    """
    if isinstance(value, int):
        return value * 60
    v = value.strip()
    if v.isdigit():
        return int(v) * 60
    days = 0
    if "-" in v:
        d, _, v = v.partition("-")
        if not d.isdigit():
            raise UsubError(f"invalid time value {value!r}")
        days = int(d)
    if ":" in v:
        parts = v.split(":")
        if len(parts) > 3 or not all(p.isdigit() for p in parts):
            raise UsubError(f"invalid time value {value!r}")
        nums = [int(p) for p in parts] + [0] * (3 - len(parts))
        h, m, s = nums
        return days * 86400 + h * 3600 + m * 60 + s
    if days and not v:
        return days * 86400
    matches = _TIME_UNITS_RE.findall(v)
    if not matches or "".join(n + u for n, u in matches).lower() != v.lower():
        raise UsubError(f"invalid time value {value!r} (expected e.g. 90, 2h, 1h30m, 1:30:00)")
    return days * 86400 + math.ceil(sum(float(n) * _TIME_UNITS_S[u.lower()] for n, u in matches))


def minutes(seconds: int) -> int:
    return max(1, math.ceil(seconds / 60))


def slurm_time(seconds: int) -> str:
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
