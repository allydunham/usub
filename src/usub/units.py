"""
Parsing and formatting of memory and time values.
"""

from __future__ import annotations

import math
import re

from .errors import UsubError

# Any number
_NUM = r"(\d+(?:\.\d+)?)"

# Memory formats
_MEM_FACTORS_MB = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
_MEM_RE = re.compile(rf"^\s*{_NUM}\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)

# Accepted time formats and conversions to seconds
_TIME_FORMATS = (
    (
        re.compile(r"^(\d+)$"),
        lambda m: seconds(m=int(m))
    ),
    (
        re.compile(r"^(\d+):([0-5]\d)$"),
        lambda m, s: seconds(m=int(m), s=int(s))
    ),
    (
        re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$"),
        lambda h, m, s: seconds(h=int(h), m=int(m), s=int(s))
    ),
    (
        re.compile(r"^(\d+)-([01]?\d|2[0-3])(?::([0-5]\d)(?::([0-5]\d))?)?$"),
        lambda d, h, m, s: seconds(int(d), int(h), int(m) if m else 0, int(s) if s else 0)
    ),
    (
        re.compile(
            rf"^(?:{_NUM}\s*d)?\s*(?:{_NUM}\s*h)?\s*(?:{_NUM}\s*m)?\s*(?:{_NUM}\s*s)?$", re.IGNORECASE
        ),
        lambda d, h, m, s: seconds(float(d) if d else 0, float(h) if h else 0,
                                   float(m) if m else 0, float(s) if s else 0)
    )
)

def parse_mem(value: str | int | float) -> int:
    """
    Parse a memory amount into megabytes (rounded up).

    Bare numbers are megabytes. Supports K, M, G, T (optionally followed by B
    or iB) which are treated as binary multiples (``"4G"`` -> 4096).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise UsubError(f"Invalid memory value {value!r}")

    mb = value
    if isinstance(value, str):
        m = _MEM_RE.match(value)
        if not m:
            raise UsubError(f"Invalid memory value {value!r} (expected e.g. 500M, 4G)")
        mb = float(m.group(1)) * _MEM_FACTORS_MB[m.group(2).upper()]

    if mb <= 0:  # zero would mean "no limit" to some schedulers
        raise UsubError(f"Memory values must be positive ({value!r})")

    return math.ceil(mb)

def parse_time(value: str | int | float) -> int:
    """
    Parse a duration into seconds (rounded up).

    Accepts:
    - ``M`` (bare minutes, as for LSF ``-W``), as a string or a number
    - ``M:SS``
    - ``H:MM:SS``
    - ``D-H``
    - ``D-H:MM``
    - ``D-H:MM:SS``
    - unit strings (``2h``, ``1h30m``, etc.)
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise UsubError(f"Invalid time value {value!r}")

    if isinstance(value, str):
        v = value.strip()
        if not v:
            raise UsubError("Empty time value")

        for pattern, to_seconds in _TIME_FORMATS:
            if m := pattern.match(v):
                total = to_seconds(*m.groups())
                break
        else:
            raise UsubError(f"Invalid time value {value!r} (expected e.g. 90, 2h, 1h30m, "
                            "1:30:00, 2-12:00:00)")
    else:
        total = math.ceil(value * 60)

    if total <= 0:  # zero would mean "no limit" to some schedulers
        raise UsubError(f"Time values must be positive ({value!r})")

    return total

def seconds(d=0, h=0, m=0, s=0) -> int:
    """
    Convert time duration from multiple units into integer seconds (rounded up)
    """
    return math.ceil(86400 * d + 3600 * h + 60 * m + s)

def minutes(seconds: int) -> int:
    """
    Convert seconds to a whole number of minutes
    """
    return max(1, math.ceil(seconds / 60))
