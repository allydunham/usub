"""
End-to-end speed checks: time the real ``usub`` command, including interpreter start-up.

Each case is compared with a bare ``python -c pass`` measured in the same session, so the
budget is the overhead usub adds and does not depend on machine speed. Run on their own
with ``uv run pytest -m benchmark -s`` to see the timings; skip with ``-m "not benchmark"``.
"""

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from usub.config import TEMPLATE

pytestmark = pytest.mark.benchmark

RUNS = 15
BUDGET_MS = 60  # allowed overhead over bare python; currently about 33 ms
USUB = Path(sys.executable).parent / "usub"

def median_ms(argv, cwd) -> float:
    """
    Median wall-clock time of running ``argv`` (after one warm-up), in milliseconds
    """
    times = []
    for _ in range(RUNS + 1):
        start = time.perf_counter()
        subprocess.run(argv, cwd=cwd, env=os.environ, stdout=subprocess.DEVNULL, check=True)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times[1:])

@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """Median start-up time of a bare Python interpreter."""
    return median_ms([sys.executable, "-c", "pass"], tmp_path_factory.mktemp("baseline"))

@pytest.mark.skipif(not USUB.exists(), reason="usub command is not installed in this environment")
@pytest.mark.parametrize("name, args", [
    ("single job", ["r", "-b", "lsf", "-m", "4G", "-t", "2h", "--", "python", "x.py"]),
    ("index array", ["r", "-b", "slurm", "-a", "1-1000", "-j", "50", "--", "echo", "{i}"]),
    ("2000 command file", ["r", "-b", "lsf", "-C", "commands.txt"]),
    ("2000 file list", ["r", "-b", "lsf", "-F", "files.txt", "--", "gzip", "{}"]),
    ("local array", ["r", "-b", "local", "-a", "1-1000", "--", "echo", "{i}"]),
    ("with user + project config", ["r", "--", "python", "x.py"]),
])
def test_overhead_over_bare_python(name, args, baseline, tmp_path, capsys):
    (tmp_path / "files.txt").write_text("".join(f"data/sample_{i}.fastq\n" for i in range(2000)))
    (tmp_path / "commands.txt").write_text("".join(f"gzip data/sample_{i}.fastq\n" for i in range(2000)))
    if name.startswith("with"):
        (tmp_path / "home" / ".usub.toml").write_text(TEMPLATE)  # USUB_CONFIG points here (conftest)
        (tmp_path / ".usub.toml").write_text('[defaults]\nmem = "4G"\n[profiles.farm]\nqueue = "long"\n')
    overhead = median_ms([str(USUB), *args], tmp_path) - baseline
    with capsys.disabled():
        print(f"\n{name}: +{overhead:.0f} ms over bare python ({baseline:.0f} ms), budget {BUDGET_MS} ms")
    assert overhead < BUDGET_MS, f"{name} adds {overhead:.0f} ms over bare python (budget {BUDGET_MS} ms)"
