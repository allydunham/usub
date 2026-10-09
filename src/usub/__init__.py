"""usub: unified job submission for LSF, Slurm and local parallel execution.

    from usub import Job, render, submit
    job = Job("samtools index {}", files=paths, mem="2G", max_parallel=50)
    print(render(job, profile="farm").script)
    job_id = submit(job, profile="farm")
"""

from __future__ import annotations

__version__ = "0.1.0"

from typing import TYPE_CHECKING

from .config import Config, load_config, resolve
from .errors import SubmitError, UsubError
from .job import IndexRange, Job

if TYPE_CHECKING:
    from .backends.base import Rendered

__all__ = ["Config", "IndexRange", "Job", "SubmitError", "UsubError", "load_config",
           "render", "resolve", "submit", "__version__"]


def render(job: Job, *, profile: str | None = None, backend: str | None = None,
           config: Config | None = None) -> Rendered:
    """Resolve ``job`` against config and render it for the selected backend."""
    from .backends.base import get_backend

    r = resolve(job, profile=profile, backend=backend, config=config)
    return get_backend(r.backend, r.options).render(r.job, r.profile)


def submit(job: Job, *, profile: str | None = None, backend: str | None = None,
           config: Config | None = None) -> str:
    """Render and submit ``job``; returns the scheduler job id."""
    return render(job, profile=profile, backend=backend, config=config).submit()
