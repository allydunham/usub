"""
usub: unified job submission for LSF, Slurm and local parallel execution.

    from usub import Job, render, submit
    job = Job(cmds="commands.txt", mem="2G", max_parallel=50)
    print(render(job, profile="farm").script)
    job_id = submit(job, profile="farm")
"""

__version__ = "0.1.0"

from .backends.base import Rendered, render, submit
from .config import Config, Profile, load_config, resolve
from .errors import SubmitError, UsubError, UsubWarning
from .job import IndexRange, Job

__all__ = ["Config", "IndexRange", "Job", "Profile", "Rendered", "SubmitError", "UsubError", "UsubWarning",
           "load_config", "render", "resolve", "submit", "__version__"]
