"""
IBM Spectrum LSF backend (``bsub``).

Options (from the config profile):
  lsf_mem_unit      unit of -M/rusage values: KB, MB (default) or GB (site LSF_UNIT_FOR_LIMITS)
  lsf_rusage        add -R "select[mem>X] rusage[mem=X]" (default true; needed on the Sanger farm)
  lsf_mem_per_core  divide the memory reservation by cores (default false)
  gpu_args          raw directive(s) for GPUs, with {gpus} for the -g value; the default is
                    -gpu "num=N" for a count N, or -gpu SPEC for any other -g value
"""

from __future__ import annotations

import math
import re

from ..errors import UsubError
from ..job import Job, Plan
from ..units import minutes, parse_mem, parse_time
from .base import Backend, Directive

_UNIT_PER_MB = {"KB": 1024, "MB": 1, "GB": 1 / 1024}

class LSF(Backend):
    """Submit with ``bsub``, feeding the script on stdin so ``#BSUB`` lines are parsed."""

    name = "lsf"
    directive_prefix = "#BSUB"
    index_var = "LSB_JOBINDEX"
    submit_program = "bsub"
    oneliner_wrap = ""
    job_id_re = re.compile(r"Job <(\d+)>")
    option_names = frozenset({"lsf_mem_unit", "lsf_rusage", "lsf_mem_per_core", "gpu_args"})

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        """
        ``#BSUB`` options: name/array, logs, queue, cores, memory, time, GPUs, dependencies
        """
        if plan.is_array:
            limit = f"%{job.max_parallel}" if job.max_parallel else ""
            d: list[Directive] = [["-J", f"{job.name}[{plan.indices.spec()}]{limit}"]]
            idx = "%I"
        else:
            d = [["-J", job.name]]
            idx = None

        d.append(["-o", self.log_path(job, "%J", idx)])

        if not job.merge_logs:  # without -e, LSF writes stderr to the -o file
            d.append(["-e", self.log_path(job, "%J", idx, err=True)])

        if job.queue:
            d.append(["-q", job.queue])

        cores = int(job.cores or 1)
        if cores > 1:
            d.append(["-n", str(cores)])
            d.append(["-R", "span[hosts=1]"])

        if job.mem:
            unit = str(self.options.get("lsf_mem_unit", "MB")).upper()
            if unit not in _UNIT_PER_MB:
                raise UsubError(f"lsf_mem_unit must be one of KB, MB, GB (got {unit!r})")
            mem = max(1, math.ceil(parse_mem(job.mem) * _UNIT_PER_MB[unit]))
            d.append(["-M", str(mem)])
            if self.options.get("lsf_rusage", True):
                reserve = math.ceil(mem / cores) if self.options.get("lsf_mem_per_core") else mem
                d.append(["-R", f"select[mem>{reserve}] rusage[mem={reserve}]"])

        if job.time:
            d.append(["-W", str(minutes(parse_time(job.time)))])

        gpus = self.gpu_directives(job)
        if gpus is None:
            gpus = [["-gpu", f"num={job.gpus}" if str(job.gpus).isdigit() else str(job.gpus)]]
        d += gpus

        if job.after:
            d.append(["-w", " && ".join(f"done({a})" for a in job.after)])

        d += job.extra
        return d
