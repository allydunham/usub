"""IBM Spectrum LSF backend (``bsub``).

Options (from the config profile):
  lsf_mem_unit      unit of -M/rusage values: KB, MB (default) or GB (site LSF_UNIT_FOR_LIMITS)
  lsf_rusage        add -R "select[mem>X] rusage[mem=X]" (default true; needed on the Sanger farm)
  lsf_mem_per_core  divide the memory reservation by cores (default false)
  gpu_args          raw directive(s) for GPUs, with {gpus}; default -gpu "num={gpus}"
"""

from __future__ import annotations

import math
import re
import shlex

from ..errors import UsubError
from ..job import Job, Plan
from ..units import minutes, parse_mem, parse_time
from .base import Backend, Directive

_UNIT_PER_MB = {"KB": 1024, "MB": 1, "GB": 1 / 1024}


class LSF(Backend):
    name = "lsf"
    directive_prefix = "#BSUB"
    index_var = "LSB_JOBINDEX"
    submit_program = "bsub"
    job_id_re = re.compile(r"Job <(\d+)>")

    def submit_argv(self, script_path: str) -> tuple[list[str], bool]:
        return ["bsub"], True  # bsub only parses #BSUB lines when the script is on stdin

    def mem_value(self, job: Job) -> int:
        unit = str(self.options.get("lsf_mem_unit", "MB")).upper()
        if unit not in _UNIT_PER_MB:
            raise UsubError(f"lsf_mem_unit must be one of KB, MB, GB (got {unit!r})")
        return max(1, math.ceil(parse_mem(job.mem) * _UNIT_PER_MB[unit]))

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        d: list[Directive] = []
        if plan.is_array:
            spec = plan.indices.spec()
            limit = f"%{job.max_parallel}" if job.max_parallel else ""
            d.append(["-J", f"{job.name}[{spec}]{limit}"])
            idx = "%I"
        else:
            d.append(["-J", job.name])
            idx = None
        d.append(["-o", self.log_path(job, ".out", "%J", idx)])
        d.append(["-e", self.log_path(job, ".err", "%J", idx)])
        if job.queue:
            d.append(["-q", job.queue])
        cores = int(job.cores or 1)
        if cores > 1:
            d.append(["-n", str(cores)])
            d.append(["-R", "span[hosts=1]"])
        if job.mem:
            mem = self.mem_value(job)
            d.append(["-M", str(mem)])
            if self.options.get("lsf_rusage", True):
                reserve = math.ceil(mem / cores) if self.options.get("lsf_mem_per_core") else mem
                d.append(["-R", f"select[mem>{reserve}] rusage[mem={reserve}]"])
        if job.time:
            d.append(["-W", str(minutes(parse_time(job.time)))])
        gpus = self.gpu_directives(job)
        d += gpus if gpus is not None else [["-gpu", f"num={job.gpus}"]]
        if job.after:
            d.append(["-w", " && ".join(f"done({a})" for a in job.after)])
        d += job.extra
        return d

    def oneliner_argv(self, job: Job, plan: Plan, command: str) -> str:
        parts = ["bsub"]
        for d in self.directives(job, plan):
            parts.append(d if isinstance(d, str) else shlex.join(d))
        parts.append(shlex.quote(command))
        return " ".join(parts)
