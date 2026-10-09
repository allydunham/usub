"""Slurm backend (``sbatch``).

Options (from the config profile):
  gpu_args  raw directive(s) for GPUs, with {gpus}; default --gres=gpu:{gpus}
"""

from __future__ import annotations

import re
import shlex

from ..job import Job, Plan
from ..units import parse_mem, parse_time, slurm_time
from .base import Backend, Directive


class Slurm(Backend):
    name = "slurm"
    directive_prefix = "#SBATCH"
    index_var = "SLURM_ARRAY_TASK_ID"
    submit_program = "sbatch"
    job_id_re = re.compile(r"Submitted batch job (\d+)")

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        if plan.is_array:
            job_token, idx = "%A", "%a"
        else:
            job_token, idx = "%j", None
        d: list[Directive] = [
            [f"--job-name={job.name}"],
            [f"--output={self.log_path(job, '.out', job_token, idx)}"],
            [f"--error={self.log_path(job, '.err', job_token, idx)}"],
        ]
        if plan.is_array:
            limit = f"%{job.max_parallel}" if job.max_parallel else ""
            d.append([f"--array={plan.indices.spec()}{limit}"])
        if job.queue:
            d.append([f"--partition={job.queue}"])
        d.append(["--nodes=1"])
        d.append(["--ntasks=1"])
        d.append([f"--cpus-per-task={int(job.cores or 1)}"])
        if job.mem:
            mb = parse_mem(job.mem)
            d.append([f"--mem={mb // 1024}G" if mb % 1024 == 0 else f"--mem={mb}M"])
        if job.time:
            d.append([f"--time={slurm_time(parse_time(job.time))}"])
        gpus = self.gpu_directives(job)
        d += gpus if gpus is not None else [[f"--gres=gpu:{job.gpus}"]]
        if job.after:
            d.append([f"--dependency=afterok:{':'.join(job.after)}"])
        d += job.extra
        return d

    def oneliner_argv(self, job: Job, plan: Plan, command: str) -> str:
        parts = ["sbatch"]
        for d in self.directives(job, plan):
            parts.append(d if isinstance(d, str) else shlex.join(d))
        parts.append(shlex.quote(f"--wrap={command}"))
        return " ".join(parts)
