"""
Slurm backend (``sbatch``).

Options (from the config profile):
  gpu_args  raw directive(s) for GPUs, with {gpus} for the -g value; the default is
            --gres=gpu:{gpus}, so -g 2 or -g a100:2 both work
"""

from __future__ import annotations

import re

from ..job import Job, Plan
from ..units import parse_mem, parse_time
from .base import Backend, Directive

class Slurm(Backend):
    """Submit with ``sbatch`` (script on stdin); the one-line form uses ``--wrap``."""

    name = "slurm"
    directive_prefix = "#SBATCH"
    index_var = "SLURM_ARRAY_TASK_ID"
    submit_program = "sbatch"
    oneliner_wrap = "--wrap="
    job_id_re = re.compile(r"Submitted batch job (\d+)")
    option_names = frozenset({"gpu_args"})

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        """
        ``#SBATCH`` options: name, logs, array, partition, cpus, memory, time, GPUs, dependencies
        """
        job_token, idx = ("%A", "%a") if plan.is_array else ("%j", None)
        d: list[Directive] = [[f"--job-name={job.name}"],
                              [f"--output={self.log_path(job, job_token, idx)}"]]

        if not job.merge_logs:  # without --error, Slurm writes stderr to the --output file
            d.append([f"--error={self.log_path(job, job_token, idx, err=True)}"])

        if plan.is_array:
            limit = f"%{job.max_parallel}" if job.max_parallel else ""
            d.append([f"--array={plan.indices.spec()}{limit}"])

        if job.queue:
            d.append([f"--partition={job.queue}"])

        d += [["--nodes=1"], ["--ntasks=1"], [f"--cpus-per-task={int(job.cores or 1)}"]]

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

def slurm_time(seconds: int) -> str:
    """
    Convert seconds into a Slurm time string
    """
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
