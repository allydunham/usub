"""
Local backend: run on this machine, arrays in parallel via ``xargs -P``.

Single jobs stream their output. Array tasks log to ``<logdir>/<name>.<index>`` and ``.err``
(one merged file with ``merge_logs``). Resource requests other than ``max_parallel`` are
ignored; ``max_parallel`` defaults to the number of CPUs.
"""

from __future__ import annotations

import os
import shlex

from ..errors import SubmitError, UsubError
from ..job import Job, Plan
from .base import Backend, Directive, Rendered

class Local(Backend):
    """Run with ``bash`` on this machine."""

    name = "local"
    submit_program = "bash"
    index_var = "1"  # tasks are run as ``bash -c 'command' _ INDEX``

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        """
        No scheduler directives locally
        """
        return []

    def render_script(self, job: Job, plan: Plan, profile: str | None) -> str:
        """
        Bash script running the command, or an array as ``seq | xargs -P`` with a log per task
        """
        if job.after:
            raise UsubError("The local backend does not support job dependencies (--after)")
        lines = ["#!/bin/bash", self.header_comment(profile), "", f"cd {shlex.quote(os.fspath(job.cwd))}",
                 *job.setup]
        if not plan.is_array:
            return "\n".join(lines + [plan.command]) + "\n"

        log = shlex.quote(os.path.join(job.logdir, job.name))
        redirect = lambda i: f"exec > {log}.{i} " + ("2>&1" if job.merge_logs else f"2> {log}.{i}.err")
        idx = plan.indices
        script = f"{redirect('${1}')}; {self.fill_index(plan.command)}"
        lines += [f"mkdir -p {shlex.quote(os.fspath(job.logdir))}",
                  f"seq {idx.start} {idx.step} {idx.end} | "
                  f"xargs -n 1 -P {job.max_parallel or os.cpu_count() or 1} bash -c {shlex.quote(script)} _"]
        return "\n".join(lines) + "\n"

    def oneliner(self, rendered: Rendered) -> str:
        """
        ``bash -c`` for single jobs, ``seq | xargs -P`` for index arrays (output not redirected)
        """
        job, plan = rendered.job, rendered.plan
        if job.after:
            raise UsubError("The local backend does not support job dependencies (--after)")
        command = "; ".join([*job.setup, self.fill_index(plan.command) if plan.is_array else plan.command])
        if not plan.is_array:
            return f"bash -c {shlex.quote(command)}"
        idx = plan.indices
        return (f"seq {idx.start} {idx.step} {idx.end} | "
                f"xargs -n 1 -P {job.max_parallel or os.cpu_count() or 1} bash -c {shlex.quote(command)} _")

    def submit_hint(self, path: str | None) -> str:
        """
        The shell command that runs the script at ``path``
        """
        return f"bash {path or 'SCRIPT'}"

    def submit(self, rendered: Rendered) -> str:
        """
        Save the script as ``<name>.script`` and run it in the foreground

        Local jobs have no scheduler id, so the id returned is the process id of the run.
        """
        import subprocess

        rendered.prepare()
        path = rendered.save(force=True)
        proc = subprocess.Popen(["bash", path])
        if (rc := proc.wait()) != 0:
            raise SubmitError(f"Local job failed (exit {rc}); script: {path}")
        return str(proc.pid)
