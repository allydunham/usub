"""Local backend: run on this machine, arrays in parallel via ``xargs -P``.

Single jobs stream their output; array tasks log to ``<logdir>/<name>.<stamp>.<pid>.<index>.out|err``.
Resource requests other than ``max_parallel`` are ignored. ``max_parallel`` defaults
to the number of CPUs.
"""

from __future__ import annotations

import os
import shlex

from ..errors import SubmitError, UsubError
from ..job import INDEX_VAR, Job, Plan
from .base import _TASKS_DELIM, Backend, Directive, Rendered


class Local(Backend):
    name = "local"
    submit_program = "bash"

    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        return []

    def parallelism(self, job: Job) -> int:
        return int(job.max_parallel or os.cpu_count() or 1)

    def _check(self, job: Job) -> None:
        if job.after:
            raise UsubError("the local backend does not support job dependencies (--after)")

    def render_script(self, job: Job, plan: Plan, manifest_path: str | None,
                      profile: str | None) -> str:
        self._check(job)
        lines = ["#!/bin/bash", self.header_comment(profile), ""]
        lines += [f"cd {shlex.quote(os.fspath(job.cwd))}", *job.setup]
        if not plan.is_array:
            lines.append(plan.command)
            return "\n".join(lines) + "\n"

        idx = plan.indices
        logdir = shlex.quote(os.fspath(job.logdir))
        lines += [
            f"usub_logdir={logdir}",
            f'usub_run="{job.name}.$(date +%Y%m%d-%H%M%S).$$"',
            'mkdir -p "$usub_logdir"',
        ]
        if plan.tasks is not None:
            if manifest_path:
                lines.append(f"usub_tasks={shlex.quote(manifest_path)}")
            else:
                lines += [
                    'usub_tasks="$usub_logdir/$usub_run.tasks"',
                    f"cat >\"$usub_tasks\" <<'{_TASKS_DELIM}'",
                    *plan.tasks,
                    _TASKS_DELIM,
                ]
            run = 'eval "$(sed -n "${1}p" "$usub_tasks")"'
        else:
            lines.append("usub_tasks=")
            run = plan.command
        lines += [
            "usub_task() {",
            f'  export {INDEX_VAR}="$1"',
            f"  ( {run}",
            '  ) >"$usub_logdir/$usub_run.$1.out" 2>"$usub_logdir/$usub_run.$1.err"',
            "  usub_rc=$?",
            '  if [ "$usub_rc" -ne 0 ]; then echo "usub: task $1 failed (exit $usub_rc)" >&2; return 1; fi',
            "}",
            "export -f usub_task",
            "export usub_logdir usub_run usub_tasks",
            f"seq {idx.start} {idx.step} {idx.end} | "
            f"xargs -P {self.parallelism(job)} -I{{}} bash -c 'usub_task \"$1\"' _ {{}}",
            "usub_rc=$?",
            'echo "usub: logs in $usub_logdir/$usub_run.*" >&2',
            "exit $usub_rc",
        ]
        return "\n".join(lines) + "\n"

    def oneliner(self, rendered: Rendered) -> str:
        job, plan = rendered.job, rendered.plan
        self._check(job)
        if plan.tasks is not None:
            raise UsubError("one-line form is only available for single jobs and index arrays; "
                            "use the script form for cmds/files arrays")
        command = "; ".join([*job.setup, plan.command])
        if not plan.is_array:
            return f"bash -c {shlex.quote(command)}"
        idx = plan.indices
        inner = shlex.quote(f'export {INDEX_VAR}="$1"; {command}')
        return (f"seq {idx.start} {idx.step} {idx.end} | "
                f"xargs -P {self.parallelism(job)} -I{{}} bash -c {inner} _ {{}}")

    def submit_hint(self, path: str | None) -> str:
        return f"bash {path or 'SCRIPT'}"

    def submit(self, rendered: Rendered) -> str:
        import subprocess

        path = rendered.save()
        proc = subprocess.Popen(["bash", path])
        rc = proc.wait()
        if rc != 0:
            raise SubmitError(f"local job failed (exit {rc}); script: {path}")
        return str(proc.pid)
