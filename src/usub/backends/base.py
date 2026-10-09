"""Backend interface and the shared job-script builder.

A backend turns a resolved :class:`~usub.job.Job` into a self-contained shell
script (scheduler directives + body) and knows how to submit it. Most of the
work is shared here; a scheduler backend mainly provides :meth:`Backend.directives`.
"""

from __future__ import annotations

import os
import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .. import __version__
from ..errors import SubmitError, UsubError
from ..job import INDEX_VAR, Job, Plan

# A directive is either argv (quoted as needed) or a raw string used verbatim.
Directive = list[str] | str

_DIRECTIVE_SAFE_RE = re.compile(r"^[\w@%+=:,./-]+$")
_TASKS_DELIM = "USUB_TASKS"
DEFAULT_EMBED_LIMIT = 1_000_000  # bytes of task text embedded before using a manifest file


def directive_quote(arg: str) -> str:
    """Quote an argument for a ``#BSUB``/``#SBATCH`` line (double quotes, as schedulers expect)."""
    if _DIRECTIVE_SAFE_RE.match(arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass
class Rendered:
    """A rendered job: the script plus everything needed to save or submit it."""

    script: str
    job: Job
    plan: Plan
    backend: Backend
    profile: str | None = None
    manifest: tuple[str, str] | None = None  # (path, content) when tasks are too big to embed

    def prepare(self) -> None:
        """Create the log directory and write the task manifest, if any."""
        os.makedirs(self.job.logdir, exist_ok=True)
        if self.manifest:
            path, content = self.manifest
            with open(path, "w") as fh:
                fh.write(content)

    def write(self, path: str | os.PathLike) -> str:
        self.prepare()
        path = os.fspath(path)
        with open(path, "w") as fh:
            fh.write(self.script)
        os.chmod(path, 0o755)
        return path

    def save(self) -> str:
        """Write the script into ``<logdir>/scripts/`` with a timestamped name."""
        import time

        scripts = os.path.join(self.job.logdir, "scripts")
        os.makedirs(scripts, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(scripts, f"{self.job.name}.{stamp}")
        path, n = f"{base}.sh", 1
        while os.path.exists(path):
            n += 1
            path = f"{base}.{n}.sh"
        return self.write(path)

    def command(self) -> str:
        """A one-line submission command (single jobs and index arrays only)."""
        return self.backend.oneliner(self)

    def submit(self) -> str:
        """Submit and return the job id."""
        return self.backend.submit(self)

    def submit_hint(self, path: str | None = None) -> str:
        return self.backend.submit_hint(path)


class Backend(ABC):
    """Base class for submission backends.

    Subclasses set the class attributes and implement :meth:`directives`;
    override :meth:`render_script`/:meth:`submit` for non-scheduler backends.
    ``options`` holds backend-specific settings from the config profile.
    """

    name: ClassVar[str]
    directive_prefix: ClassVar[str] = ""
    index_var: ClassVar[str] = INDEX_VAR  # scheduler's array index env var
    submit_program: ClassVar[str] = ""
    job_id_re: ClassVar[re.Pattern] = re.compile(r"(\d+)")

    def __init__(self, options: dict | None = None):
        self.options = dict(options or {})

    # -- to implement -------------------------------------------------------
    @abstractmethod
    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        """Scheduler options for this job, one directive per entry."""

    def submit_argv(self, script_path: str) -> tuple[list[str], bool]:
        """Return (argv, feed_script_on_stdin)."""
        return [self.submit_program, script_path], False

    def oneliner_argv(self, job: Job, plan: Plan, command: str) -> str:
        raise UsubError(f"backend {self.name!r} has no one-line form; use the script")

    def submit_hint(self, path: str | None) -> str:
        argv, stdin = self.submit_argv(path or "SCRIPT")
        return f"{shlex.join(argv)} < {path or 'SCRIPT'}" if stdin else shlex.join(argv)

    # -- shared -------------------------------------------------------------
    def gpu_directives(self, job: Job) -> list[Directive] | None:
        """Raw directives from the ``gpu_args`` option, or None to use the backend default."""
        if not job.gpus:
            return []
        if tmpl := self.options.get("gpu_args"):
            tmpl = [tmpl] if isinstance(tmpl, str) else tmpl
            return [t.format(gpus=job.gpus) for t in tmpl]
        return None

    def log_path(self, job: Job, suffix: str, job_token: str, index_token: str | None) -> str:
        parts = [job.name, job_token] + ([index_token] if index_token else [])
        return os.path.join(job.logdir, ".".join(parts) + suffix)

    def render(self, job: Job, profile: str | None = None) -> Rendered:
        plan = job.plan()
        manifest = self._manifest(job, plan)
        script = self.render_script(job, plan, manifest[0] if manifest else None, profile)
        return Rendered(script, job, plan, self, profile, manifest)

    def render_script(self, job: Job, plan: Plan, manifest_path: str | None,
                      profile: str | None) -> str:
        lines = ["#!/bin/bash"]
        for d in self.directives(job, plan):
            text = d if isinstance(d, str) else " ".join(directive_quote(a) for a in d)
            lines.append(f"{self.directive_prefix} {text}")
        lines.append(self.header_comment(profile))
        lines.append("")
        lines += self.body_lines(job, plan, manifest_path)
        return "\n".join(lines) + "\n"

    def header_comment(self, profile: str | None) -> str:
        where = f"backend={self.name}" + (f", profile={profile}" if profile else "")
        return f"# generated by usub {__version__} ({where})"

    def body_lines(self, job: Job, plan: Plan, manifest_path: str | None) -> list[str]:
        lines = [f"cd {shlex.quote(os.fspath(job.cwd))}", *job.setup]
        if plan.is_array:
            lines.append(f'export {INDEX_VAR}="${{{self.index_var}}}"')
        return lines + task_lines(plan, manifest_path)

    def _manifest(self, job: Job, plan: Plan) -> tuple[str, str] | None:
        if plan.tasks is None:
            return None
        content = "\n".join(plan.tasks) + "\n"
        limit = int(self.options.get("embed_limit", DEFAULT_EMBED_LIMIT))
        if len(content) <= limit and _TASKS_DELIM not in plan.tasks:
            return None
        import hashlib

        digest = hashlib.sha1(content.encode()).hexdigest()[:10]
        return os.path.join(job.logdir, f"{job.name}.{digest}.tasks"), content

    def oneliner(self, rendered: Rendered) -> str:
        plan, job = rendered.plan, rendered.job
        if plan.tasks is not None:
            raise UsubError("one-line form is only available for single jobs and index arrays; "
                            "use the script form for cmds/files arrays")
        command = "; ".join([*job.setup, plan.command])
        if plan.is_array:
            command = f'export {INDEX_VAR}="${{{self.index_var}}}"; {command}'
        return self.oneliner_argv(job, plan, command)

    def submit(self, rendered: Rendered) -> str:
        import subprocess

        path = rendered.save()
        argv, use_stdin = self.submit_argv(path)
        try:
            with open(path) as fh:
                proc = subprocess.run(argv, stdin=fh if use_stdin else subprocess.DEVNULL,
                                      capture_output=True, text=True)
        except FileNotFoundError:
            raise SubmitError(f"{argv[0]!r} not found; is {self.name} available on this host?") from None
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise SubmitError(f"{argv[0]} failed (exit {proc.returncode}): {output.strip()}\n"
                              f"script: {path}")
        m = self.job_id_re.search(output)
        if not m:
            raise SubmitError(f"could not find job id in {argv[0]} output: {output.strip()!r}")
        return m.group(1)


def task_lines(plan: Plan, manifest_path: str | None) -> list[str]:
    """Shell lines that run the task for ``$USUB_INDEX`` (or the single command)."""
    if plan.tasks is None:
        return [plan.command]
    select = f'usub_cmd=$(sed -n "${{{INDEX_VAR}}}p"'
    if manifest_path:
        lines = [f"{select} {shlex.quote(manifest_path)})"]
    else:
        # Brace group feeds the heredoc to sed; avoids heredoc-inside-$() parsing bugs in old bash.
        lines = [f"{{ {select}); }} <<'{_TASKS_DELIM}'", *plan.tasks, _TASKS_DELIM]
    return lines + ['eval "$usub_cmd"']


_BUILTIN = {
    "lsf": ("usub.backends.lsf", "LSF"),
    "slurm": ("usub.backends.slurm", "Slurm"),
    "local": ("usub.backends.local", "Local"),
}


def get_backend(name: str, options: dict | None = None) -> Backend:
    """Instantiate a backend by name: built-ins first, then ``usub.backends`` entry points."""
    if name in _BUILTIN:
        import importlib

        module, attr = _BUILTIN[name]
        cls = getattr(importlib.import_module(module), attr)
    else:
        from importlib.metadata import entry_points

        eps = entry_points(group="usub.backends", name=name)
        if not eps:
            known = ", ".join(sorted(set(_BUILTIN) | {e.name for e in entry_points(group="usub.backends")}))
            raise UsubError(f"unknown backend {name!r} (available: {known})")
        cls = next(iter(eps)).load()
    return cls(options)
