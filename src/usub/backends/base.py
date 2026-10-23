"""
Backend interface, the shared job-script builder and the render/submit entry points.

A backend turns a resolved :class:`~usub.job.Job` into a self-contained shell
script (scheduler directives + body) and knows how to submit it. Most of the
work is shared here; a scheduler backend mainly provides :meth:`Backend.directives`.
"""

from __future__ import annotations

import os
import re
import shlex
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .. import __version__
from ..config import Config, resolve
from ..errors import SubmitError, UsubError, UsubWarning
from ..job import PLACEHOLDER_RE, Job, Plan

# A directive is either argv (quoted as needed) or a raw string used verbatim
Directive = list[str] | str

_DIRECTIVE_SAFE_RE = re.compile(r"^[\w@%+=:,./-]+$")
_SCRIPT_DELIM = "SCRIPT"

@dataclass
class Rendered:
    """A rendered job: the script plus everything needed to save or submit it."""
    script: str
    job: Job
    plan: Plan
    backend: Backend

    def prepare(self) -> None:
        """
        Create the log directory
        """
        os.makedirs(self.job.logdir, exist_ok=True)

    def write(self, path: str | os.PathLike) -> str:
        """
        Write the (executable) script to ``path`` after preparing the log directory
        """
        self.prepare()
        path = os.fspath(path)
        with open(path, "w") as fh:
            fh.write(self.script)
        os.chmod(path, 0o755)
        return path

    def save(self, job_id: str | None = None, force: bool = False) -> str | None:
        """
        Write the script to ``<logdir>/<name>[.<job_id>].script`` and return its path

        Merged single jobs already have the script at the top of their log, so nothing is
        saved unless ``force`` is set.
        """
        if self.job.merge_logs and not self.plan.is_array and not force:
            return None
        name = ".".join(p for p in (self.job.name, job_id, "script") if p)
        return self.write(os.path.join(self.job.logdir, name))

    def command(self) -> str:
        """
        A one-line submission command (the script's command, passed to the scheduler as an argument)
        """
        return self.backend.oneliner(self)

    def submit(self) -> str:
        """
        Submit and return the job id
        """
        return self.backend.submit(self)

class Backend(ABC):
    """
    Base class for submission backends.

    Subclasses set the class attributes (``option_names`` lists the config keys they understand,
    so that other keys can be reported as likely typos) and implement :meth:`directives`;
    override :meth:`render_script`/:meth:`submit` for non-scheduler backends.
    ``options`` holds backend-specific settings from the config profile.
    """

    name: ClassVar[str]
    directive_prefix: ClassVar[str] = ""
    index_var: ClassVar[str]  # shell variable holding the array index inside a task
    option_names: ClassVar[frozenset[str]] = frozenset()  # config keys this backend understands
    submit_program: ClassVar[str] = ""
    oneliner_wrap: ClassVar[str | None] = None  # prefix for the command in one-line form
    job_id_re: ClassVar[re.Pattern] = re.compile(r"(\d+)")

    def __init__(self, options: dict | None = None):
        """
        Store backend-specific ``options`` from the config profile
        """
        self.options = dict(options or {})

    @abstractmethod
    def directives(self, job: Job, plan: Plan) -> list[Directive]:
        """
        Scheduler options for this job, one directive per entry
        """

    def submit_hint(self, path: str | None) -> str:
        """
        The shell command that submits the script at ``path`` (schedulers read it from stdin)
        """
        return f"{self.submit_program} < {path or 'SCRIPT'}"

    def gpu_directives(self, job: Job) -> list[Directive] | None:
        """
        Raw directives from the ``gpu_args`` option, or None to use the backend default

        ``job.gpus`` is a count or the scheduler's own spec, and fills ``{gpus}`` in the template.
        """
        if not job.gpus or str(job.gpus) == "0":
            return []
        if tmpl := self.options.get("gpu_args"):
            tmpl = [tmpl] if isinstance(tmpl, str) else tmpl
            try:
                return [t.format(gpus=job.gpus) for t in tmpl]
            except (KeyError, IndexError, ValueError) as e:
                raise UsubError(f"Invalid gpu_args template {tmpl!r}: only {{gpus}} may be used ({e!r})") from None
        return None

    def log_path(self, job: Job, job_token: str, index_token: str | None = None,
                 err: bool = False) -> str:
        """
        Log file path ``<logdir>/<name>.<id>[.<index>][.err]`` from the scheduler's id/index tokens
        """
        parts = [job.name, job_token] + ([index_token] if index_token else [])
        return os.path.join(job.logdir, ".".join(parts) + (".err" if err else ""))

    def header_comment(self, profile: str | None) -> str:
        """
        Comment line identifying the generator, backend, profile and generation time
        """
        import time

        where = f"backend={self.name}" + (f", profile={profile}" if profile else "")
        return f"# generated by usub {__version__} ({where}) on {time.strftime('%Y-%m-%d %H:%M:%S')}"

    def render(self, job: Job, profile: str | None = None) -> Rendered:
        """
        Render a resolved job
        """
        plan = job.plan()
        return Rendered(self.render_script(job, plan, profile), job, plan, self)

    def fill_index(self, command: str) -> str:
        """
        Replace ``{}``/``{i}`` in an index array's command with the scheduler's index variable
        """
        return PLACEHOLDER_RE.sub("${" + self.index_var + "}", command)

    def render_script(self, job: Job, plan: Plan, profile: str | None) -> str:
        """
        Build the job script: shebang, directives, then cd, setup and the command

        With ``merge_logs`` a single job also prints its own script (as submitted, minus this
        printing) at the start of its log; arrays get one merged log per index instead.
        """
        head = ["#!/bin/bash"]
        for d in self.directives(job, plan):
            if not isinstance(d, str):  # schedulers expect double quotes in directive lines
                d = " ".join(a if _DIRECTIVE_SAFE_RE.match(a)
                             else '"' + a.replace("\\", "\\\\").replace('"', '\\"') + '"'
                             for a in d)
            if "\n" in d:
                raise UsubError(f"Directives cannot contain newlines: {d!r}")
            head.append(f"{self.directive_prefix} {d}")
        head += [self.header_comment(profile), ""]

        body = [f"cd {shlex.quote(os.fspath(job.cwd))}", *job.setup,
                self.fill_index(plan.command) if plan.is_array else plan.command]

        show = []
        if job.merge_logs and not plan.is_array:
            show = [f"cat <<'{_SCRIPT_DELIM}'", *head, *body, _SCRIPT_DELIM]
        return "\n".join(head + show + body) + "\n"

    def oneliner(self, rendered: Rendered) -> str:
        """
        One-line submission command, with the command passed as a single quoted argument
        """
        plan, job = rendered.plan, rendered.job
        if self.oneliner_wrap is None:
            raise UsubError(f"Backend {self.name!r} has no one-line form; use the script")
        command = "; ".join([*job.setup, self.fill_index(plan.command) if plan.is_array else plan.command])
        parts = [self.submit_program]
        parts += [d if isinstance(d, str) else shlex.join(d) for d in self.directives(job, plan)]
        return " ".join(parts + [shlex.quote(self.oneliner_wrap + command)])

    def submit(self, rendered: Rendered) -> str:
        """
        Submit the script on stdin, parse the job id and save the script under that id

        A script that could not be submitted is kept as ``<name>.failed-<time>.script``.
        """
        import subprocess
        import time

        rendered.prepare()
        argv = [self.submit_program]
        try:
            proc = subprocess.run(argv, input=rendered.script, capture_output=True, text=True)
        except FileNotFoundError:
            raise SubmitError(f"{argv[0]!r} not found; is {self.name} available on this host?") from None
        output = (proc.stdout + proc.stderr).strip()
        m = self.job_id_re.search(output)
        if proc.returncode != 0 or not m:
            path = rendered.save(f"failed-{time.strftime('%Y%m%d-%H%M%S')}", force=True)
            reason = (f"{argv[0]} failed (exit {proc.returncode}): {output}" if proc.returncode
                      else f"could not find job id in {argv[0]} output: {output!r}")
            raise SubmitError(f"{reason}\nscript: {path}")
        job_id = m.group(1)
        try:
            rendered.save(job_id)
        except OSError as e:  # the job is already submitted, so don't lose its id
            warnings.warn(f"Job {job_id} was submitted but its script could not be saved: {e}",
                          UsubWarning, stacklevel=2)
        return job_id

_BUILTIN = {
    "lsf": ("usub.backends.lsf", "LSF"),
    "slurm": ("usub.backends.slurm", "Slurm"),
    "local": ("usub.backends.local", "Local"),
}

def _backend_class(name: str) -> type[Backend]:
    """
    The class of a built-in backend, or of a ``usub.backends`` entry point
    """
    if name in _BUILTIN:
        import importlib

        module, attr = _BUILTIN[name]
        return getattr(importlib.import_module(module), attr)

    from importlib.metadata import entry_points

    eps = entry_points(group="usub.backends", name=name)
    if not eps:
        known = ", ".join(sorted(set(_BUILTIN) | {e.name for e in entry_points(group="usub.backends")}))
        raise UsubError(f"Unknown backend {name!r} (available: {known})")

    return next(iter(eps)).load()

def get_backend(name: str, options: dict | None = None) -> Backend:
    """
    Instantiate a backend by name, warning about options that no built-in backend recognises

    Options belonging to another built-in backend are not warned about, as profiles can share
    defaults (for example ``lsf_*`` options in ``[defaults]`` used by a Slurm profile).
    """
    cls = _backend_class(name)

    if unknown := set(options or {}) - cls.option_names:
        unknown -= set().union(*(_backend_class(n).option_names for n in _BUILTIN))

    if unknown:
        accepted = ", ".join(sorted(cls.option_names)) or "none"
        warnings.warn(f"Unknown option(s) {', '.join(sorted(unknown))} for the {name} backend "
                      f"(it accepts: {accepted}); check for typos", UsubWarning, stacklevel=2)

    return cls(options)

def render(job: Job, *, profile: str | None = None, backend: str | None = None,
           config: Config | None = None) -> Rendered:
    """
    Resolve ``job`` against config and render it for the selected backend
    """
    r = resolve(job, profile=profile, backend=backend, config=config)
    return get_backend(r.backend, r.options).render(r.job, r.profile)

def submit(job: Job, *, profile: str | None = None, backend: str | None = None,
           config: Config | None = None) -> str:
    """
    Render and submit ``job``; returns the scheduler job id
    """
    return render(job, profile=profile, backend=backend, config=config).submit()
