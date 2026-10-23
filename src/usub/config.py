"""
Configuration: profiles, config files and resolution of a Job's effective settings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .errors import UsubError
from .job import Job, safe_name

PROJECT_CONFIG = ".usub.toml"

TEMPLATE = """\
# usub configuration. Settings apply in order: built-ins < [defaults] < selected profile <
# command line. Set USUB_CONFIG to use a different file instead of ~/.usub.toml.
# A .usub.toml in a project directory (or any parent) is applied on top of this file.

# Profile to use when none is given; "auto" picks the first profile whose
# host_match matches this host, else lsf/slurm if bsub/sbatch is on PATH, else local.
default_profile = "auto"

[defaults]
logdir = "~/usub_logs"   # relative paths are relative to the submission directory
mem = "1G"
cores = 1
# time = "1h"
# merge_logs = true      # one log per job: script, then stdout and stderr together
# setup = ["source ~/.bashrc"]   # lines run before the command in every job

[profiles.lsf]
backend = "lsf"
host_match = ["farm*", "*.my.server.com"]
queue = "normal"
lsf_mem_unit = "MB"      # site LSF_UNIT_FOR_LIMITS
lsf_rusage = true        # always add -R "select[mem>X] rusage[mem=X]"
# gpu_args = ['-gpu "num={gpus}"']

[profiles.slurm]
backend = "slurm"
# host_match = ["login*.mycluster.org"]
# queue = "standard"
# extra = ["--account=myproject"]

[profiles.local]
backend = "local"
# max_parallel = 8
"""

PROJECT_TEMPLATE = """\
# usub project configuration: applied on top of ~/.usub.toml for jobs run in this directory
# or any subdirectory. Only list what differs; profiles with the same name are merged.

[defaults]
# logdir = "logs"
# mem = "4G"

# [profiles.farm]
# queue = "long"
# extra = ["-G", "my-project"]
"""

# How each Profile setting is checked when read from a config file: (test, description)
_FIELD_CHECKS = {
    "backend": (lambda v: isinstance(v, str), "a string"),
    "queue": (lambda v: isinstance(v, str), "a string"),
    "logdir": (lambda v: isinstance(v, str), "a string"),
    "cores": (lambda v: type(v) is int and v > 0, "a positive integer"),
    "max_parallel": (lambda v: type(v) is int and v > 0, "a positive integer"),
    "merge_logs": (lambda v: type(v) is bool, "true or false"),
    "mem": (lambda v: type(v) in (int, float, str), 'a string such as "4G" or a number of MB'),
    "time": (lambda v: type(v) in (int, float, str), 'a string such as "2h" or a number of minutes'),
    "gpus": (lambda v: type(v) in (int, str), "a count or a string in the scheduler's syntax"),
}

@dataclass
class Profile:
    """
    A group of settings: one cluster profile or ``[defaults]``.

    Unset fields (``None``/empty) inherit from lower layers when merged. Keys in a config
    file that are not fields (e.g. ``lsf_mem_unit``) are backend options, which the backend
    checks when it is created.
    """
    backend: str | None = None
    host_match: list[str] = field(default_factory=list)
    queue: str | None = None
    cores: int | None = None
    mem: str | int | None = None
    time: str | int | None = None
    gpus: int | str | None = None
    logdir: str | None = None
    max_parallel: int | None = None
    merge_logs: bool | None = None
    extra: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)

    @classmethod
    def builtin(cls) -> Profile:
        """
        Settings used when nothing else sets them
        """
        return cls(logdir="~/usub_logs", cores=1, mem="1G", merge_logs=False)

    @classmethod
    def from_dict(cls, data: Mapping) -> Profile:
        """
        Build from a config file table: settings are type-checked, list settings may be
        single strings, and keys that are not fields become backend options
        """
        data = dict(data)

        names = {f.name for f in fields(cls)} - {"options"}
        kwargs = {k: data.pop(k) for k in list(data) if k in names}

        for key in ("host_match", "extra", "setup"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = [kwargs[key]]
            value = kwargs.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise UsubError(f"{key} must be a string or a list of strings (got {value!r})")

        for key, value in kwargs.items():
            check, expected = _FIELD_CHECKS.get(key, (None, None))
            if check and not check(value):
                raise UsubError(f"{key} must be {expected} (got {value!r})")

        return cls(**kwargs, options=data)

    def merge(self, *others: Profile) -> Profile:
        """
        Layer ``others`` on top of this profile: set values override, ``extra``/``setup``
        accumulate, options are combined and ``host_match`` is replaced if given
        """
        out = replace(self, extra=list(self.extra), setup=list(self.setup),
                      host_match=list(self.host_match), options=dict(self.options))
        for other in others:
            for f in fields(self):
                value = getattr(other, f.name)
                if f.name in ("extra", "setup"):
                    getattr(out, f.name).extend(value)
                elif f.name == "options":
                    out.options.update(value)
                elif f.name == "host_match":
                    out.host_match = list(value) or out.host_match
                elif value is not None:
                    setattr(out, f.name, value)
        return out

@dataclass
class Config:
    """
    Settings from config files: ``[defaults]``, ``[profiles.<name>]`` and ``default_profile``.

    ``sources`` lists the files the config was loaded from, in the order applied.
    """

    defaults: Profile = field(default_factory=Profile)
    profiles: dict[str, Profile] = field(default_factory=dict)
    default_profile: str | None = None  # None means auto-detect
    sources: list[Path] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping) -> Config:
        """
        Build from parsed TOML; ``default_profile = "auto"`` means auto-detect
        """
        if unknown := set(data) - {"default_profile", "defaults", "profiles"}:
            raise UsubError(f"Unknown key {', '.join(sorted(map(repr, unknown)))} "
                            "(expected default_profile, [defaults] and [profiles.NAME])")

        defaults = data.get("defaults", {})
        profiles = data.get("profiles", {})
        default = data.get("default_profile")

        if not isinstance(defaults, Mapping):
            raise UsubError("[defaults] must be a table")

        if not isinstance(profiles, Mapping) or not all(isinstance(p, Mapping) for p in profiles.values()):
            raise UsubError("[profiles.<name>] entries must be tables")

        if default is not None and not isinstance(default, str):
            raise UsubError('default_profile must be a profile name or "auto"')

        return cls(Profile.from_dict(defaults),
                   {name: Profile.from_dict(p) for name, p in profiles.items()},
                   None if default == "auto" else default)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> Config:
        """
        Load one TOML config file, reporting errors with the file name
        """
        import tomllib

        path = Path(path)
        try:
            config = cls.from_dict(tomllib.loads(path.read_text()))
        except (tomllib.TOMLDecodeError, UsubError) as e:
            raise UsubError(f"{path}: {e}") from None

        config.sources = [path]
        return config

    def merge(self, other: Config) -> Config:
        """
        Layer ``other`` on top: defaults and same-named profiles are merged, new profiles added
        """
        profiles = {name: self.profiles[name].merge(p) if name in self.profiles else p
                    for name, p in other.profiles.items()}
        return Config(self.defaults.merge(other.defaults), {**self.profiles, **profiles},
                      other.default_profile or self.default_profile, self.sources + other.sources)

def config_path() -> Path:
    """
    Path of the user config: ``$USUB_CONFIG``, else ``~/.usub.toml``
    """
    return Path(os.environ.get("USUB_CONFIG") or "~/.usub.toml").expanduser()

def load_config(path: str | os.PathLike | None = None, cwd: str | os.PathLike | None = None) -> Config:
    """
    Load the user config, then the nearest project ``.usub.toml`` on top of it

    The project file is the first ``.usub.toml`` found in ``cwd`` (default: the current
    directory) or its parents, ignoring the user config itself. An explicit ``path`` is
    loaded alone; missing files are an error then, but not for the default locations.
    """
    if path:
        paths = [Path(path).expanduser()]
        if not paths[0].is_file():
            raise UsubError(f"Config file not found: {paths[0]}")
    else:
        user = config_path()
        paths = [user] if user.is_file() else []
        skip = {user.resolve(), (Path.home() / PROJECT_CONFIG).resolve()}
        start = Path(cwd or os.getcwd()).resolve()
        for directory in (start, *start.parents):
            if (found := directory / PROJECT_CONFIG).is_file() and found not in skip:
                paths.append(found)
                break

    config = Config()
    for p in paths:
        config = config.merge(Config.from_file(p))

    return config

@dataclass
class Resolved:
    """A Job with all settings filled in, plus the backend and options to render it with."""
    job: Job
    backend: str
    profile: str | None
    options: dict

def select_profile(config: Config, profile: str | None = None,
                   hostname: str | None = None) -> str | None:
    """
    Pick a profile name, or None to use [defaults] with an auto-detected backend
    """
    name = profile or config.default_profile
    if name:
        if name not in config.profiles:
            known = ", ".join(config.profiles) or "none defined"
            raise UsubError(f"Unknown profile {name!r} (known: {known})")
        return name
    from fnmatch import fnmatch

    if hostname is None and any(p.host_match for p in config.profiles.values()):
        import socket
        hostname = socket.gethostname()

    for pname, prof in config.profiles.items():
        if any(fnmatch(hostname, pat) for pat in prof.host_match):
            return pname

    return None

def resolve(job: Job, *, profile: str | None = None, backend: str | None = None,
            config: Config | None = None, hostname: str | None = None) -> Resolved:
    """
    Fill unset Job fields from builtins < [defaults] < profile

    ``extra`` and ``setup`` accumulate across layers instead of replacing; the job's own
    values come last. Relative ``logdir`` is relative to the job's working directory.
    """
    if config is None:
        config = load_config(cwd=job.cwd)

    pname = select_profile(config, profile, hostname)
    merged = Profile.builtin().merge(config.defaults, config.profiles.get(pname, Profile()))

    backend_name = backend or merged.backend
    if not backend_name:
        from shutil import which
        backend_name = "lsf" if which("bsub") else "slurm" if which("sbatch") else "local"

    job_fields = {f.name for f in fields(Job)}
    updates = {f.name: getattr(merged, f.name) for f in fields(merged)
               if f.name in job_fields and f.name not in ("extra", "setup")
               and getattr(job, f.name) is None}

    resolved = replace(job, **updates, extra=merged.extra + job.extra, setup=merged.setup + job.setup)

    resolved.name = safe_name(str(resolved.name)) if resolved.name else job.default_name()
    resolved.cwd = os.path.abspath(os.path.expanduser(os.fspath(resolved.cwd or os.getcwd())))

    logdir = os.path.expanduser(os.fspath(resolved.logdir))
    resolved.logdir = os.path.normpath(os.path.join(resolved.cwd, logdir))

    return Resolved(resolved, backend_name, pname, merged.options)
