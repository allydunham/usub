"""User configuration: TOML profiles and resolution of a Job's effective settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .errors import UsubError
from .job import Job, safe_name

# Settings that map onto Job fields; anything else in a profile is a backend option.
JOB_KEYS = {"name", "cores", "mem", "time", "queue", "gpus", "logdir", "max_parallel"}
LIST_KEYS = {"after", "extra", "setup"}
META_KEYS = {"backend", "host_match"}
ALIASES = {"partition": "queue"}

BUILTIN_DEFAULTS: dict = {"logdir": "usub-logs", "cores": 1, "mem": "1G"}

# Environment variables that override config (but not explicit job/CLI values).
ENV_KEYS = {"USUB_LOGDIR": "logdir", "USUB_QUEUE": "queue"}

TEMPLATE = """\
# usub configuration. Settings apply in order: [defaults] < selected profile <
# environment (USUB_PROFILE, USUB_BACKEND, USUB_LOGDIR, USUB_QUEUE) < command line.

# Profile to use when none is given; "auto" picks the first profile whose
# host_match matches this host, else lsf/slurm if bsub/sbatch is on PATH, else local.
default_profile = "auto"

[defaults]
logdir = "~/usub-logs"   # relative paths are relative to the submission directory
mem = "1G"
cores = 1
# time = "1h"
# setup = ["source ~/.bashrc"]   # lines run before the command in every job

[profiles.farm]
backend = "lsf"
host_match = ["farm*", "*.internal.sanger.ac.uk"]
queue = "normal"
lsf_mem_unit = "MB"      # site LSF_UNIT_FOR_LIMITS
lsf_rusage = true        # add -R "select[mem>X] rusage[mem=X]" (required on the Sanger farm)
# gpu_args = ['-gpu "num={gpus}"']

[profiles.slurm]
backend = "slurm"
# host_match = ["login*.mycluster.org"]
# partition = "standard"
# extra = ["--account=myproject"]

[profiles.local]
backend = "local"
# max_parallel = 8
"""


def config_path() -> Path:
    if p := os.environ.get("USUB_CONFIG"):
        return Path(p).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "usub" / "config.toml"


@dataclass
class Config:
    defaults: dict = field(default_factory=dict)
    profiles: dict[str, dict] = field(default_factory=dict)
    default_profile: str = "auto"

    @classmethod
    def from_dict(cls, data: Mapping) -> Config:
        profiles = data.get("profiles", {})
        if not isinstance(profiles, Mapping) or not all(isinstance(p, Mapping) for p in profiles.values()):
            raise UsubError("config: [profiles.<name>] entries must be tables")
        return cls(
            defaults=_normalise(dict(data.get("defaults", {}))),
            profiles={k: _normalise(dict(v)) for k, v in profiles.items()},
            default_profile=data.get("default_profile", "auto"),
        )


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the user config; a missing default config file gives an empty Config."""
    p = Path(path) if path else config_path()
    try:
        text = p.read_text()
    except FileNotFoundError:
        if path:
            raise UsubError(f"config file not found: {p}") from None
        return Config()
    import tomllib

    try:
        return Config.from_dict(tomllib.loads(text))
    except tomllib.TOMLDecodeError as e:
        raise UsubError(f"config {p}: {e}") from None


def _normalise(d: dict) -> dict:
    for alias, key in ALIASES.items():
        if alias in d:
            d[key] = d.pop(alias)
    for key in LIST_KEYS | {"host_match"}:
        if isinstance(d.get(key), str):
            d[key] = [d[key]]
    return d


@dataclass
class Resolved:
    job: Job
    backend: str
    profile: str | None
    options: dict


def select_profile(config: Config, profile: str | None = None, env: Mapping = os.environ,
                   hostname: str | None = None) -> str | None:
    """Pick a profile name, or None to use [defaults] with auto-detected backend."""
    name = profile or env.get("USUB_PROFILE") or config.default_profile
    if name and name != "auto":
        if name not in config.profiles:
            known = ", ".join(config.profiles) or "none defined"
            raise UsubError(f"unknown profile {name!r} (known: {known}; config: {config_path()})")
        return name
    if not config.profiles:
        return None
    from fnmatch import fnmatch

    if hostname is None:
        import socket

        hostname = socket.gethostname()
    for pname, prof in config.profiles.items():
        if any(fnmatch(hostname, pat) for pat in prof.get("host_match", [])):
            return pname
    return None


def detect_backend() -> str:
    from shutil import which

    if which("bsub"):
        return "lsf"
    if which("sbatch"):
        return "slurm"
    return "local"


def resolve(job: Job, *, profile: str | None = None, backend: str | None = None,
            config: Config | None = None, env: Mapping = os.environ,
            hostname: str | None = None) -> Resolved:
    """Fill unset Job fields from builtins < [defaults] < profile < environment."""
    if config is None:
        config = load_config()
    pname = select_profile(config, profile, env, hostname)
    layers = [BUILTIN_DEFAULTS, config.defaults, config.profiles.get(pname, {}) if pname else {}]
    layers.append({key: env[var] for var, key in ENV_KEYS.items() if env.get(var)})

    merged: dict = {}
    lists: dict[str, list] = {k: [] for k in LIST_KEYS}
    for layer in layers:
        for k, v in layer.items():
            if k in LIST_KEYS:
                lists[k].extend(v)
            else:
                merged[k] = v

    backend_name = backend or env.get("USUB_BACKEND") or merged.get("backend") or detect_backend()
    options = {k: v for k, v in merged.items() if k not in JOB_KEYS | META_KEYS}

    updates = {}
    for f in fields(Job):
        if f.name in LIST_KEYS:
            updates[f.name] = lists[f.name] + list(getattr(job, f.name))
        elif f.name in JOB_KEYS and getattr(job, f.name) is None and f.name in merged:
            updates[f.name] = merged[f.name]
    resolved = replace(job, **updates)
    resolved.name = safe_name(str(resolved.name)) if resolved.name else job.default_name()
    resolved.cwd = os.path.abspath(os.path.expanduser(os.fspath(resolved.cwd or os.getcwd())))
    logdir = os.path.expanduser(os.fspath(resolved.logdir))
    resolved.logdir = os.path.normpath(os.path.join(resolved.cwd, logdir))
    return Resolved(resolved, backend_name, pname, options)
