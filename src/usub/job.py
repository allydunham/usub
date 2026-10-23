"""
Scheduler-neutral job description and array expansion.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

from .errors import UsubError

# Placeholders for the array index in an index array's command
PLACEHOLDER_RE = re.compile(r"\{i?\}")

# Placeholders for the current file of a file list, as shell expressions on the variable f
FILE_PLACEHOLDER_RE = re.compile(r"\{(//|/\.|/|\.|)\}")
_FILE_EXPRS = {
    "": '"$f"',
    "/": '"$(basename "$f")"',
    "//": '"$(dirname "$f")"',
    ".": '"${f%.*}"',
    "/.": '"$(basename "${f%.*}")"',
}
_RANGE_RE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?(?:\s*:\s*(\d+))?\s*$")
_NAME_UNSAFE_RE = re.compile(r"[^\w.-]+")

@dataclass(frozen=True)
class IndexRange:
    """An inclusive array index range, e.g. ``1-100:2``."""
    start: int
    end: int
    step: int = 1

    def __post_init__(self):
        """
        Reject negative, reversed or zero-step ranges
        """
        if self.start < 0 or self.end < self.start or self.step < 1:
            raise UsubError(f"Invalid array range {self.spec()!r}")

    @classmethod
    def parse(cls, value: str | int | range | IndexRange) -> IndexRange:
        """
        Parse ``"1-100"``, ``"1-100:2"``, ``"N"`` or ``N`` (meaning ``1-N``) or a ``range``
        """
        if isinstance(value, IndexRange):
            return value

        if isinstance(value, range):
            if not len(value):
                raise UsubError("Empty array range")
            return cls(value.start, value[-1], value.step)

        if isinstance(value, int):
            return cls(1, value)

        m = _RANGE_RE.match(value)

        if not m:
            raise UsubError(f"Invalid array range {value!r} (expected e.g. 1-100 or 1-100:2)")

        start, end, step = m.groups()

        if end is None:
            return cls(1, int(start), int(step or 1))

        return cls(int(start), int(end), int(step or 1))

    def __len__(self) -> int:
        """
        Number of indices in the range
        """
        return (self.end - self.start) // self.step + 1

    def spec(self) -> str:
        """
        Range in scheduler syntax, e.g. ``1-100`` or ``1-100:2``
        """
        s = f"{self.start}-{self.end}"
        return s if self.step == 1 else f"{s}:{self.step}"

@dataclass(frozen=True)
class Plan:
    """
    What the job runs: ``command`` once, or once per index in ``indices`` for an array.
    """

    command: str
    indices: IndexRange | None

    @property
    def is_array(self) -> bool:
        """
        Whether the job runs as an array
        """
        return self.indices is not None

@dataclass
class Job:
    """
    A scheduler-neutral job description.

    Array forms (mutually exclusive):
      * ``array="1-100"`` - ``command`` runs once per index; ``{i}`` or ``{}`` in it gives the
        index, as the scheduler's own index variable.
      * ``cmds="commands.txt"`` - task *i* runs line *i* of that file, so ``command`` must be
        unset.
      * ``file_list="files.txt"`` - task *i* runs ``command`` for the path on line *i* of that
        file: ``{}`` is the path, ``{/}`` its basename, ``{//}`` its directory, ``{.}`` it without
        extension and ``{/.}`` the basename without extension (the path is appended if there is
        no placeholder). Placeholders expand to quoted shell expressions, so don't quote them.

    Relative file paths are relative to ``cwd``, and the files must be readable by the job.

    Resource fields left as ``None`` are filled from config when rendered.
    """
    command: str | Sequence[str] | None = None
    array: str | int | range | IndexRange | None = None
    cmds: str | os.PathLike | None = None
    file_list: str | os.PathLike | None = None
    name: str | None = None
    cores: int | None = None
    mem: str | int | None = None
    time: str | int | None = None
    queue: str | None = None
    gpus: int | str | None = None
    logdir: str | os.PathLike | None = None
    cwd: str | os.PathLike | None = None
    max_parallel: int | None = None
    merge_logs: bool | None = None
    after: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """
        One ``name = value`` line for each field that is set
        """
        return "\n".join(f"{f.name} = {getattr(self, f.name)!r}" for f in fields(self)
                         if getattr(self, f.name) not in (None, [], ()))

    def command_str(self) -> str | None:
        """
        The command as a single shell string (argv lists are joined with spaces, like bsub)
        """
        if self.command is None or isinstance(self.command, str):
            return self.command or None
        return " ".join(self.command) or None

    def plan(self) -> Plan:
        """
        Work out what runs; command and file lists become arrays that read their line at run time
        """
        command = self.command_str()
        modes = [m for m, v in (("array", self.array), ("cmds", self.cmds), ("file_list", self.file_list))
                 if v is not None]
        if len(modes) > 1:
            raise UsubError(f"Only one of array, cmds and file_list can be given (got {', '.join(modes)})")

        if self.cmds is not None:
            if command:
                raise UsubError("Command cannot be combined with cmds")
            path, lines = _read_list(self.cwd, self.cmds)
            return Plan(f'sed -n "{{i}}p" {shlex.quote(path)} | bash', IndexRange(1, lines))

        if command is None:
            raise UsubError("No command given")

        if self.file_list is not None:
            path, lines = _read_list(self.cwd, self.file_list)
            if FILE_PLACEHOLDER_RE.search(command):
                command = FILE_PLACEHOLDER_RE.sub(lambda m: _FILE_EXPRS[m.group(1)], command)
            else:
                command += ' "$f"'
            return Plan(f'f=$(sed -n "{{i}}p" {shlex.quote(path)}); {command}', IndexRange(1, lines))

        if self.array is not None:
            return Plan(command, IndexRange.parse(self.array))
        return Plan(command, None)

    def default_name(self) -> str:
        """
        The job name, or else the program's (or command file's) basename, or ``usub``
        """
        if self.name:
            return safe_name(self.name)

        if self.cmds and not self.command_str():
            return safe_name(os.path.splitext(os.path.basename(os.fspath(self.cmds)))[0]) or "usub"

        try:
            first = shlex.split(self.command_str() or "")[0]
        except (ValueError, IndexError):
            return "usub"

        return safe_name(os.path.basename(first)) or "usub"

def _read_list(cwd: str | os.PathLike | None, path: str | os.PathLike) -> tuple[str, int]:
    """
    Absolute path of a list file (relative to ``cwd``) and its number of lines, not counting
    trailing blank lines
    """
    path = os.path.abspath(os.path.join(os.fspath(cwd or os.getcwd()), os.path.expanduser(os.fspath(path))))
    lines = 0
    try:
        with open(path, "rb") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    lines = n
    except OSError as e:
        raise UsubError(f"Cannot read {path}: {e.strerror}") from None

    if not lines:
        raise UsubError(f"{path} is empty")

    return path, lines

def safe_name(name: str) -> str:
    """
    Make a name safe for job names and log file names
    """
    return _NAME_UNSAFE_RE.sub("_", name).strip("_")
