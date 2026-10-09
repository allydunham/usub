"""Scheduler-neutral job description and array expansion."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field

from .errors import UsubError

# GNU parallel style placeholders: {} {/} {//} {.} {/.} {i}
PLACEHOLDER_RE = re.compile(r"\{(//|/\.|/|\.|i|)\}")
_RANGE_RE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?(?:\s*:\s*(\d+))?\s*$")
_NAME_UNSAFE_RE = re.compile(r"[^\w.-]+")

INDEX_VAR = "USUB_INDEX"


@dataclass(frozen=True)
class IndexRange:
    """An inclusive array index range, e.g. ``1-100:2``."""

    start: int
    end: int
    step: int = 1

    def __post_init__(self):
        if self.start < 0 or self.end < self.start or self.step < 1:
            raise UsubError(f"invalid array range {self.spec()!r}")

    @classmethod
    def parse(cls, value: str | int | range | IndexRange) -> IndexRange:
        """Parse ``"1-100"``, ``"1-100:2"`` or ``"N"`` (meaning ``1-N``)."""
        if isinstance(value, IndexRange):
            return value
        if isinstance(value, range):
            if not len(value):
                raise UsubError("empty array range")
            return cls(value.start, value[-1], value.step)
        if isinstance(value, int):
            return cls(1, value)
        m = _RANGE_RE.match(value)
        if not m:
            raise UsubError(f"invalid array range {value!r} (expected e.g. 1-100 or 1-100:2)")
        start, end, step = m.groups()
        if end is None:
            return cls(1, int(start), int(step or 1))
        return cls(int(start), int(end), int(step or 1))

    def __len__(self) -> int:
        return (self.end - self.start) // self.step + 1

    def spec(self) -> str:
        s = f"{self.start}-{self.end}"
        return s if self.step == 1 else f"{s}:{self.step}"


@dataclass(frozen=True)
class Plan:
    """What each task runs.

    Exactly one of ``command`` (single job or index range) and ``tasks`` (one
    pre-expanded command per array element, selected by 1-based index) is set.
    """

    command: str | None
    tasks: tuple[str, ...] | None
    indices: IndexRange | None

    @property
    def is_array(self) -> bool:
        return self.indices is not None


@dataclass
class Job:
    """A scheduler-neutral job description.

    Array forms (mutually exclusive):
      * ``array="1-100"`` – ``command`` runs once per index; ``{i}``/``{}`` and
        ``$USUB_INDEX`` give the index.
      * ``cmds=[...]`` – one shell command per task; ``command`` must be unset.
      * ``files=[...]`` – ``command`` is a template; ``{}`` etc. are replaced with
        each (shell-quoted) item, which is appended if there is no placeholder.

    Resource fields left as ``None`` are filled from config when rendered.
    """

    command: str | Sequence[str] | None = None
    array: str | int | range | IndexRange | None = None
    cmds: Sequence[str] | None = None
    files: Sequence[str | os.PathLike] | None = None
    name: str | None = None
    cores: int | None = None
    mem: str | int | None = None
    time: str | int | None = None
    queue: str | None = None
    gpus: int | None = None
    logdir: str | os.PathLike | None = None
    cwd: str | os.PathLike | None = None
    max_parallel: int | None = None
    after: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)

    def command_str(self) -> str | None:
        if self.command is None or isinstance(self.command, str):
            return self.command or None
        return " ".join(self.command) or None

    def plan(self) -> Plan:
        command = self.command_str()
        modes = [m for m, v in (("array", self.array), ("cmds", self.cmds), ("files", self.files))
                 if v is not None]
        if len(modes) > 1:
            raise UsubError(f"only one of array/cmds/files may be given (got {', '.join(modes)})")
        mode = modes[0] if modes else None

        if mode == "cmds":
            if command:
                raise UsubError("a command cannot be combined with cmds; put it in the command list")
            tasks = tuple(c for c in (c.strip() for c in self.cmds) if c and not c.startswith("#"))
            return _task_plan(tasks)
        if command is None:
            raise UsubError("no command given")
        if mode == "files":
            items = [os.fspath(f) for f in self.files]
            return _task_plan(tuple(expand(command, item, i) for i, item in enumerate(items, 1)))
        if mode == "array":
            indices = IndexRange.parse(self.array)
            cmd = PLACEHOLDER_RE.sub(lambda m: _range_sub(m, command), command)
            return Plan(cmd, None, indices)
        return Plan(command, None, None)

    def default_name(self) -> str:
        if self.name:
            return safe_name(self.name)
        command = self.command_str()
        if command:
            try:
                first = shlex.split(command)[0]
            except (ValueError, IndexError):
                first = ""
            if first:
                return safe_name(os.path.basename(first)) or "usub"
        return "usub"


def _task_plan(tasks: tuple[str, ...]) -> Plan:
    if not tasks:
        raise UsubError("array has no tasks (empty command or file list)")
    for t in tasks:
        if "\n" in t:
            raise UsubError(f"task commands must be single lines: {t!r}")
    return Plan(None, tasks, IndexRange(1, len(tasks)))


def _range_sub(m: re.Match, command: str) -> str:
    if m.group(1) not in ("", "i"):
        raise UsubError(f"placeholder {m.group(0)} is only valid with files: {command!r}")
    return "${" + INDEX_VAR + "}"


def _strip_ext(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root if ext else path


def expand(template: str, item: str, index: int) -> str:
    """Substitute placeholders in ``template`` for one item (appending it if none)."""

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key == "i":
            return str(index)
        if key == "":
            value = item
        elif key == "/":
            value = os.path.basename(item)
        elif key == "//":
            value = os.path.dirname(item) or "."
        elif key == ".":
            value = _strip_ext(item)
        else:  # "/."
            value = _strip_ext(os.path.basename(item))
        return shlex.quote(value)

    if not PLACEHOLDER_RE.search(template):
        return f"{template} {shlex.quote(item)}"
    return PLACEHOLDER_RE.sub(sub, template)


def safe_name(name: str) -> str:
    return _NAME_UNSAFE_RE.sub("_", name).strip("_")
