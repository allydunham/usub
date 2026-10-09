"""Command-line interface: ``usub run`` / ``usub config``."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .errors import UsubError

RUN_EPILOG = """\
examples:
  usub r -m 4G -c 2 -- python train.py                     print an LSF/Slurm/bash script
  usub r -s -m 4G -t 2h -- python train.py                 submit it, print the job id
  usub r -a 1-100 -j 10 -- python sim.py --seed {i}        index array, 10 at a time
  usub r -C commands.txt -s                                one task per line of commands.txt
  usub r -f data/*.bam -- samtools index {}                one task per file
  usub r -F samples.txt -- 'zcat {} | wc -l > {/.}.n'      one task per path listed in a file
  usub r -f *.fq.gz -- fastqc | bsub                       pipe the script to the scheduler
  usub r -b local -s -j 8 -f *.txt -- gzip                 run locally, 8 in parallel

placeholders (--files): {} item, {/} basename, {//} dirname, {.} without extension,
  {/.} basename without extension, {i} task index; item is appended if none is used.
  Values are shell-quoted, so do not wrap placeholders in quotes.
  With --array, {} and {i} (or $USUB_INDEX) give the index.

Use -- before COMMAND when using -f (or when COMMAND starts with '-').
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="usub", allow_abbrev=False,
                                description="Unified job submission for LSF, Slurm and local runs.")
    p.add_argument("--version", action="version", version=f"usub {__version__}")
    sub = p.add_subparsers(dest="action", metavar="ACTION")

    r = sub.add_parser("run", aliases=["r"], allow_abbrev=False, epilog=RUN_EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       help="render (default) or submit a job",
                       description="Render a job script for COMMAND, or submit it with -s.")
    a = r.add_argument_group("arrays (one kind; default is a single job)")
    a.add_argument("-a", "--array", metavar="RANGE", help="index range, e.g. 1-100 or 1-100:2")
    a.add_argument("-C", "--cmds", metavar="FILE", help="file of commands, one per line ('-' = stdin)")
    a.add_argument("-f", "--files", nargs="+", metavar="PATH",
                   help="paths or quoted globs; COMMAND is a template")
    a.add_argument("-F", "--file-list", metavar="FILE", help="file of paths, one per line ('-' = stdin)")
    a.add_argument("-j", "--max-parallel", type=int, metavar="N", help="max tasks running at once")

    res = r.add_argument_group("resources")
    res.add_argument("-c", "--cores", type=int, metavar="N")
    res.add_argument("-m", "--mem", metavar="MEM", help="total memory, e.g. 500M, 4G")
    res.add_argument("-t", "--time", metavar="TIME", help="time limit, e.g. 90 (min), 2h, 1:30:00")
    res.add_argument("-q", "--queue", metavar="Q", help="queue / partition")
    res.add_argument("-g", "--gpus", type=int, metavar="N")
    res.add_argument("-J", "--name", metavar="NAME", help="job name (default: program name)")
    res.add_argument("-o", "--logdir", metavar="DIR")
    res.add_argument("--cwd", metavar="DIR", help="working directory (default: current)")
    res.add_argument("-d", "--after", action="append", default=[], metavar="JOBID",
                     help="start after JOBID succeeds (repeatable)")
    res.add_argument("--setup", action="append", default=[], metavar="LINE",
                     help="shell line to run before the command (repeatable)")
    res.add_argument("-x", "--extra", action="append", default=[], metavar="ARGS",
                     help="raw scheduler arguments, e.g. -x '-R \"select[avx2]\"' or "
                          "-x=--account=abc (repeatable)")

    sel = r.add_argument_group("selection and output")
    sel.add_argument("-p", "--profile", help="config profile (default: auto)")
    sel.add_argument("-b", "--backend", help="lsf, slurm, local, or a plugin backend")
    sel.add_argument("--config", metavar="FILE", help="config file (default: ~/.config/usub/config.toml)")
    sel.add_argument("-s", "--submit", action="store_true", help="submit and print the job id")
    sel.add_argument("--emit", choices=["script", "cmd"], default="script",
                     help="print the job script (default) or a one-line submit command")
    sel.add_argument("--write", metavar="PATH", help="write the script to PATH instead of stdout")
    r.add_argument("command", nargs=argparse.REMAINDER, metavar="COMMAND")

    c = sub.add_parser("config", help="show or create the config file",
                       description="show: resolved profile and settings; init: write a template; "
                                   "path: print the config path")
    c.add_argument("what", nargs="?", choices=["show", "init", "path"], default="show")
    c.add_argument("-p", "--profile")
    c.add_argument("--config", metavar="FILE")
    c.add_argument("--force", action="store_true", help="init: overwrite an existing file")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = None
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action in ("run", "r"):
            return cmd_run(args, tail)
        if tail:
            parser.error("unexpected arguments after --")
        if args.action == "config":
            return cmd_config(args)
        parser.print_help()
        return 0
    except UsubError as e:
        print(f"usub: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


def _read_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    try:
        with open(path) as fh:
            return fh.read().splitlines()
    except OSError as e:
        raise UsubError(f"cannot read {path}: {e.strerror}") from None


def _expand_files(patterns: list[str]) -> list[str]:
    import glob

    out: list[str] = []
    for pat in patterns:
        if glob.has_magic(pat):
            matches = sorted(glob.glob(pat))
            if not matches:
                raise UsubError(f"no files match {pat!r}")
            out += matches
        else:
            out.append(pat)
    return out


def cmd_run(args: argparse.Namespace, tail: list[str] | None) -> int:
    from . import Job, load_config, render

    command = args.command
    if tail is not None:
        if command:
            raise UsubError(f"unexpected arguments before --: {' '.join(command)}")
        command = tail
    if args.cmds == "-" and args.file_list == "-":
        raise UsubError("--cmds and --file-list cannot both read stdin")

    files = None
    if args.files is not None or args.file_list is not None:
        files = _expand_files(args.files or [])
        if args.file_list:
            files += [ln.strip() for ln in _read_lines(args.file_list) if ln.strip()]

    job = Job(
        command=command or None,
        array=args.array,
        cmds=_read_lines(args.cmds) if args.cmds else None,
        files=files,
        name=args.name, cores=args.cores, mem=args.mem, time=args.time, queue=args.queue,
        gpus=args.gpus, logdir=args.logdir, cwd=args.cwd, max_parallel=args.max_parallel,
        after=args.after, extra=args.extra, setup=args.setup,
    )
    rendered = render(job, profile=args.profile, backend=args.backend,
                      config=load_config(args.config))

    if args.submit:
        job_id = rendered.submit()
        print(job_id)
        print(f"usub: submitted {rendered.job.name} ({rendered.backend.name} job {job_id}); "
              f"logs in {rendered.job.logdir}", file=sys.stderr)
    elif args.emit == "cmd":
        line = rendered.command()
        rendered.prepare()
        print(line)
    elif args.write:
        path = rendered.write(args.write)
        print(f"usub: wrote {path}; submit with: {rendered.submit_hint(path)}", file=sys.stderr)
    else:
        rendered.prepare()
        sys.stdout.write(rendered.script)
        sys.stdout.flush()
        if sys.stdout.isatty():
            print(f"# submit with: usub r -s ... or pipe to: {rendered.submit_hint(None)}",
                  file=sys.stderr)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from .config import TEMPLATE, config_path, load_config, resolve
    from .job import Job

    path = args.config and os.path.expanduser(args.config) or str(config_path())
    if args.what == "path":
        print(path)
        return 0
    if args.what == "init":
        if os.path.exists(path) and not args.force:
            raise UsubError(f"{path} already exists (use --force to overwrite)")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(TEMPLATE)
        print(f"wrote {path}", file=sys.stderr)
        return 0

    config = load_config(args.config)
    r = resolve(Job(command="true", name="-"), profile=args.profile, config=config)
    exists = "" if os.path.exists(path) else " (not found; using built-in defaults)"
    print(f"config:  {path}{exists}")
    print(f"profile: {r.profile or '(none)'}")
    print(f"backend: {r.backend}")
    for key in ("cores", "mem", "time", "queue", "gpus", "logdir", "max_parallel", "setup", "extra"):
        value = getattr(r.job, key)
        if value not in (None, []):
            print(f"  {key} = {value!r}")
    for key, value in sorted(r.options.items()):
        print(f"  {key} = {value!r}")
    if config.profiles:
        print(f"profiles: {', '.join(config.profiles)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
