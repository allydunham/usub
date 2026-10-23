"""Command-line interface: ``usub run`` / ``usub config``."""

from __future__ import annotations

import argparse
import os
import sys
import warnings

from . import __version__
from .errors import UsubError, UsubWarning

RUN_EPILOG = """\
examples:
  usub r -m 4G -c 2 -- python train.py                     print an LSF/Slurm/bash script
  usub r -s -m 4G -t 2h -- python train.py                 submit it, print the job id
  usub r -g 2 -- python train.py                           two GPUs (-g 'num=2:gmodel=A100' on LSF, -g a100:2 on Slurm)
  usub r -a 1-100 -j 10 -- python sim.py --seed {i}        index array, 10 at a time
  usub r -C commands.txt -j 10 -s                          one task per line of commands.txt
  usub r -F bams.txt -- samtools index {}                  the command for each path in bams.txt
  usub r -F reads.txt -- 'zcat {} | wc -l > {/.}.n'        placeholders can be used anywhere
  usub r -m 4G -- python train.py | bsub                   pipe the script to the scheduler
  usub r -b local -s -j 8 -C commands.txt                  run the commands locally, 8 at a time

With --array, {} and {i} in COMMAND give the index (as the scheduler's own index variable).
With --cmds, task N runs line N of the file; with --file-list it runs COMMAND for the path on line N:
  {} path, {/} basename, {//} directory, {.} without extension, {/.} basename without
  extension (the path is appended if COMMAND has none). Do not quote placeholders yourself.
Both files are read when each task starts, so they must be readable by the job.

Use -- before COMMAND when it starts with '-'. Use only one of -s, --emit cmd and --write.
"""

def positive_int(value: str) -> int:
    """
    argparse type for options that must be a positive integer
    """
    try:
        number = int(value)
    except ValueError:
        number = 0

    if number < 1:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")

    return number

def arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for ``usub run`` and ``usub config``
    """
    parser = argparse.ArgumentParser(prog="usub", allow_abbrev=False,
                                     description="Unified job submission tool")
    parser.add_argument("--version", action="version", version=f"usub {__version__}")

    # Create sub parsers for different verbs
    sub = parser.add_subparsers(dest="action", metavar="ACTION")

    # Run command
    r = sub.add_parser("run", aliases=["r"], allow_abbrev=False, epilog=RUN_EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       help="render (default) or submit a job",
                       description="Render a job script for COMMAND, or submit it with -s.")

    a = r.add_argument_group("arrays (one kind; default is a single job)")
    a.add_argument("-a", "--array", metavar="RANGE", help="index range, e.g. 1-100 or 1-100:2")
    a.add_argument("-C", "--cmds", metavar="FILE",
                   help="file of commands, one per line; task N runs line N")
    a.add_argument("-F", "--file-list", metavar="FILE",
                   help="file of paths, one per line; COMMAND is run for each")
    a.add_argument("-j", "--max-parallel", type=positive_int, metavar="N", help="max tasks running at once")

    res = r.add_argument_group("resources")
    res.add_argument("-c", "--cores", type=positive_int, metavar="N")
    res.add_argument("-m", "--mem", metavar="MEM", help="total memory, e.g. 500M, 4G")
    res.add_argument("-t", "--time", metavar="TIME", help="time limit, e.g. 90 (min), 2h, 1:30:00")
    res.add_argument("-q", "--queue", metavar="Q", help="queue / partition")
    res.add_argument("-g", "--gpus", metavar="SPEC",
                     help="GPUs: a count (2) or the scheduler's own spec, e.g. LSF 'num=2:gmodel=A100' "
                          "or Slurm 'a100:2'")
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
    sel.add_argument("--config", metavar="FILE",
                     help="use only this config file (default: ~/.usub.toml, then a project .usub.toml)")
    sel.add_argument("--merge-logs", action=argparse.BooleanOptionalAction, default=None,
                     help="one log per job (per index for arrays) with stdout and stderr together, "
                          "and the script at its top for single jobs; default: separate .err and .script files")
    sel.add_argument("-s", "--submit", action="store_true", help="submit and print the job id")
    sel.add_argument("--emit", choices=["script", "cmd"], default="script",
                     help="print the job script (default) or a one-line submit command")
    sel.add_argument("--write", metavar="PATH", help="write the script to PATH instead of stdout")

    r.add_argument("command", nargs=argparse.REMAINDER, metavar="COMMAND")

    # Config command
    c = sub.add_parser("config", help="show or create the config file",
                       description="show: resolved profile and settings; init: write a template; "
                                   "path: print the config files in use (or the user config location)")
    c.add_argument("what", nargs="?", choices=["show", "init", "path"], default="show")
    c.add_argument("-p", "--profile")
    c.add_argument("--config", metavar="FILE")
    c.add_argument("--project", action="store_true",
                   help="init: create .usub.toml in the current directory instead of ~/.usub.toml")
    c.add_argument("--force", action="store_true", help="init: overwrite an existing file")

    return parser

def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """
    Print usub warnings as ``usub: warning: ...`` and format any others as Python normally does
    """
    if issubclass(category, UsubWarning):
        print(f"usub: warning: {message}", file=sys.stderr)
    else:
        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and dispatch to commands; returns the exit code
    """
    parser = arg_parser()
    args = parser.parse_args(argv)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always", UsubWarning)
            warnings.showwarning = _show_warning

            if args.action in ("run", "r"):
                return cmd_run(args)
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

    except OSError as e:
        print(f"usub: error: {e}", file=sys.stderr)
        return 2

def cmd_run(args: argparse.Namespace) -> int:
    """
    Build a Job from the arguments, then print, write or submit it
    """
    from . import Job, load_config, render

    if sum([args.submit, args.emit == "cmd", bool(args.write)]) > 1:
        raise UsubError("Use only one of --submit, --emit cmd and --write")

    command = args.command[1:] if args.command[:1] == ["--"] else args.command  # REMAINDER keeps it
    job = Job(
        command=command or None,
        array=args.array,
        cmds=os.path.abspath(args.cmds) if args.cmds else None,
        file_list=os.path.abspath(args.file_list) if args.file_list else None,
        name=args.name, cores=args.cores, mem=args.mem, time=args.time, queue=args.queue,
        gpus=args.gpus, logdir=args.logdir, cwd=args.cwd, max_parallel=args.max_parallel,
        merge_logs=args.merge_logs,
        after=args.after, extra=args.extra, setup=args.setup,
    )

    rendered = render(job, profile=args.profile, backend=args.backend,
                      config=load_config(args.config, args.cwd))

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
        print(f"usub: wrote {path}; submit with: {rendered.backend.submit_hint(path)}", file=sys.stderr)

    else:  # print the script
        rendered.prepare()
        sys.stdout.write(rendered.script)
        sys.stdout.flush()
        if sys.stdout.isatty():
            print(f"# submit with: usub r -s ... or pipe to: {rendered.backend.submit_hint(None)}",
                  file=sys.stderr)
    return 0

def cmd_config(args: argparse.Namespace) -> int:
    """
    ``config path`` / ``init`` / ``show``: locate, create or summarise the config
    """
    from .config import PROJECT_CONFIG, PROJECT_TEMPLATE, TEMPLATE, config_path, load_config, resolve
    from .job import Job

    if args.what == "init":
        path = os.path.join(os.getcwd(), PROJECT_CONFIG) if args.project else args.config or str(config_path())
        path = os.path.abspath(os.path.expanduser(path))

        if os.path.exists(path) and not args.force:
            raise UsubError(f"{path} already exists (use --force to overwrite)")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(PROJECT_TEMPLATE if args.project else TEMPLATE)

        print(f"wrote {path}", file=sys.stderr)
        return 0

    config = load_config(args.config)
    if args.what == "path":
        print("\n".join(map(str, config.sources)) or config_path())
        return 0

    r = resolve(Job(), profile=args.profile, config=config)
    print(f"config:  {', '.join(map(str, config.sources)) or '(none found; using built-in defaults)'}")
    print(f"profile: {r.profile or '(none)'}")
    print(f"backend: {r.backend}")
    print(r.job)

    for key, value in sorted(r.options.items()):
        print(f"{key} = {value!r}")

    if config.profiles:
        print(f"profiles: {', '.join(config.profiles)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
