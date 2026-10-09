# usub — unified job submitter

One interface for submitting commands to **LSF**, **Slurm** or **local parallel** execution, with
personal defaults per cluster, easy job arrays, and the generated script always available for
fine-tuning. It submits jobs; it does not monitor them or run workflows.

## Install

```bash
uv tool install git+<repo-url>        # or, from a checkout: uv tool install -e .
usub config init                      # writes ~/.config/usub/config.toml to edit
```

Requires Python ≥ 3.11 and has no dependencies, so startup is ~60 ms.

## Usage

`usub run` (alias `usub r`) **prints** a job script by default; add `-s` to submit it.

```bash
usub r -m 4G -c 2 -t 2h -- python train.py            # print the script for the current cluster
usub r -s -m 4G -c 2 -t 2h -- python train.py         # submit; prints the job id
usub r -m 4G -- python train.py | bsub                # or pipe the script yourself (sbatch works too)
usub r --write job.sh -- python train.py              # save the script to edit by hand
usub r --emit cmd -- python train.py                  # a one-line bsub/sbatch command instead

# arrays
usub r -a 1-100 -j 10 -- python sim.py --seed {i}     # index array, at most 10 running at once
usub r -C commands.txt                                # one task per line of commands.txt
usub r -f data/*.bam -- samtools index {}             # one task per file
usub r -F samples.txt -- 'zcat {} | wc -l > {/.}.n'   # one task per path listed in a file

usub r -b local -s -j 8 -f *.fq.gz -- fastqc          # same thing on this machine, 8 at a time
```

| Option | Meaning |
|---|---|
| `-a RANGE` | index array `1-100` or `1-100:2`; `{i}`, `{}` or `$USUB_INDEX` in the command give the index |
| `-C FILE` | one command per line (`-` = stdin; blank and `#` lines skipped) |
| `-f PATH…` / `-F FILE` | file array from paths/globs, or from a file of paths |
| `-j N` | max tasks running at once |
| `-c` `-m` `-t` `-q` `-g` | cores, memory (`4G`), time (`90`, `2h`, `1:30:00`), queue/partition, GPUs |
| `-J NAME` `-o DIR` `--cwd DIR` | job name, log directory, working directory |
| `-d JOBID` | start after `JOBID` succeeds (repeatable) |
| `--setup LINE` | shell line run before the command, e.g. `module load …` (repeatable) |
| `-x ARGS` | raw scheduler arguments for anything not modelled, e.g. `-x '-R "select[avx2]"'`, `-x=--account=abc` |
| `-p` `-b` | config profile, backend (`lsf`, `slurm`, `local`) |
| `-s` / `--write PATH` / `--emit cmd` | submit / write script / print one-line command |

Use `--` before the command when using `-f`, or when the command starts with `-`.

**Placeholders** (with `-f`/`-F`, GNU parallel style): `{}` path, `{/}` basename, `{//}` dirname,
`{.}` without extension, `{/.}` basename without extension, `{i}` task index. If the command has no
placeholder, the path is appended. Values are shell-quoted automatically, so **don't put quotes
around placeholders**.

Logs go to `<logdir>/<name>.<jobid>[.<index>].out|err`. Submitted scripts are kept in
`<logdir>/scripts/`. Array tasks are embedded in the script itself, so a printed script is
complete and reproducible. Very large task lists (>1 MB) go to a `.tasks` file next to the logs.

## Configuration

`~/.config/usub/config.toml` (or `$XDG_CONFIG_HOME/usub/…`, or `$USUB_CONFIG`):

```toml
default_profile = "auto"     # first profile whose host_match matches, else bsub/sbatch on PATH, else local

[defaults]
logdir = "~/usub-logs"
mem = "1G"

[profiles.farm]
backend = "lsf"
host_match = ["farm*"]
queue = "normal"
lsf_mem_unit = "MB"          # site LSF_UNIT_FOR_LIMITS
lsf_rusage = true            # add -R "select[mem>X] rusage[mem=X]"

[profiles.hpc]
backend = "slurm"
host_match = ["login*.hpc.example.org"]
partition = "standard"
extra = ["--account=myproject"]
```

Later sources override earlier ones: built-ins → `[defaults]` → profile → environment
(`USUB_PROFILE`, `USUB_BACKEND`, `USUB_LOGDIR`, `USUB_QUEUE`) → command line. `setup` and `extra`
lists accumulate instead of replacing. Run `usub config show [-p NAME]` to see the resolved settings.

Backend options: `lsf_mem_unit`, `lsf_rusage`, `lsf_mem_per_core`, `gpu_args` (raw directives
with `{gpus}`, e.g. `['-gpu "num={gpus}:j_exclusive=yes"']`), `embed_limit`.

## Python API

```python
from usub import Job, render, submit

job = Job("samtools index {}", files=bam_paths, mem="2G", max_parallel=50)
print(render(job, profile="farm").script)   # inspect
job_id = submit(job, profile="farm")        # or render(...).submit()
```

`Job` takes the same settings as the CLI (`array=`, `cmds=`, `files=`, `cores=`, `mem=`, `time=`,
`queue=`, `gpus=`, `after=`, `extra=`, `setup=`, …). `render()` returns a `Rendered` object with
`.script`, `.command()`, `.write(path)` and `.submit()`.

## Adding a backend

Subclass `usub.backends.Backend`: set `name`, `directive_prefix`, `index_var`, `submit_program`
and `job_id_re`, then implement `directives(job, plan)`, returning one argv list or raw string
per directive line. See [lsf.py](src/usub/backends/lsf.py) (~80 lines). Built-in backends are
registered in `backends/base.py`. External packages can register one through the entry point
group `usub.backends`:

```toml
[project.entry-points."usub.backends"]
pbs = "my_pkg.pbs:PBS"
```

## Development

```bash
uv run pytest                         # UPDATE_GOLDEN=1 regenerates tests/golden/*.sh
```
