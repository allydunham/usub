# usub — unified job submitter

One interface for submitting commands to HPC executors, currently supporting **LSF**, **Slurm** and **local parallel** execution.
Includes personal defaults per cluster and easy job arrays, and makes the generated script available for
fine-tuning.
Its purpose is to generate submission commands/scripts and optionally submit them rather than monitor jobs or run workflows.

## Install

```bash
uv tool install git+https://github.com/allydunham/usub   # or, from a checkout: uv tool install -e .
usub config init                                         # writes ~/.usub.toml to edit
```

Requires Python ≥ 3.11 with no dependencies. It adds about 30 ms to Python's own start-up (roughly 60 ms in
total), and the benchmark tests keep it that way.

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
usub r -C commands.txt -j 10                          # one task per line of commands.txt
usub r -F bams.txt -- samtools index {}               # the command for each path listed in bams.txt
usub r -F reads.txt -- 'zcat {} | wc -l > {/.}.n'     # placeholders can go anywhere in the command

usub r -b local -s -j 8 -C commands.txt               # same thing on this machine, 8 at a time
```

| Option | Meaning |
|---|---|
| `-a RANGE` | index array `1-100` or `1-100:2`; `{i}` or `{}` in the command give the index |
| `-C FILE` | array where task *N* runs line *N* of `FILE` (so it must be readable by the job) |
| `-F FILE` | array where task *N* runs the command for the path on line *N* of `FILE`; see Arrays |
| `-j N` | max tasks running at once |
| `-c` `-m` `-t` `-q` | cores, memory (`4G`), time (`90`, `2h`, `1:30:00`), queue/partition |
| `-g SPEC` | GPUs: a count (`2`) or the scheduler's own spec; see GPUs below |
| `-J NAME` `-o DIR` `--cwd DIR` | job name, log directory, working directory |
| `-d JOBID` | start after `JOBID` succeeds (repeatable) |
| `--setup LINE` | shell line run before the command, e.g. `module load …` (repeatable) |
| `-x ARGS` | raw scheduler arguments for anything not modelled, e.g. `-x '-R "select[avx2]"'`, `-x=--account=abc` |
| `-p` `-b` | config profile, backend (`lsf`, `slurm`, `local`) |
| `--merge-logs` | stdout, stderr (and the script, for a single job) in one log; see below |
| `-s` / `--write PATH` / `--emit cmd` | submit / write script / print one-line command (use only one) |

Use `--` before the command when it starts with `-`.

### GPUs

`-g 2` asks for two GPUs with the backend's default directive (`#BSUB -gpu num=2`,
`#SBATCH --gres=gpu:2`). Anything that isn't a plain number is passed through in the scheduler's own
syntax, so the request can change job by job without touching your config:

```bash
usub r -b lsf   -g "num=2:gmodel=A100:mode=exclusive_process" -- python train.py   # #BSUB -gpu num=2:gmodel=A100:mode=exclusive_process
usub r -b slurm -g a100:2 -- python train.py                                       # #SBATCH --gres=gpu:a100:2
```

If your site needs a different directive altogether, set `gpu_args` in the profile: a list of
directive templates in which `{gpus}` stands for the `-g` value, for example
`gpu_args = ["--gpus-per-node={gpus}"]`, and keep writing `-g 2`. For anything else, add raw
directives with `-x`. `gpus` can also be given a default in config, and `-g 0` switches that off.

## Arrays

There are three ways to specify an array and what each task runs, and they are mutually exclusive:

- `-a RANGE` runs one command for every index (write `{i}` or `{}` where the index goes),
- `-C FILE` runs line *N* of a file of commands as task *N*, and
- `-F FILE` runs the command once for each path in a file with one path per line, task *N* using
  line *N*. In the command, `{}` is the path, `{/}` its basename, `{//}` its directory, `{.}` the
  path without its extension (everything after the last dot) and `{/.}` the basename without it;
  if there is no placeholder the path is appended. They expand to quoted shell expressions, so
  **don't put quotes around them**. To make the list from a glob, use `ls data/*.bam > bams.txt`.

Both files are read when each task starts, so they must be readable by the job and shouldn't be
edited while the array is running. Every line is a task, so blank and `#` lines in the middle of a
`-C` file are tasks that do nothing; blank lines at the end of either file are ignored.

### Logs and scripts

Logs and scripts all go in the log directory (`-o`, default `~/usub_logs`), named after the
job name and the id the scheduler returns:

| File | Contents |
|---|---|
| `<name>.<id>` | stdout |
| `<name>.<id>.err` | stderr |
| `<name>.<id>.script` | the script that was submitted, with its generation time in the header |

Arrays write `<name>.<id>.<index>` and `<name>.<id>.<index>.err` for each index but keep a single
`.script`. With `--merge-logs` (or `merge_logs = true` in config; `--no-merge-logs` to undo) there
is no `.err`: stderr goes to the same log as stdout, and a single job's log starts with its own
script instead of having a `.script` file. A script that is rejected by the scheduler is kept as
`<name>.failed-<time>.script`. The local backend has no scheduler id, so its logs and script are
just `<name>.<index>[.err]` and `<name>.script`, and are overwritten by the next run of the same name.

Generated scripts stay close to what you would write by hand. A script is the scheduler's
`#BSUB`/`#SBATCH` lines, a `cd`, your `setup` lines and your command. An index array uses the
scheduler's own variable (`${LSB_JOBINDEX}`, `${SLURM_ARRAY_TASK_ID}`) where you wrote `{i}`, and a
commands file is the same thing with the command `sed -n "${LSB_JOBINDEX}p" commands.txt | bash`,
and a file list puts `f=$(sed -n "${LSB_JOBINDEX}p" files.txt);` in front of your command.
Locally an array is `seq … | xargs -P N bash -c '…'`, so `usub r … | bash` is all you need to run one.
This all makes it easier to customise and edit if the generated commands are not quite what you need.

## Configuration

Settings are applied with the following priority:

- Command line flags
- Local project `.usub.toml` (current directory or any parent)
- Your global `~/.usub.toml` (or the file named by the `USUB_CONFIG` environment variable)
- Built-in defaults

Higher priority settings override lower ones and anything unset looks through to the next level.
`setup` and `extra` lists accumulate instead of replacing. Run `usub config show [-p NAME]` to see the
files used and the resolved settings.

`usub config init` writes a commented template to `~/.usub.toml`; `usub config init --project` writes a minimal
`.usub.toml` in the current directory (neither overwrites an existing file unless you add `--force`).
`usub config path` lists the config files in use. `--config FILE` uses only that file.

```toml
default_profile = "auto"     # first profile whose host_match matches, else bsub/sbatch on PATH, else local

[defaults] # Defaults for all profiles
logdir = "~/usub_logs"
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
queue = "standard"
extra = ["--account=myproject"]
```

### Config fields

A config file has one top-level key and two kinds of table:

| Key | Meaning |
|---|---|
| `default_profile` | Name of the profile to use when `-p` isn't given. `"auto"` (the default) picks the first profile whose `host_match` matches this host, and otherwise uses just `[defaults]`. |
| `[defaults]` | Settings applied to every job, whichever profile is used. |
| `[profiles.NAME]` | A named set of settings, normally one per cluster. It is applied on top of `[defaults]`, and selected by `-p NAME`, `default_profile` or `host_match`. |

Both kinds of table accept these settings. Each can also be given on the command line, which wins:

| Field | Type, default | What it does | Command line |
|---|---|---|---|
| `backend` | `"lsf"`, `"slurm"`, `"local"` or a plugin name. Default: `lsf` if `bsub` is on `PATH`, else `slurm` if `sbatch` is, else `local` | Which scheduler to generate a script for and submit to | `-b` |
| `host_match` | list of globs, profiles only | Hostname patterns (e.g. `["farm*", "*.hpc.example.org"]`) that make this profile the automatic choice; the first matching profile in the file wins | |
| `queue` | string | LSF queue (`-q`) or Slurm partition (`--partition`) | `-q` |
| `cores` | integer, `1` | CPU cores on a single node (`-n` with `span[hosts=1]` on LSF, `--cpus-per-task` on Slurm) | `-c` |
| `mem` | `"4G"`, `"500M"` or MB as a number, `"1G"` | Total memory for the job; units `K` `M` `G` `T` are binary | `-m` |
| `time` | `"2h"`, `"1h30m"`, `"1:30:00"`, `"2-12:00:00"` or minutes, none | Wall-clock limit | `-t` |
| `gpus` | integer or string, none | GPUs to request: a count, or the scheduler's own spec; see GPUs | `-g` |
| `logdir` | path, `"~/usub_logs"` | Where logs and saved scripts go (`~` is expanded; relative paths are relative to the working directory) | `-o` |
| `max_parallel` | integer, none | Most array tasks running at once; locally it defaults to the number of CPUs | `-j` |
| `merge_logs` | boolean, `false` | Put stdout and stderr in one log (and the script at the top of a single job's log) | `--merge-logs` / `--no-merge-logs` |
| `extra` | list of strings | Raw scheduler options added to every job. Each entry becomes one directive line written as is, e.g. `["--account=myproject"]` or `['-R "select[avx2]"']`. Lists from every level are combined | `-x` |
| `setup` | list of strings | Shell lines run before the command, after the `cd`, e.g. `["module load python/3.12"]`. Lists from every level are combined | `--setup` |

Some things are per job only and can't be set in config: the command itself, `-a`/`-C`/`-F`,
the job name (`-J`), the working directory (`--cwd`) and dependencies (`-d`). The local backend
ignores `queue`, `cores`, `mem`, `time`, `gpus` and `extra`, and doesn't support `-d`.

Settings are type-checked when the file is read, and an unknown top-level key (such as a table
misspelt `[default]`) is an error. Any other key inside a table is a **backend option**: it is handed
to the backend, which uses the ones it recognises. A key that no built-in backend recognises is most
likely a typo, so it is reported as a warning (`usub: warning: Unknown option(s) foo for the slurm
backend ...`) and otherwise ignored. A key that belongs to a different built-in backend, such as
`lsf_mem_unit` in `[defaults]` while using a Slurm profile, is accepted quietly so that defaults can be
shared. `usub config show` lists every setting and option in effect.

| Backend option | Backend | Default | What it does |
|---|---|---|---|
| `lsf_mem_unit` | LSF | `"MB"` | Unit your site uses for `-M` and `rusage` memory values (`KB`, `MB` or `GB`, i.e. `LSF_UNIT_FOR_LIMITS`) |
| `lsf_rusage` | LSF | `true` | Also add `-R "select[mem>X] rusage[mem=X]"` next to `-M`, which some sites require so that memory is reserved as well as limited |
| `lsf_mem_per_core` | LSF | `false` | Divide the `select`/`rusage` reservation by `cores`, for sites that count reservations per core |
| `gpu_args` | LSF, Slurm | the default GPU directive | List of directive templates that replace it, with `{gpus}` standing for the `-g` value, e.g. `['-gpu "{gpus}:j_exclusive=yes"']` |

In Python, a config is built from `Profile` objects, which take the same fields (with backend
options in `options={...}`):
`Config(defaults=Profile(mem="2G"), profiles={"farm": Profile(backend="lsf", options={"lsf_mem_unit": "GB"})})`.

## Python API

```python
from usub import Job, render, submit

job = Job("samtools index {}", file_list="bams.txt", mem="2G", max_parallel=50)
print(render(job, profile="farm").script)   # inspect
job_id = submit(job, profile="farm")        # or render(...).submit()
```

`Job` takes the same settings as the CLI (`array=`, `cmds=`, `file_list=`, `cores=`, `mem=`, `time=`,
`queue=`, `gpus=`, `after=`, `extra=`, `setup=`, …). `render()` returns a `Rendered` object with
`.script`, `.command()`, `.write(path)` and `.submit()`. Bad input raises `usub.UsubError` (and
`usub.SubmitError` if the scheduler refuses a job); probable mistakes such as unknown options are
reported with `warnings.warn` as `usub.UsubWarning`.

## Adding a backend

Subclass `usub.backends.Backend`: set `name`, `directive_prefix`, `index_var` (the shell variable
holding the array index), `submit_program`, `job_id_re` (a regex capturing the job id from the
submit command's output), `oneliner_wrap` (if it has a one-line form) and `option_names` (the config
keys it understands, so that other keys can be reported as typos). Then implement
`directives(job, plan)`, returning one argv list or raw string per directive line. See
[lsf.py](src/usub/backends/lsf.py) (~80 lines). Built-in backends are registered in
`backends/base.py`. External packages can register one through the entry point group
`usub.backends`:

```toml
[project.entry-points."usub.backends"]
pbs = "my_pkg.pbs:PBS"
```

## Development

```bash
uv run pytest                         # UPDATE_GOLDEN=1 regenerates tests/golden/*.sh
uv run pytest -m benchmark -s         # end-to-end speed (overhead over bare python, budget 60 ms)
uv run pytest -m 'not benchmark'      # skip the timing tests
```

Development is done on the dev branch with releases periodically merged into release.
Issues and pull requests welcome.