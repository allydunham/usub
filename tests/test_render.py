import os
import re
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from usub import Config, Job, Profile, UsubError, UsubWarning, render
from usub.backends import Backend, get_backend

GOLDEN = Path(__file__).parent / "golden"
CMDS = Path(__file__).parent / "data" / "commands.txt"
FILES = Path(__file__).parent / "data" / "files.txt"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

BASE = dict(cwd="/work", logdir="logs", name="job")
JOBS = {
    "single": Job("python train.py --lr 0.1 | tee out.txt", cores=4, mem="4G", time="2h",
                  queue="normal", gpus=1, after=["123"], extra=["-x-raw 1"], setup=["module load py"],
                  **BASE),
    "range": Job("python sim.py --seed {i}", array="1-100:2", max_parallel=10, **BASE),
    "cmds": Job(cmds=str(CMDS), max_parallel=3, **BASE),
    "files": Job("samtools index {}", file_list=str(FILES), max_parallel=5, **BASE),
    "single_merged": Job("python train.py", merge_logs=True, **BASE),
    "cmds_merged": Job(cmds=str(CMDS), max_parallel=3, merge_logs=True, **BASE),
}

def masked(script):
    """Replace the generation time and the location of the commands file so scripts can be compared."""
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "<TIME>", script).replace(str(CMDS.parent), "<DATA>")

def rendered(backend, kind, **options):
    config = Config(defaults=Profile(backend=backend, options=options))
    job = JOBS[kind]
    if backend == "local" and job.after:
        job = Job(**{**job.__dict__, "after": []})
    return render(job, config=config)

@pytest.mark.parametrize("backend", ["lsf", "slurm", "local"])
@pytest.mark.parametrize("kind", list(JOBS))
def test_golden(backend, kind):
    script = rendered(backend, kind).script
    path = GOLDEN / f"{backend}_{kind}.sh"
    if UPDATE:
        path.write_text(masked(script))
    assert masked(script) == path.read_text()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)

def test_lsf_options():
    s = rendered("lsf", "single", lsf_mem_unit="GB", lsf_mem_per_core=True,
                 gpu_args=['-gpu "num={gpus}:j_exclusive=yes"']).script
    assert "#BSUB -M 4\n" in s
    assert '#BSUB -R "select[mem>1] rusage[mem=1]"' in s
    assert '#BSUB -gpu "num=1:j_exclusive=yes"' in s
    assert "rusage" not in rendered("lsf", "single", lsf_rusage=False).script
    with pytest.raises(UsubError, match="lsf_mem_unit"):
        rendered("lsf", "single", lsf_mem_unit="PB")

@pytest.mark.parametrize("backend, gpus, directive", [
    ("lsf", 2, "#BSUB -gpu num=2\n"),
    ("lsf", "2", "#BSUB -gpu num=2\n"),
    ("lsf", "num=2:gmodel=A100:mode=exclusive_process", "#BSUB -gpu num=2:gmodel=A100:mode=exclusive_process\n"),
    ("slurm", 2, "#SBATCH --gres=gpu:2\n"),
    ("slurm", "a100:2", "#SBATCH --gres=gpu:a100:2\n"),
])
def test_gpu_spec(backend, gpus, directive):
    """A count uses the backend's default directive; any other spec is passed through to it."""
    script = render(Job("x", gpus=gpus, **BASE), config=Config(defaults=Profile(backend=backend))).script
    assert directive in script

@pytest.mark.parametrize("gpus", [None, 0, "0"])
@pytest.mark.parametrize("backend", ["lsf", "slurm"])
def test_no_gpu_directive_without_gpus(backend, gpus):
    script = render(Job("x", gpus=gpus, **BASE), config=Config(defaults=Profile(backend=backend))).script
    assert "gpu" not in script and "gres" not in script

def test_gpu_args_template_receives_the_spec():
    config = Config(defaults=Profile(backend="lsf", options={"gpu_args": ['-gpu "{gpus}:j_exclusive=yes"']}))
    assert '#BSUB -gpu "num=2:j_exclusive=yes"\n' in render(Job("x", gpus="num=2", **BASE), config=config).script

def test_slurm_mem_units():
    s = render(Job("x", mem="1500M", **BASE), config=Config(defaults=Profile(backend="slurm"))).script
    assert "--mem=1500M" in s

@pytest.mark.parametrize("backend, expected", [
    ("lsf", "bsub -J 'job[1-100:2]%10' -o /work/logs/job.%J.%I -e /work/logs/job.%J.%I.err "
            "-M 1024 -R 'select[mem>1024] rusage[mem=1024]' 'python sim.py --seed ${LSB_JOBINDEX}'"),
    ("slurm", "sbatch --job-name=job --output=/work/logs/job.%A.%a --error=/work/logs/job.%A.%a.err "
              "--array=1-100:2%10 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G "
              "'--wrap=python sim.py --seed ${SLURM_ARRAY_TASK_ID}'"),
    ("local", "seq 1 2 100 | xargs -n 1 -P 10 bash -c 'python sim.py --seed ${1}' _"),
])
def test_oneliner_range(backend, expected):
    assert rendered(backend, "range").command() == expected

@pytest.mark.parametrize("backend, prefix, index_var", [
    ("lsf", "", "${LSB_JOBINDEX}"), ("slurm", "--wrap=", "${SLURM_ARRAY_TASK_ID}"), ("local", "", "${1}"),
])
def test_oneliner_cmds_file(backend, prefix, index_var):
    task = shlex.quote(f'{prefix}sed -n "{index_var}p" {CMDS} | bash')
    assert task in rendered(backend, "cmds").command()

def test_local_rejects_dependencies():
    with pytest.raises(UsubError, match="dependencies"):
        render(JOBS["single"], config=Config(defaults=Profile(backend="local")))

@pytest.mark.parametrize("backend", ["lsf", "slurm", "local"])
@pytest.mark.parametrize("kind", list(JOBS))
def test_scripts_have_no_usub_variables_or_helpers(backend, kind):
    """Scripts should read like what you would type: only the job's own commands and the scheduler's."""
    assert "USUB" not in rendered(backend, kind).script.replace("generated by usub", "")

def run_task(script, index):
    """Run a Slurm script as task ``index`` of an array and return its stdout."""
    return subprocess.run(["bash"], input=script, text=True, capture_output=True, check=True,
                          env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index)}).stdout

def test_commands_file_runs_the_right_line_per_index(tmp_path):
    (tmp_path / "cmds.txt").write_text("echo 'a b'\necho c | tr a-z A-Z\n")
    r = render(Job(cmds="cmds.txt", cwd=tmp_path, logdir="l", name="t"),
               config=Config(defaults=Profile(backend="slurm")))
    assert [run_task(r.script, i) for i in (1, 2)] == ["a b\n", "C\n"]

def test_file_list_runs_the_command_for_the_right_path(tmp_path):
    (tmp_path / "files.txt").write_text("d.x/a b.fq.gz\nd.x/c.tar.gz\n")
    r = render(Job("echo {i}: {} {/} {//} {.} {/.}", file_list="files.txt", cwd=tmp_path, logdir="l", name="t"),
               config=Config(defaults=Profile(backend="slurm")))
    assert [run_task(r.script, i) for i in (1, 2)] == [
        "1: d.x/a b.fq.gz a b.fq.gz d.x d.x/a b.fq a b.fq\n",
        "2: d.x/c.tar.gz c.tar.gz d.x d.x/c.tar c.tar\n",
    ]

def test_file_list_path_without_placeholder_is_appended(tmp_path):
    (tmp_path / "files.txt").write_text("my file.txt\n")
    r = render(Job("echo", file_list="files.txt", cwd=tmp_path, logdir="l", name="t"),
               config=Config(defaults=Profile(backend="slurm")))
    assert run_task(r.script, 1) == "my file.txt\n"

def test_unknown_backend():
    with pytest.raises(UsubError, match="Unknown backend 'pbs'"):
        get_backend("pbs")

def test_plugin_backend(monkeypatch):
    class Fake(Backend):
        name = "fake"
        directive_prefix = "#FAKE"

        def directives(self, job, plan):
            return [["--name", job.name], "raw line"]

    class EP:
        name = "fake"

        def load(self):
            return Fake

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group, name=None: [EP()] if name in (None, "fake") else [])
    s = render(Job("x", **BASE), backend="fake", config=Config()).script
    assert "#FAKE --name job\n#FAKE raw line\n" in s

@pytest.mark.parametrize("backend, kind, out_line, err_line", [
    ("lsf", "single", "#BSUB -o /work/logs/job.%J\n", "#BSUB -e /work/logs/job.%J.err\n"),
    ("lsf", "range", "#BSUB -o /work/logs/job.%J.%I\n", "#BSUB -e /work/logs/job.%J.%I.err\n"),
    ("slurm", "single", "#SBATCH --output=/work/logs/job.%j\n", "#SBATCH --error=/work/logs/job.%j.err\n"),
    ("slurm", "range", "#SBATCH --output=/work/logs/job.%A.%a\n", "#SBATCH --error=/work/logs/job.%A.%a.err\n"),
])
def test_log_directives(backend, kind, out_line, err_line):
    split = rendered(backend, kind).script
    assert out_line in split and err_line in split
    merged = render(replace(JOBS[kind], merge_logs=True), config=Config(defaults=Profile(backend=backend)))
    assert out_line in merged.script and err_line not in merged.script  # no -e/--error: stderr joins stdout

def test_save_names_script_after_job_id(tmp_path):
    r = render(Job("echo hi", cwd=tmp_path, logdir="logs", name="job"),
               config=Config(defaults=Profile(backend="slurm")))
    path = r.save("42")
    assert path == str(tmp_path / "logs" / "job.42.script")
    assert Path(path).read_text() == r.script and os.access(path, os.X_OK)

def test_save_skips_merged_single_job_unless_forced(tmp_path):
    kw = dict(cwd=tmp_path, logdir="logs", name="job", merge_logs=True)
    config = Config(defaults=Profile(backend="slurm"))
    single = render(Job("echo hi", **kw), config=config)
    assert single.save("42") is None and not (tmp_path / "logs").exists()
    assert single.save("42", force=True) == str(tmp_path / "logs" / "job.42.script")
    array = render(Job("echo {i}", array=2, **kw), config=config)
    assert array.save("43") == str(tmp_path / "logs" / "job.43.script")  # arrays keep one script

def test_merged_single_job_log_starts_with_script(tmp_path):
    """Run the script as a scheduler would: stderr and stdout go to the same log."""
    r = render(Job("echo hello", cwd=tmp_path, logdir="logs", name="job", merge_logs=True),
               config=Config(defaults=Profile(backend="slurm")))
    log = subprocess.run(["bash"], input=r.script, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=True).stdout
    lines = log.splitlines()
    assert lines[0] == "#!/bin/bash" and "#SBATCH --output=" in log and "# generated by usub" in log
    assert lines.count("echo hello") == 1 and lines[-1] == "hello"  # script copy, then output
    assert "SCRIPT" not in log

def test_unmerged_script_does_not_print_itself(tmp_path):
    r = render(Job("echo hello", cwd=tmp_path, logdir="logs", name="job"),
               config=Config(defaults=Profile(backend="slurm")))
    out = subprocess.run(["bash"], input=r.script, text=True, capture_output=True, check=True)
    assert out.stdout == "hello\n" and out.stderr == ""

def test_unknown_options_warn_with_the_accepted_ones():
    config = Config(defaults=Profile(backend="lsf", options={"tipo": 1, "lsf_mem_unit": "GB", "also_wrong": 2}))
    with pytest.warns(UsubWarning, match=r"Unknown option\(s\) also_wrong, tipo for the lsf backend .*gpu_args, "
                                         r"lsf_mem_per_core, lsf_mem_unit, lsf_rusage"):
        render(Job("x", **BASE), config=config)

@pytest.mark.parametrize("backend, options", [
    ("lsf", {"lsf_mem_unit": "GB", "lsf_rusage": True, "lsf_mem_per_core": False, "gpu_args": []}),
    ("slurm", {"gpu_args": ["--gres=gpu:{gpus}"]}),
    ("slurm", {"lsf_mem_unit": "GB"}),  # another built-in backend's option: profiles can share defaults
    ("local", {"lsf_rusage": False, "gpu_args": []}),
    ("local", {}),
])
def test_known_options_do_not_warn(backend, options, recwarn):
    render(Job("x", **BASE), config=Config(defaults=Profile(backend=backend, options=options)))
    assert not [w for w in recwarn if issubclass(w.category, UsubWarning)]

def test_plugin_backend_options_are_checked_against_the_plugin(monkeypatch, recwarn):
    class Fake(Backend):
        name = "fake"
        directive_prefix = "#FAKE"
        option_names = frozenset({"fake_option"})

        def directives(self, job, plan):
            return []

    class EP:
        name = "fake"

        def load(self):
            return Fake

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group, name=None: [EP()] if name in (None, "fake") else [])
    render(Job("x", **BASE), backend="fake", config=Config(defaults=Profile(options={"fake_option": 1})))
    assert not [w for w in recwarn if issubclass(w.category, UsubWarning)]
    with pytest.warns(UsubWarning, match="Unknown option.* nope for the fake backend .*fake_option"):
        render(Job("x", **BASE), backend="fake", config=Config(defaults=Profile(options={"nope": 1})))

def test_invalid_gpu_args_template_is_a_clear_error():
    config = Config(defaults=Profile(backend="slurm", options={"gpu_args": ["--x={gpus}{oops}"]}))
    with pytest.raises(UsubError, match=r"Invalid gpu_args template .*only \{gpus\}"):
        render(Job("x", gpus=1, **BASE), config=config)

@pytest.mark.parametrize("backend", ["lsf", "slurm"])
def test_newlines_cannot_inject_directives(backend):
    with pytest.raises(UsubError, match="cannot contain newlines"):
        render(Job("x", queue="a\n#BSUB -x", **BASE), config=Config(defaults=Profile(backend=backend)))
    with pytest.raises(UsubError, match="cannot contain newlines"):
        render(Job("x", extra=["-x\n-y"], **BASE), config=Config(defaults=Profile(backend=backend)))
