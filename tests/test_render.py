import os
import subprocess
from pathlib import Path

import pytest

from usub import Config, Job, UsubError, render
from usub.backends import Backend, get_backend

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

BASE = dict(cwd="/work", logdir="logs", name="job")
JOBS = {
    "single": Job("python train.py --lr 0.1 | tee out.txt", cores=4, mem="4G", time="2h",
                  queue="normal", gpus=1, after=["123"], extra=["-x-raw 1"], setup=["module load py"],
                  **BASE),
    "range": Job("python sim.py --seed {i}", array="1-100:2", max_parallel=10, **BASE),
    "cmds": Job(cmds=["echo one", "echo 'two words'"], max_parallel=3, **BASE),
    "files": Job("samtools index {}", files=["a.bam", "b c.bam"], max_parallel=5, **BASE),
}


def rendered(backend, kind, **options):
    config = Config(defaults={"backend": backend, **options})
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
        path.write_text(script)
    assert script == path.read_text()
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


def test_slurm_mem_units():
    s = render(Job("x", mem="1500M", **BASE), config=Config(defaults={"backend": "slurm"})).script
    assert "--mem=1500M" in s


@pytest.mark.parametrize("backend, expected", [
    ("lsf", "bsub -J 'job[1-100:2]%10' -o /work/logs/job.%J.%I.out -e /work/logs/job.%J.%I.err "
            "-M 1024 -R 'select[mem>1024] rusage[mem=1024]' "
            "'export USUB_INDEX=\"${LSB_JOBINDEX}\"; python sim.py --seed ${USUB_INDEX}'"),
    ("slurm", "sbatch --job-name=job --output=/work/logs/job.%A.%a.out --error=/work/logs/job.%A.%a.err "
              "--array=1-100:2%10 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G "
              "'--wrap=export USUB_INDEX=\"${SLURM_ARRAY_TASK_ID}\"; python sim.py --seed ${USUB_INDEX}'"),
    ("local", "seq 1 2 100 | xargs -P 10 -I{} bash -c "
              "'export USUB_INDEX=\"$1\"; python sim.py --seed ${USUB_INDEX}' _ {}"),
])
def test_oneliner_range(backend, expected):
    assert rendered(backend, "range").command() == expected


def test_oneliner_rejects_task_arrays():
    for backend in ("lsf", "slurm", "local"):
        with pytest.raises(UsubError, match="one-line"):
            rendered(backend, "files").command()


def test_local_rejects_dependencies():
    with pytest.raises(UsubError, match="dependencies"):
        render(JOBS["single"], config=Config(defaults={"backend": "local"}))


def test_manifest_fallback(tmp_path):
    job = Job("echo {}", files=["a", "b"], cwd=str(tmp_path), logdir="logs", name="m")
    for backend in ("lsf", "local"):
        r = render(job, config=Config(defaults={"backend": backend, "embed_limit": 3}))
        path, content = r.manifest
        assert content == "echo a\necho b\n"
        assert "USUB_TASKS" not in r.script and path in r.script
        r.prepare()
        assert Path(path).read_text() == content


def test_manifest_used_when_task_equals_delimiter():
    r = render(Job(cmds=["USUB_TASKS"], **BASE), config=Config(defaults={"backend": "lsf"}))
    assert r.manifest is not None


def test_task_selection_runs_in_bash(tmp_path):
    """The embedded heredoc picks the right line for each index."""
    r = render(Job("echo {/}", files=["d/a b", "d/c"], cwd=str(tmp_path), logdir="l", name="t"),
               config=Config(defaults={"backend": "slurm"}))
    for index, expected in ((1, "a b"), (2, "c")):
        out = subprocess.run(["bash"], input=r.script, text=True, capture_output=True, check=True,
                             env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index)})
        assert out.stdout == expected + "\n"


def test_unknown_backend():
    with pytest.raises(UsubError, match="unknown backend 'pbs'"):
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
