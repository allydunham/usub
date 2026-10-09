import os
import stat
from pathlib import Path

import pytest

from usub.cli import main


@pytest.fixture
def fake_schedulers(tmp_path, monkeypatch):
    """bsub/sbatch stand-ins that record their stdin/argv and print a job id."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, reply in (("bsub", "Job <4242> is submitted to queue <normal>."),
                        ("sbatch", "Submitted batch job 777")):
        script = bindir / name
        script.write_text(f'#!/bin/bash\necho "$@" > "{tmp_path}/{name}.args"\n'
                          f'cat > "{tmp_path}/{name}.stdin"\necho "{reply}"\n')
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


def test_print_script(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["r", "-b", "lsf", "-m", "2G", "--", "echo", "hi"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("#!/bin/bash\n#BSUB -J echo\n")
    assert out.rstrip().endswith("echo hi")
    assert (tmp_path / "usub-logs").is_dir()  # created so a piped submission can log


def test_command_without_separator(capsys, tmp_path):
    assert main(["r", "-b", "slurm", "--cwd", str(tmp_path), "python", "x.py", "-v"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("python x.py -v")


def test_files_with_glob(capsys, tmp_path):
    for n in ("b.txt", "a.txt", "c.dat"):
        (tmp_path / n).touch()
    assert main(["r", "-b", "lsf", "--cwd", str(tmp_path), "-f", str(tmp_path / "*.txt"),
                 "--", "gzip"]) == 0
    out = capsys.readouterr().out
    assert f"gzip {tmp_path}/a.txt\ngzip {tmp_path}/b.txt\nUSUB_TASKS" in out
    assert '#BSUB -J "gzip[1-2]"' in out


def test_file_list_and_cmds_from_file(capsys, tmp_path):
    lst = tmp_path / "list"
    lst.write_text("x.fq\n\ny.fq\n")
    assert main(["r", "-b", "slurm", "--cwd", str(tmp_path), "-F", str(lst), "--", "wc", "{}"]) == 0
    assert "wc x.fq\nwc y.fq\n" in capsys.readouterr().out
    assert main(["r", "-b", "slurm", "--cwd", str(tmp_path), "-C", str(lst)]) == 0
    assert "--array=1-2" in capsys.readouterr().out


def test_write_and_emit_cmd(capsys, tmp_path):
    target = tmp_path / "job.sh"
    assert main(["r", "-b", "lsf", "--cwd", str(tmp_path), "--write", str(target), "--", "true"]) == 0
    assert target.read_text().startswith("#!/bin/bash")
    assert "bsub < " in capsys.readouterr().err
    assert main(["r", "-b", "slurm", "--cwd", str(tmp_path), "--emit", "cmd", "--", "true"]) == 0
    assert capsys.readouterr().out.startswith("sbatch --job-name=true ")


@pytest.mark.parametrize("backend, job_id", [("lsf", "4242"), ("slurm", "777")])
def test_submit(fake_schedulers, capsys, backend, job_id):
    tmp = fake_schedulers
    assert main(["r", "-s", "-b", backend, "--cwd", str(tmp), "-a", "3", "--", "echo", "{i}"]) == 0
    assert capsys.readouterr().out == f"{job_id}\n"
    scripts = list((tmp / "usub-logs" / "scripts").glob("echo.*.sh"))
    assert len(scripts) == 1
    if backend == "lsf":  # script fed on stdin so #BSUB lines are parsed
        assert (tmp / "bsub.stdin").read_text() == scripts[0].read_text()
    else:
        assert (tmp / "sbatch.args").read_text().strip() == str(scripts[0])


def test_submit_failure(tmp_path, capsys, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "bsub").write_text("#!/bin/bash\necho 'Bad queue' >&2\nexit 255\n")
    (bindir / "bsub").chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    assert main(["r", "-s", "-b", "lsf", "--cwd", str(tmp_path), "--", "true"]) == 2
    assert "bsub failed (exit 255): Bad queue" in capsys.readouterr().err


def test_local_end_to_end(tmp_path, capsys):
    for n in ("one.txt", "two words.txt"):
        (tmp_path / n).write_text("a\nb\n")
    argv = ["r", "-s", "-b", "local", "-j", "2", "--cwd", str(tmp_path),
            "-f", str(tmp_path / "*.txt"), "--", "wc -l < {} > {.}.count"]
    assert main(argv) == 0
    assert (tmp_path / "one.count").read_text().strip() == "2"
    assert (tmp_path / "two words.count").read_text().strip() == "2"
    logs = sorted(p.name for p in (tmp_path / "usub-logs").iterdir())
    assert sum(n.endswith(".out") for n in logs) == 2
    assert any(n.endswith(".tasks") for n in logs)


def test_local_failure_reported(tmp_path, capsys):
    cmds = tmp_path / "cmds"
    cmds.write_text("true\nexit 3\n")
    assert main(["r", "-s", "-b", "local", "--cwd", str(tmp_path), "-C", str(cmds)]) == 2
    err = capsys.readouterr().err
    assert "local job failed" in err


@pytest.mark.parametrize("argv, message", [
    (["r", "-b", "lsf"], "no command given"),
    (["r", "-b", "lsf", "-a", "1-2", "-C", "x", "--", "echo"], "cannot read x"),
    (["r", "-b", "lsf", "-f", "/nonexistent/*.q", "--", "echo"], "no files match"),
    (["r", "-b", "lsf", "-m", "lots", "--", "echo"], "invalid memory"),
    (["r", "-p", "nope", "--", "echo"], "unknown profile"),
    (["r", "-C", "-", "-F", "-", "--", "x"], "both read stdin"),
])
def test_errors(argv, message, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_config_init_show_path(tmp_path, capsys, monkeypatch):
    path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setenv("USUB_CONFIG", str(path))
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(path)
    assert main(["config", "init"]) == 0
    assert "[profiles.farm]" in path.read_text()
    assert main(["config", "init"]) == 2
    assert main(["config", "show", "-p", "farm"]) == 0
    out = capsys.readouterr().out
    assert "profile: farm" in out and "backend: lsf" in out and "lsf_rusage = True" in out


def test_no_action_prints_help(capsys):
    assert main([]) == 0
    assert "usage: usub" in capsys.readouterr().out
