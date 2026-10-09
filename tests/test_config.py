import pytest

from usub import Config, Job, UsubError, load_config, resolve
from usub.config import TEMPLATE, select_profile

CONFIG = Config.from_dict({
    "default_profile": "auto",
    "defaults": {"mem": "2G", "logdir": "/logs", "time": "1h", "setup": "source a"},
    "profiles": {
        "farm": {"backend": "lsf", "host_match": ["farm5-*", "*.sanger.ac.uk"], "queue": "normal",
                 "mem": "3G", "lsf_rusage": True, "setup": ["source b"]},
        "hpc": {"backend": "slurm", "host_match": "login*", "partition": "short",
                "extra": ["--account=abc"]},
    },
})


def test_load_template(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TEMPLATE)
    config = load_config(path)
    assert set(config.profiles) == {"farm", "slurm", "local"}
    assert config.profiles["farm"]["backend"] == "lsf"


def test_load_missing_default_is_empty():
    assert load_config() == Config()


def test_load_errors(tmp_path):
    with pytest.raises(UsubError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("x = ")
    with pytest.raises(UsubError, match="bad.toml"):
        load_config(bad)


@pytest.mark.parametrize("host, profile", [
    ("farm5-head1", "farm"), ("node.sanger.ac.uk", "farm"), ("login02", "hpc"), ("laptop", None),
])
def test_auto_profile_by_host(host, profile):
    assert select_profile(CONFIG, env={}, hostname=host) == profile


def test_profile_precedence_arg_env_default():
    config = Config(profiles=CONFIG.profiles, default_profile="hpc")
    assert select_profile(config, env={}) == "hpc"
    assert select_profile(config, env={"USUB_PROFILE": "farm"}) == "farm"
    assert select_profile(config, "hpc", env={"USUB_PROFILE": "farm"}) == "hpc"
    with pytest.raises(UsubError, match="unknown profile 'nope'"):
        select_profile(config, "nope", env={})


def test_resolve_layers():
    r = resolve(Job("x", cwd="/work"), config=CONFIG, env={}, hostname="farm5-head1")
    assert (r.profile, r.backend) == ("farm", "lsf")
    assert r.job.mem == "3G"            # profile beats defaults
    assert r.job.time == "1h"           # from defaults
    assert r.job.cores == 1             # built-in
    assert r.job.queue == "normal"
    assert r.job.setup == ["source a", "source b"]  # lists accumulate
    assert r.job.logdir == "/logs"
    assert r.options == {"lsf_rusage": True}


def test_resolve_job_and_env_override():
    job = Job("x", cwd="/work", mem="8G", setup=["source c"], logdir="out")
    env = {"USUB_LOGDIR": "/env-logs", "USUB_QUEUE": "long", "USUB_BACKEND": "local"}
    r = resolve(job, profile="farm", config=CONFIG, env=env)
    assert r.job.mem == "8G"
    assert r.job.queue == "long"
    assert r.job.logdir == "/work/out"  # relative to cwd; job beats env
    assert r.job.setup == ["source a", "source b", "source c"]
    assert r.backend == "local"
    assert resolve(job, profile="farm", backend="slurm", config=CONFIG, env=env).backend == "slurm"


def test_resolve_alias_and_extra():
    r = resolve(Job("x", cwd="/w", extra=["--qos=high"]), profile="hpc", config=CONFIG, env={})
    assert r.job.queue == "short"
    assert r.job.extra == ["--account=abc", "--qos=high"]


def test_resolve_names_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = resolve(Job("samtools sort", name=None), config=Config(), env={}, backend="lsf")
    assert r.job.name == "samtools"
    assert r.job.cwd == str(tmp_path.resolve()) or r.job.cwd == str(tmp_path)
    assert r.job.logdir.endswith("usub-logs")
    assert resolve(Job("x", name="a b"), config=Config(), env={}, backend="lsf").job.name == "a_b"


def test_resolve_detects_backend(monkeypatch):
    monkeypatch.setattr("usub.config.detect_backend", lambda: "slurm")
    assert resolve(Job("x"), config=Config(), env={}).backend == "slurm"
