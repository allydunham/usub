import os

import pytest

from usub import Config, Job, Profile, UsubError, load_config, resolve
from usub.config import PROJECT_TEMPLATE, TEMPLATE, select_profile

CONFIG = Config(
    defaults=Profile(mem="2G", logdir="/logs", time="1h", setup=["source a"]),
    profiles={
        "farm": Profile(backend="lsf", host_match=["farm5-*", "*.sanger.ac.uk"], queue="normal",
                        mem="3G", setup=["source b"], options={"lsf_rusage": True}),
        "hpc": Profile(backend="slurm", host_match=["login*"], queue="short",
                       extra=["--account=abc"]),
    },
)

def test_profile_from_dict():
    p = Profile.from_dict({"backend": "lsf", "queue": "q", "host_match": "farm*",
                           "setup": "module load x", "lsf_mem_unit": "GB", "gpu_args": ["a"]})
    assert p == Profile(backend="lsf", queue="q", host_match=["farm*"], setup=["module load x"],
                        options={"lsf_mem_unit": "GB", "gpu_args": ["a"]})

@pytest.mark.parametrize("data", [
    {"extra": [1, 2]}, {"extra": 5}, {"setup": {"a": 1}}, {"host_match": 3}, {"setup": ["ok", None]},
])
def test_profile_from_dict_rejects_bad_lists(data):
    with pytest.raises(UsubError, match=f"{next(iter(data))} must be a string or a list of strings"):
        Profile.from_dict(data)

def test_profile_merge():
    base = Profile(backend="lsf", mem="1G", cores=1, host_match=["a*"], extra=["-x"],
                   options={"a": 1, "b": 1})
    top = Profile(mem="2G", host_match=[], extra=["-y"], options={"b": 2})
    merged = base.merge(top, Profile(queue="q"))
    assert merged == Profile(backend="lsf", mem="2G", cores=1, queue="q", host_match=["a*"],
                             extra=["-x", "-y"], options={"a": 1, "b": 2})
    assert base.extra == ["-x"] and base.options == {"a": 1, "b": 1}  # inputs untouched
    assert base.merge(Profile(host_match=["z*"])).host_match == ["z*"]

def test_load_template(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TEMPLATE)
    config = load_config(path)
    assert set(config.profiles) == {"lsf", "slurm", "local"}
    assert config.profiles["lsf"].backend == "lsf"
    assert config.profiles["lsf"].options["lsf_rusage"] is True
    assert config.default_profile is None
    assert config.sources == [path]

def test_load_project_template(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(PROJECT_TEMPLATE)
    assert load_config(path).profiles == {}

def test_load_missing_default_is_empty():
    assert load_config() == Config()

def test_load_errors(tmp_path):
    with pytest.raises(UsubError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("x = ")
    with pytest.raises(UsubError, match="bad.toml"):
        load_config(bad)
    bad.write_text("[profiles]\nfarm = 1\n")
    with pytest.raises(UsubError, match="bad.toml: .*must be tables"):
        load_config(bad)

def test_project_config_overrides_user_config(tmp_path, monkeypatch):
    user = tmp_path / "home" / ".usub.toml"
    user.write_text('default_profile = "farm"\n[defaults]\nmem = "2G"\nsetup = ["a"]\n'
                    '[profiles.farm]\nbackend = "lsf"\nqueue = "normal"\nlsf_mem_unit = "GB"\n'
                    '[profiles.other]\nbackend = "slurm"\n')
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".usub.toml").write_text('default_profile = "other"\n[defaults]\nmem = "8G"\nsetup = ["b"]\n'
                                        '[profiles.farm]\nqueue = "long"\n[profiles.new]\nbackend = "local"\n')
    monkeypatch.chdir(project)
    config = load_config()
    assert config.sources == [user, project / ".usub.toml"]
    assert config.default_profile == "other"
    assert config.defaults.mem == "8G" and config.defaults.setup == ["a", "b"]
    assert config.profiles["farm"].queue == "long"          # overridden
    assert config.profiles["farm"].backend == "lsf"         # kept from the user config
    assert config.profiles["farm"].options == {"lsf_mem_unit": "GB"}
    assert set(config.profiles) == {"farm", "other", "new"}

def test_project_config_found_in_parent_directories(tmp_path):
    (tmp_path / "proj" / "a" / "b").mkdir(parents=True)
    (tmp_path / "proj" / ".usub.toml").write_text('[defaults]\nmem = "5G"\n')
    (tmp_path / "proj" / "a" / ".usub.toml").write_text('[defaults]\nmem = "6G"\n')
    assert load_config(cwd=tmp_path / "proj" / "a" / "b").defaults.mem == "6G"  # nearest wins
    assert load_config(cwd=tmp_path / "proj").defaults.mem == "5G"
    assert load_config(cwd=tmp_path).sources == []

def test_user_config_is_not_also_a_project_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".usub.toml").write_text('[defaults]\nmem = "2G"\n')
    (home / "work").mkdir()
    assert load_config(cwd=home / "work").sources == [home / ".usub.toml"]
    # even when USUB_CONFIG points elsewhere, ~/.usub.toml is never treated as a project file
    other = tmp_path / "other.toml"
    other.write_text('[defaults]\ncores = 3\n')
    monkeypatch.setenv("USUB_CONFIG", str(other))
    assert load_config(cwd=home / "work").sources == [other]

def test_explicit_path_replaces_discovery(tmp_path, monkeypatch):
    (tmp_path / ".usub.toml").write_text('[defaults]\nmem = "5G"\n')
    explicit = tmp_path / "x.toml"
    explicit.write_text('[defaults]\ncores = 9\n')
    monkeypatch.chdir(tmp_path)
    config = load_config(explicit)
    assert config.sources == [explicit] and config.defaults.mem is None

@pytest.mark.parametrize("host, profile", [
    ("farm5-head1", "farm"), ("node.sanger.ac.uk", "farm"), ("login02", "hpc"), ("laptop", None),
])
def test_auto_profile_by_host(host, profile):
    assert select_profile(CONFIG, hostname=host) == profile

def test_profile_precedence_arg_over_default():
    config = Config(profiles=CONFIG.profiles, default_profile="hpc")
    assert select_profile(config) == "hpc"
    assert select_profile(config, "farm") == "farm"
    with pytest.raises(UsubError, match="Unknown profile 'nope'"):
        select_profile(config, "nope")

def test_resolve_layers():
    r = resolve(Job("x", cwd="/work"), config=CONFIG, hostname="farm5-head1")
    assert (r.profile, r.backend) == ("farm", "lsf")
    assert r.job.mem == "3G"            # profile beats defaults
    assert r.job.time == "1h"           # from defaults
    assert r.job.cores == 1             # built-in
    assert r.job.queue == "normal"
    assert r.job.setup == ["source a", "source b"]  # lists accumulate
    assert r.job.logdir == "/logs"
    assert r.options == {"lsf_rusage": True}

def test_resolve_job_overrides_config():
    job = Job("x", cwd="/work", mem="8G", setup=["source c"], logdir="out")
    r = resolve(job, profile="farm", config=CONFIG)
    assert r.job.mem == "8G"
    assert r.job.queue == "normal"
    assert r.job.logdir == "/work/out"  # relative to cwd; job beats config
    assert r.job.setup == ["source a", "source b", "source c"]
    assert r.backend == "lsf"
    assert resolve(job, profile="farm", backend="slurm", config=CONFIG).backend == "slurm"

def test_resolve_gpus_from_config_and_job_override():
    config = Config(defaults=Profile(gpus="num=1:gmodel=A100"))
    assert resolve(Job("x", cwd="/w"), config=config, backend="lsf").job.gpus == "num=1:gmodel=A100"
    assert resolve(Job("x", cwd="/w", gpus=2), config=config, backend="lsf").job.gpus == 2

def test_resolve_extra():
    r = resolve(Job("x", cwd="/w", extra=["--qos=high"]), profile="hpc", config=CONFIG)
    assert r.job.queue == "short"
    assert r.job.extra == ["--account=abc", "--qos=high"]

def test_resolve_loads_project_config_from_job_cwd(tmp_path):
    (tmp_path / ".usub.toml").write_text('[defaults]\nmem = "7G"\nlogdir = "plogs"\n')
    r = resolve(Job("x", cwd=tmp_path), backend="lsf")
    assert r.job.mem == "7G"
    assert r.job.logdir == str(tmp_path.resolve() / "plogs") or r.job.logdir == str(tmp_path / "plogs")

def test_resolve_names_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = resolve(Job("samtools sort", name=None), config=Config(), backend="lsf")
    assert r.job.name == "samtools"
    assert r.job.cwd == str(tmp_path.resolve()) or r.job.cwd == str(tmp_path)
    assert r.job.logdir == os.path.expanduser("~/usub_logs")  # default is absolute, in $HOME
    assert resolve(Job("x", name="a b"), config=Config(), backend="lsf").job.name == "a_b"

@pytest.mark.parametrize("on_path, backend", [({"bsub", "sbatch"}, "lsf"), ({"sbatch"}, "slurm"),
                                               (set(), "local")])
def test_resolve_detects_backend(monkeypatch, on_path, backend):
    monkeypatch.setattr("shutil.which", lambda prog: prog if prog in on_path else None)
    assert resolve(Job("x"), config=Config()).backend == backend

@pytest.mark.parametrize("data", [
    {"cores": "four"}, {"cores": 0}, {"cores": True}, {"cores": 1.5}, {"max_parallel": -1},
    {"merge_logs": "yes"}, {"mem": ["4G"]}, {"time": [1]}, {"gpus": 1.5}, {"backend": 3},
    {"queue": 1}, {"logdir": ["x"]},
])
def test_profile_rejects_wrongly_typed_settings(data):
    with pytest.raises(UsubError, match=f"{next(iter(data))} must be"):
        Profile.from_dict(data)

def test_profile_accepts_documented_types():
    p = Profile.from_dict({"cores": 4, "mem": "4G", "time": 90, "gpus": "num=2", "merge_logs": True,
                           "max_parallel": 10, "queue": "q", "logdir": "~/l", "backend": "slurm"})
    assert (p.cores, p.mem, p.time, p.gpus, p.merge_logs) == (4, "4G", 90, "num=2", True)
    assert Profile.from_dict({"mem": 4096.5, "gpus": 2}).mem == 4096.5

@pytest.mark.parametrize("data, message", [
    ({"default": {"mem": "9G"}}, "Unknown key 'default'"),
    ({"defualts": {}, "profiles": {}}, "Unknown key 'defualts'"),
    ({"defaults": 3}, r"\[defaults\] must be a table"),
    ({"profiles": {"farm": 1}}, "must be tables"),
    ({"default_profile": 3}, "default_profile must be"),
])
def test_config_rejects_bad_structure(data, message):
    with pytest.raises(UsubError, match=message):
        Config.from_dict(data)

def test_config_errors_name_the_file(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[defaults]\ncores = 'x'\n")
    with pytest.raises(UsubError, match=r"bad\.toml: cores must be a positive integer"):
        load_config(path)
