import pytest

from usub import Config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Never read the real user config or USUB_* settings during tests."""
    for var in ("USUB_PROFILE", "USUB_BACKEND", "USUB_LOGDIR", "USUB_QUEUE", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USUB_CONFIG", str(tmp_path / "no-config.toml"))


@pytest.fixture
def empty_config():
    return Config()
