import pytest

from usub import Config

@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Never read the real user or project config during tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USUB_CONFIG", str(home / ".usub.toml"))

@pytest.fixture
def empty_config():
    return Config()
