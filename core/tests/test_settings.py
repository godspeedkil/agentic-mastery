import pytest

from core.settings import Settings


def test_require_returns_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings(_env_file=None).require("anthropic_api_key") == "sk-test"


def test_require_fails_loudly_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings(_env_file=None).require("openai_api_key")
