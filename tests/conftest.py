import asyncio
import importlib.util
import os
from unittest.mock import MagicMock

import pytest
from dynaconf import Dynaconf

def has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def require_pkgs(names: list[str]):
    missing = [n for n in names if not has_pkg(n)]
    if missing:
        pytest.skip(f"Skipping optional tools test; missing packages: {', '.join(missing)}")

def require_env(var: str):
    if not os.environ.get(var):
        pytest.skip(f"Skipping optional tools test; missing env var: {var}")


@pytest.fixture
def dynaconf_test_settings(tmp_path, monkeypatch):
    """Provide a minimal Dynaconf test settings for ModelManager to use."""
    settings_toml = tmp_path / "settings.toml"
    settings_content = """
    [models.chat.gpt-4_1]
    api_key = "dummy_key"
    model = "gpt-4.1"
    api_type = "open_ai"
    base_url = "https://api.openai.com/v1"
    _tool_support = true
    _streaming_support = true
    _system_prompt_support = true
    _cost_input = 2.00
    _cost_output = 8.00

    [defaults]
    chat_model = "gpt-4_1"
    chat_temperature = 0.7
    mini_model = "gpt-4_1"
    google_api_key = "dummy_google_key"
    """
    settings_toml.write_text(settings_content)
    test_settings = Dynaconf(settings_files=[str(settings_toml)])

    def dummy_get_settings(*args, **kwargs):
        return test_settings

    monkeypatch.setattr("mchat_core.config.get_settings", dummy_get_settings)
    monkeypatch.setenv("SETTINGS_FILE_FOR_DYNACONF", str(settings_toml))
    return test_settings


@pytest.fixture
def patch_tools(monkeypatch):
    """Patch tool loading to return fake tools for tests."""
    google_search = MagicMock(name="google_search")
    generate_image = MagicMock(name="generate_image")
    today = MagicMock(name="today")

    fake_tools = {
        "google_search": google_search,
        "generate_image": generate_image,
        "today": today,
    }

    monkeypatch.setattr(
        "mchat_core.agent_manager.load_tools",
        lambda *a, **kw: fake_tools,
        raising=False,
    )
    return fake_tools