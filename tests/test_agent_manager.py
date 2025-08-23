from unittest.mock import MagicMock

import pytest
import yaml
from dynaconf import Dynaconf


@pytest.fixture
def dynaconf_test_settings(tmp_path, monkeypatch):
    """Create a minimal TOML config and patch settings to force ModelManager to use it."""
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
    memory_model = "gpt-4_1"
    memory_model_temperature = 0.1
    memory_model_max_tokens = 2048
    google_api_key = "dummy_google_key"
    """

    settings_toml.write_text(settings_content)
    test_settings = Dynaconf(settings_files=[str(settings_toml)])

    # Patch get_settings to always return this instance, ignoring args
    def dummy_get_settings(*args, **kwargs):
        return test_settings

    monkeypatch.setattr("mchat_core.config.get_settings", dummy_get_settings)
    monkeypatch.setenv("DYNACONF_SETTINGS", str(settings_toml))
    return test_settings


@pytest.fixture
def agents_yaml(tmp_path):
    agent_conf = {
        "default_with_tools": {
            "type": "agent",
            "description": "A general-purpose bot",
            "prompt": "Please ask me anything.",
            "oneshot": False,
            "max_rounds": 10,
            "tools": ["google_search", "generate_image", "today"],
        },
        "research_team": {
            "type": "team",
            "team_type": "selector",
            "chooseable": False,
            "agents": ["default_with_tools", "ai2", "ai3"],
            "description": "Team research.",
            "max_rounds": 5,
            "oneshot": False,
        },
        "ai2": {
            "type": "agent",
            "description": "The second agent.",
            "prompt": "I am AI2.",
            "chooseable": False,
            "tools": ["google_search"],
        },
        "ai3": {
            "type": "agent",
            "description": "The third agent.",
            "prompt": "I am AI3.",
            "chooseable": True,
        },
    }
    path = tmp_path / "agents.yaml"
    with open(path, "w") as f:
        yaml.dump(agent_conf, f)
    return str(path)


@pytest.fixture
def patch_tools(monkeypatch):
    """Patch out actual tool loading in AutogenManager for test speed and independence."""
    # Patch out the importlib logic that loads tools from filesystem
    dummy_tool = MagicMock(is_callable=True, name="google_search", description="desc")
    dummy_tool.__class__.name = "google_search"
    tool_dict = {"google_search": dummy_tool}
    monkeypatch.setattr("mchat_core.agent_manager.BaseTool", object)
    monkeypatch.setattr(
        "mchat_core.agent_manager.FunctionTool", MagicMock(return_value=dummy_tool)
    )
    return tool_dict


def test_init_and_properties(dynaconf_test_settings, agents_yaml, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    # Check loaded agents
    assert "default_with_tools" in manager.agents
    assert "research_team" in manager.agents
    assert sorted(manager.chooseable_agents) == ["ai3", "default_with_tools"]
    assert manager.mm.available_chat_models == ["gpt-4_1"]


def test_agent_model_manager_isolated(dynaconf_test_settings, agents_yaml, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    # ModelManager in agent_manager points to test-only config!
    assert manager.mm.available_chat_models == ["gpt-4_1"]


@pytest.mark.tools
def test_tool_loading_real(dynaconf_test_settings, agents_yaml):
    from .conftest import require_pkgs

    require_pkgs(["tzlocal"])  # minimal for the "today" tool
    from mchat_core.agent_manager import AutogenManager

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    assert "today" in manager.tools


def test_stream_tokens_property(dynaconf_test_settings, agents_yaml, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    # Add a dummy agent so the setter doesn't error
    from unittest.mock import MagicMock

    manager.agent = MagicMock(_model_client_stream=True, name="dummy_agent")
    manager.stream_tokens = False
    assert not manager.agent._model_client_stream


def test_agents_property_and_chooseable(
    dynaconf_test_settings, agents_yaml, patch_tools
):
    from mchat_core.agent_manager import AutogenManager

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    # Agents property returns agent dict
    assert isinstance(manager.agents, dict)
    assert "default_with_tools" in manager.chooseable_agents


def test_error_on_both_agents_and_agent_paths(
    monkeypatch, dynaconf_test_settings, agents_yaml
):
    from mchat_core.agent_manager import AutogenManager

    dummy_agents = {"foo": {"prompt": "bar"}}
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None,
            agents=dummy_agents,
            agent_paths=[agents_yaml],
        )


def test_load_agents_from_json_string(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    json_str = (
        '{"json_agent": {"type": "agent", "description": "json agent", "prompt": "hi"}}'
    )
    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[json_str]
    )
    assert "json_agent" in manager.agents
    assert manager.agents["json_agent"]["description"] == "json agent"


def test_load_agents_from_yaml_string(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    yaml_str = """
yaml_agent:
  type: agent
  description: yaml agent
  prompt: hello
"""
    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[yaml_str]
    )
    assert "yaml_agent" in manager.agents
    assert manager.agents["yaml_agent"]["description"] == "yaml agent"


def test_load_agents_invalid_yaml_string(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    non_agent_str = "this is not json or yaml"
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None, agent_paths=[non_agent_str]
        )


def test_load_agents_invalid_json_string(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    non_agent_str = '{"foo": "bar"}'  # Looks like JSON but not a valid agent definition
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None, agent_paths=[non_agent_str]
        )


def test_load_agents_team_without_prompt_is_ok(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    yaml_str = """
teamy:
  type: team
  team_type: round_robin
  description: a team
  agents: []
"""
    manager = AutogenManager(
        message_callback=lambda *a, **kw: None,
        agent_paths=[yaml_str],
    )
    assert "teamy" in manager.agents
    assert manager.agents["teamy"]["description"] == "a team"


def test_load_agents_team_missing_agents_raises(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    yaml_str = """
bad_team:
  type: team
  team_type: round_robin
  description: missing agents list
"""
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None,
            agent_paths=[yaml_str],
        )


def test_load_agents_invalid_json_non_mapping_list(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    non_mapping_json = '["a", "b"]'
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None,
            agent_paths=[non_mapping_json],
        )


def test_new_conversation_model_fallbacks(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    # Provide two agents: one with explicit model, one without (should use default)
    agents = {
        "with_model": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "model": "gpt-4_1",
        },
        "no_model": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
        },
    }
    manager = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)

    # Uses model from agent config
    import asyncio

    asyncio.run(manager.new_conversation(agent="with_model"))
    assert manager.model == "gpt-4_1"

    # Uses default when not in agent config
    asyncio.run(manager.new_conversation(agent="no_model"))
    assert manager.model == dynaconf_test_settings.defaults.chat_model


@pytest.mark.asyncio
async def test_consume_agent_stream_stopmessage_returns_taskresult(
    dynaconf_test_settings, patch_tools
):
    from autogen_agentchat.messages import StopMessage

    from mchat_core.agent_manager import AutogenManager

    async def runner(task: str, cancellation_token):
        yield StopMessage(content="done", source="unit")

    async def cb(*args, **kwargs):
        return None

    agents = {
        "a": {"type": "agent", "description": "d", "prompt": "p"},
    }
    m = AutogenManager(message_callback=cb, agents=agents)
    result = await m._consume_agent_stream(
        agent_runner=runner, oneshot=False, task="t", cancellation_token=None
    )
    assert result.stop_reason == "presumed done"
    assert result.messages and result.messages[0].content.lower().startswith("presumed")


@pytest.mark.asyncio
async def test_consume_agent_stream_end_of_stream_returns_taskresult(
    dynaconf_test_settings, patch_tools
):
    from mchat_core.agent_manager import AutogenManager

    async def runner(task: str, cancellation_token):
        if False:
            yield None  # no yields -> end of stream

    async def cb(*args, **kwargs):
        return None

    agents = {
        "a": {"type": "agent", "description": "d", "prompt": "p"},
    }
    m = AutogenManager(message_callback=cb, agents=agents)
    result = await m._consume_agent_stream(
        agent_runner=runner, oneshot=False, task="t", cancellation_token=None
    )
    assert result.stop_reason == "end_of_stream"
    assert result.messages and result.messages[0].content.lower().startswith(
        "end of stream"
    )


@pytest.mark.asyncio
async def test_unknown_event_is_reported_to_callback(
    dynaconf_test_settings, patch_tools
):
    from mchat_core.agent_manager import AutogenManager

    class Unknown:
        def __repr__(self):
            return "<UnknownEvent>"

    async def runner(task: str, cancellation_token):
        yield Unknown()

    calls = []

    async def cb(msg, *_, **kw):
        calls.append(msg)

    agents = {
        "a": {"type": "agent", "description": "d", "prompt": "p"},
    }
    m = AutogenManager(message_callback=cb, agents=agents)
    # It will iterate once, then end and return end_of_stream
    await m._consume_agent_stream(
        agent_runner=runner, oneshot=False, task="t", cancellation_token=None
    )
    # First callback is "<unknown>", second is repr(response)
    assert any("<unknown>" in str(c) for c in calls)
    assert any("UnknownEvent" in str(c) for c in calls)
