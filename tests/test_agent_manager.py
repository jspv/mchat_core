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
    mini_model = "gpt-4_1"
    google_api_key = "dummy_google_key"
    """

    settings_toml.write_text(settings_content)
    test_settings = Dynaconf(settings_files=[str(settings_toml)])

    # Patch get_settings to always return this instance, ignoring args
    def dummy_get_settings(*args, **kwargs):
        return test_settings

    monkeypatch.setattr("mchat_core.config.get_settings", dummy_get_settings)
    monkeypatch.setenv("SETTINGS_FILE_FOR_DYNACONF", str(settings_toml))
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
    """Patch out actual tool loading in AutogenManager for test speed and isolation.

    Returns a dict of fake tools keyed by the names referenced in tests/fixtures.
    """
    from unittest.mock import MagicMock

    google_search = MagicMock(name="google_search")
    generate_image = MagicMock(name="generate_image")
    today = MagicMock(name="today")

    fake_tools = {
        "google_search": google_search,
        "generate_image": generate_image,
        "today": today,
    }

    # AutogenManager uses load_tools from its own module namespace; accept any args.
    monkeypatch.setattr(
        "mchat_core.agent_manager.load_tools", lambda *a, **kw: fake_tools, raising=False
    )
    return fake_tools


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

    # Point AutogenManager at the package tools directory explicitly
    import os
    import mchat_core as pkg
    tools_dir = os.path.join(os.path.dirname(pkg.__file__), "tools")

    from mchat_core.agent_manager import AutogenManager
    manager = AutogenManager(
        message_callback=lambda *a, **kw: None,
        agent_paths=[agents_yaml],
        tools_directory=tools_dir,
    )
    assert "today" in manager.tools


@pytest.mark.asyncio
async def test_stream_tokens_property(dynaconf_test_settings, agents_yaml, patch_tools):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    manager = AutogenManager(
        message_callback=lambda *a, **kw: None, agent_paths=[agents_yaml]
    )
    # Use a bare AgentSession with a dummy agent to test the property
    from unittest.mock import MagicMock

    session = await AgentSession.create(manager=manager)
    session.agent = MagicMock(_model_client_stream=True, name="dummy_agent")
    session._stream_tokens = True
    session.stream_tokens = False
    assert not session.agent._model_client_stream


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


def test_load_agents_non_agent_json_like_string_raises(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    # Looks like JSON but is not a valid agents mapping (values must be dicts)
    bad_json = '{"foo": "bar"}'
    with pytest.raises(ValueError):
        AutogenManager(
            message_callback=lambda *a, **kw: None, agent_paths=[bad_json]
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

    session1 = asyncio.run(manager.new_conversation(agent="with_model"))
    assert session1.model == "gpt-4_1"

    # Uses default when not in agent config
    session2 = asyncio.run(manager.new_conversation(agent="no_model"))
    assert session2.model == dynaconf_test_settings.defaults.chat_model


@pytest.mark.asyncio
async def test_consume_agent_stream_stopmessage_returns_taskresult(
    dynaconf_test_settings, patch_tools
):
    from autogen_agentchat.messages import StopMessage

    from mchat_core.agent_manager import AutogenManager, AgentSession

    async def runner(task: str, cancellation_token):
        yield StopMessage(content="done", source="unit")

    async def cb(*args, **kwargs):
        return None

    agents = {
        "a": {"type": "agent", "description": "d", "prompt": "p"},
    }
    m = AutogenManager(message_callback=cb, agents=agents)
    s = await AgentSession.create(manager=m)
    result = await s._consume_agent_stream(
        agent_runner=runner, oneshot=False, task="t", cancellation_token=None
    )
    assert result.stop_reason == "presumed done"
    assert result.messages and result.messages[0].content.lower().startswith("presumed")


@pytest.mark.asyncio
async def test_consume_agent_stream_end_of_stream_returns_taskresult(
    dynaconf_test_settings, patch_tools
):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    async def runner(task: str, cancellation_token):
        if False:
            yield None  # no yields -> end of stream

    async def cb(*args, **kwargs):
        return None

    agents = {
        "a": {"type": "agent", "description": "d", "prompt": "p"},
    }
    m = AutogenManager(message_callback=cb, agents=agents)
    s = await AgentSession.create(manager=m)
    result = await s._consume_agent_stream(
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
    from mchat_core.agent_manager import AutogenManager, AgentSession

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
    s = await AgentSession.create(manager=m)
    # It will iterate once, then end and return end_of_stream
    await s._consume_agent_stream(
        agent_runner=runner, oneshot=False, task="t", cancellation_token=None
    )
    # First callback is "<unknown>", second is repr(response)
    assert any("<unknown>" in str(c) for c in calls)
    assert any("UnknownEvent" in str(c) for c in calls)


# --- New tests: verify handler methods call the session-level callback ---

from types import SimpleNamespace


@pytest.mark.asyncio
async def test_handler_text_message_non_stream_uses_session_callback(
    dynaconf_test_settings,
):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb, stream_tokens=False)

    msg = SimpleNamespace(content="hello", source="a")
    await s._handle_text_message(msg, oneshot=False)

    assert calls and calls[-1][0] == "hello"
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is True


@pytest.mark.asyncio
async def test_handler_stream_chunk_uses_session_callback(dynaconf_test_settings):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    evt = SimpleNamespace(content="chunk", source="a")
    await s._handle_stream_chunk(evt)

    assert calls and calls[-1][0] == "chunk"
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is False


@pytest.mark.asyncio
async def test_handler_multi_modal_uses_session_callback(dynaconf_test_settings):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    evt = SimpleNamespace(content="image-bytes", source="a")
    await s._handle_multi_modal(evt)

    assert calls and calls[-1][0].startswith("MM:")
    assert "image-bytes" in calls[-1][0]
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is True


@pytest.mark.asyncio
async def test_handler_thought_event_uses_session_callback(dynaconf_test_settings):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    evt = SimpleNamespace(content="thinking...", source="a")
    await s._handle_thought_event(evt)

    assert calls and "(Thinking: thinking...)" in calls[-1][0]
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is False


@pytest.mark.asyncio
async def test_handler_tool_call_request_uses_session_callback(dynaconf_test_settings):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    tool = SimpleNamespace(name="toolA", arguments={"q": 1})
    evt = SimpleNamespace(content=[tool], source="a")
    await s._handle_tool_call_request(evt)

    assert calls and "with arguments:" in calls[-1][0]
    assert "toolA" in calls[-1][0]
    assert calls[-1][1]["agent"] == "a"


@pytest.mark.asyncio
async def test_handler_tool_call_execution_uses_session_callback(
    dynaconf_test_settings,
):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    evt = SimpleNamespace(content={"ok": True}, source="a")
    await s._handle_tool_call_execution(evt)

    assert calls and calls[-1][0] == "done"
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is True


@pytest.mark.asyncio
async def test_handler_user_input_request_uses_session_callback(
    dynaconf_test_settings,
):
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb)

    evt = SimpleNamespace(content="input please", source="a")
    await s._handle_user_input_request(evt)

    assert calls and calls[-1][0] == "input please"
    assert calls[-1][1]["agent"] == "a"
    assert calls[-1][1]["complete"] is True


@pytest.mark.asyncio
async def test_text_message_suppressed_when_streaming(dynaconf_test_settings):
    """When stream_tokens=True, TextMessage should not trigger a full message callback."""
    from mchat_core.agent_manager import AutogenManager, AgentSession

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    agents = {"a": {"type": "agent", "description": "d", "prompt": "p"}}
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    s = await AgentSession.create(manager=m, message_callback=cb, stream_tokens=True)

    # With streaming on, _handle_text_message should not emit a 'complete' message
    msg = SimpleNamespace(content="hello", source="a")
    await s._handle_text_message(msg, oneshot=False)

    assert calls == []
