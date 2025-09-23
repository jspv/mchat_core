import asyncio
import pytest

# Module under test
from mchat_core import agent_manager as am


class FakeModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_info = {"function_calling": True}


class FakeMM:
    default_chat_model = "fake-chat"
    default_chat_temperature = 0.2
    default_mini_model = "fake-mini"

    def open_model(self, model_id: str):
        return FakeModel(model_id)

    def get_tool_support(self, model_id: str) -> bool:
        return True

    def get_system_prompt_support(self, model_id: str) -> bool:
        return True

    def get_streaming_support(self, model_id: str) -> bool:
        return True


class FakeCond:
    def __init__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        return self


class FakeContext:
    async def get_messages(self):
        return []


class FakeAssistantAgent:
    instances = []

    def __init__(
        self,
        name,
        model_client,
        tools,
        model_context,
        system_message=None,
        model_client_stream=None,
        reflect_on_tool_use=None,
        description=None,
        **kwargs,
    ):
        self.name = name
        self.model_client = model_client
        self.tools = tools
        self._model_context = model_context if model_context is not None else FakeContext()
        self.system_message = system_message
        self._model_client_stream = model_client_stream
        self.reflect_on_tool_use = reflect_on_tool_use
        self.description = description
        self.extra_kwargs = kwargs
        FakeAssistantAgent.instances.append(self)


class FakeGroupChat:
    def __init__(self, participants, termination_condition=None, **kwargs):
        self._participants = participants
        self.termination_condition = termination_condition

    # Not exercised by these tests
    async def run_stream(self, *args, **kwargs):
        yield None


@pytest.fixture(autouse=True)
def patch_autogen_bits(monkeypatch):
    # Patch runtime dependencies inside the module under test
    monkeypatch.setattr(am, "AssistantAgent", FakeAssistantAgent)
    monkeypatch.setattr(am, "RoundRobinGroupChat", FakeGroupChat)
    monkeypatch.setattr(am, "StopMessageTermination", FakeCond)
    monkeypatch.setattr(am, "MaxMessageTermination", lambda *a, **k: FakeCond())
    monkeypatch.setattr(am, "ExternalTermination", FakeCond)
    # SmartReflectorTermination used only in solo path
    monkeypatch.setattr(am, "SmartReflectorTermination", FakeCond)

    # Ensure a fresh instance list each test
    FakeAssistantAgent.instances = []

    yield


@pytest.mark.asyncio
async def test_solo_agent_extra_kwargs_passed_and_reserved_filtered(monkeypatch):
    # Build manager with a single (solo) agent
    agents = {
        "solo": {
            "type": "agent",
            "model": "fake-chat",
            "prompt": "Hello",
            "description": "A solo agent",
            "extra_kwargs": {
                "foo": 123,                         # should pass through
                "bar": "ok",                        # should pass through
                # Reserved keys below should be filtered out:
                "name": "OVERRIDE_ME",
                "model_client": "OVERRIDE",
                "tools": ["x"],
                "model_context": "OVERRIDE",
                "system_message": "OVERRIDE",
                "model_client_stream": False,
                "reflect_on_tool_use": False,
            },
        }
    }

    async def noop(*args, **kwargs):
        pass

    manager = am.AutogenManager(message_callback=noop, agents=agents, stream_tokens=False)
    # Patch manager.mm with our fake model manager
    manager.mm = FakeMM()

    session = await am.AgentSession.create(
        manager=manager,
        agent_name="solo",
        model_id=None,
        temperature=None,
        stream_tokens=False,
        message_callback=noop,
    )

    # Validate FakeAssistantAgent got created with filtered kwargs
    assert len(FakeAssistantAgent.instances) == 1
    inst = FakeAssistantAgent.instances[0]

    # Extra kwargs should include only non-reserved keys
    assert inst.extra_kwargs == {"foo": 123, "bar": "ok"}

    # Explicit params should be preserved (not overridden by extra_kwargs)
    assert inst.name == "solo"
    assert inst.system_message == "Hello"  # system prompt supported path
    # reflect_on_tool_use is set explicitly by implementation
    assert inst.reflect_on_tool_use is True


@pytest.mark.asyncio
async def test_team_subagent_extra_kwargs_passed_and_reserved_filtered(monkeypatch):
    # Two subagents with extra_kwargs, nested under a team
    agents = {
        "a1": {
            "type": "agent",
            "model": "fake-chat",
            "prompt": "Hi from a1",
            "description": "Agent 1",
            "extra_kwargs": {"x": 1, "name": "nope"},
        },
        "a2": {
            "type": "agent",
            "model": "fake-chat",
            "prompt": "Hi from a2",
            "description": "Agent 2",
            "extra_kwargs": {"y": 2, "model_client": "nope"},
        },
        "team": {
            "type": "team",
            "team_type": "round_robin",
            "agents": ["a1", "a2"],
            "description": "A team",
            "max_rounds": 2,
        },
    }

    async def noop(*args, **kwargs):
        pass

    manager = am.AutogenManager(message_callback=noop, agents=agents, stream_tokens=False)
    manager.mm = FakeMM()

    session = await am.AgentSession.create(
        manager=manager,
        agent_name="team",
        model_id=None,
        temperature=None,
        stream_tokens=False,
        message_callback=noop,
    )

    # Two AssistantAgent instances should be created for subagents
    assert len(FakeAssistantAgent.instances) == 2
    # Find by name to be explicit
    a1_inst = next(i for i in FakeAssistantAgent.instances if i.name == "a1")
    a2_inst = next(i for i in FakeAssistantAgent.instances if i.name == "a2")

    # Extra kwargs should include only non-reserved keys for each subagent
    assert a1_inst.extra_kwargs == {"x": 1}
    assert a2_inst.extra_kwargs == {"y": 2}

    # Check explicitly set args still intact
    assert a1_inst.description == "Agent 1"
    assert a2_inst.description == "Agent 2"