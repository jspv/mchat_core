import pytest

from autogen_core.model_context import (
    BufferedChatCompletionContext,
    HeadAndTailChatCompletionContext,
    TokenLimitedChatCompletionContext,
    UnboundedChatCompletionContext,
)


@pytest.mark.asyncio
async def test_unbounded_default_context_for_agent(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "unbounded": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
        }
    }
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="unbounded")
    assert isinstance(session.agent._model_context, UnboundedChatCompletionContext)


@pytest.mark.asyncio
async def test_buffered_context_for_agent(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager
    from autogen_core.models import UserMessage

    agents = {
        "buf": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "buffered", "buffer_size": 3},
        }
    }
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="buf")
    ctx = session.agent._model_context
    assert isinstance(ctx, BufferedChatCompletionContext)

    # Add more than buffer_size messages and verify get_messages is <= buffer_size
    for i in range(5):
        await ctx.add_message(UserMessage(content=f"m{i}", source="user"))
    msgs = await ctx.get_messages()
    assert len(msgs) <= 3


@pytest.mark.asyncio
async def test_token_limited_context_for_agent(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "tok": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "token", "token_limit": 1234},
        }
    }
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="tok")
    assert isinstance(session.agent._model_context, TokenLimitedChatCompletionContext)


@pytest.mark.asyncio
async def test_head_tail_context_for_agent(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "ht": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "head_tail", "head_size": 1, "tail_size": 2},
        }
    }
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="ht")
    assert isinstance(session.agent._model_context, HeadAndTailChatCompletionContext)


@pytest.mark.asyncio
async def test_team_subagent_contexts(dynaconf_test_settings, patch_tools):
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "a1": {
            "type": "agent",
            "description": "a1",
            "prompt": "p1",
            "context": {"type": "buffered", "buffer_size": 2},
        },
        "a2": {
            "type": "agent",
            "description": "a2",
            "prompt": "p2",
            "context": {"type": "head_tail", "head_size": 1, "tail_size": 1},
        },
        "team": {
            "type": "team",
            "team_type": "round_robin",
            "description": "t",
            "agents": ["a1", "a2"],
        },
    }
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="team")

    parts = session.agent_team._participants
    assert len(parts) == 2
    assert isinstance(parts[0]._model_context, BufferedChatCompletionContext)
    assert isinstance(parts[1]._model_context, HeadAndTailChatCompletionContext)
