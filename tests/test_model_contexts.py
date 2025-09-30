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


@pytest.mark.asyncio
async def test_context_type_aliases(dynaconf_test_settings, patch_tools):
    """Test that various aliases for context types work correctly"""
    from mchat_core.agent_manager import AutogenManager

    # Test buffered aliases
    agents = {
        "buf1": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "buffer", "buffer_size": 5},
        },
        "buf2": {
            "type": "agent", 
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "buffer_size", "buffer_size": 5},
        },
        # Test token limit aliases
        "tok1": {
            "type": "agent",
            "description": "desc", 
            "prompt": "hi",
            "context": {"type": "token_limited", "token_limit": 1000},
        },
        "tok2": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi", 
            "context": {"type": "tokenlimited", "token_limit": 1000},
        },
        # Test head_tail aliases
        "ht1": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "headandtail", "head_size": 2, "tail_size": 3},
        },
        "ht2": {
            "type": "agent",
            "description": "desc", 
            "prompt": "hi",
            "context": {"type": "head_and_tail", "head_size": 2, "tail_size": 3},
        },
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    
    # Test buffered aliases
    session1 = await m.new_conversation(agent="buf1")
    session2 = await m.new_conversation(agent="buf2")
    assert isinstance(session1.agent._model_context, BufferedChatCompletionContext)
    assert isinstance(session2.agent._model_context, BufferedChatCompletionContext)
    
    # Test token limit aliases
    session3 = await m.new_conversation(agent="tok1")
    session4 = await m.new_conversation(agent="tok2")
    assert isinstance(session3.agent._model_context, TokenLimitedChatCompletionContext)
    assert isinstance(session4.agent._model_context, TokenLimitedChatCompletionContext)
    
    # Test head_tail aliases
    session5 = await m.new_conversation(agent="ht1")
    session6 = await m.new_conversation(agent="ht2")
    assert isinstance(session5.agent._model_context, HeadAndTailChatCompletionContext)
    assert isinstance(session6.agent._model_context, HeadAndTailChatCompletionContext)


@pytest.mark.asyncio
async def test_invalid_context_configurations(dynaconf_test_settings, patch_tools):
    """Test that invalid context configurations fallback to unbounded"""
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "invalid_buffer": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "buffered", "buffer_size": 0},  # Invalid: <= 0
        },
        "invalid_head_tail": {
            "type": "agent",
            "description": "desc", 
            "prompt": "hi",
            "context": {"type": "head_tail", "head_size": 0, "tail_size": 5},  # Invalid: head_size must be > 0
        },
        "invalid_type": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi", 
            "context": {"type": "nonexistent_type"},  # Invalid: unknown type
        },
        "non_dict_context": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": "invalid",  # Invalid: not a dict
        },
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    
    # All should fallback to unbounded context
    session1 = await m.new_conversation(agent="invalid_buffer")
    session2 = await m.new_conversation(agent="invalid_head_tail") 
    session3 = await m.new_conversation(agent="invalid_type")
    session4 = await m.new_conversation(agent="non_dict_context")
    
    assert isinstance(session1.agent._model_context, UnboundedChatCompletionContext)
    assert isinstance(session2.agent._model_context, UnboundedChatCompletionContext)
    assert isinstance(session3.agent._model_context, UnboundedChatCompletionContext)
    assert isinstance(session4.agent._model_context, UnboundedChatCompletionContext)


@pytest.mark.asyncio
async def test_token_limit_edge_cases(dynaconf_test_settings, patch_tools):
    """Test token limit context with various edge cases"""
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "tok_null": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi", 
            "context": {"type": "token", "token_limit": None},  # Should use model's remaining tokens
        },
        "tok_empty": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "token", "token_limit": ""},  # Should use model's remaining tokens  
        },
        "tok_null_str": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "token", "token_limit": "null"},  # Should use model's remaining tokens
        },
        "tok_no_limit": {
            "type": "agent", 
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "token"},  # No token_limit specified
        },
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    
    # All should create TokenLimitedChatCompletionContext with None token_limit
    session1 = await m.new_conversation(agent="tok_null")
    session2 = await m.new_conversation(agent="tok_empty")
    session3 = await m.new_conversation(agent="tok_null_str")
    session4 = await m.new_conversation(agent="tok_no_limit")
    
    assert isinstance(session1.agent._model_context, TokenLimitedChatCompletionContext)
    assert isinstance(session2.agent._model_context, TokenLimitedChatCompletionContext)
    assert isinstance(session3.agent._model_context, TokenLimitedChatCompletionContext)
    assert isinstance(session4.agent._model_context, TokenLimitedChatCompletionContext)


@pytest.mark.asyncio
async def test_head_tail_edge_cases(dynaconf_test_settings, patch_tools):
    """Test head_tail context with edge cases"""
    from mchat_core.agent_manager import AutogenManager

    agents = {
        "ht_zero_head": {
            "type": "agent",
            "description": "desc",
            "prompt": "hi",
            "context": {"type": "head_tail", "head_size": 0, "tail_size": 5},  # Invalid: head_size must be > 0, should fallback to unbounded
        },
        "ht_defaults": {
            "type": "agent",
            "description": "desc", 
            "prompt": "hi",
            "context": {"type": "head_tail"},  # Should use defaults: head_size=3, tail_size=20
        },
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    
    session1 = await m.new_conversation(agent="ht_zero_head")
    session2 = await m.new_conversation(agent="ht_defaults")
    
    # head_size=0 is invalid, should fallback to unbounded
    assert isinstance(session1.agent._model_context, UnboundedChatCompletionContext)
    # defaults should work fine
    assert isinstance(session2.agent._model_context, HeadAndTailChatCompletionContext)


@pytest.mark.asyncio  
async def test_context_message_persistence(dynaconf_test_settings, patch_tools):
    """Test that context maintains messages across multiple interactions"""
    from mchat_core.agent_manager import AutogenManager
    from autogen_core.models import UserMessage, AssistantMessage
    
    agents = {
        "persistent": {
            "type": "agent",
            "description": "desc",
            "prompt": "You always respond with 'OK'",
            "context": {"type": "buffered", "buffer_size": 10},
        }
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="persistent")
    
    # Add messages to context and verify they persist
    ctx = session.agent._model_context
    
    # Add some messages
    await ctx.add_message(UserMessage(content="Hello", source="user"))
    await ctx.add_message(AssistantMessage(content="Hi there", source="assistant"))
    await ctx.add_message(UserMessage(content="How are you?", source="user"))
    
    # Verify messages are retained
    messages = await ctx.get_messages()
    assert len(messages) >= 3  # Should include system prompt + our messages
    
    # Check that our messages are in there
    content_strings = [str(msg.content) for msg in messages]
    assert any("Hello" in content for content in content_strings)
    assert any("Hi there" in content for content in content_strings) 
    assert any("How are you?" in content for content in content_strings)


@pytest.mark.asyncio
async def test_buffered_context_exact_behavior(dynaconf_test_settings, patch_tools):
    """Test exact buffering behavior - only latest N messages should be kept"""
    from mchat_core.agent_manager import AutogenManager
    from autogen_core.models import UserMessage, AssistantMessage

    agents = {
        "buffer_1": {
            "type": "agent",
            "description": "desc",
            "prompt": "test",
            "context": {"type": "buffered", "buffer_size": 1},
        },
        "buffer_2": {
            "type": "agent", 
            "description": "desc",
            "prompt": "test",
            "context": {"type": "buffered", "buffer_size": 2},
        }
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    
    # Test buffer_size = 1: should only keep the latest message
    session1 = await m.new_conversation(agent="buffer_1")
    ctx1 = session1.agent._model_context
    
    # Add 5 messages
    await ctx1.add_message(UserMessage(content="Message 1", source="user"))
    await ctx1.add_message(UserMessage(content="Message 2", source="user"))
    await ctx1.add_message(UserMessage(content="Message 3", source="user"))
    await ctx1.add_message(UserMessage(content="Message 4", source="user"))
    await ctx1.add_message(UserMessage(content="Message 5", source="user"))
    
    messages1 = await ctx1.get_messages()
    # Should have system prompt + 1 buffered message = 2 total (or just 1 if no system prompt)
    user_messages1 = [msg for msg in messages1 if hasattr(msg, 'source') and msg.source == 'user']
    assert len(user_messages1) == 1, f"Expected 1 user message, got {len(user_messages1)}"
    assert "Message 5" in str(user_messages1[0].content), "Should only have the latest message"
    assert not any("Message 1" in str(msg.content) for msg in messages1), "Should not have old messages"
    assert not any("Message 2" in str(msg.content) for msg in messages1), "Should not have old messages"
    
    # Test buffer_size = 2: should only keep the latest 2 messages
    session2 = await m.new_conversation(agent="buffer_2")
    ctx2 = session2.agent._model_context
    
    # Add 5 messages
    await ctx2.add_message(UserMessage(content="Msg A", source="user"))
    await ctx2.add_message(UserMessage(content="Msg B", source="user"))
    await ctx2.add_message(UserMessage(content="Msg C", source="user"))
    await ctx2.add_message(UserMessage(content="Msg D", source="user"))
    await ctx2.add_message(UserMessage(content="Msg E", source="user"))
    
    messages2 = await ctx2.get_messages()
    user_messages2 = [msg for msg in messages2 if hasattr(msg, 'source') and msg.source == 'user']
    assert len(user_messages2) == 2, f"Expected 2 user messages, got {len(user_messages2)}"
    
    # Should have the latest 2 messages (D and E)
    content_strings2 = [str(msg.content) for msg in user_messages2]
    assert any("Msg D" in content for content in content_strings2), "Should have second-to-last message"
    assert any("Msg E" in content for content in content_strings2), "Should have latest message"
    assert not any("Msg A" in str(msg.content) for msg in messages2), "Should not have old messages"
    assert not any("Msg B" in str(msg.content) for msg in messages2), "Should not have old messages"
    assert not any("Msg C" in str(msg.content) for msg in messages2), "Should not have old messages"


@pytest.mark.asyncio
async def test_head_tail_context_exact_behavior(dynaconf_test_settings, patch_tools):
    """Test exact head/tail behavior - should keep first N and last M messages"""
    from mchat_core.agent_manager import AutogenManager
    from autogen_core.models import UserMessage
    
    agents = {
        "head_tail": {
            "type": "agent",
            "description": "desc", 
            "prompt": "test",
            "context": {"type": "head_tail", "head_size": 2, "tail_size": 2},
        }
    }
    
    m = AutogenManager(message_callback=lambda *a, **kw: None, agents=agents)
    session = await m.new_conversation(agent="head_tail")
    ctx = session.agent._model_context
    
    # Add many messages to exceed head + tail capacity
    for i in range(10):
        await ctx.add_message(UserMessage(content=f"Message {i}", source="user"))
    
    messages = await ctx.get_messages()
    user_messages = [msg for msg in messages if hasattr(msg, 'source') and msg.source == 'user']
    
    # Should have at most head_size + tail_size = 4 user messages
    assert len(user_messages) <= 4, f"Expected at most 4 user messages, got {len(user_messages)}"
    
    # Check that we have the first 2 (head) and last 2 (tail)
    content_strings = [str(msg.content) for msg in user_messages]
    
    # Should have early messages (head)
    assert any("Message 0" in content for content in content_strings), "Should have first message in head"
    assert any("Message 1" in content for content in content_strings), "Should have second message in head"
    
    # Should have late messages (tail)  
    assert any("Message 8" in content for content in content_strings), "Should have second-to-last message in tail"
    assert any("Message 9" in content for content in content_strings), "Should have last message in tail"
    
    # Should NOT have middle messages
    assert not any("Message 4" in content for content in content_strings), "Should not have middle messages"
    assert not any("Message 5" in content for content in content_strings), "Should not have middle messages"
