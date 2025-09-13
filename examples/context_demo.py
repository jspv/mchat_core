"""Context demo using print statements.

This script prints model context details before and after a sample ask.
"""

# ruff: noqa: T201
import asyncio
from contextlib import asynccontextmanager

from autogen_core.models import UserMessage

from mchat_core.agent_manager import AutogenManager
from mchat_core.model_manager import ModelManager

AGENTS_YAML = """
demo_unbounded:
    type: agent
    description: Demo agent using unbounded context
    prompt: |
        DEMO PROMPT: You will see how the prompt is injected.
    context:
        type: unbounded

demo_buffered:
    type: agent
    description: Demo agent using buffered context
    prompt: |
        DEMO PROMPT: You will see how the prompt is injected.
    context:
        type: buffered
        buffer_size: 3

demo_token:
    type: agent
    description: Demo agent using token-limited context
    prompt: |
        DEMO PROMPT: You will see how the prompt is injected.
    context:
        type: token
        token_limit: 2048

demo_head_tail:
    type: agent
    description: Demo agent using head/tail context
    prompt: |
        DEMO PROMPT: You will see how the prompt is injected.
    context:
        type: head_tail
        head_size: 1
        tail_size: 2
"""


@asynccontextmanager
async def patched_system_prompt_support(mm: ModelManager, supported: bool):
    """Temporarily override system-prompt support detection for demonstration."""
    original = mm.get_system_prompt_support

    def _patched(model_id: str) -> bool:  # type: ignore[override]
        return supported

    mm.get_system_prompt_support = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        mm.get_system_prompt_support = original  # type: ignore[assignment]


async def show_agent_context(manager: AutogenManager, name: str):
    session = await manager.new_conversation(agent=name, stream_tokens=False)
    ctx = session.agent._model_context
    mm = manager.mm

    # BEFORE ask: inspect context and effective system prompt
    msgs_before = await ctx.get_messages()
    sys_prompt_supported = mm.get_system_prompt_support(session.model)
    effective_system_prompt = session.prompt if sys_prompt_supported else None

    print()
    print(f"Agent: {name}")
    print(f"  Context type: {ctx.__class__.__name__}")
    print(f"  System prompt supported: {sys_prompt_supported}")
    print(f"  Configured prompt: {repr((session.prompt or '')[:80])}")
    eff = (effective_system_prompt or "")[:80]
    print(f"  Effective system prompt this run: {repr(eff) or 'None'}")
    print(f"  Messages BEFORE ask: {len(msgs_before)}")
    for i, m in enumerate(msgs_before):
        src = getattr(m, "source", "?")
        content = getattr(m, "content", "")
        print(f"    [{i}] {m.__class__.__name__} from={src} :: {content}")

    # Demonstrate context windowing by adding messages
    for i in range(5):
        await ctx.add_message(UserMessage(content=f"demo m{i}", source="user"))
    msgs_demo = await ctx.get_messages()
    print(f"  After adding 5 demo messages -> context size: {len(msgs_demo)}")
    for i, m in enumerate(msgs_demo):
        src = getattr(m, "source", "?")
        content = getattr(m, "content", "")
        print(f"    [{i}] {m.__class__.__name__} from={src} :: {content}")

    # Perform a small ask; ModelManager handles API/model setup and errors
    try:
        result = await session.ask("Say 'hello' only.")
        last = result.messages[-1] if result.messages else None
        if last is not None:
            tail = getattr(last, "content", "")
            print(f"  Model reply: {repr((tail or '')[:120])}")
        else:
            print("  Model reply: <no messages>")
    except Exception as e:
        print(f"  Ask failed: {e}")

    # AFTER ask: inspect context again
    msgs_after = await ctx.get_messages()
    print(f"  Messages AFTER ask: {len(msgs_after)}")
    for i, m in enumerate(msgs_after):
        src = getattr(m, "source", "?")
        content = getattr(m, "content", "")
        print(f"    [{i}] {m.__class__.__name__} from={src} :: {content}")


async def demo_once(force_supported: bool | None):
    print("=" * 80)
    label = (
        (
            "(force system-prompt supported)"
            if force_supported
            else "(force unsupported)"
        )
        if force_supported is not None
        else "(auto-detect)"
    )
    print(f"Context demo {label}")

    async def _noop(*_a, **_kw):
        return None

    manager = AutogenManager(message_callback=_noop, agent_paths=[AGENTS_YAML])

    if force_supported is None:
        # Auto-detect: just run once with the environment's model settings
        for name in (
            "demo_unbounded",
            "demo_buffered",
            "demo_token",
            "demo_head_tail",
        ):
            await show_agent_context(manager, name)
        return

    # Patch support flag for demo purposes
    async with patched_system_prompt_support(manager.mm, supported=force_supported):
        for name in (
            "demo_unbounded",
            "demo_buffered",
            "demo_token",
            "demo_head_tail",
        ):
            await show_agent_context(manager, name)


async def main():
    # 1) Auto-detect as configured
    await demo_once(None)
    # 2) Force unsupported to show the alternate path
    await demo_once(False)
    # 3) Force supported to show the supported path
    await demo_once(True)


if __name__ == "__main__":
    asyncio.run(main())
