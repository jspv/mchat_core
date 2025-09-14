import asyncio
import os
import textwrap

import pytest
from dynaconf import Dynaconf


def require_env(var: str):
    if not os.environ.get(var):
        pytest.skip(f"Skipping live LLM tests; missing env var: {var}")


@pytest.fixture(scope="function")
def live_settings(monkeypatch):
    """Load example_settings.toml with a real OpenAI key and patch get_settings.

    Skips when OPENAI_API_KEY is not set.
    """
    require_env("OPENAI_API_KEY")
    root = os.path.dirname(os.path.dirname(__file__))  # repo root
    settings_path = os.path.join(root, "example_settings.toml")
    if not os.path.exists(settings_path):
        pytest.skip("example_settings.toml not found; skipping live tests")

    # Load example settings and inject the key
    cfg = Dynaconf(settings_files=[settings_path])
    # Provide variables referenced by @format in example_settings.toml
    cfg.set("openai_api_key", os.environ["OPENAI_API_KEY"])  # used via @format
    cfg.set("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", "dummy_openrouter_key"))
    cfg.set("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT", "https://example.azure.com"))
    # Provide dummy Azure AD client credentials when azure models are present
    cfg.set("azure_tenant_id", os.environ.get("AZURE_TENANT_ID", "00000000-0000-0000-0000-000000000000"))
    cfg.set("azure_client_id", os.environ.get("AZURE_CLIENT_ID", "11111111-1111-1111-1111-111111111111"))
    cfg.set("azure_client_id_secret", os.environ.get("AZURE_CLIENT_SECRET", "dummy-secret"))

    # Prefer the cheaper/faster mini model by default if available
    if "gpt-4o-mini" in cfg["models"]["chat"]:
        cfg.defaults.chat_model = "gpt-4o-mini"
        cfg.defaults.mini_model = "gpt-4o-mini"
    # Provide a dummy google_api_key default to satisfy code paths that might access it
    cfg.defaults.google_api_key = os.environ.get("GOOGLE_API_KEY", "dummy_google_key")

    # Ensure minimal noise from tools: tests choose agents without tools

    # Patch the package settings loader to return this instance
    monkeypatch.setattr("mchat_core.config.get_settings", lambda *a, **kw: cfg)
    return cfg


@pytest.fixture(scope="function")
def example_agents_path():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "example_agents.yaml")
    if not os.path.exists(path):
        pytest.skip("example_agents.yaml not found; skipping live tests")
    return path


# Pretty-print helper for manual validation; run pytest with -s to see output
def _dump_messages(result, calls=None, title: str = None, max_chars: int = 500):
    try:
        if title:
            print(f"\n===== {title} =====")
        print("=== LLM TaskResult ===")
        print(f"stop_reason={getattr(result, 'stop_reason', None)}")
        msgs = getattr(result, "messages", None) or []
        for i, m in enumerate(msgs):
            src = getattr(m, "source", "?")
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else str(content)
            print(f"[{i}] {src}: {text[:max_chars]}")
        if calls is not None:
            print("=== Streaming/Callback events ===")
            for i, (msg, kw) in enumerate(calls):
                agent = kw.get("agent") if isinstance(kw, dict) else None
                complete = kw.get("complete") if isinstance(kw, dict) else None
                mtxt = msg if isinstance(msg, str) else str(msg)
                print(f"[{i}] agent={agent} complete={complete} msg={mtxt[:200]}")
    except Exception as e:
        print(f"[pp-error] {type(e).__name__}: {e}")


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_solo_agent_default_no_tools_simple(live_settings):
    from mchat_core.agent_manager import AutogenManager

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    yaml_str = textwrap.dedent(
        """
        solo_agent:
          type: agent
          description: A general-purpose chatbot
          prompt: "Follow instructions exactly."
          oneshot: false
          max_rounds: 5
        """
    )

    m = AutogenManager(message_callback=cb, agent_paths=[yaml_str])
    session = await m.new_conversation(agent="solo_agent")

    # Keep it simple to speed up completion
    task = "Reply with the exact word: OK"

    result = await asyncio.wait_for(session.ask(task), timeout=90)
    assert result is not None
    assert isinstance(result.stop_reason, str)
    assert result.messages, f"No messages; stop_reason={result.stop_reason}"
    # At least one callback should have fired (streaming or final)
    assert calls, "Expected at least one callback invocation"
    _dump_messages(result, calls=calls, title="solo agent: solo_agent")
    # Validate the agent replied with OK as requested (strict normalized check)
    def _norm(s: str) -> str:
        return s.strip().strip("'\"`.,! ").upper()

    contents = [getattr(m, "content", "") for m in getattr(result, "messages", [])]
    normalized = [_norm(c) for c in contents if isinstance(c, str)]
    assert any(c == "OK" for c in normalized), (
        f"Expected an exact 'OK' reply, got: {contents}"
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_round_robin_team_ctx_team_mixed(live_settings):
        from mchat_core.agent_manager import AutogenManager

        yaml_str = textwrap.dedent(
                """
                rr_a:
                    type: agent
                    description: a
                    prompt: "No matter what the user says, respond with 'Hello from A.'"
                    oneshot: false
                rr_b:
                    type: agent
                    description: b
                    prompt: "No matter what the user says, respond with 'Hello from B.'"
                    oneshot: false
                ctx_team_mixed:
                    type: team
                    team_type: round_robin
                    description: Team example
                    max_rounds: 3
                    oneshot: false
                    agents: [rr_a, rr_b]
                """
        )

        calls: list[tuple[str, dict]] = []

        async def cb(msg, **kw):
            calls.append((msg, kw))

        m = AutogenManager(message_callback=cb, agent_paths=[yaml_str])
        session = await m.new_conversation(agent="ctx_team_mixed")

        # Encourage the orchestrator to proceed to the next participant
        result = await asyncio.wait_for(
            session.ask("Have each team member greet once, then stop."),
            timeout=120,
        )
        assert result is not None
        assert result.messages and any(getattr(msg, "content", "").strip() for msg in result.messages)
        # Verify that both rr_a and rr_b produced output (round-robin alternation)
        spoken = {kw.get("agent") for _, kw in calls if isinstance(kw, dict)}
        assert {"rr_a", "rr_b"}.issubset(spoken), f"Expected both agents to speak, got: {spoken}"
        # Verify each agent followed its instruction exactly enough (normalized check)
        def _norm_line(s: str) -> str:
            # Lowercase, drop quotes/punct, collapse whitespace
            import re
            t = s.lower()
            t = t.replace("'", "").replace('"', "").replace("`", "")
            t = re.sub(r"[\.,!?:;]+", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        msgs_a = [
            _norm_line(msg)
            for msg, kw in calls
            if isinstance(kw, dict) and kw.get("agent") == "rr_a" and isinstance(msg, str)
        ]
        msgs_b = [
            _norm_line(msg)
            for msg, kw in calls
            if isinstance(kw, dict) and kw.get("agent") == "rr_b" and isinstance(msg, str)
        ]
        assert any("hello from a" in m for m in msgs_a), (
            f"rr_a did not respond with expected phrase. Got: {msgs_a}"
        )
        assert any("hello from b" in m for m in msgs_b), (
            f"rr_b did not respond with expected phrase. Got: {msgs_b}"
        )
        _dump_messages(result, calls=calls, title="round_robin: ctx_team_mixed")


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_selector_team_minimal(live_settings):
    from mchat_core.agent_manager import AutogenManager

    # Build a tiny selector team with no tools and an explicit termination word
    yaml_str = textwrap.dedent(
        """
        sel_a:
          type: agent
          description: first team member
          prompt: >
            No matter what the user says,
            respond with 'Hello from A.'
          oneshot: false
        sel_b:
          type: agent
          description: second team member
          prompt: >
            No matter what the user says,
            respond with 'Hello from B.'
          oneshot: false
        last_speaker:
          type: agent
          description: last speaker, speaks after everyone else has spoken
          prompt: >
            No matter what the user says,
            respond with 'TERMINATE' from the last speaker.
          oneshot: false
        sel_c:
          type: agent
          description: third team member
          prompt: >
            No matter what the user says,
            respond with 'Hello from C.'
          oneshot: false
        sel_team:
          type: team
          team_type: selector
          description: selector demo
          allow_repeated_speakers: true
          selector_prompt: >
            Select an agent to speak next.
            {roles}

            Current conversation context:
            {history}

            Read the above conversation, then select an agent from {participants} to 
            speak next. Make sure everyone speaks at least once and last_speaker 
            should speak last. Only select one agent.
          agents: [sel_a, sel_b, last_speaker, sel_c]
          team_model: gpt-4o-mini
          termination_message: TERMINATE
          oneshot: false
          max_rounds: 6
        """
    )

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    m = AutogenManager(message_callback=cb, agent_paths=[yaml_str])
    session = await m.new_conversation(agent="sel_team")

    result = await asyncio.wait_for(session.ask("Hello"), timeout=120)
    assert result is not None
    assert isinstance(result.stop_reason, str)
    assert result.messages
    _dump_messages(result, calls=calls, title="selector team: sel_team")

    msgs = getattr(result, "messages", [])
    assert len(msgs) == 5, f"Expected 5 messages (user + four agents), got {len(msgs)}"
    speakers = [getattr(m, "source", "?") for m in msgs]
    assert speakers[-1] == "last_speaker", (
        f"Expected last message from last_speaker, got: {speakers[-1]}"
    )

    # Ensure each agent spoke at least once (besides the initial user message)
    participant_speakers = set(speakers[1:])
    expected_participants = {"sel_a", "sel_b", "sel_c", "last_speaker"}
    assert expected_participants.issubset(participant_speakers), (
        f"Expected all agents to speak at least once; got: {participant_speakers}"
    )

    # Validate callbacks: 4 final completions in any order
    final_calls = [
        kw.get("agent")
        for (msg, kw) in calls
        if isinstance(kw, dict) and kw.get("complete") is True
    ]
    assert len(final_calls) == 4, (
        f"Expected 4 final callback events, got {len(final_calls)}: {final_calls}"
    )
    assert expected_participants == set(final_calls), (
        f"Callback final agents mismatch. Expected {expected_participants}, got {set(final_calls)}"
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_streaming_callbacks_observed(live_settings):
    from mchat_core.agent_manager import AutogenManager

    calls = []

    async def cb(msg, **kw):
        calls.append((msg, kw))

    # stream_tokens defaults to True when callback provided
    yaml_str = textwrap.dedent(
        """
        default_no_tools:
          type: agent
          description: A general-purpose chatbot
          prompt: "You are helpful."
          oneshot: false
          max_rounds: 5
        """
    )
    m = AutogenManager(message_callback=cb, agent_paths=[yaml_str])
    session = await m.new_conversation(agent="default_no_tools")

    result = await asyncio.wait_for(session.ask("Give a very short answer: 'done'"), timeout=90)

    # We don't strictly require streaming chunks (depends on model and latency),
    # but we should observe at least one callback call overall.
    assert calls, "Expected callback to receive at least one event"
    assert result is not None and result.messages
    _dump_messages(result, calls=calls, title="streaming: default_no_tools")


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_magentic_one_team_minimal(live_settings):
    from mchat_core.agent_manager import AutogenManager
    import time

    # Minimal MagenticOne team with three simple agents
    yaml_str = textwrap.dedent(
        """
        m1_a:
          type: agent
          description: Knows a piece of the puzzle
          prompt: >
            Follow the instructions of the MagenticOneOrchestrator. If asked to share 
            your piece of the puzzle, provide the answer "wolf" otherwise be helpful.
          context:
            type: buffered
            buffer_size: 1
          oneshot: false
        m2_a:
          type: agent
          description: Knows a piece of the puzzle
          prompt: >
            Follow the instructions of the MagenticOneOrchestrator. If asked to share 
            your piece of the puzzle, provide the answer "ram" otherwise be helpful.
          context:
            type: buffered
            buffer_size: 1          
          oneshot: false
        m3_a:
          type: agent
          description: Knows a piece of the puzzle
          prompt: > 
            Follow the instructions of the MagenticOneOrchestrator. If asked 
            to share your piece of the puzzle, provide the answer "hart" otherwise be 
            helpful. - important, this is "hart" not "heart"
          context:
            type: buffered
            buffer_size: 1
          oneshot: false
        m1_team:
          type: team
          team_type: magnetic_one
          description: magnetic one demo
          agents: [m1_a, m2_a, m3_a]
          team_model: gpt-4_1
        """
    )

    # Verbose status callback to observe progress in real time
    start_ts = time.monotonic()

    async def cb(msg, **kw):
        agent = kw.get("agent") if isinstance(kw, dict) else None
        complete = kw.get("complete") if isinstance(kw, dict) else None
        try:
            text = msg if isinstance(msg, str) else str(msg)
        except Exception:
            text = repr(msg)
        elapsed = time.monotonic() - start_ts
        # Print a detailed status block with full message (no truncation)
        print(
            f"\n[m1 status +{elapsed:.2f}s] agent={agent} complete={complete} msg:\n{text}\n",
            flush=True,
        )

    m = AutogenManager(message_callback=cb, agent_paths=[yaml_str])
    session = await m.new_conversation(agent="m1_team")

    result = await asyncio.wait_for(
        session.ask("The solution to the puzzle is three words, work with your team to find it.  When you have the three words you are done."),
        timeout=180,
    )
    assert result is not None
    assert result.messages and any(getattr(msg, "content", "").strip() for msg in result.messages)
    _dump_messages(result, title="magentic_one: m1_team")

    # Success criteria: the last message should contain all three words
    # 'wolf', 'ram', and 'hart' (case-insensitive)
    last_msg = getattr(result, "messages", [])[-1]
    last_text = str(getattr(last_msg, "content", ""))
    low = last_text.lower()
    for token in ("wolf", "ram", "hart"):
        assert token in low, f"Expected '{token}' to appear in the final message, got: {last_text}"
