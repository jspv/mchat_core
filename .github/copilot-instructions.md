# Copilot instructions for mchat_core

Rules for AI coding agents in this repo. Keep edits small, follow existing patterns, and reference concrete files when changing behavior.

## Big picture
- Purpose: utility layer for Autogen-driven multi-agent chats (Dynaconf + YAML config).
- Key modules: `agent_manager.py` (sessions, teams, streaming, tools/MCP), `model_manager.py` (Dynaconf -> OpenAI/Azure clients + feature flags), `tool_utils.py` (Python tools + MCP parsing/validation/loading), `logging_utils.py` (TRACE logger + rich formatting).

## Architecture & data flow
- Agents are defined in YAML/JSON or strings; loaded via `AgentManager._load_agents(...)` when constructing `AutogenManager(message_callback, agent_paths=[...])` or with in-memory dict.
- Create conversations only via `await AgentManager.new_conversation(...)` to get an `AgentSession` (direct `AgentSession(...)` raises RuntimeError).
- Solo agents use `ModelManager.open_model(model_id)` to build clients. If a model lacks system-prompt support, the agent prompt is injected as the first user message; ensure your context window preserves it.
- Tools per agent = Python tools from `mchat_core/tools/` + MCP tools resolved at session start. If the selected model lacks tool support, tools are omitted.
- Teams use Autogen group chats (`RoundRobinGroupChat`, `SelectorGroupChat`, `MagenticOneGroupChat`). Streaming is disabled for teams.

## Conventions that matter
- Context strategies (agent `context`): `unbounded` (default) | `buffered` (buffer_size>0) | `token` (optional token_limit) | `head_tail` (head>=0, tail>0).
  - When system prompts are unsupported: use `buffered` with buffer_size>=2 or `head_tail` with head>=1 to retain the injected prompt; invalid configs raise `ValueError`.
- Termination: compose `StopMessageTermination` + `MaxMessageTermination` (+ optional `TextMentionTermination`) and a `SmartReflectorTermination` using `default_mini_model`.
- Streaming: enabled only if supported by the model; `AgentSession.stream_tokens` is per-session and becomes `None` when unsupported.
- MCP tools in `tools:`:
  - String: `mcp: uvx --from . server` or `mcp:http://localhost:8000/mcp`
  - Dict: `{ mcp: "...", cwd: "...", env: {...} }`
  - Placeholders are registered at startup; replaced with real tools on session start.
- Python tools: subclass `BaseTool` in `mchat_core/tools`, implement `run(...)`, optionally `verify_setup()` to set `is_callable`; wrapped as `autogen_core.tools.FunctionTool`.

## Dev workflows
- Deps: managed with `uv`. Optional tool deps group: `tools` in `pyproject.toml` (install: `uv sync --group tools`).
- Tests (pytest): default excludes live LLM (`-m "not live_llm"`). Run: `pytest`; skip reasons: `pytest -rs`; only tools: `pytest -m tools -rs`; live LLM: set `OPENAI_API_KEY` and run `pytest -m live_llm -rs`.
- Lint/type-check: Ruff + mypy configured in `pyproject.toml`.

## Minimal examples
- Agent YAML (mixing tools + MCP):
  ```yaml
  helpful:
    type: agent
    description: General chatbot
    prompt: Follow instructions exactly.
    tools:
      - google_search
      - mcp: uvx --from . my-mcp_server
    context:
      type: head_tail
      head_size: 1
      tail_size: 20 
  ```
- Start a session:
  ```python
  from mchat_core.agent_manager import AutogenManager
  m = AutogenManager(message_callback=lambda *a, **k: None, agent_paths=["agents.yaml"])
  session = await m.new_conversation(agent="helpful")
  result = await session.ask("Reply with: OK")
  ```

## Gotchas
- Don’t instantiate `AgentSession` directly; always use `new_conversation`.
- Models with `_tool_support=false` ignore tools even if configured.
- When system prompts are unsupported, misconfigured context can drop your prompt—prefer `head_tail` or `buffered (>=2)`.
- Team agents don’t stream; don’t rely on token chunks in callbacks.

Questions or gaps (e.g., selector prompts, MCP HTTP vs stdio nuances, logger setup)? Point them out and we’ll expand this doc with concrete examples.
