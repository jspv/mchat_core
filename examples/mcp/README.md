# MCP Date/Time Demo Server

A minimal Model Context Protocol (MCP) stdio server written in Python (using FastMCP). It exposes a single tool that returns the current date/time in ISO 8601 format.

## What it provides

- Tool: `now(tz: str | None = None) -> str`
  - `tz`: Use "UTC" for UTC; anything else is treated as local time
  - Returns: ISO 8601 string, e.g. `2025-10-01T12:34:56.789012+00:00`

## Run it (stdio)

- Using your local Python (ensure the `fastmcp` package is installed):

```bash
pip install fastmcp
python examples/mcp/mcp_datetime_server.py
```

- Using `uvx` (no local install, ephemeral env, recommended):

```bash
uvx --with fastmcp python examples/mcp/mcp_datetime_server.py
```

The server speaks MCP over stdio and is intended to be launched by an MCP client.

## Use it from Agent YAML (stdio)

```yaml
tools:
  - mcp:uvx --with fastmcp python examples/mcp/mcp_datetime_server.py
```

Or, if you've installed `fastmcp` locally and want to call Python directly:

```yaml
tools:
  - mcp:python examples/mcp/mcp_datetime_server.py
```

Optionally set a working directory if needed (e.g., when referencing relative paths with `uvx`):

```yaml
tools:
  - mcp: uvx --with fastmcp python examples/mcp/mcp_datetime_server.py
    cwd: /Users/justin/src/mchat/mchat_core
```

## Serve over HTTP-stream (optional)

Expose the same tool over HTTP-stream (not SSE) using the wrapper:

```bash
# No local install: add required deps to the ephemeral env
uvx --with fastmcp --with uvicorn --with starlette \
  python examples/mcp/mcp_datetime_http_stream.py

# Or, install locally and run
pip install fastmcp uvicorn starlette
python examples/mcp/mcp_datetime_http_stream.py
```

Then reference it from YAML using an HTTP MCP URL:

```yaml
tools:
  - mcp:http://127.0.0.1:8000/mcp
```

## Notes

- This server is intentionally minimal and suitable for demonstrations and tests.
