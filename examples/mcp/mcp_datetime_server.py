"""
Minimal MCP stdio server for demo and testing.

Exposes one tool:
    - now(): returns current date/time in ISO 8601 format (UTC by default)

Recommended server SDK: FastMCP (simple decorator-based API)

Run locally (any of the following):
    - uvx --with fastmcp python examples/mcp/mcp_datetime_server.py
    - pip install fastmcp && python examples/mcp/mcp_datetime_server.py

Use from Agent YAML (stdio MCP):
    tools:
        - mcp:uvx --with fastmcp python examples/mcp/mcp_datetime_server.py
    # or, if you installed fastmcp locally
        - mcp:python examples/mcp/mcp_datetime_server.py
"""

from __future__ import annotations

import datetime as _dt

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception as e:  # pragma: no cover - example script
    raise SystemExit(
        "This example uses the 'fastmcp' package (Python MCP server).\n"
        "Run without local installs using:\n"
        "  uvx --with fastmcp python examples/mcp/mcp_datetime_server.py\n"
        "\nOr install it locally:\n"
        "  pip install fastmcp\n"
        f"\nImport error: {type(e).__name__}: {e}"
    ) from e


mcp = FastMCP("datetime-mcp")


@mcp.tool()
def now(tz: str | None = None) -> str:
    """Return the current date/time.

    Args:
        tz: Optional timezone string (e.g., 'UTC'). For this simple demo,
            only 'UTC' is recognized; anything else is treated as local time.

    Returns:
        ISO 8601 string, e.g., '2025-10-01T12:34:56.789012+00:00'
    """
    if (tz or "").upper() == "UTC":
        now_dt = _dt.datetime.now(tz=_dt.timezone.utc)
    else:
        # Local time with offset if available
        try:
            now_dt = _dt.datetime.now().astimezone()
        except Exception:
            now_dt = _dt.datetime.now()
    return now_dt.isoformat()


def main() -> None:  # pragma: no cover - example script
    # Run the MCP server over stdio (FastMCP provides a simple runner)
    try:
        mcp.run()
    except TypeError:
        mcp.run_stdio()  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover - example script
    main()
