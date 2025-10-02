"""
HTTP-stream MCP wrapper for the datetime demo server.

This exposes the same MCP tools over HTTP (streamable HTTP, not SSE)
by wrapping the FastMCP ASGI app and serving it with uvicorn.

Environment variables:
    - HOST (default: 127.0.0.1)
    - PORT (default: 8000)
    - CORS_ORIGINS (default: "*" to allow all; comma-separated list otherwise)

Run examples:
      - uvx --with fastmcp --with uvicorn --with starlette \
          python examples/mcp/mcp_datetime_http_stream.py
      - pip install fastmcp uvicorn starlette \
          && python examples/mcp/mcp_datetime_http_stream.py

From YAML (client side):
    tools:
        - mcp:http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import os

try:
    import uvicorn  # type: ignore
    from starlette.middleware.cors import CORSMiddleware  # type: ignore
except Exception as e:  # pragma: no cover - example script
    raise SystemExit(
        "This wrapper requires 'uvicorn' and 'starlette'.\n"
        "Run without local installs using:\n"
        "  uvx --with fastmcp --with uvicorn --with starlette "
        "python examples/mcp/mcp_datetime_http_stream.py\n"
        "\nOr install locally:\n"
        "  pip install fastmcp uvicorn starlette\n"
        f"\nImport error: {type(e).__name__}: {e}"
    ) from e

# Import the FastMCP app from the stdio server module (same tools)
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mcp_datetime_server as mcp_server  # noqa: E402


def get_app():
    """Return the FastMCP streamable HTTP ASGI app with optional CORS."""
    app = mcp_server.mcp.streamable_http_app()

    cors_origins = os.environ.get("CORS_ORIGINS", "*")
    origins = (
        ["*"]
        if cors_origins.strip() == "*"
        else [o.strip() for o in cors_origins.split(",") if o.strip()]
    )
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


def main() -> None:  # pragma: no cover - example script
    host = os.environ.get("HOST", "127.0.0.1")
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    # Run using the ASGI app factory; pass the callable directly to avoid
    # import-string issues
    uvicorn.run(
        get_app,
        factory=True,
        host=host,
        port=port,
        log_level="warning",  # type: ignore[arg-type]
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover - example script
    main()
