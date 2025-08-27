import os
from textwrap import dedent
from unittest.mock import patch

import pytest

from mchat_core.tool_utils import load_tools, BaseTool


# Helper to create a Python module file in a temp directory
def write_module(p, name, content):
    f = p / f"{name}.py"
    f.write_text(dedent(content))
    return f


def test_load_tools_missing_dir_returns_empty(tmp_path):
    # When the target directory does not exist, load_tools should return an empty dict.
    missing = tmp_path / "nope"
    tools = load_tools(str(missing))
    assert tools == {}


@patch("mchat_core.tool_utils.FunctionTool")
def test_load_tools_discovers_valid_base_tool(mock_ft, tmp_path):
    # Create a valid, callable tool class that inherits from BaseTool.
    write_module(
        tmp_path,
        "valid_tool",
        """
        from mchat_core.tool_utils import BaseTool
        class MyTool(BaseTool):
            name = "my_tool"
            description = "My test tool"
            is_callable = True
            def run(self, x: int = 1) -> int:
                return x + 1
        """,
    )

    # Mock the FunctionTool wrapper to avoid depending on autogen internals.
    mock_tool_instance = object()
    mock_ft.return_value = mock_tool_instance

    # load_tools should discover the tool module, instantiate the tool, and wrap it.
    tools = load_tools(str(tmp_path))

    # The discovered tool should be present under its declared name.
    assert "my_tool" in tools
    # The value should be whatever FunctionTool returned (mocked here).
    assert tools["my_tool"] is mock_tool_instance

    # Verify we created the FunctionTool with expected args.
    assert mock_ft.called
    _, kwargs = mock_ft.call_args
    assert kwargs["name"] == "my_tool"
    assert kwargs["description"] == "My test tool"
    # Different autogen versions use different kwarg names; accept any callable.
    assert callable(kwargs["func"]) or callable(kwargs.get("function") or kwargs.get("run", None))


@patch("mchat_core.tool_utils.FunctionTool")
def test_load_tools_skips_not_callable_tools(mock_ft, tmp_path):
    # Tools with is_callable=False should be skipped (not wrapped or returned).
    write_module(
        tmp_path,
        "not_callable_tool",
        """
        from mchat_core.tool_utils import BaseTool
        class NotCallable(BaseTool):
            name = "bad_tool"
            description = "Not callable"
            is_callable = False
        """,
    )
    tools = load_tools(str(tmp_path))
    assert "bad_tool" not in tools
    mock_ft.assert_not_called()



def test_load_tools_module_import_failure_is_ignored(tmp_path):
    # If importing a tool module raises, load_tools should catch and continue.
    write_module(
        tmp_path,
        "boom",
        """
        raise RuntimeError("boom at import")
        """,
    )
    tools = load_tools(str(tmp_path))
    # The faulty module should not break discovery; result remains empty.
    assert tools == {}