"""Simplified tests for MCP (Model Context Protocol) tool functionality."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, Mock

from mchat_core.tool_utils import (
    MCPToolSpec,
    parse_mcp_tools,
    MCPToolManager,
    validate_and_load_mcp_tools,
    load_agent_mcp_tools,
)


class TestMCPToolSpec:
    """Test the MCPToolSpec class."""

    def test_web_mcp_spec_parsing(self):
        """Test parsing of web-based MCP tool specifications."""
        spec = MCPToolSpec("mcp:http://localhost:3000")
        
        assert spec.type == "web"
        assert spec.connection_info["url"] == "http://localhost:3000"
        assert spec.connection_info["host"] == "localhost"
        assert spec.connection_info["port"] == 3000
        assert spec.connection_info["scheme"] == "http"

    def test_https_mcp_spec_parsing(self):
        """Test parsing of HTTPS MCP tool specifications."""
        spec = MCPToolSpec("mcp:https://api.example.com:8080/mcp")
        
        assert spec.type == "web"
        assert spec.connection_info["url"] == "https://api.example.com:8080/mcp"
        assert spec.connection_info["host"] == "api.example.com"
        assert spec.connection_info["port"] == 8080
        assert spec.connection_info["scheme"] == "https"

    def test_stdio_mcp_spec_parsing(self):
        """Test parsing of stdio MCP tool specifications."""
        spec = MCPToolSpec("mcp:uvx --from . google-search-mcp")
        
        assert spec.type == "stdio"
        assert spec.connection_info["command"] == "uvx"
        assert spec.connection_info["args"] == ["--from", ".", "google-search-mcp"]
        assert spec.connection_info["full_command"] == "uvx --from . google-search-mcp"

    def test_stdio_mcp_spec_with_path(self):
        """Test parsing of stdio MCP tool with absolute path."""
        spec = MCPToolSpec("mcp:/usr/local/bin/my-tool --arg1 --arg2")
        
        assert spec.type == "stdio"
        assert spec.connection_info["command"] == "/usr/local/bin/my-tool"
        assert spec.connection_info["args"] == ["--arg1", "--arg2"]

    def test_invalid_mcp_spec(self):
        """Test that invalid MCP specs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid MCP spec"):
            MCPToolSpec("not-mcp:something")

    def test_invalid_port_handling(self):
        """Test that invalid ports are handled gracefully."""
        spec = MCPToolSpec("mcp:http://localhost:99999")
        
        assert spec.type == "web"
        assert spec.connection_info["port"] is None  # Should be None due to invalid port

    def test_config_values(self):
        """Test configuration value handling."""
        config = {"timeout": 10, "cwd": "/tmp"}
        spec = MCPToolSpec("mcp:python --version", config)
        
        assert spec.get_config_value("timeout") == 10
        assert spec.get_config_value("cwd") == "/tmp"
        assert spec.get_config_value("nonexistent", "default") == "default"

    def test_repr(self):
        """Test string representation."""
        spec = MCPToolSpec("mcp:python --version")
        repr_str = repr(spec)
        
        assert "MCPToolSpec" in repr_str
        assert "stdio" in repr_str
        assert "mcp:python --version" in repr_str


class TestParseMCPTools:
    """Test the parse_mcp_tools function."""

    def test_parse_mixed_tools(self):
        """Test parsing mixed MCP and regular tools."""
        tool_specs = [
            "regular_tool",
            "mcp:http://localhost:3000",
            "another_regular_tool",
            "mcp:uvx --from . search-mcp",
            "mcp:python -m my_server",
        ]
        
        mcp_tools = parse_mcp_tools(tool_specs)
        
        assert len(mcp_tools) == 3
        assert all(isinstance(tool, MCPToolSpec) for tool in mcp_tools)
        assert mcp_tools[0].spec_string == "mcp:http://localhost:3000"
        assert mcp_tools[1].spec_string == "mcp:uvx --from . search-mcp"
        assert mcp_tools[2].spec_string == "mcp:python -m my_server"

    def test_parse_no_mcp_tools(self):
        """Test parsing when no MCP tools are present."""
        tool_specs = ["regular_tool1", "regular_tool2"]
        
        mcp_tools = parse_mcp_tools(tool_specs)
        
        assert len(mcp_tools) == 0

    def test_parse_with_config(self):
        """Test parsing with agent configuration."""
        tool_specs = ["mcp:python --version"]
        agent_config = {
            "mcp_config": {
                "timeout": 15,
                "cwd": "/custom/path"
            }
        }
        
        mcp_tools = parse_mcp_tools(tool_specs, agent_config)
        
        assert len(mcp_tools) == 1
        assert mcp_tools[0].get_config_value("timeout") == 15
        assert mcp_tools[0].get_config_value("cwd") == "/custom/path"

    def test_parse_invalid_spec_continues(self):
        """Test that invalid specs are skipped but parsing continues."""
        tool_specs = [
            "mcp:valid_command",
            "invalid:spec",  # This should be skipped
            "mcp:another_valid_command"
        ]
        
        with patch("mchat_core.tool_utils.logger") as mock_logger:
            mcp_tools = parse_mcp_tools(tool_specs)
        
        assert len(mcp_tools) == 2
        assert mcp_tools[0].spec_string == "mcp:valid_command"
        assert mcp_tools[1].spec_string == "mcp:another_valid_command"


class TestMCPToolManager:
    """Test the MCPToolManager class."""

    def test_init(self):
        """Test MCPToolManager initialization."""
        manager = MCPToolManager()
        
        assert manager.active_connections == {}
        assert manager.validated_tools == {}

    @pytest.mark.asyncio
    async def test_validate_tools_success(self):
        """Test successful validation of multiple tools."""
        manager = MCPToolManager()
        specs = [
            MCPToolSpec("mcp:python --version"),
            MCPToolSpec("mcp:echo test"),
        ]
        
        # Mock the validate_mcp_tool function directly
        async def mock_validate(spec):
            if "python" in spec.spec_string:
                return True
            return False
        
        with patch("mchat_core.tool_utils.validate_mcp_tool", side_effect=mock_validate):
            results = await manager.validate_tools(specs)
        
        assert len(results) == 2
        assert results["mcp:python --version"] is True
        assert results["mcp:echo test"] is False
        assert manager.validated_tools["mcp:python --version"] is True
        assert manager.validated_tools["mcp:echo test"] is False

    @pytest.mark.asyncio
    async def test_validate_tools_with_exceptions(self):
        """Test validation handling exceptions."""
        manager = MCPToolManager()
        specs = [MCPToolSpec("mcp:python --version")]
        
        async def mock_validate_error(spec):
            raise Exception("Test error")
        
        with patch("mchat_core.tool_utils.validate_mcp_tool", side_effect=mock_validate_error), \
             patch("mchat_core.tool_utils.logger") as mock_logger:
            results = await manager.validate_tools(specs)
        
        assert results["mcp:python --version"] is False
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_mcp_tools_no_autogen_ext(self):
        """Test loading MCP tools when autogen_ext is not available."""
        manager = MCPToolManager()
        manager.validated_tools = {"mcp:python --version": True}
        specs = [MCPToolSpec("mcp:python --version")]
        
        # Mock the import to fail for the entire module
        original_import = __builtins__['__import__']
        def mock_import(name, *args, **kwargs):
            if 'autogen_ext.tools.mcp' in name:
                raise ImportError("No module named 'autogen_ext.tools.mcp'")
            return original_import(name, *args, **kwargs)
        
        with patch("builtins.__import__", side_effect=mock_import), \
             patch("mchat_core.tool_utils.logger") as mock_logger:
            tools = await manager.load_mcp_tools_for_conversation(specs)
        
        assert tools == {}
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_stdio_tools_with_real_imports(self):
        """Test that stdio tools can be loaded with actual autogen_ext imports.
        
        This test would have caught the WebServerParams vs StreamableHttpServerParams issue
        by actually trying to import the classes rather than mocking the entire module.
        """
        manager = MCPToolManager()
        manager.validated_tools = {"mcp:python --version": True}
        specs = [MCPToolSpec("mcp:python --version")]
        
        # Mock only the mcp_server_tools call, not the imports
        # This forces the real import to happen and would fail if classes don't exist
        with patch("autogen_ext.tools.mcp.mcp_server_tools", return_value=[]) as mock_mcp_tools:
            tools = await manager.load_mcp_tools_for_conversation(specs)
        
        # Should succeed - imports should work with correct class names
        assert tools == {}
        mock_mcp_tools.assert_called_once()
        
        # Verify the call was made with StdioServerParams
        call_args = mock_mcp_tools.call_args[0][0]
        assert hasattr(call_args, 'command')
        assert call_args.command == "python"

    @pytest.mark.asyncio
    async def test_load_mcp_tools_skips_unvalidated(self):
        """Test that unvalidated tools are skipped."""
        manager = MCPToolManager()
        manager.validated_tools = {"mcp:python --version": False}
        specs = [MCPToolSpec("mcp:python --version")]
        
        with patch("mchat_core.tool_utils.logger") as mock_logger:
            tools = await manager.load_mcp_tools_for_conversation(specs)
        
        assert tools == {}
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_connections(self):
        """Test cleanup of MCP connections."""
        manager = MCPToolManager()
        
        # Mock connections with and without cleanup method
        mock_conn1 = Mock()
        mock_conn1.cleanup = Mock()
        mock_conn2 = Mock()  # No cleanup method
        
        manager.active_connections = {
            "spec1": mock_conn1,
            "spec2": mock_conn2,
        }
        
        await manager.cleanup_connections()
        
        mock_conn1.cleanup.assert_called_once()
        assert manager.active_connections == {}

    @pytest.mark.asyncio
    async def test_cleanup_connections_with_exception(self):
        """Test cleanup handling exceptions."""
        manager = MCPToolManager()
        
        mock_conn = Mock()
        mock_conn.cleanup.side_effect = Exception("Cleanup failed")
        manager.active_connections = {"spec1": mock_conn}
        
        with patch("mchat_core.tool_utils.logger") as mock_logger:
            await manager.cleanup_connections()
        
        mock_logger.warning.assert_called_once()
        assert manager.active_connections == {}

    @pytest.mark.asyncio 
    async def test_import_class_compatibility_check(self):
        """Test that verifies correct autogen_ext class names are used.
        
        This test would have caught the WebServerParams vs StreamableHttpServerParams issue
        by checking that the code uses the correct class names available in the installed version.
        """
        # Verify that StreamableHttpServerParams exists (new correct name)
        try:
            from autogen_ext.tools.mcp import StreamableHttpServerParams
            assert StreamableHttpServerParams is not None
        except ImportError:
            pytest.skip("autogen_ext.tools.mcp not available")
        
        # Verify that WebServerParams does NOT exist (old incorrect name)
        with pytest.raises(ImportError):
            from autogen_ext.tools.mcp import WebServerParams
        
        # Verify all classes our code actually imports exist
        from autogen_ext.tools.mcp import (
            StdioServerParams,
            StreamableHttpServerParams, 
            mcp_server_tools,
        )
        
        assert StdioServerParams is not None
        assert StreamableHttpServerParams is not None
        assert mcp_server_tools is not None

    @pytest.mark.asyncio
    async def test_load_web_tools_with_real_imports(self):
        """Test that web tools can be loaded with actual autogen_ext imports.
        
        This test ensures that web-based MCP tools use the correct class names
        (StreamableHttpServerParams, not WebServerParams).
        """
        manager = MCPToolManager()
        manager.validated_tools = {"mcp:http://localhost:3000": True}
        specs = [MCPToolSpec("mcp:http://localhost:3000")]
        
        # Mock only the mcp_server_tools call, not the imports
        # This forces the real import to happen and would fail if classes don't exist
        with patch("autogen_ext.tools.mcp.mcp_server_tools", return_value=[]) as mock_mcp_tools:
            tools = await manager.load_mcp_tools_for_conversation(specs)
        
        # Should not fail due to import errors - the import should succeed
        # and mcp_server_tools should be called with StreamableHttpServerParams
        assert tools == {}  # Empty because we mocked mcp_server_tools to return []
        mock_mcp_tools.assert_called_once()
        
        # Verify the call was made with StreamableHttpServerParams (not WebServerParams)
        call_args = mock_mcp_tools.call_args[0][0]
        assert hasattr(call_args, 'url')
        assert call_args.url == "http://localhost:3000"


class TestHighLevelFunctions:
    """Test high-level MCP integration functions."""

    @pytest.mark.asyncio
    async def test_validate_and_load_mcp_tools_success(self):
        """Test successful validation and loading setup."""
        agents_config = {
            "agent1": {
                "tools": ["regular_tool", "mcp:python --version"],
                "mcp_config": {"timeout": 10}
            },
            "agent2": {
                "tools": ["mcp:echo test"]
            }
        }
        
        # Mock validation results
        async def mock_validate(specs):
            return {
                "mcp:python --version": True,
                "mcp:echo test": False
            }
        
        with patch.object(MCPToolManager, "validate_tools", side_effect=mock_validate):
            manager, results = await validate_and_load_mcp_tools(agents_config)
        
        assert isinstance(manager, MCPToolManager)
        assert results["mcp:python --version"] is True
        assert results["mcp:echo test"] is False

    @pytest.mark.asyncio
    async def test_validate_and_load_mcp_tools_no_mcp_tools(self):
        """Test when no MCP tools are found."""
        agents_config = {
            "agent1": {"tools": ["regular_tool1", "regular_tool2"]}
        }
        
        with patch("mchat_core.tool_utils.logger") as mock_logger:
            manager, results = await validate_and_load_mcp_tools(agents_config)
        
        assert isinstance(manager, MCPToolManager)
        assert results == {}
        mock_logger.debug.assert_called_with("No MCP tools found in agent configurations")

    @pytest.mark.asyncio
    async def test_load_agent_mcp_tools_success(self):
        """Test loading MCP tools for a specific agent."""
        agent_config = {
            "tools": ["regular_tool", "mcp:python --version"],
            "mcp_config": {"timeout": 15}
        }
        
        mock_manager = Mock()
        # Make the mock method async
        async def async_mock(specs):
            return {"test_tool": Mock()}
        mock_manager.load_mcp_tools_for_conversation = async_mock
        
        tools = await load_agent_mcp_tools(agent_config, mock_manager)
        
        assert len(tools) == 1
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_load_agent_mcp_tools_no_mcp_tools(self):
        """Test loading when agent has no MCP tools."""
        agent_config = {"tools": ["regular_tool1", "regular_tool2"]}
        mock_manager = Mock()
        
        tools = await load_agent_mcp_tools(agent_config, mock_manager)
        
        assert tools == {}
        mock_manager.load_mcp_tools_for_conversation.assert_not_called()


# Integration test fixtures
@pytest.fixture
def sample_agents_config():
    """Sample agent configuration for integration tests."""
    return {
        "agent_with_mcp": {
            "description": "Agent with MCP tools",
            "tools": [
                "regular_tool",
                "mcp:python --version",
                "mcp:http://localhost:3000"
            ],
            "mcp_config": {
                "timeout": 10,
                "cwd": "/tmp"
            }
        },
        "agent_without_mcp": {
            "description": "Agent without MCP tools",
            "tools": ["regular_tool1", "regular_tool2"]
        }
    }


class TestIntegration:
    """Integration tests for MCP functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_mcp_workflow(self, sample_agents_config):
        """Test complete MCP workflow from parsing to loading."""
        # Mock validation results
        validation_results = {
            "mcp:python --version": True,
            "mcp:http://localhost:3000": False
        }
        
        async def mock_validate(specs):
            return {spec.spec_string: validation_results.get(spec.spec_string, False) for spec in specs}
        
        async def mock_load_tools(specs):
            return {"python_version_tool": Mock()}
        
        with patch.object(MCPToolManager, "validate_tools", side_effect=mock_validate), \
             patch.object(MCPToolManager, "load_mcp_tools_for_conversation", side_effect=mock_load_tools):
            
            # Step 1: Validate and setup
            manager, results = await validate_and_load_mcp_tools(sample_agents_config)
            
            # Step 2: Load tools for specific agent
            agent_config = sample_agents_config["agent_with_mcp"]
            tools = await load_agent_mcp_tools(agent_config, manager)
        
        # Verify results
        assert results["mcp:python --version"] is True
        assert results["mcp:http://localhost:3000"] is False
        assert len(tools) == 1
        assert "python_version_tool" in tools

    def test_mcp_tool_spec_config_inheritance(self):
        """Test that MCP config is properly inherited from agent config."""
        agent_config = {
            "mcp_config": {
                "timeout": 20,
                "cwd": "/custom/path",
                "env": {"DEBUG": "1"}
            }
        }
        
        mcp_tools = parse_mcp_tools(["mcp:python --version"], agent_config)
        
        assert len(mcp_tools) == 1
        spec = mcp_tools[0]
        assert spec.get_config_value("timeout") == 20
        assert spec.get_config_value("cwd") == "/custom/path"
        assert spec.get_config_value("env") == {"DEBUG": "1"}


class TestMCPValidationSimplified:
    """Simplified tests for MCP validation without complex mocking."""

    @pytest.mark.asyncio
    async def test_validate_stdio_tool_command_exists(self):
        """Test stdio validation when command exists."""
        from mchat_core.tool_utils import validate_mcp_tool
        
        spec = MCPToolSpec("mcp:python --version")
        
        # Mock shutil.which to return a path and avoid real MCP calls
        with patch("shutil.which", return_value="/usr/bin/python"), \
             patch("autogen_ext.tools.mcp.mcp_server_tools", return_value=[]):
            result = await validate_mcp_tool(spec)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_stdio_tool_command_not_exists(self):
        """Test stdio validation when command doesn't exist."""
        from mchat_core.tool_utils import validate_mcp_tool
        
        spec = MCPToolSpec("mcp:nonexistent_command")
        
        # Mock shutil.which to return None
        with patch("shutil.which", return_value=None):
            result = await validate_mcp_tool(spec)
        
        assert result is False

    @pytest.mark.asyncio 
    async def test_validate_stdio_tool_absolute_path_exists(self):
        """Test stdio validation with absolute path that exists."""
        from mchat_core.tool_utils import validate_mcp_tool
        
        spec = MCPToolSpec("mcp:/usr/bin/python --version")
        
        with patch("os.path.isabs", return_value=True), \
             patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch("autogen_ext.tools.mcp.mcp_server_tools", return_value=[]):
            result = await validate_mcp_tool(spec)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_stdio_tool_absolute_path_not_exists(self):
        """Test stdio validation with absolute path that doesn't exist."""
        from mchat_core.tool_utils import validate_mcp_tool
        
        spec = MCPToolSpec("mcp:/nonexistent/path/command")
        
        with patch("os.path.isabs", return_value=True), \
             patch("os.path.isfile", return_value=False):
            result = await validate_mcp_tool(spec)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_web_tool_unreachable_returns_false(self):
        """Web validation returns False when MCP server cannot be queried."""
        from mchat_core.tool_utils import validate_mcp_tool

        spec = MCPToolSpec("mcp:http://localhost:3000")

        with patch("autogen_ext.tools.mcp.mcp_server_tools", side_effect=Exception("connect error")), \
             patch("mchat_core.tool_utils.logger") as mock_logger:
            result = await validate_mcp_tool(spec)

        assert result is False
        # We log at debug level for validation failure details
        mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_validate_web_tool_success_returns_true(self):
        """Web validation returns True when MCP server responds to tool query."""
        from mchat_core.tool_utils import validate_mcp_tool

        spec = MCPToolSpec("mcp:http://localhost:3000")

        with patch("autogen_ext.tools.mcp.mcp_server_tools", return_value=[]):
            result = await validate_mcp_tool(spec)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_tool_exception_handling(self):
        """Test that exceptions during validation are handled gracefully."""
        from mchat_core.tool_utils import validate_mcp_tool
        
        spec = MCPToolSpec("mcp:python --version")
        
        with patch("shutil.which", side_effect=Exception("Unexpected error")), \
             patch("mchat_core.tool_utils.logger") as mock_logger:
            result = await validate_mcp_tool(spec)
        
        assert result is False
        mock_logger.warning.assert_called_once()