import pytest
from unittest.mock import Mock
from mchat_core.agent_manager import AgentManager


@pytest.fixture
def agent_manager():
    """Create an AgentManager instance for testing."""
    agents_yaml = """
searcher:
  type: agent
  description: A search agent
  prompt: You are a helpful search agent

analyzer:
  type: agent
  description: An analysis agent
  prompt: You analyze data
  tools:
    - existing_tool
    """
    
    mock_callback = Mock()
    am = AgentManager(message_callback=mock_callback, agent_paths=[agents_yaml])
    
    # Add a pre-existing tool
    am.tools["existing_tool"] = Mock()
    
    return am


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""
    return {
        "search_tool": Mock(),
        "analysis_tool": Mock(),
        "data_tool": Mock()
    }


class TestGlobalToolManagement:
    """Test global tool registry management."""
    
    def test_add_tool(self, agent_manager, sample_tools):
        """Test adding a tool to the global registry."""
        tool_name = "test_tool"
        tool_func = sample_tools["search_tool"]
        
        agent_manager.add_tool(tool_name, tool_func)
        
        assert tool_name in agent_manager.tools
        assert agent_manager.tools[tool_name] == tool_func
    
    def test_add_tool_overwrites_existing(self, agent_manager, sample_tools):
        """Test that adding an existing tool overwrites it."""
        tool_name = "test_tool"
        old_tool = Mock()
        new_tool = sample_tools["search_tool"]
        
        # Add initial tool
        agent_manager.add_tool(tool_name, old_tool)
        assert agent_manager.tools[tool_name] == old_tool
        
        # Overwrite with new tool
        agent_manager.add_tool(tool_name, new_tool)
        assert agent_manager.tools[tool_name] == new_tool
    
    def test_remove_tool_existing(self, agent_manager, sample_tools):
        """Test removing an existing tool."""
        tool_name = "test_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        result = agent_manager.remove_tool(tool_name)
        
        assert result is True
        assert tool_name not in agent_manager.tools
    
    def test_remove_tool_nonexistent(self, agent_manager):
        """Test removing a non-existent tool."""
        result = agent_manager.remove_tool("nonexistent_tool")
        
        assert result is False
    
    def test_list_tools_empty(self):
        """Test listing tools when registry is empty."""
        agents_yaml = """
test_agent:
  type: agent
  description: Test agent
  prompt: Test prompt
"""
        am = AgentManager(message_callback=Mock(), agent_paths=[agents_yaml])
        
        tools = am.list_tools()
        assert tools == []
    
    def test_list_tools_with_tools(self, agent_manager, sample_tools):
        """Test listing tools when registry has tools."""
        for name, tool in sample_tools.items():
            agent_manager.add_tool(name, tool)
        
        tools = agent_manager.list_tools()  # Fixed: use agent_manager instead of am
        expected_tools = ["existing_tool"] + list(sample_tools.keys())
        
        assert len(tools) == len(expected_tools)
        assert all(tool in tools for tool in expected_tools)


class TestAgentToolManagement:
    """Test per-agent tool management."""
    
    def test_add_agent_tool_success(self, agent_manager, sample_tools):
        """Test successfully adding a tool to an agent."""
        tool_name = "new_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        agent_manager.add_agent_tool("searcher", tool_name)
        
        agent_tools = agent_manager.list_agent_tools("searcher")
        assert tool_name in agent_tools
    
    def test_add_agent_tool_creates_tools_list(self, agent_manager, sample_tools):
        """Test that adding a tool creates the tools list if it doesn't exist."""
        tool_name = "new_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        # Ensure searcher has no tools initially
        assert "tools" not in agent_manager._agents["searcher"]
        
        agent_manager.add_agent_tool("searcher", tool_name)
        
        assert "tools" in agent_manager._agents["searcher"]
        assert tool_name in agent_manager._agents["searcher"]["tools"]
    
    def test_add_agent_tool_duplicate(self, agent_manager, sample_tools):
        """Test adding the same tool twice to an agent."""
        tool_name = "duplicate_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        agent_manager.add_agent_tool("searcher", tool_name)
        agent_manager.add_agent_tool("searcher", tool_name)  # Add again
        
        agent_tools = agent_manager.list_agent_tools("searcher")
        assert agent_tools.count(tool_name) == 1
    
    def test_add_agent_tool_nonexistent_agent(self, agent_manager, sample_tools):
        """Test adding a tool to a non-existent agent."""
        tool_name = "test_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        with pytest.raises(ValueError, match="Agent 'nonexistent' does not exist"):
            agent_manager.add_agent_tool("nonexistent", tool_name)
    
    def test_add_agent_tool_nonexistent_tool(self, agent_manager):
        """Test adding a non-existent tool to an agent."""
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found in global registry"):
            agent_manager.add_agent_tool("searcher", "nonexistent_tool")
    
    def test_remove_agent_tool_success(self, agent_manager):
        """Test successfully removing a tool from an agent."""
        # The analyzer already has existing_tool
        result = agent_manager.remove_agent_tool("analyzer", "existing_tool")
        
        assert result is True
        agent_tools = agent_manager.list_agent_tools("analyzer")
        assert "existing_tool" not in agent_tools
    
    def test_remove_agent_tool_not_assigned(self, agent_manager, sample_tools):
        """Test removing a tool that's not assigned to the agent."""
        tool_name = "unassigned_tool"
        agent_manager.add_tool(tool_name, sample_tools["search_tool"])
        
        result = agent_manager.remove_agent_tool("searcher", tool_name)
        
        assert result is False
    
    def test_remove_agent_tool_nonexistent_agent(self, agent_manager):
        """Test removing a tool from a non-existent agent."""
        result = agent_manager.remove_agent_tool("nonexistent", "any_tool")
        
        assert result is False
    
    def test_list_agent_tools_with_tools(self, agent_manager):
        """Test listing tools for an agent that has tools."""
        tools = agent_manager.list_agent_tools("analyzer")
        
        assert "existing_tool" in tools
    
    def test_list_agent_tools_no_tools(self, agent_manager):
        """Test listing tools for an agent with no tools."""
        tools = agent_manager.list_agent_tools("searcher")
        
        assert tools == []
    
    def test_list_agent_tools_nonexistent_agent(self, agent_manager):
        """Test listing tools for a non-existent agent."""
        with pytest.raises(ValueError, match="Agent 'nonexistent' does not exist"):
            agent_manager.list_agent_tools("nonexistent")


class TestToolInfo:
    """Test tool information retrieval."""
    
    def test_get_tool_info_success(self, agent_manager, sample_tools):
        """Test getting information about an existing tool."""
        tool_name = "info_tool"
        tool_func = sample_tools["search_tool"]
        agent_manager.add_tool(tool_name, tool_func)
        agent_manager.add_agent_tool("searcher", tool_name)
        agent_manager.add_agent_tool("analyzer", tool_name)
        
        info = agent_manager.get_tool_info(tool_name)
        
        assert info["name"] == tool_name
        assert info["function"] == tool_func
        assert "searcher" in info["used_by_agents"]
        assert "analyzer" in info["used_by_agents"]
    
    def test_get_tool_info_no_agents(self, agent_manager, sample_tools):
        """Test getting info for a tool not used by any agents."""
        tool_name = "unused_tool"
        tool_func = sample_tools["search_tool"]
        agent_manager.add_tool(tool_name, tool_func)
        
        info = agent_manager.get_tool_info(tool_name)
        
        assert info["name"] == tool_name
        assert info["function"] == tool_func
        assert info["used_by_agents"] == []
    
    def test_get_tool_info_nonexistent(self, agent_manager):
        """Test getting info for a non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found"):
            agent_manager.get_tool_info("nonexistent_tool")


class TestIntegrationScenarios:
    """Test complex scenarios involving multiple operations."""
    
    def test_tool_lifecycle(self, agent_manager, sample_tools):
        """Test complete tool lifecycle: add, assign, reassign, remove."""
        tool_name = "lifecycle_tool"
        tool_func = sample_tools["search_tool"]
        
        # Add tool to registry
        agent_manager.add_tool(tool_name, tool_func)
        assert tool_name in agent_manager.list_tools()
        
        # Assign to agents
        agent_manager.add_agent_tool("searcher", tool_name)
        agent_manager.add_agent_tool("analyzer", tool_name)
        
        searcher_tools = agent_manager.list_agent_tools("searcher")
        analyzer_tools = agent_manager.list_agent_tools("analyzer")
        assert tool_name in searcher_tools
        assert tool_name in analyzer_tools
        
        # Check tool info
        info = agent_manager.get_tool_info(tool_name)
        assert len(info["used_by_agents"]) == 2
        
        # Remove from one agent
        agent_manager.remove_agent_tool("searcher", tool_name)
        searcher_tools = agent_manager.list_agent_tools("searcher")
        analyzer_tools = agent_manager.list_agent_tools("analyzer")
        assert tool_name not in searcher_tools
        assert tool_name in analyzer_tools
        
        # Remove from global registry
        agent_manager.remove_tool(tool_name)
        assert tool_name not in agent_manager.list_tools()
        
        # Tool should still be in agent's list (allows for cleanup scenarios)
        analyzer_tools = agent_manager.list_agent_tools("analyzer")
        assert tool_name in analyzer_tools
    
    def test_multiple_tools_multiple_agents(self, agent_manager, sample_tools):
        """Test managing multiple tools across multiple agents."""
        # Add all tools
        for name, func in sample_tools.items():
            agent_manager.add_tool(name, func)
        
        # Assign different combinations to agents
        agent_manager.add_agent_tool("searcher", "search_tool")
        agent_manager.add_agent_tool("searcher", "data_tool")
        agent_manager.add_agent_tool("analyzer", "analysis_tool")
        agent_manager.add_agent_tool("analyzer", "data_tool")
        
        # Verify assignments
        searcher_tools = agent_manager.list_agent_tools("searcher")
        analyzer_tools = agent_manager.list_agent_tools("analyzer")
        
        assert "search_tool" in searcher_tools
        assert "data_tool" in searcher_tools
        assert "analysis_tool" not in searcher_tools
        
        assert "analysis_tool" in analyzer_tools
        assert "data_tool" in analyzer_tools
        assert "search_tool" not in analyzer_tools
        
        # Check shared tool info
        data_tool_info = agent_manager.get_tool_info("data_tool")
        assert len(data_tool_info["used_by_agents"]) == 2
        assert "searcher" in data_tool_info["used_by_agents"]
        assert "analyzer" in data_tool_info["used_by_agents"]