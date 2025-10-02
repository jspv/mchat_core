import asyncio
import json
import os
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from functools import reduce
from typing import Any

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import (
    TaskResult,
    TerminatedException,
    TerminationCondition,
)
from autogen_agentchat.conditions import (
    ExternalTermination,
    MaxMessageTermination,
    StopMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_agentchat.teams import (
    MagenticOneGroupChat,
    RoundRobinGroupChat,
    SelectorGroupChat,
)
from autogen_core import CancellationToken
from autogen_core.model_context import (
    BufferedChatCompletionContext,
    HeadAndTailChatCompletionContext,
    TokenLimitedChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai._openai_client import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)

from .logging_utils import get_logger, trace  # noqa: F401
from .model_manager import ModelManager
from .terminator import SmartReflectorTermination
from .tool_utils import (
    create_mcp_validation_task,
    load_agent_mcp_tools,
    load_tools,
    register_mcp_placeholders,
    replace_mcp_placeholders_with_real_tools,
    validate_and_load_mcp_tools,
)

logger = get_logger(__name__)


# default_mini_model is configured (or falls back) in ModelManager


label_prompt = (
    "Here is a conversation between a Human and AI. Give me no more than 10 words which"
    " will be used to remind the user of the conversation.  Your reply should be no"
    " more than 10 words and at most 66 total characters. Here is the conversation"
    " {conversation}"
)

summary_prompt = (
    "Here is a conversation between a Human and AI. Provide a detailed summary of the "
    "Conversation: "
    "{conversation}"
)


class IsCompleteTermination(TerminationCondition):
    mm = ModelManager()
    memory_model = mm.open_model(mm.default_mini_model)
    prompt = """ Take a look at the conversation so far. If the conversation has
    reached a natural conclusion and it is the clear that the agent had completed
    the task and has provided a clear response, return True. Otherwise, return False.
    ONLY return the word 'True' or 'False'.  Here is the conversation so far:
    {conversation}"""

    def __init__(self):
        self._is_terminated = False

    @property
    def terminated(self) -> bool:
        return self._is_terminated

    async def __call__(
        self, messages: Sequence[AgentEvent | ChatMessage]
    ) -> StopMessage | None:
        if self._is_terminated:
            raise TerminatedException("Termination condition has already been reached")
        system_message = IsCompleteTermination.prompt.format(conversation=messages)
        out = await IsCompleteTermination.memory_model.create(
            [SystemMessage(content=system_message)]
        )
        logger.debug(f"IsCompleteTermination response: {out.content.strip()}")
        if out.content.strip().lower() == "true":
            self._is_terminated = True
            return StopMessage(
                content="Agent is complete termination condition met",
                source="IsCompleteTermination",
            )
        return None

    async def reset(self):
        self._is_terminated = False


class AgentManager:
    # createa a model_manager for use with llmtools and AgentManager
    ag_mm = ModelManager()
    ag_memory_model = ag_mm.open_model(ag_mm.default_mini_model)

    def __init__(
        self,
        agent_callback: Callable | None = None,
        agents: dict | None = None,
        agent_paths: list[str] | None = None,
        stream_tokens: bool = None,
        message_callback: Callable | None = None,
        tools_directory: str | None = None,
        load_default_tools: bool = False,
    ):
        """
        Initialize the AgentManager.

        Exactly one of `agents` or `agent_paths` must be provided.

        Args:
            agent_callback:
                Async callable invoked with each agent message/event.
                Signature: async def cb(message: str, **kwargs) -> None
            agents:
                In-memory mapping of agent definitions. Keys are agent names; values are
                dicts describing agents or teams. For agents, common fields include:
                type: "agent", description: str, prompt: str, model: str,
                temperature: float, tools: list[str], extra_kwargs: dict.
                For teams: type: "team", team_type: "round_robin" | "selector"
            agent_paths:
                List of file-system paths (files or directories) or raw YAML/JSON
                strings containing agent definitions. Directories will be scanned for
                ".yaml/.json files.
            stream_tokens:
                Enable token/message streaming for compatible models.
                If True, `message_callback`
                must be provided. If None (default) and `message_callback` is provided,
                streaming defaults to True.
            message_callback:
                Async callable that receives streamed output/messages.
                Expected usage:
                  await message_callback(
                      message: str,
                      agent: str | None = None,
                      complete: bool | None = None,
                      flush: bool | None = None,
                  )
                - message: the text chunk or final message
                - agent: source agent name (when available)
                - complete: True when a message is complete;
                  False for incremental chunks
                - flush: hint for UIs to flush output
            tools_directory:
                Directory containing custom tool specs to load. If None, only default
                tools may be loaded (see `load_default_tools`).
            load_default_tools:
                When True, load built-in default tools.
                If `tools_directory` is also provided,
                defaults and custom tools are merged.

        Raises:
            ValueError: If both or neither of `agents` and `agent_paths` are provided,
                        or if `stream_tokens` is True without a `message_callback`.
        """
        # Ensure that exactly one of agents or agent_paths is provided
        if (agents is None) == (
            agent_paths is None
        ):  # Both are None or both are provided
            raise ValueError(
                "You must specify exactly one of 'agents' or 'agent_paths'"
            )

        # if no callback is provided, disable streaming; nowhere to send tokens
        if message_callback is None and stream_tokens is True:
            raise ValueError("stream_tokens cannot be True if message_callback is None")
        # default to streaming if a callback is provided and no preference is given
        if stream_tokens is None and message_callback:
            stream_tokens = True

        # create noop callback if none is provided, keeps rest of the code cleaner
        async def noop_callback(*args, **kwargs):
            pass

        agent_callback = noop_callback if agent_callback is None else agent_callback
        message_callback = (
            noop_callback if message_callback is None else message_callback
        )

        self.mm = ModelManager()
        # Default callback and streaming setting for sessions
        self._default_agent_callback = agent_callback
        self._default_message_callback = message_callback
        self._default_stream_tokens = stream_tokens

        # Load agents
        self._agents = agents if agents is not None else {}
        self._agents = self._load_agents(agent_paths) if agent_paths else self._agents
        self._chooseable_agents = [
            agent_name
            for agent_name, val in self._agents.items()
            if val.get("chooseable", True)
        ]

        # Initialize available tools
        if tools_directory is None:
            # Load only default tools or none at all
            self.tools = load_tools(None) if load_default_tools else {}
        else:
            # Load custom tools; optionally merge default tools
            custom_tools = load_tools(tools_directory)
            if load_default_tools:
                default_tools = load_tools(None)
                self.tools = {**default_tools, **custom_tools}
            else:
                self.tools = custom_tools

        # Initialize MCP tool manager and validate MCP tools
        self.mcp_manager = None
        self._mcp_validation_task = None
        self._mcp_placeholder_tools = {}  # spec_string -> placeholder tool name
        # Register MCP placeholders via tool_utils
        self._mcp_placeholder_tools = register_mcp_placeholders(
            self._agents, self.tools
        )
        self._start_mcp_validation()

    def new_agent(
        self, agent_name, model_name, prompt, tools: list | None = None
    ) -> None:
        """Create a new agent with the given name, model, tools, and prompt"""
        pass

    @property
    def agents(self) -> dict:
        """Return the agent structure"""
        return self._agents

    @property
    def chooseable_agents(self) -> list:
        """Return list of agents the UI can choose from"""
        return self._chooseable_agents

    async def new_conversation(
        self,
        agent: str,
        model_id: str | None = None,
        temperature: float | None = None,
        stream_tokens: bool | None = None,
        message_callback: Callable | None = None,
        agent_callback: Callable | None = None,
    ) -> "AgentSession":
        """Create and return a new per-agent conversation session.

        The returned AgentSession encapsulates state and methods such as ask(),
        memory management, streaming control, and cancel/terminate.

        Args:
            stream_tokens: Override manager default if provided
            message_callback: Override manager default if provided (streaming callback)
            agent_callback: Override manager default if provided
        """
        # Use session-specific values or fall back to manager defaults
        effective_stream_tokens = (
            stream_tokens if stream_tokens is not None else self._default_stream_tokens
        )
        effective_message_callback = (
            message_callback
            if message_callback is not None
            else self._default_message_callback
        )
        effective_agent_callback = (
            agent_callback
            if agent_callback is not None
            else self._default_agent_callback
        )

        session = await AgentSession.create(
            manager=self,
            agent_name=agent,
            model_id=model_id,
            temperature=temperature,
            stream_tokens=effective_stream_tokens,
            message_callback=effective_message_callback,
            agent_callback=effective_agent_callback,
        )
        return session

    def _load_agents(self, paths: list[str]) -> dict:
        """Read the agent definition files and load the agents, or parse agent
        definitions from strings"""
        agent_files = []
        agent_strings = []
        for path in paths:
            if isinstance(path, str):
                logger.debug(f"Loading agent from path: {path}")
                # Check if it's a directory
                if os.path.isdir(path):
                    logger.debug(f"Loading agents from directory: {path}")
                    for filename in os.listdir(path):
                        if filename.endswith(".json") or filename.endswith(".yaml"):
                            agent_files.append(os.path.join(path, filename))
                # Check if it's a file
                elif (
                    path.endswith(".json") or path.endswith(".yaml")
                ) and os.path.exists(path):
                    logger.debug(f"Loading agent from file: {path}")
                    agent_files.append(path)
                else:
                    logger.debug(f"Assuming {path} is a YAML or JSON string")
                    # Try to detect if it's a JSON string, otherwise assume YAML
                    s = path.strip()
                    if (s.startswith("{") and s.endswith("}")) or (
                        s.startswith("[") and s.endswith("]")
                    ):
                        agent_strings.append(("json", s))
                    else:
                        agent_strings.append(("yaml", s))
        agents = {}
        # Load from files
        for file in agent_files:
            extension = os.path.splitext(file)[1]
            with open(file, encoding="utf8") as f:
                logger.debug(f"Loading agent file {file}")
                try:
                    if extension == ".json":
                        file_agents = json.load(f)
                    elif extension == ".yaml":
                        file_agents = yaml.safe_load(f)
                except FileNotFoundError:
                    logger.error(f"Error: file {file} not found.")
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML file {file}: {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON file {file}: {e}")
                except Exception as e:
                    logger.exception(
                        f"An unexpected error occurred processing {file}: {e}"
                    )
                if not isinstance(file_agents, dict):
                    error_msg = (
                        f"Agent file {file} must contain a mapping of agents, "
                        f"got {type(file_agents)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                agents.update(file_agents)
        # Load from strings
        for fmt, data in agent_strings:
            logger.debug(f"Loading agent from string ({fmt})")
            try:
                if fmt == "json":
                    file_agents = json.loads(data)
                elif fmt == "yaml":
                    file_agents = yaml.safe_load(data)
                if not isinstance(file_agents, dict):
                    error_msg = (
                        f"Agent string ({fmt}) must be a mapping of agents, "
                        f"got {type(file_agents)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                agents.update(file_agents)
            except (ValueError, AttributeError) as e:
                error_msg = f"Error parsing agent string ({fmt}): {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # Validate agents
        for agent_name, agent_data in agents.items():
            if not isinstance(agent_data, dict):
                raise ValueError(
                    f"Agent '{agent_name}' must be a dict, got {type(agent_data)}"
                )
            if "type" not in agent_data:
                raise ValueError(f"Agent '{agent_name}' missing required 'type' field")
            # 'prompt' is not required for team agents
            if agent_data.get("type") == "agent" and "prompt" not in agent_data:
                raise ValueError(
                    f"Agent '{agent_name}' missing required 'prompt' field"
                )
            if "description" not in agent_data:
                raise ValueError(
                    f"Agent '{agent_name}' missing required 'description' field"
                )
            if "type" in agent_data and agent_data["type"] == "team":
                if "agents" not in agent_data or not isinstance(
                    agent_data["agents"], list
                ):
                    raise ValueError(
                        f"Team agent '{agent_name}' must have an 'agents' list"
                    )
        return agents

    # Tool management methods
    def add_tool(self, tool_name: str, tool_function) -> None:
        """Add a tool to the global tool registry.

        Args:
            tool_name: Unique identifier for the tool
            tool_function: The tool function/callable to register
        """
        if tool_name in self.tools:
            logger.warning(f"Tool '{tool_name}' already exists, overwriting")
        self.tools[tool_name] = tool_function
        logger.debug(f"Added tool '{tool_name}' to global registry")

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the global tool registry.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if it didn't exist
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.debug(f"Removed tool '{tool_name}' from global registry")
            return True
        return False

    def list_tools(self) -> list[str]:
        """Get list of all available tools.

        Returns:
            List of tool names in the global registry
        """
        return list(self.tools.keys())

    def add_agent_tool(self, agent_name: str, tool_name: str) -> None:
        """Add a tool to a specific agent's tool list.

        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool to add (must exist in global registry)

        Raises:
            ValueError: If agent doesn't exist or tool not in global registry
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' does not exist")

        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in global registry")

        # Ensure the agent has a tools list
        if "tools" not in self._agents[agent_name]:
            self._agents[agent_name]["tools"] = []

        # Add tool if not already present
        if tool_name not in self._agents[agent_name]["tools"]:
            self._agents[agent_name]["tools"].append(tool_name)
            logger.debug(f"Added tool '{tool_name}' to agent '{agent_name}'")
        else:
            logger.warning(
                f"Tool '{tool_name}' already assigned to agent '{agent_name}'"
            )

    def remove_agent_tool(self, agent_name: str, tool_name: str) -> bool:
        """Remove a tool from a specific agent's tool list.

        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if agent doesn't exist or tool wasn't
            assigned
        """
        if agent_name not in self._agents:
            return False

        agent_tools = self._agents[agent_name].get("tools", [])
        if tool_name in agent_tools:
            agent_tools.remove(tool_name)
            logger.debug(f"Removed tool '{tool_name}' from agent '{agent_name}'")
            return True

        return False

    def list_agent_tools(self, agent_name: str) -> list[str]:
        """Get list of tools assigned to a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of tool names assigned to the agent

        Raises:
            ValueError: If agent doesn't exist
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' does not exist")

        return self._agents[agent_name].get("tools", [])

    def get_tool_info(self, tool_name: str) -> dict:
        """Get information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool information including which agents use it

        Raises:
            ValueError: If tool doesn't exist
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Find which agents use this tool
        agents_using_tool = []
        for agent_name, agent_data in self._agents.items():
            agent_tools = agent_data.get("tools", [])
            if tool_name in agent_tools:
                agents_using_tool.append(agent_name)

        return {
            "name": tool_name,
            "function": self.tools[tool_name],
            "used_by_agents": agents_using_tool,
        }

    # removed: deprecated _register_mcp_placeholder_tools
    # (use tool_utils.register_mcp_placeholders)

    def _start_mcp_validation(self):
        """Start MCP tool validation in the background (delegates to tool_utils)."""
        self._mcp_validation_task = create_mcp_validation_task(self._agents)

    # removed: _validate_mcp_tools (handled by tool_utils.run_mcp_validation)

    async def get_agent_mcp_tools(self, agent_name: str) -> dict[str, any]:
        """Get MCP tools for a specific agent.

        This is called when starting a conversation to load MCP tools.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary of MCP tool name -> tool function mappings
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' does not exist")

        # Ensure MCP manager is initialized
        if self.mcp_manager is None:
            # If validation didn't run during init, do it now
            self.mcp_manager, _ = await validate_and_load_mcp_tools(self._agents)

        # Wait for validation to complete if it's still running
        if self._mcp_validation_task and not self._mcp_validation_task.done():
            logger.debug("Waiting for MCP validation to complete...")
            try:
                self.mcp_manager, _ = await self._mcp_validation_task
            except Exception as e:
                logger.error(f"MCP validation task failed: {e}")

        agent_config = self._agents[agent_name]
        mcp_tools = await load_agent_mcp_tools(agent_config, self.mcp_manager)

        # Replace placeholder tools with real tools in the global registry
        self._replace_placeholder_tools_with_real_tools(mcp_tools)

        return mcp_tools

    def _replace_placeholder_tools_with_real_tools(self, mcp_tools: dict):
        """Replace placeholders with actual loaded MCP tools in the global registry.

        For MCP tools, placeholders are regsistered at startup, but the stdio tools are
        not fully loaded until a conversation starts"""
        replace_mcp_placeholders_with_real_tools(
            mcp_tools, self.tools, self._mcp_placeholder_tools
        )

    async def cleanup_mcp_connections(self):
        """Clean up MCP connections when shutting down."""
        # Cancel the validation task if it's still running
        if self._mcp_validation_task and not self._mcp_validation_task.done():
            self._mcp_validation_task.cancel()
            try:
                await self._mcp_validation_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Exception during MCP validation task cleanup: {e}")

        if self.mcp_manager:
            await self.mcp_manager.cleanup_connections()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_mcp_connections()

    def __del__(self):
        """Destructor that cleans up pending tasks."""
        if (
            hasattr(self, "_mcp_validation_task")
            and self._mcp_validation_task
            and not self._mcp_validation_task.done()
        ):
            # Cancel the task if it's still running
            self._mcp_validation_task.cancel()


class AutogenManager(AgentManager):
    """Alias for AgentManager"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AgentSession:
    """A per-agent, per-conversation session.

    Encapsulates the active autogen agent/team, model context, streaming,
    termination, and memory for this conversation.

    **IMPORTANT**: Do not instantiate this class directly. Use the async factory
    method `AgentSession.create()` instead, which properly handles all async
    initialization required for the session.

    Example:
        # Correct usage:
        session = await AgentSession.create(
            manager=my_manager,
            agent_name="my_agent"
        )

        # Incorrect usage (will raise RuntimeError):
        session = AgentSession(manager=my_manager, agent_name="my_agent")
    """

    def __init__(
        self,
        manager: AutogenManager,
        agent_name: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        stream_tokens: bool = True,
        message_callback: Callable | None = None,
        agent_callback: Callable | None = None,
        _internal: bool = False,
    ) -> None:
        """Initialize AgentSession - Internal use only.

        **WARNING**: This constructor is for internal use only. Direct instantiation
        will result in an incomplete, non-functional session. Always use
        AgentSession.create() for proper initialization.

        Args:
            _internal: Private flag to prevent direct instantiation. Only the
                      create() factory method should set this to True.

        Raises:
            RuntimeError: If called directly (when _internal=False)
        """
        if not _internal:
            raise RuntimeError(
                "AgentSession cannot be instantiated directly. "
                "Use 'await AgentManager.new_conversation(...)' instead for proper "
                " initialization."
            )
        self.manager = manager
        self.agent_name = agent_name
        self._model_id = model_id
        self._temperature = temperature
        self._stream_tokens = stream_tokens
        self._streaming_preference = stream_tokens
        # per-session callback; use provided callback or fall back to manager default
        self._agent_callback = (
            agent_callback
            if agent_callback is not None
            else manager._default_agent_callback
        )
        self._message_callback = (
            message_callback
            if message_callback is not None
            else manager._default_message_callback
        )

        # Will be used to cancel ongoing tasks for this session
        self._cancelation_token: CancellationToken | None = None

        # These will be set during initialization
        self.agent = None
        self.agent_team = None
        self.terminator = None
        self.oneshot = False
        self._prompt = ""
        self._description = ""

    @classmethod
    async def create(
        cls,
        manager: AgentManager,
        agent_name: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
        stream_tokens: bool = True,
        message_callback: Callable | None = None,
        agent_callback: Callable | None = None,
    ) -> "AgentSession":
        """Async factory method to create and initialize an AgentSession.

        Args:
            manager: The AutogenManager instance
            agent_name: Name of the agent to create session for
            model_id: Override model for this session
            temperature: Override temperature for this session
            stream_tokens: Override streaming setting for this session
            message_callback: Override manager default if provided
            agent_callback: Override callback for this session

        Returns:
            An initialized AgentSession

        Example:
            session = await AgentSession.create(
                manager=my_manager,
                agent_name="helpful_assistant"
            )
        """
        session = cls(
            manager=manager,
            agent_name=agent_name,
            model_id=model_id,
            temperature=temperature,
            stream_tokens=stream_tokens,
            message_callback=message_callback,
            agent_callback=agent_callback,
            _internal=True,  # Allow internal instantiation
        )
        await session._initialize()
        return session

    async def _initialize(self) -> None:
        if self.agent_name is None:
            # nothing to initialize
            return

        agents = self.manager._agents
        mm = self.manager.mm
        tools_map = self.manager.tools

        agent = self.agent_name
        agent_data = agents[agent]

        # Use model from argument, agent config, or default
        if self._model_id is None:
            self._model_id = agent_data.get("model", mm.default_chat_model)
        # Use temperature from argument, agent config, or default
        if self._temperature is None:
            self._temperature = agent_data.get(
                "temperature", mm.default_chat_temperature
            )

        self._prompt = agent_data.get("prompt", "")
        self._description = agent_data.get("description", "")
        self._extra_kwargs = agent_data.get("extra_kwargs", {})

        if agent_data.get("type") == "team":
            # Team-based Agents
            self.agent_team = self._create_team(agent_data["team_type"], agent_data)
            # Assign first participant as primary agent for name/stream toggles
            self.agent = self.agent_team._participants[0]

            # currently not streaming tokens for team agents
            self._stream_tokens = None
            self.manager._stream_tokens = None
            logger.info("token streaming for team-based agents currently disabled")

        else:
            # Solo Agent
            model_client = mm.open_model(self._model_id)

            # Validate context capacity if system prompts are unsupported
            self._validate_prompt_context(self._model_id, agent_data)

            # don't use tools if the model does't support them
            if (
                not mm.get_tool_support(self._model_id)
                or not model_client.model_info.get("function_calling", False)
                or "tools" not in agent_data
            ):
                tools = None
            else:
                # Load regular Python tools (filter out dict-based MCP tools)
                string_tools = [t for t in agent_data["tools"] if isinstance(t, str)]
                regular_tools = [tools_map[t] for t in string_tools if t in tools_map]

                # Load MCP tools for this agent
                mcp_tools = await self.manager.get_agent_mcp_tools(agent)
                mcp_tool_list = list(mcp_tools.values())

                # Combine regular and MCP tools
                tools = regular_tools + mcp_tool_list

                logger.debug(
                    f"Loaded {len(regular_tools)} regular tools and "
                    f"{len(mcp_tool_list)} MCP tools for agent {agent}"
                )

            # system message if supported; else pass prompt as initial user message
            if mm.get_system_prompt_support(self._model_id):
                system_message = self._prompt
                initial_messages = None
            else:
                system_message = None
                initial_messages = [UserMessage(content=self._prompt, source="user")]

            # Build the model_context (with optional initial messages)
            model_context = self._make_model_context(
                agent_data=agent_data,
                model_client=model_client,
                tools=tools,
                initial_messages=initial_messages,
            )

            # Load Extra multi-shot messages if they exist
            if "extra_context" not in agents[agent]:
                agents[agent]["extra_context"] = []
            for extra in agents[agent]["extra_context"]:
                if extra[0] == "ai":
                    await model_context.add_message(
                        AssistantMessage(content=extra[1], source=agent)
                    )
                elif extra[0] == "human":
                    await model_context.add_message(
                        UserMessage(content=extra[1], source="user")
                    )
                elif extra[0] == "system":
                    raise ValueError(f"system message not implemented: {extra[0]}")
                else:
                    raise ValueError(f"Unknown extra context type {extra[0]}")

            # build the agent
            if agent_data.get("type") == "autogen-agent":
                if agent_data.get("name") == "websurfer":
                    self.agent = MultimodalWebSurfer(
                        model_client=model_client,
                        name=agent,
                    )
                    # not streaming builtin autogen agents right now
                    logger.info(
                        f"token streaming agent:{agent} disabled or not supported"
                    )
                    self._stream_tokens = None
                    self.manager._stream_tokens = None
                else:
                    raise ValueError(f"Unknown autogen agent type for agent:{agent}")
            else:
                # Merge in any extra args, avoiding conflicts with explicit params
                extra_kwargs = {}
                if isinstance(self._extra_kwargs, dict) and self._extra_kwargs:
                    reserved = {
                        "name",
                        "model_client",
                        "tools",
                        "model_context",
                        "system_message",
                        "model_client_stream",
                        "reflect_on_tool_use",
                    }
                    extra_kwargs = {
                        k: v for k, v in self._extra_kwargs.items() if k not in reserved
                    }
                    dropped = set(self._extra_kwargs.keys()) - set(extra_kwargs.keys())
                    if dropped:
                        logger.debug(
                            "Skipping conflicting extra_kwargs for AssistantAgent: %s",
                            sorted(dropped),
                        )

                self.agent = AssistantAgent(
                    name=agent,
                    model_client=model_client,
                    tools=tools,
                    model_context=model_context,
                    system_message=system_message,
                    model_client_stream=True,
                    reflect_on_tool_use=True,
                    **extra_kwargs,
                )

                messages = await self.agent._model_context.get_messages()
                logger.trace(f"messages: {messages}")

            # disable streaming if not supported by the model, otherwise use preference
            if not mm.get_streaming_support(self._model_id):
                self._stream_tokens = None
                try:
                    self.agent._model_client_stream = None
                except Exception:
                    pass
                self.manager._stream_tokens = None
                logger.info(f"token streaming for model {self._model_id} not supported")
            else:
                self._stream_tokens = self._streaming_preference
                try:
                    self.agent._model_client_stream = self._stream_tokens
                except Exception:
                    pass
                self.manager._stream_tokens = self._stream_tokens

            # Build the termination conditions
            terminators = [StopMessageTermination()]
            max_rounds = agent_data.get("max_rounds", 5)
            terminators.append(MaxMessageTermination(max_rounds))
            if "termination_message" in agent_data:
                terminators.append(
                    TextMentionTermination(agent_data["termination_message"])
                )
            self.terminator = ExternalTermination()  # for custom terminations
            terminators.append(self.terminator)

            # default oneshot to false unless specified
            self.oneshot = agent_data.get("oneshot", False)

            # Smart terminator to reflect on the conversation if not complete
            terminators.append(
                SmartReflectorTermination(
                    model_client=mm.open_model(mm.default_mini_model),
                    oneshot=self.oneshot,
                    agent_name=agent,
                )
            )

            logger.debug("creating RR group chat")
            termination = reduce(lambda x, y: x | y, terminators)

            self.agent_team = RoundRobinGroupChat(
                participants=[self.agent],
                termination_condition=termination,
            )

    def _validate_prompt_context(self, model_id: str, agent_data: dict) -> None:
        """Validate context when system prompts are unsupported.

        Ensures an injected prompt can be preserved. Raises ValueError when
        configuration is insufficient (e.g., buffered with buffer_size < 2, or
        head_tail with head_size < 1).
        """
        mm = self.manager.mm
        # If system prompts are supported, nothing to validate
        if mm.get_system_prompt_support(model_id):
            return

        ctx_cfg = (agent_data or {}).get("context") or {}
        if not isinstance(ctx_cfg, dict):
            # Defaults to unbounded; acceptable
            return
        ctx_type = str(ctx_cfg.get("type", "unbounded")).strip().lower()

        try:
            if ctx_type in ("buffered", "buffer", "buffer_size"):
                buffer_size = int(ctx_cfg.get("buffer_size", 20))
                if buffer_size < 2:
                    raise ValueError(
                        f"Model '{model_id}' does not support system prompts and "
                        f"buffered context size {buffer_size} is too small to "
                        "retain the injected prompt. Set buffer_size >= 2 or "
                        "use a model that supports system prompts."
                    )
            elif ctx_type in ("head_tail", "headandtail", "head_and_tail", "head-tail"):
                head_size = int(ctx_cfg.get("head_size", 3))
                if head_size < 1:
                    raise ValueError(
                        f"Model '{model_id}' does not support system prompts and "
                        "head_tail context requires head_size >= 1 to retain the "
                        "injected prompt."
                    )
            # token-limited and unbounded are acceptable; trimming is token-based
            # and not deterministic here
        except Exception as e:
            # Surface as configuration error with proper chaining
            name = agent_data.get("name", "unknown")
            raise ValueError(
                f"Invalid context configuration for agent '{name}': {e}"
            ) from e

    def _make_model_context(
        self,
        agent_data: dict,
        model_client,
        tools,
        initial_messages: list | None = None,
    ) -> Any:
        """Create a ChatCompletionContext based on agent configuration.

        Supported configuration on the agent:
          context:
            type: unbounded | buffered | token | head_tail
            # buffered
            buffer_size: int
            # token
            token_limit: int | null
            # head_tail
            head_size: int
            tail_size: int

        Falls back to UnboundedChatCompletionContext if unset or invalid.
        """
        ctx_cfg = (agent_data or {}).get("context") or {}
        if not isinstance(ctx_cfg, dict):
            logger.warning(
                "agent context config must be a mapping; defaulting to unbounded"
            )
            return UnboundedChatCompletionContext(initial_messages=initial_messages)

        ctx_type = str(ctx_cfg.get("type", "unbounded")).strip().lower()

        try:
            if ctx_type in ("unbounded", "default", "all"):
                return UnboundedChatCompletionContext(initial_messages=initial_messages)

            if ctx_type in ("buffered", "buffer", "buffer_size"):
                buffer_size = int(ctx_cfg.get("buffer_size", 20))
                if buffer_size <= 0:
                    raise ValueError("buffer_size must be > 0")
                return BufferedChatCompletionContext(
                    buffer_size=buffer_size, initial_messages=initial_messages
                )

            if ctx_type in (
                "token",
                "token_limited",
                "tokenlimited",
                "token_limit",
            ):
                # token_limit is optional; if None, model's remaining_tokens will be
                # used
                token_limit_raw = ctx_cfg.get("token_limit", None)
                token_limit = (
                    None
                    if token_limit_raw in (None, "", "null")
                    else int(token_limit_raw)
                )
                # tool_schema is optional; for now we let it be None (tools are
                # passed to agent separately)
                return TokenLimitedChatCompletionContext(
                    model_client=model_client,
                    token_limit=token_limit,
                    initial_messages=initial_messages,
                )

            if ctx_type in ("head_tail", "headandtail", "head_and_tail", "head-tail"):
                head_size = int(ctx_cfg.get("head_size", 3))
                tail_size = int(ctx_cfg.get("tail_size", 20))
                if head_size < 0 or tail_size <= 0:
                    raise ValueError("head_size must be >= 0 and tail_size must be > 0")
                return HeadAndTailChatCompletionContext(
                    head_size=head_size,
                    tail_size=tail_size,
                    initial_messages=initial_messages,
                )

        except Exception as e:
            logger.warning(
                "invalid context config for agent '%s': %s; defaulting to unbounded",
                agent_data.get("name", "unknown"),
                e,
            )
            return UnboundedChatCompletionContext(initial_messages=initial_messages)

        # Fallback
        logger.warning(
            "unknown context type '%s' for agent '%s'; defaulting to unbounded",
            ctx_type,
            agent_data.get("name", "unknown"),
        )
        return UnboundedChatCompletionContext(initial_messages=initial_messages)

    # --- Agent-specific properties (session-scoped)

    @property
    def prompt(self) -> str:
        """Returns the current prompt for this session's agent"""
        return getattr(self, "_prompt", "")

    @property
    def description(self) -> str:
        """Returns the current description for this session's agent"""
        return getattr(self, "_description", "")

    @property
    def model(self) -> str | None:
        """Returns the current model id for this session"""
        return getattr(self, "_model_id", None)

    @property
    def stream_tokens(self) -> bool | None:
        """Current token streaming state for this session.

        Returns True/False, or None when unsupported by the model.
        """
        return self._stream_tokens

    @stream_tokens.setter
    def stream_tokens(self, value: bool) -> None:
        """Enable or disable token streaming for this session.

        If the model doesn't support streaming, value is ignored and stays None.
        """
        if not isinstance(value, bool):
            raise ValueError("stream_tokens must be a boolean")

        # If currently disabled by logic, don't allow it to be enabled
        if self._stream_tokens is None:
            logger.info(
                f"token streaming disabled, setting stream_tokens to {value} ignored"
            )
            return

        # Remember the last setting; apply to agent if available
        self._streaming_preference = value
        self._stream_tokens = value
        try:
            if hasattr(self, "agent") and self.agent is not None:
                self.agent._model_client_stream = value
                logger.info(
                    "token streaming for %s set to %s",
                    getattr(self.agent, "name", "agent"),
                    value,
                )
        except Exception:
            logger.debug("unable to set agent streaming flag", exc_info=True)

    def _create_team(
        self, team_type: str, agent_data: dict
    ) -> RoundRobinGroupChat | SelectorGroupChat | MagenticOneGroupChat:
        # agent_data needs to be a team
        if "type" not in agent_data or agent_data["type"] != "team":
            raise ValueError("agent_data 'type' for team must be 'team'")

        mm = self.manager.mm
        tools_map = self.manager.tools
        agents_dict = self.manager._agents

        # build the agents
        agents = []
        for agent in agent_data["agents"]:
            subagent_data = agents_dict[agent]
            if "model" not in subagent_data:
                subagent_data["model"] = mm.default_chat_model
            model_client = mm.open_model(subagent_data["model"])

            # Validate context capacity for subagents when system prompts unsupported
            self._validate_prompt_context(subagent_data["model"], subagent_data)

            if subagent_data.get("type") == "autogen-agent":
                if subagent_data.get("name") == "websurfer":
                    agents.append(
                        MultimodalWebSurfer(
                            model_client=model_client,
                            name=agent,
                        )
                    )
                else:
                    raise ValueError(f"Unknown autogen agent type for agent:{agent}")
            else:
                # don't use tools if the model does't support them
                if (
                    not mm.get_tool_support(subagent_data["model"])
                    or not model_client.model_info.get("function_calling", False)
                    or "tools" not in subagent_data
                ):
                    tools = None
                else:
                    # load the tools (skip unknown tool ids gracefully)
                    tools = []
                    for tool in subagent_data["tools"]:
                        if tool in tools_map:
                            tools.append(tools_map[tool])
                        else:
                            logger.warning(f"Tool {tool} not found; skipping.")

                # system message if supported; else include prompt as initial
                # user message in the context
                if mm.get_system_prompt_support(subagent_data["model"]):
                    system_message = subagent_data["prompt"]
                    sub_initial_messages = None
                else:
                    system_message = None
                    sub_initial_messages = [
                        UserMessage(content=subagent_data["prompt"], source="user")
                    ]

                # Build model context with any initial messages
                sub_model_context = self._make_model_context(
                    agent_data=subagent_data,
                    model_client=model_client,
                    tools=tools,
                    initial_messages=sub_initial_messages,
                )

                extra_kwargs = {}
                if isinstance(subagent_data.get("extra_kwargs", None), dict):
                    reserved = {
                        "name",
                        "model_client",
                        "tools",
                        "model_context",
                        "system_message",
                        "model_client_stream",
                        "reflect_on_tool_use",
                    }
                    extra_kwargs = {
                        k: v
                        for k, v in subagent_data["extra_kwargs"].items()
                        if k not in reserved
                    }
                    dropped = set(subagent_data["extra_kwargs"].keys()) - set(
                        extra_kwargs.keys()
                    )
                    if dropped:
                        logger.debug(
                            "Skipping conflicting extra_kwargs for AssistantAgent: %s",
                            sorted(dropped),
                        )

                agents.append(
                    AssistantAgent(
                        name=agent,
                        model_client=model_client,
                        tools=tools,
                        model_context=sub_model_context,
                        system_message=system_message,
                        description=subagent_data["description"],
                        reflect_on_tool_use=True,
                        **extra_kwargs,
                    )
                )

        # construct the team
        terminators = []
        max_rounds = agent_data.get("max_rounds", 5)
        terminators.append(MaxMessageTermination(max_rounds))
        if "termination_message" in agent_data:
            terminators.append(
                TextMentionTermination(agent_data["termination_message"])
            )
        self.terminator = ExternalTermination()  # for custom terminations
        terminators.append(self.terminator)
        termination = reduce(lambda x, y: x | y, terminators)

        if "oneshot" in agent_data:
            self.oneshot = agent_data["oneshot"]
        else:
            self.oneshot = True if len(agents) == 1 else False

        if team_type == "round_robin":
            return RoundRobinGroupChat(agents, termination_condition=termination)
        elif team_type == "selector":
            if "team_model" not in agent_data:
                team_model = mm.open_model(mm.default_chat_model)
            else:
                team_model = mm.open_model(agent_data["team_model"])
            allow_repeated_speaker = agent_data.get("allow_repeated_speaker", False)

            if "selector_prompt" in agent_data:
                return SelectorGroupChat(
                    agents,
                    model_client=team_model,
                    selector_prompt=agent_data["selector_prompt"],
                    termination_condition=termination,
                    allow_repeated_speaker=allow_repeated_speaker,
                )
            else:
                return SelectorGroupChat(
                    agents,
                    model_client=team_model,
                    allow_repeated_speaker=allow_repeated_speaker,
                    termination_condition=termination,
                )
        elif team_type == "magnetic_one":
            if "team_model" not in agent_data:
                team_model = mm.open_model(mm.default_chat_model)
            else:
                team_model = mm.open_model(agent_data["team_model"])
            return MagenticOneGroupChat(
                agents, model_client=team_model, termination_condition=termination
            )
        else:
            raise ValueError(f"Unknown team type {team_type}")

    async def ask(self, task: str) -> TaskResult:
        self._cancelation_token = CancellationToken()

        try:
            result: TaskResult = await self._consume_agent_stream(
                agent_runner=self.agent_team.run_stream,
                oneshot=self.oneshot,
                task=task,
                cancellation_token=self._cancelation_token,
            )
        except Exception as e:
            logger.error(f"Error in agent stream: {e}")
            result = TaskResult(
                messages=[
                    TextMessage(  # ensure a sequence of TextMessage
                        source="System",
                        content=f"Error in agent stream: {e}",
                    )
                ],
                stop_reason="error",
            )
            await self._message_callback(
                "Error in response from AI, see debug", flush=True
            )

        self._cancelation_token = None
        if result.stop_reason.startswith("Exception occurred"):
            # notify the UI that an exception occurred
            logger.warning(
                f"Exception occurred talking to the AI: {result.stop_reason}"
            )
            await self._message_callback(
                "Error in response from AI, see debug", flush=True
            )
        logger.debug(f"Final result: {result.stop_reason}")
        await asyncio.sleep(0.1)  # Allow for autogen to cleanup/end the conversation
        return result

    def cancel(self) -> None:
        if self._cancelation_token:
            self._cancelation_token.cancel()

    def terminate(self) -> None:
        if hasattr(self, "terminator") and self.terminator:
            self.terminator.set()

    async def clear_memory(self) -> None:
        """Clear the agent's model context memory (async-only)."""
        if hasattr(self, "agent") and hasattr(self.agent, "_model_context"):
            try:
                ctx = self.agent._model_context
                clear = getattr(ctx, "clear", None)
                if callable(clear):
                    await clear()
            except Exception:
                logger.debug("Failed to clear model context", exc_info=True)

    async def update_memory(self, state: dict) -> None:
        await self.agent.load_state(state)

    async def get_memory(self) -> Mapping[str, Any]:
        return await self.agent.save_state()

    async def _consume_agent_stream(
        self,
        agent_runner: Callable[..., AsyncIterable],  # type of run_stream
        oneshot: bool,
        task: str,
        cancellation_token: CancellationToken,
    ) -> TaskResult:
        try:
            async for response in agent_runner(
                task=task, cancellation_token=cancellation_token
            ):
                # call agent_callback to notifiy application if needed, this is
                # intended to allow the calling app to see the raw responses for
                # debugging or monitoring
                await self._agent_callback(response)
                # Dispatch to correct handler
                if response is None:
                    logger.debug("Ignoring None response")
                    continue

                if isinstance(response, ModelClientStreamingChunkEvent):
                    await self._handle_stream_chunk(response)
                    continue

                if isinstance(response, MultiModalMessage):
                    await self._handle_multi_modal(response)
                    continue

                if isinstance(response, StopMessage):
                    logger.debug(f"Received StopMessage: {response.content}")
                    return TaskResult(
                        messages=[
                            TextMessage(  # ensure a sequence of TextMessage
                                source="System",
                                content="Presumed done",
                            )
                        ],
                        stop_reason="presumed done",
                    )

                if isinstance(response, TaskResult):
                    return response

                if isinstance(response, TextMessage):
                    handled = await self._handle_text_message(response, oneshot=oneshot)
                    if handled:
                        return handled
                    continue

                if isinstance(response, ThoughtEvent):
                    await self._handle_thought_event(response)
                    continue

                if isinstance(response, ToolCallExecutionEvent):
                    await self._handle_tool_call_execution(response)
                    continue

                if isinstance(response, ToolCallRequestEvent):
                    await self._handle_tool_call_request(response)
                    continue

                if isinstance(response, ToolCallSummaryMessage):
                    await self._handle_tool_summary(response)
                    continue

                if isinstance(response, UserInputRequestedEvent):
                    await self._handle_user_input_request(response)
                    continue

                # Unknown event
                logger.warning(f"Received unknown response type: {response!r}")
                await self._message_callback("<unknown>", flush=True)
                await self._message_callback(repr(response), flush=True)

            # TODO: This is a placeholder for when the stream ends without a TaskResult
            logger.error("Stream ended without a TaskResult")
            return TaskResult(
                messages=[TextMessage(source="System", content="End of stream")],
                stop_reason="end_of_stream",
            )
        except Exception as e:
            logger.error(f"Error in agent stream: {e}")
            await self._message_callback("Error in responses, see debug", flush=True)
            return TaskResult(
                messages=[
                    TextMessage(source="System", content=f"Error in agent stream: {e}")
                ],
                stop_reason="error",
            )

    # - - Handlers for different response types

    async def _handle_multi_modal(self, response: MultiModalMessage) -> None:
        await self._message_callback(
            f"MM:{response.content}", agent=response.source, complete=True
        )

    async def _handle_stream_chunk(
        self, response: ModelClientStreamingChunkEvent
    ) -> None:
        await self._message_callback(
            response.content, agent=response.source, complete=False
        )

    async def _handle_text_message(
        self, response: TextMessage, oneshot: bool
    ) -> TaskResult | None:
        if response.source == "user":
            return None

        logger.trace(f"TextMessage from {response.source}: {response.content}")

        # Only show the message if we're not streaming; otherwise streaming handles it
        if not self._stream_tokens:
            await self._message_callback(
                response.content, agent=response.source, complete=True
            )

    async def _handle_thought_event(self, response: ThoughtEvent) -> None:
        logger.debug(f"ThoughtEvent: {response.content}")
        await self._message_callback(
            f"\n\n*(Thinking: {response.content})*\n\n",
            agent=response.source,
            complete=False,
        )

    async def _handle_tool_summary(self, response: ToolCallSummaryMessage) -> None:
        logger.debug(f"Ignoring tool call summary message: {response.content}")

    async def _handle_tool_call_execution(
        self, response: ToolCallExecutionEvent
    ) -> None:
        logger.info(f"Tool call result: {response.content}")
        await self._message_callback("done", agent=response.source, complete=True)

    async def _handle_tool_call_request(self, response: ToolCallRequestEvent) -> None:
        tool_message = "\n\ncalling tool"
        for tool in response.content:
            tool_message += f"{tool.name} with arguments:\n{tool.arguments}\n"
        tool_message += "..."
        await self._message_callback(tool_message, agent=response.source)

    async def _handle_user_input_request(
        self, response: UserInputRequestedEvent
    ) -> None:
        logger.info(f"UserInputRequestedEvent: {response.content}")
        await self._message_callback(
            response.content, agent=response.source, complete=True
        )


class LLMTools:
    llmtools_mm = ModelManager()
    llmtools_summary_model: (
        AzureOpenAIChatCompletionClient | OpenAIChatCompletionClient
    ) = llmtools_mm.open_model(llmtools_mm.default_mini_model)

    def __init__(self):
        """Builds the intent prompt template"""
        pass

    @staticmethod
    async def aget_summary_label(conversation: str) -> str:
        """Returns the a very short summary of a conversation suitable for a label"""
        system_message = label_prompt.format(conversation=conversation)
        try:
            out = await LLMTools.llmtools_summary_model.create(
                [SystemMessage(content=system_message)]
            )
        except Exception as e:
            logger.error(f"Error getting summary label: {type(e)}:{e}")
            raise
        return out.content

    @staticmethod
    async def get_conversation_summary(conversation: str) -> str:
        """Returns the summary of a conversation"""
        system_message = summary_prompt.format(conversation=conversation)
        try:
            out = await LLMTools.llmtools_summary_model.create(
                [SystemMessage(content=system_message)]
            )
        except Exception as e:
            logger.error(f"Error getting conversation summary: {type(e)}:{e}")
            raise
        return out.content
