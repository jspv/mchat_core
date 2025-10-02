import asyncio
import importlib.util
import os
import shlex
from urllib.parse import urlparse

from autogen_core.tools import FunctionTool

from .logging_utils import get_logger

logger = get_logger(__name__)


class MCPToolPlaceholder:
    """Placeholder for MCP tools that will be loaded at conversation time."""

    def __init__(self, spec: "MCPToolSpec"):
        self.spec = spec
        self.name = f"mcp_placeholder_{hash(spec.spec_string) % 10000}"
        self.description = f"MCP tool: {spec.spec_string}"

    async def __call__(self, *args, **kwargs):
        """This should never be called - placeholders are replaced before use."""
        raise RuntimeError(
            f"MCP placeholder tool {self.name} was called before being replaced "
            f"with real tool. This indicates a bug in the MCP loading logic."
        )


class MCPToolSpec:
    """Represents an MCP tool specification."""

    def __init__(self, spec_string: str, config: dict | None = None):
        self.spec_string = spec_string
        self.config = config or {}
        self.type = None
        self.connection_info = None
        self._parse_spec()

    def _parse_spec(self):
        """Parse MCP tool specification string."""
        if not self.spec_string.startswith("mcp:"):
            raise ValueError(f"Invalid MCP spec: {self.spec_string}")

        spec_content = self.spec_string[4:]  # Remove 'mcp:' prefix

        # Check if it's a URL (web-based MCP)
        if spec_content.startswith(("http://", "https://")):
            self.type = "web"
            parsed_url = urlparse(spec_content)
            # Handle port validation gracefully
            port = None
            try:
                port = parsed_url.port
            except ValueError:
                # Invalid port range, will be caught during validation
                pass

            self.connection_info = {
                "url": spec_content,
                "host": parsed_url.hostname,
                "port": port,
                "scheme": parsed_url.scheme,
            }
        else:
            # It's a command (stdio MCP)
            self.type = "stdio"
            # Split command and arguments safely
            cmd_parts = shlex.split(spec_content)
            self.connection_info = {
                "command": cmd_parts[0] if cmd_parts else "",
                "args": cmd_parts[1:] if len(cmd_parts) > 1 else [],
                "full_command": spec_content,
            }

    def get_config_value(self, key: str, default=None):
        """Get configuration value with fallback to default."""
        return self.config.get(key, default)

    def __repr__(self):
        return f"MCPToolSpec(type={self.type}, spec='{self.spec_string}')"


def register_mcp_placeholders(
    agents_config: dict, tools_registry: dict
) -> dict[str, str]:
    """Register placeholder tools for all MCP specs found in agents.

    This allows MCP tools to appear immediately in list_tools(), while actual
    MCP connections are deferred until conversation time.

    Args:
        agents_config: mapping of agent name -> agent configuration
        tools_registry: the global tools registry (name -> callable) to update

    Returns:
        Mapping of MCP spec string -> placeholder tool name
    """
    placeholder_map: dict[str, str] = {}
    try:
        for agent_name, agent_config in (agents_config or {}).items():
            tools = (
                agent_config.get("tools", []) if isinstance(agent_config, dict) else []
            )
            try:
                mcp_specs = parse_mcp_tools(tools, agent_config)
                for spec in mcp_specs:
                    placeholder = MCPToolPlaceholder(spec)
                    tool_name = placeholder.name
                    tools_registry[tool_name] = placeholder
                    placeholder_map[spec.spec_string] = tool_name
                    logger.debug(
                        "Registered MCP placeholder '%s' for spec '%s' (agent=%s)",
                        tool_name,
                        spec.spec_string,
                        agent_name,
                    )
            except Exception as e:
                logger.error("Error parsing MCP tools for agent %s: %s", agent_name, e)
    except Exception as e:
        logger.error("Error during MCP placeholder registration: %s", e)

    return placeholder_map


def replace_mcp_placeholders_with_real_tools(
    mcp_tools: dict, tools_registry: dict, placeholder_map: dict[str, str]
) -> None:
    """Replace any registered MCP placeholders with real loaded tools.

    Current behavior: replaces the first available placeholder with each real
    tool found (maintains backward compatibility with previous logic that did
    not track a 1:1 spec->tool mapping from server results).

    Args:
        mcp_tools: dict of tool name -> real tool callable/adapter
        tools_registry: global tool registry to mutate
        placeholder_map: mapping of spec string -> placeholder tool name
    """
    try:
        # Import here to avoid circulars at module import time
        from .tool_utils import MCPToolPlaceholder as _Placeholder
    except Exception:
        _Placeholder = MCPToolPlaceholder

    for _tool_name, tool_func in (mcp_tools or {}).items():
        # Find an un-replaced placeholder
        placeholders_to_replace: list[tuple[str, str]] = []
        for spec_string, placeholder_name in placeholder_map.items():
            if placeholder_name in tools_registry:
                obj = tools_registry[placeholder_name]
                if isinstance(obj, _Placeholder):
                    placeholders_to_replace.append((spec_string, placeholder_name))

        if placeholders_to_replace:
            _spec, placeholder_name = placeholders_to_replace[0]
            tools_registry[placeholder_name] = tool_func
            logger.debug(
                "Replaced MCP placeholder '%s' with real tool.", placeholder_name
            )


def parse_mcp_tools(
    tool_specs: list[str | dict], agent_config: dict | None = None
) -> list[MCPToolSpec]:
    """Parse list of tool specifications and extract MCP tools.

    Args:
        tool_specs: List of tool specification strings or dictionaries
        agent_config: Agent configuration that may contain MCP-specific options

    Returns:
        List of MCPToolSpec objects for MCP tools found
    """
    mcp_tools = []
    global_mcp_config = agent_config.get("mcp_config", {}) if agent_config else {}

    for spec in tool_specs:
        # Handle both string and dictionary formats
        if isinstance(spec, str) and spec.startswith("mcp:"):
            try:
                mcp_spec = MCPToolSpec(spec, global_mcp_config)
                mcp_tools.append(mcp_spec)
                logger.debug(f"Parsed MCP tool: {mcp_spec}")
            except Exception as e:
                logger.warning(f"Failed to parse MCP tool spec '{spec}': {e}")
        elif isinstance(spec, dict) and spec.get("mcp"):
            # Handle structured format: {mcp: "command", cwd: "/path", env: {...}}
            try:
                mcp_command = spec["mcp"]
                tool_config = {k: v for k, v in spec.items() if k != "mcp"}
                # Merge with global config (tool-specific config takes precedence)
                combined_config = {**global_mcp_config, **tool_config}
                mcp_spec = MCPToolSpec(f"mcp:{mcp_command}", combined_config)
                mcp_tools.append(mcp_spec)
                logger.debug(f"Parsed structured MCP tool: {mcp_spec}")
            except Exception as e:
                logger.warning(
                    f"Failed to parse structured MCP tool spec '{spec}': {e}"
                )

    return mcp_tools


async def validate_mcp_tool(mcp_spec: MCPToolSpec) -> bool:
    """Validate that an MCP tool specification is reachable/valid.

    Args:
        mcp_spec: MCP tool specification to validate

    Returns:
        True if the MCP server appears to be valid/reachable
    """
    try:
        # Import MCP functionality
        try:
            from autogen_ext.tools.mcp import (
                StdioServerParams,
                StreamableHttpServerParams,
                mcp_server_tools,
            )
        except ImportError:
            logger.warning("autogen_ext.tools.mcp not available for MCP validation")
            return False

        # Create appropriate server params based on type
        if mcp_spec.type == "web":
            server_params = StreamableHttpServerParams(
                url=mcp_spec.connection_info["url"],
                timeout=mcp_spec.get_config_value("timeout", 5.0),
            )
        elif mcp_spec.type == "stdio":
            # For stdio MCP, first check if the command exists and is executable
            import shutil

            command = mcp_spec.connection_info["command"]

            # Check if command exists in PATH or is an absolute path
            if os.path.isabs(command):
                if not (os.path.isfile(command) and os.access(command, os.X_OK)):
                    return False
            else:
                if shutil.which(command) is None:
                    return False

            server_params = StdioServerParams(
                command=command,
                args=mcp_spec.connection_info.get("args", []),
                cwd=mcp_spec.get_config_value("cwd", os.getcwd()),
                env=mcp_spec.get_config_value("env", None),
            )
        else:
            logger.warning(f"Unknown MCP type: {mcp_spec.type}")
            return False

        # Actually try to query the MCP server for its tools
        # This is the proper way to validate - use the MCP protocol
        try:
            tools = await mcp_server_tools(server_params)
            # If we can get tools without error, the server is valid
            return len(tools) >= 0  # Even 0 tools is valid
        except Exception as e:
            logger.debug(f"MCP validation failed for {mcp_spec}: {e}")
            return False

    except Exception as e:
        logger.warning(f"Error validating MCP tool {mcp_spec}: {e}")
        return False


class MCPToolManager:
    """Manages MCP tool connections and lifecycle."""

    def __init__(self):
        self.active_connections = {}
        self.validated_tools = {}

    async def validate_tools(self, mcp_specs: list[MCPToolSpec]) -> dict[str, bool]:
        """Validate multiple MCP tools concurrently.

        Args:
            mcp_specs: List of MCP tool specifications to validate

        Returns:
            Dictionary mapping spec string to validation result
        """
        results = {}

        # Run validations concurrently
        validation_tasks = [self._validate_and_store(spec) for spec in mcp_specs]

        validation_results = await asyncio.gather(
            *validation_tasks, return_exceptions=True
        )

        for spec, result in zip(mcp_specs, validation_results, strict=True):
            if isinstance(result, Exception):
                logger.warning(f"Validation failed for {spec}: {result}")
                results[spec.spec_string] = False
            else:
                results[spec.spec_string] = result

        return results

    async def _validate_and_store(self, spec: MCPToolSpec) -> bool:
        """Validate a single MCP tool and store the result."""
        is_valid = await validate_mcp_tool(spec)
        self.validated_tools[spec.spec_string] = is_valid
        return is_valid

    async def load_mcp_tools_for_conversation(
        self, mcp_specs: list[MCPToolSpec]
    ) -> dict[str, any]:
        """Load MCP tools for a conversation session.

        This is where we'll actually connect to MCP servers and get their tools.
        Called when starting a conversation.

        Args:
            mcp_specs: List of validated MCP tool specifications

        Returns:
            Dictionary of tool name -> tool function mappings
        """
        tools = {}

        for spec in mcp_specs:
            # Skip if not validated
            if not self.validated_tools.get(spec.spec_string, False):
                logger.warning(f"Skipping unvalidated MCP tool: {spec.spec_string}")
                continue

            try:
                spec_tools = await self._load_mcp_server_tools(spec)
                tools.update(spec_tools)
                logger.info(f"Loaded {len(spec_tools)} tools from {spec.spec_string}")
            except Exception as e:
                logger.error(f"Failed to load tools from {spec.spec_string}: {e}")

        return tools

    async def _load_mcp_server_tools(self, spec: MCPToolSpec) -> dict[str, any]:
        """Load tools from a specific MCP server.

        Args:
            spec: MCP tool specification

        Returns:
            Dictionary of tool name -> tool function mappings
        """
        try:
            # Import MCP functionality
            from autogen_ext.tools.mcp import (
                StdioServerParams,
                StreamableHttpServerParams,
                mcp_server_tools,
            )
        except ImportError:
            logger.error("autogen_ext.tools.mcp not available - MCP tools disabled")
            return {}

        if spec.type == "stdio":
            # Set up stdio MCP server
            server_params = StdioServerParams(
                command=spec.connection_info["command"],
                args=spec.connection_info["args"],
                cwd=spec.get_config_value("cwd", os.getcwd()),
                env=spec.get_config_value("env", None),
            )
        elif spec.type == "web":
            # Set up web MCP server
            server_params = StreamableHttpServerParams(url=spec.connection_info["url"])
        else:
            logger.error(f"Unknown MCP type: {spec.type}")
            return {}

        # Load tools from the MCP server
        mcp_tools = await mcp_server_tools(server_params)

        # Convert to dictionary with proper naming
        tools_dict = {}
        for i, tool in enumerate(mcp_tools):
            # Use the tool's name if available, otherwise generate one
            tool_name = getattr(tool, "name", f"mcp_tool_{spec.type}_{i}")
            tools_dict[tool_name] = tool
            logger.debug(f"Registered MCP tool: {tool_name}")

        # Store connection for cleanup later
        self.active_connections[spec.spec_string] = server_params

        return tools_dict

    async def cleanup_connections(self):
        """Clean up any active MCP connections."""
        # This would be called when shutting down conversations or the manager
        # Specific cleanup depends on the MCP implementation
        for spec_string, connection in self.active_connections.items():
            try:
                # Attempt cleanup if the connection supports it
                if hasattr(connection, "cleanup"):
                    await connection.cleanup()
                logger.debug(f"Cleaned up MCP connection: {spec_string}")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP connection {spec_string}: {e}")

        self.active_connections.clear()


async def validate_and_load_mcp_tools(agents_config: dict) -> tuple[dict, dict]:
    """Validate and prepare MCP tools from agent configurations.

    This function should be called during agent manager initialization to
    validate MCP tools and prepare them for later loading.

    Args:
        agents_config: Dictionary of agent configurations

    Returns:
        Tuple of (mcp_manager, validation_results) where:
        - mcp_manager: MCPToolManager instance with validated tools
        - validation_results: Dict mapping spec strings to validation results
    """
    mcp_manager = MCPToolManager()
    all_mcp_specs = []

    # Collect all MCP tools from all agents
    for _agent_name, agent_config in agents_config.items():
        tools = agent_config.get("tools", [])
        mcp_specs = parse_mcp_tools(tools, agent_config)
        all_mcp_specs.extend(mcp_specs)

    validation_results = {}

    if all_mcp_specs:
        logger.info(f"Found {len(all_mcp_specs)} MCP tools to validate")
        validation_results = await mcp_manager.validate_tools(all_mcp_specs)

        valid_count = sum(1 for is_valid in validation_results.values() if is_valid)
        total_count = len(validation_results)

        logger.info(f"MCP validation complete: {valid_count}/{total_count} tools valid")

        for spec_string, is_valid in validation_results.items():
            if is_valid:
                logger.debug(f"MCP tool validated: {spec_string}")
            else:
                logger.warning(f"MCP tool validation failed: {spec_string}")
    else:
        logger.debug("No MCP tools found in agent configurations")

    return mcp_manager, validation_results


async def load_agent_mcp_tools(agent_config: dict, mcp_manager) -> dict[str, any]:
    """Load MCP tools for a specific agent.

    This is called when starting a conversation to load MCP tools.

    Args:
        agent_config: Configuration for the specific agent
        mcp_manager: MCPToolManager instance with validated tools

    Returns:
        Dictionary of MCP tool name -> tool function mappings
    """
    tools = agent_config.get("tools", [])
    mcp_specs = parse_mcp_tools(tools, agent_config)

    if not mcp_specs:
        return {}

    # Load the validated MCP tools
    mcp_tools = await mcp_manager.load_mcp_tools_for_conversation(mcp_specs)

    return mcp_tools


async def run_mcp_validation(agents_config: dict) -> tuple["MCPToolManager", dict]:
    """Run MCP validation flow and return (manager, results).

    Centralized entry point used by higher-level managers to validate all
    configured MCP tools.

    Args:
        agents_config: mapping of agent name -> agent configuration

    Returns:
        (MCPToolManager, validation_results)
    """
    return await validate_and_load_mcp_tools(agents_config)


def create_mcp_validation_task(agents_config: dict) -> asyncio.Task | None:
    """Create a background task to validate MCP tools.

    This utility schedules MCP validation without blocking initialization. If no
    running event loop is available, returns None and the caller can defer to a
    later explicit validation.

    The resulting task resolves to (MCPToolManager, validation_results).
    """
    try:
        # Only schedule if there's an active running loop in this thread.
        # Avoid deprecated get_event_loop() behavior when no loop is set.
        loop = asyncio.get_running_loop()
        return loop.create_task(run_mcp_validation(agents_config))
    except RuntimeError:
        # No running loop available; let caller defer validation explicitly.
        return None
    except Exception as e:
        logger.debug(f"MCP validation task creation failed: {e}")
        return None


def load_tools(tools_directory: str | None = None) -> dict[str, FunctionTool]:
    """Discover and load tools from a directory.

    - Scans the tools directory for .py modules.
    - Imports modules and finds classes that subclass BaseTool
      (excluding BaseTool itself).
    - Instantiates each tool; if tool_instance.is_callable is True and a callable
      'run' method exists, wraps it in a FunctionTool and returns it in a dict.

    Args:
        tools_directory: Optional explicit path to scan. Defaults to the package's
            'tools' subdirectory.

    Returns:
        dict mapping tool name -> FunctionTool
    """
    from .tool_utils import BaseTool  # local import to avoid cycles

    if tools_directory is None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        tools_directory = os.path.join(current_directory, "tools")

    tools: dict[str, FunctionTool] = {}

    if not os.path.isdir(tools_directory):
        logger.warning(f"Tools directory not found: {tools_directory}")
        return tools

    for filename in os.listdir(tools_directory):
        if not filename.endswith(".py"):
            continue

        file_path = os.path.join(tools_directory, filename)
        mod_name = filename[:-3]

        try:
            spec = importlib.util.spec_from_file_location(mod_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Failed to create spec for tool module {mod_name}")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"Failed to load tool module {mod_name}: {e}")
            continue

        for item_name in dir(module):
            try:
                item = getattr(module, item_name)
            except Exception:
                continue

            if (
                isinstance(item, type)
                and issubclass(item, BaseTool)
                and item is not BaseTool
            ):
                try:
                    tool_instance: BaseTool = item()  # type: ignore[call-arg]
                except Exception as e:
                    logger.warning(f"Failed to instantiate tool {item_name}: {e}")
                    continue

                if tool_instance.is_callable:
                    run_fn = getattr(tool_instance, "run", None)
                    if not callable(run_fn):
                        logger.warning(
                            f"Tool {getattr(tool_instance, 'name', item_name)} "
                            f"marked callable but has no callable 'run'"
                        )
                        continue
                    name = getattr(tool_instance, "name", mod_name)
                    desc = getattr(tool_instance, "description", "")
                    try:
                        # Prefer explicit kwargs for broader compatibility
                        # across versions
                        tools[name] = FunctionTool(
                            func=run_fn, description=desc, name=name
                        )
                    except TypeError:
                        # Fallback for older/newer signatures
                        tools[name] = FunctionTool(run_fn, description=desc, name=name)
                else:
                    logger.warning(
                        f"Tool {getattr(tool_instance, 'name', item_name)} "
                        f"not loaded: not callable or failed setup "
                        f"({getattr(tool_instance, 'load_error', 'unknown')})"
                    )

    return tools


class BaseTool:
    name = "Base Tool"
    description = "Description of the base tool"

    def __init__(self):
        self.load_error = None
        # Respect class-level default if provided
        # (e.g., subclass sets is_callable = False)
        self.is_callable = getattr(self, "is_callable", True)
        try:
            self.verify_setup()
        except Exception as e:
            self.load_error = f"Setup verification failed: {e}"
            self.is_callable = False

    def verify_setup(self):
        """
        Override this method in derived classes to implement setup verification logic.
        """
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
