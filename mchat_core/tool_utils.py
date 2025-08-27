import importlib.util
import os
from typing import Dict

from autogen_core.tools import FunctionTool

from .logging_utils import get_logger

logger = get_logger(__name__)


def load_tools(tools_directory: str | None = None) -> dict[str, FunctionTool]:
    """Discover and load tools from a directory.

    - Scans the tools directory for .py modules.
    - Imports modules and finds classes that subclass BaseTool (excluding BaseTool itself).
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
                        # Prefer explicit kwargs for broader compatibility across versions
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
        # Respect class-level default if provided (e.g., subclass sets is_callable = False)
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
