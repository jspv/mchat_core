# Manual test of trace logging functionality
# uv run python -m tests.tracetest from mchat_core directory

from mchat_core.logging_utils import (
    TRACE,
    LoggerConfigurator,
    get_logger,
    trace,  # noqa: F401
)

TRACE_LEVEL_NUM = 5
logger = get_logger(__name__)

_ = LoggerConfigurator(
    log_to_console=True,
    console_log_level=TRACE,  # 5
)

@trace(logger)
def main():
    logger = get_logger("test.module")
    logger.trace("Hello from TRACE!")  # Should appear
    logger.debug("Hello from DEBUG!")  # Should appear as well, since DEBUG=10 >= 5
    logger.info("Hello from INFO!")
    logger.error("Hello from ERROR!")
    logger.warning("Hello from WARNING!")  # definitely should appear


if __name__ == "__main__":
    main()
