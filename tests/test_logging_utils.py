import logging
import sys
import tempfile
import os
import asyncio

import pytest

from mchat_core.logging_utils import (
    TRACE_LEVEL_NUM,
    get_logger,
    trace,
    LoggerConfigurator,
    TraceLogger,
)


def test_trace_level_exists():
    assert logging.getLevelName(TRACE_LEVEL_NUM) == "TRACE"

def test_get_logger_returns_tracelogger():
    logger = get_logger("test.trace")
    assert isinstance(logger, TraceLogger)
    assert hasattr(logger, "trace")

def test_trace_logger_traces(caplog):
    logger = get_logger("test.trace_output")
    logger.setLevel(TRACE_LEVEL_NUM)
    with caplog.at_level(TRACE_LEVEL_NUM):
        logger.trace("trace message %d", 42)
    found = any("trace message" in rec.message for rec in caplog.records)
    assert found

def test_trace_decorator_sync(caplog):
    logger = get_logger("test.trace_deco")
    logger.setLevel(TRACE_LEVEL_NUM)

    @trace(logger)
    def f(a, b=2):
        return a + b

    with caplog.at_level(TRACE_LEVEL_NUM):
        result = f(1, b=3)
    assert result == 4
    entries = [rec.message for rec in caplog.records if "f" in rec.message]
    assert any("Calling f" in m for m in entries)
    assert any("returned:" in m for m in entries)

@pytest.mark.asyncio
async def test_trace_decorator_async(caplog):
    logger = get_logger("test.trace_async")
    logger.setLevel(TRACE_LEVEL_NUM)

    @trace(logger)
    async def afunc(x):
        await asyncio.sleep(0.01)
        return x + 10

    with caplog.at_level(TRACE_LEVEL_NUM):
        result = await afunc(5)
    assert result == 15
    entries = [rec.message for rec in caplog.records if "afunc" in rec.message]
    assert any("Calling afunc" in m for m in entries)
    assert any("returned:" in m for m in entries)

def test_logger_configurator_console_and_file(tmp_path):
    log_file = tmp_path / "test.log"
    config = LoggerConfigurator(
        log_to_console=True,
        log_to_file=True,
        file_path=str(log_file),
        console_log_level=TRACE_LEVEL_NUM,
        file_log_level=TRACE_LEVEL_NUM,
    )

    logger = get_logger("test.configured")
    logger.setLevel(TRACE_LEVEL_NUM)
    logger.trace("Hello TRACE to file & console")

    # flush file
    for h in logging.getLogger().handlers:
        if hasattr(h, "flush"):
            h.flush()

    with open(log_file, "r") as f:
        content = f.read()
    assert "Hello TRACE" in content

def test_level_override_filters(caplog):
    logger_name = "test.overrides"
    config = LoggerConfigurator(
        log_to_console=True,
        log_to_file=False,
        console_log_level=logging.WARNING,
    )
    config.add_console_filter(logger_name, TRACE_LEVEL_NUM)
    logger = get_logger(logger_name)
    logger.setLevel(TRACE_LEVEL_NUM)
    with caplog.at_level(TRACE_LEVEL_NUM):
        logger.trace("Should be visible because of override")
    assert any("Should be visible" in rec.message for rec in caplog.records)

def test_removing_filters():
    logger_name = "test.remove.filters"
    config = LoggerConfigurator(
        log_to_console=True,
        log_to_file=False,
        console_log_level=logging.WARNING,
    )
    config.add_console_filter(logger_name, TRACE_LEVEL_NUM)
    assert logger_name in config.get_console_filters()
    config.remove_console_filter(logger_name)
    assert logger_name not in config.get_console_filters()

# --- Trace Decorator/funcname tests ---

def test_trace_decorator_sync_funcname(caplog):
    logger = get_logger("test.trace_deco_funcname")
    logger.setLevel(TRACE_LEVEL_NUM)

    @trace(logger)
    def f(a, b=2):
        logger.trace("internal fn trace")
        return a + b

    with caplog.at_level(TRACE_LEVEL_NUM):
        result = f(1, b=3)
    assert result == 4

    # Entry/exit logs should be attributed to *this* test function
    calling = next(rec for rec in caplog.records if "Calling f" in rec.message)
    ret     = next(rec for rec in caplog.records if "returned:" in rec.message)
    assert calling.funcName == "test_trace_decorator_sync_funcname"
    assert ret.funcName == "test_trace_decorator_sync_funcname"

    in_func = next(rec for rec in caplog.records if rec.message == "internal fn trace")
    assert in_func.funcName == "f"

@pytest.mark.asyncio
async def test_trace_decorator_async_funcname(caplog):
    logger = get_logger("test.trace_async_funcname")
    logger.setLevel(TRACE_LEVEL_NUM)

    @trace(logger)
    async def afunc(x):
        await asyncio.sleep(0.01)
        logger.trace("async internal fn trace")
        return x + 10

    with caplog.at_level(TRACE_LEVEL_NUM):
        result = await afunc(5)
    assert result == 15

    calling = next(rec for rec in caplog.records if "Calling afunc" in rec.message)
    ret     = next(rec for rec in caplog.records if "returned:" in rec.message)
    assert calling.funcName == "test_trace_decorator_async_funcname"
    assert ret.funcName == "test_trace_decorator_async_funcname"

    in_func = next(rec for rec in caplog.records if rec.message == "async internal fn trace")
    assert in_func.funcName == "afunc"

def test_trace_decorator_nested_funcname(caplog):
    logger = get_logger("test.trace_nest_funcname")
    logger.setLevel(TRACE_LEVEL_NUM)

    @trace(logger)
    def inner(x):
        logger.trace("inner trace")
        return x * 2

    def caller():
        return inner(21)

    with caplog.at_level(TRACE_LEVEL_NUM):
        result = caller()
    assert result == 42

    # Since inner() is called from caller(), entry/exit should be attributed to 'caller'
    calling = next(rec for rec in caplog.records if "Calling inner" in rec.message)
    ret     = next(rec for rec in caplog.records if "returned:" in rec.message and "inner" in rec.message)
    assert calling.funcName == "caller"
    assert ret.funcName == "caller"

    in_func = next(rec for rec in caplog.records if rec.message == "inner trace")
    assert in_func.funcName == "inner"

# Optionally allow CLI test running:
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))