"""Tests for LoggingThrottleHook."""
import pytest
from unittest.mock import MagicMock
from src.hooks.logging_throttle import LoggingThrottleHook


def make_before_tool_event(tool_name: str, tool_input: dict = None):
    event = MagicMock()
    event.tool_use = {"name": tool_name, "input": tool_input or {}}
    event.cancel_tool = None
    return event


def make_before_invocation_event():
    return MagicMock()


class TestLoggingThrottleHook:
    def test_first_call_is_allowed(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        event = make_before_tool_event("retrieve_qa")
        hook.check_and_log(event)
        assert event.cancel_tool is None

    def test_call_at_limit_is_allowed(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        for _ in range(9):
            e = make_before_tool_event("retrieve_qa")
            hook.check_and_log(e)
        event = make_before_tool_event("retrieve_qa")
        hook.check_and_log(event)
        assert event.cancel_tool is None

    def test_call_over_limit_is_cancelled(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        for _ in range(10):
            e = make_before_tool_event("retrieve_qa")
            hook.check_and_log(e)
        event = make_before_tool_event("retrieve_qa")
        hook.check_and_log(event)
        assert event.cancel_tool is not None
        assert "limit" in event.cancel_tool.lower()

    def test_reset_clears_count(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        for _ in range(10):
            e = make_before_tool_event("retrieve_qa")
            hook.check_and_log(e)
        hook.reset_counts(make_before_invocation_event())
        event = make_before_tool_event("retrieve_qa")
        hook.check_and_log(event)
        assert event.cancel_tool is None

    def test_path_traversal_blocked(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        event = make_before_tool_event(
            "file_write",
            {"path": "../../etc/passwd", "content": "hacked"}
        )
        hook.check_and_log(event)
        assert event.cancel_tool is not None

    def test_safe_path_allowed(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        event = make_before_tool_event(
            "file_write",
            {"path": "./data/generated/my_module.md", "content": "# Module"}
        )
        hook.check_and_log(event)
        assert event.cancel_tool is None
