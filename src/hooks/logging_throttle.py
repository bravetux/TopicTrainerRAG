# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Logging and throttle hook for all Strands agents."""
import logging
import time
from pathlib import Path

from strands.hooks import (
    HookProvider, HookRegistry,
    BeforeInvocationEvent, BeforeToolCallEvent, AfterToolCallEvent,
)
from src.config import MAX_TOOLS_PER_TURN, GENERATED_DIR

logger = logging.getLogger(__name__)

_SAFE_WRITE_DIR = Path(GENERATED_DIR).resolve()


class LoggingThrottleHook(HookProvider):
    """Logs every tool call and enforces a per-turn call limit."""

    def __init__(self, max_tools: int = MAX_TOOLS_PER_TURN):
        self.max_tools = max_tools
        self._call_count = 0
        self._call_start_times: dict = {}

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.check_and_log)
        registry.add_callback(AfterToolCallEvent, self.log_result)

    def reset_counts(self, event: BeforeInvocationEvent) -> None:
        self._call_count = 0
        self._call_start_times.clear()
        logger.debug("Tool call counter reset for new invocation")

    def check_and_log(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "unknown")
        tool_input = event.tool_use.get("input", {})

        # Path traversal check for file_write calls
        if tool_name == "file_write":
            write_path = tool_input.get("path", "")
            if not self._is_safe_path(write_path):
                msg = f"File write to '{write_path}' not allowed — path outside {GENERATED_DIR}"
                logger.warning(msg)
                event.cancel_tool = msg
                return

        # Throttle check
        self._call_count += 1
        if self._call_count > self.max_tools:
            msg = f"Tool call limit reached ({self.max_tools} per turn). Cancelling '{tool_name}'."
            logger.warning(msg)
            event.cancel_tool = msg
            return

        self._call_start_times[tool_name] = time.time()
        logger.info(
            "→ TOOL CALL [%d/%d]: %s | input=%r",
            self._call_count, self.max_tools, tool_name, tool_input,
        )

    def log_result(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "unknown")
        start = self._call_start_times.pop(tool_name, None)
        duration = f"{(time.time() - start) * 1000:.0f}ms" if start else "?"
        logger.info("← TOOL DONE: %s | duration=%s", tool_name, duration)

    @staticmethod
    def _is_safe_path(path_str: str) -> bool:
        """Return True if path resolves inside the allowed generated content directory."""
        if not path_str:
            return True
        try:
            resolved = Path(path_str).resolve()
            return resolved.is_relative_to(_SAFE_WRITE_DIR)
        except Exception:
            return False
