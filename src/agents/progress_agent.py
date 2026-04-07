"""Progress agent — reads and writes quiz progress via SQLite."""
import logging
from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager

from src.tools.provider_manager import get_model
from src.tools.progress_db import progress_reader, progress_writer
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a progress tracking assistant.
You read and write quiz scores and study history for employees.
Use progress_reader to fetch history and progress_writer to save new quiz results.
Return concise JSON-formatted summaries when reporting progress.
"""


def build_progress_agent() -> Agent:
    model = get_model(temperature=0.1)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[progress_reader, progress_writer],
        conversation_manager=NullConversationManager(),
        hooks=[LoggingThrottleHook()],
    )


@tool
def progress_agent(request: str, session_id: str) -> str:
    """Manage user quiz progress — read history or save new quiz results.

    Args:
        request: Natural language request about progress (e.g. 'save selenium quiz score 80/100' or 'show my progress').
        session_id: The user's browser session identifier.

    Returns:
        Progress summary or confirmation of saved result.
    """
    agent = build_progress_agent()
    result = agent(f"{request}\nSession ID: {session_id}")
    return str(result)
