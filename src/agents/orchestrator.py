"""Orchestrator agent — single entry point routing to specialist sub-agents."""
import logging
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session.file_session_manager import FileSessionManager

from src.config import ORCHESTRATOR_WINDOW_SIZE, SESSIONS_DIR
from src.tools.provider_manager import get_model
from src.hooks.logging_throttle import LoggingThrottleHook
from src.agents.qa_agent import qa_training_agent
from src.agents.etl_agent import etl_training_agent
from src.agents.quiz_agent import quiz_agent
from src.agents.learning_path_agent import learning_path_agent
from src.agents.content_author_agent import content_author_agent
from src.agents.progress_agent import progress_agent
from src.tools.retrieval import retrieve_topic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = """You are TechTrainer AI, an intelligent training assistant for employees.

## Available Topics
{available_section}

## Routing Rules

Route every request to the appropriate specialist tool:

| Request type | Tool to call |
|---|---|
| Questions about Selenium, Tosca, or Playwright | qa_training_agent |
| Questions about AWS Glue, Spark, dbt, Informatica, Talend, SSIS, ADF | etl_training_agent |
| Questions about a custom topic | retrieve_topic with the correct topic_id |
| Quiz requests ("quiz me", "test my knowledge") | quiz_agent |
| Learning path ("what should I study", "recommend topics") | learning_path_agent |
| Content creation ("create a module", "write training material") | content_author_agent |
| Progress queries ("my scores", "how am I doing") | progress_agent |

## Response Style
- Always cite source documents when the specialist provides them
- Keep responses concise — employees want quick, actionable answers
- For code: use markdown code blocks with language identifiers
- Never answer from memory alone — always route to the appropriate specialist

## CRITICAL: Structured Data Pass-Through
When quiz_agent, learning_path_agent, or content_author_agent return a JSON string,
you MUST return that JSON string EXACTLY as-is — no paraphrasing, no summarising,
no wrapping in markdown code fences. The UI parses the raw JSON directly.
Only add a short plain-text line AFTER the JSON if you wish to comment on it.

## Boundaries
- Only answer questions related to the available topics listed above
- For topics with no material, tell the user to visit the 📚 Knowledge Base tab to upload documents
- For completely off-topic requests, politely decline and suggest relevant topics you can help with
"""


def _build_system_prompt() -> str:
    """Build the system prompt with current available/unavailable topic lists."""
    try:
        from src.tools.kb_manager import load_all_topics
        all_topics = load_all_topics()
        available = [t for t in all_topics if t["status"] == "AVAILABLE"]
        unavailable = [t for t in all_topics if t["status"] != "AVAILABLE"]

        if available:
            avail_lines = "\n".join(
                f"- {t['display_name']} (topic_id: {t['id']})"
                for t in available
            )
        else:
            avail_lines = "- None yet. Ask the user to upload documents in the 📚 Knowledge Base tab."

        if unavailable:
            unavail_names = ", ".join(t["display_name"] for t in unavailable)
            unavail_section = (
                f"\nTopics with no material (direct users to Knowledge Base tab): {unavail_names}"
            )
        else:
            unavail_section = ""

        available_section = avail_lines + unavail_section
    except Exception as exc:
        logger.warning("Could not load topic status for system prompt: %s", exc)
        available_section = "- Topic status unavailable. Route normally."

    return _SYSTEM_PROMPT_BASE.format(available_section=available_section)


_ALL_TOOLS = [
    qa_training_agent,
    etl_training_agent,
    quiz_agent,
    learning_path_agent,
    content_author_agent,
    progress_agent,
    retrieve_topic,
]


def build_orchestrator(session_id: str) -> Agent:
    """Build and return the orchestrator agent for a given session."""
    model = get_model()
    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=SESSIONS_DIR,
    )
    return Agent(
        model=model,
        system_prompt=_build_system_prompt(),
        tools=_ALL_TOOLS,
        conversation_manager=SlidingWindowConversationManager(window_size=ORCHESTRATOR_WINDOW_SIZE),
        session_manager=session_manager,
        hooks=[LoggingThrottleHook()],
        trace_attributes={
            "session.id": session_id,
            "app.name": "techtrainer-ai",
            "app.version": "1.0.0",
        },
    )
