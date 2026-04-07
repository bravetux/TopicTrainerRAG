# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Learning path agent — recommends personalised study plans based on progress."""
import json
import logging
from strands import Agent, tool
from strands.agent.conversation_manager import NullConversationManager

from src.tools.provider_manager import get_model
from src.models.schemas import LearningPath
from src.tools.progress_db import progress_reader
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_TECHNOLOGY_LIST = (
    "selenium, tosca, playwright, "
    "aws_glue, spark, dbt, informatica, ssis, talend, adf"
)

_SYSTEM_PROMPT = f"""You are a personalised learning path advisor for employee training.

Available technologies: {_TECHNOLOGY_LIST}

To recommend a learning path:
1. Call progress_reader with the session_id to get the user's quiz history
2. Analyse scores: <50% = weak area, 50-75% = developing, >75% = strong
3. Recommend topics ordered from foundational to advanced
4. For beginners (no history): start with selenium basics and aws_glue basics
5. Estimated hours: assume 2h per beginner topic, 4h per intermediate, 6h per advanced

Return a LearningPath with current_level, recommended_topics (ordered list), next_milestone, estimated_hours, weak_areas, strong_areas.
"""


def build_learning_path_agent() -> Agent:
    model = get_model(temperature=0.2)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[progress_reader],
        conversation_manager=NullConversationManager(),
        hooks=[LoggingThrottleHook()],
    )


@tool
def learning_path_agent(session_id: str) -> str:
    """Generate a personalised learning path based on the user's quiz history.

    Args:
        session_id: The user's browser session identifier.

    Returns:
        JSON string representing a LearningPath with recommended topics, current level, and time estimates.
    """
    logger.info("Learning path agent: session_id=%s", session_id)
    agent = build_learning_path_agent()
    try:
        result = agent(
            f"Generate a learning path for session_id='{session_id}'",
            structured_output_model=LearningPath,
        )
        path: LearningPath = result.structured_output
        return path.model_dump_json()
    except Exception as exc:
        logger.error("Learning path generation failed: %s", exc)
        fallback = LearningPath(
            session_id=session_id,
            current_level="beginner",
            recommended_topics=["selenium basics", "aws_glue basics", "dbt fundamentals"],
            next_milestone="Complete first Selenium beginner quiz",
            estimated_hours=6.0,
            weak_areas=[],
            strong_areas=[],
        )
        return fallback.model_dump_json()
