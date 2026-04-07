# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Content author agent — generates Markdown training modules."""
import json
import logging
from pathlib import Path
from strands import Agent, tool
from strands.agent.conversation_manager import SummarizingConversationManager

from src.config import GENERATED_DIR
from src.tools.provider_manager import get_model
from src.models.schemas import TrainingModule
from src.tools.retrieval import retrieve_qa, retrieve_etl
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a technical training content author.
You write clear, well-structured Markdown training modules for employees.

When generating a training module:
1. Call retrieve_qa or retrieve_etl FIRST to gather existing content as reference
2. Structure content with: Introduction, Prerequisites, Core Concepts (with examples), Hands-On Exercises, Summary, References
3. Include practical, runnable code examples for every concept
4. Calibrate content to the requested difficulty level
5. Exercises should be practical tasks (not just questions)
6. References must only cite actual source documents found via retrieval tools

Return a TrainingModule with all fields populated. The content field should be full Markdown.
"""


def _save_module(module: TrainingModule) -> str:
    """Write generated module to ./data/generated/ and return the file path."""
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in module.title.lower())
    safe_tech = "".join(c if c.isalnum() or c in "-_" else "_" for c in module.technology.lower())
    filename = f"{safe_tech}_{safe_title}.md"
    output_path = Path(GENERATED_DIR) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module.content, encoding="utf-8")
    logger.info("Module saved: %s", output_path)
    return str(output_path)


def build_content_author_agent() -> Agent:
    model = get_model(temperature=0.6)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_qa, retrieve_etl],
        conversation_manager=SummarizingConversationManager(summary_ratio=0.4),
        hooks=[LoggingThrottleHook()],
    )


@tool
def content_author_agent(
    title: str,
    technology: str,
    difficulty: str,
    objectives: str,
) -> str:
    """Generate a complete Markdown training module on a specific technology topic.

    Args:
        title: Title of the training module (e.g. 'Introduction to Selenium Locators').
        technology: Technology domain (e.g. 'selenium', 'aws_glue', 'dbt').
        difficulty: Difficulty level: beginner, intermediate, or advanced.
        objectives: Comma-separated learning objectives for this module.

    Returns:
        JSON string with the generated TrainingModule including title, content, exercises, and file path.
    """
    logger.info("Content author: title=%r technology=%s difficulty=%s", title, technology, difficulty)
    prompt = (
        f"Create a training module:\n"
        f"Title: {title}\n"
        f"Technology: {technology}\n"
        f"Difficulty: {difficulty}\n"
        f"Learning objectives: {objectives}\n"
        f"First retrieve relevant content for context, then write the full module."
    )
    agent = build_content_author_agent()
    try:
        result = agent(prompt, structured_output_model=TrainingModule)
        module: TrainingModule = result.structured_output
        file_path = _save_module(module)
        output = module.model_dump()
        output["saved_to"] = file_path
        return json.dumps(output)
    except Exception as exc:
        logger.error("Content generation failed: %s", exc)
        return json.dumps({"error": str(exc)})
