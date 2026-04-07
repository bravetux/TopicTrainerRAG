# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""QA training agent — expert on Selenium, Tosca, Playwright."""
import logging
from pathlib import Path
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.config import QA_AGENT_WINDOW_SIZE
from src.tools.provider_manager import get_model
from src.tools.retrieval import retrieve_qa
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SKILLS_DIR = str(Path(__file__).parent.parent / "skills")

_SYSTEM_PROMPT = """You are a QA testing training specialist with deep expertise in:
- Selenium WebDriver (Python and Java)
- Tricentis Tosca
- Playwright (Python and TypeScript)
- General test automation best practices

When answering questions:
1. ALWAYS call retrieve_qa first to find relevant content from the training materials
2. Cite your sources by mentioning the document name
3. Provide practical, runnable code examples
4. If training materials don't cover the topic, say so clearly — do not fabricate content
"""


def build_qa_agent() -> Agent:
    model = get_model(temperature=0.3)
    try:
        from strands import AgentSkills
        skills = AgentSkills(skills=_SKILLS_DIR)
        plugins = [skills]
    except Exception:
        plugins = []
        logger.warning("AgentSkills not available, proceeding without skills plugin")
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_qa],
        plugins=plugins,
        conversation_manager=SlidingWindowConversationManager(window_size=QA_AGENT_WINDOW_SIZE),
        hooks=[LoggingThrottleHook()],
    )


@tool
def qa_training_agent(query: str) -> str:
    """Answer questions about QA testing technologies: Selenium, Tosca, and Playwright.

    Args:
        query: The employee's question about QA testing tools, frameworks, or best practices.

    Returns:
        Detailed answer with code examples and source citations from training materials.
    """
    logger.info("QA agent query: %r", query)
    agent = build_qa_agent()
    result = agent(query)
    return str(result)
