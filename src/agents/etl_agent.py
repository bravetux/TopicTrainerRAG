# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""ETL training agent — expert on Glue, Spark, dbt, Informatica, Talend, SSIS, ADF."""
import logging
from pathlib import Path
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.config import ETL_AGENT_WINDOW_SIZE
from src.tools.provider_manager import get_model
from src.tools.retrieval import retrieve_etl
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SKILLS_DIR = str(Path(__file__).parent.parent / "skills")

_SYSTEM_PROMPT = """You are a data engineering and ETL training specialist with deep expertise in:
- AWS Glue (Spark jobs, crawlers, DynamicFrames, Data Catalog)
- Apache Spark (PySpark, DataFrames, SparkSQL)
- dbt (models, tests, macros, materializations)
- Informatica PowerCenter (mappings, sessions, transformations)
- Talend (jobs, components, tMap)
- SQL Server Integration Services (SSIS packages, data flows)
- Azure Data Factory (pipelines, activities, linked services)

When answering questions:
1. ALWAYS call retrieve_etl first to find relevant content from the training materials
2. Cite your sources by mentioning the document name
3. Provide practical code/configuration examples
4. If training materials don't cover the topic, say so clearly — do not fabricate content
"""


def build_etl_agent() -> Agent:
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
        tools=[retrieve_etl],
        plugins=plugins,
        conversation_manager=SlidingWindowConversationManager(window_size=ETL_AGENT_WINDOW_SIZE),
        hooks=[LoggingThrottleHook()],
    )


@tool
def etl_training_agent(query: str) -> str:
    """Answer questions about ETL and data engineering: AWS Glue, Spark, dbt, Informatica, Talend, SSIS, ADF.

    Args:
        query: The employee's question about data engineering tools, ETL patterns, or pipeline best practices.

    Returns:
        Detailed answer with code examples and source citations from training materials.
    """
    logger.info("ETL agent query: %r", query)
    agent = build_etl_agent()
    result = agent(query)
    return str(result)
