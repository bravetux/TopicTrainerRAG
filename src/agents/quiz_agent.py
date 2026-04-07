# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Quiz agent — generates MCQ quizzes with structured output."""
import json
import logging
from strands import Agent, tool
from strands.agent.conversation_manager import SummarizingConversationManager

from src.tools.provider_manager import get_model
from src.models.schemas import QuizResult
from src.tools.retrieval import retrieve_qa, retrieve_etl
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a quiz generator for employee training.
Generate multiple-choice questions (MCQs) with exactly 4 options per question.
Each question must have a clear correct answer and a helpful explanation.

Rules:
1. Always call retrieve_qa or retrieve_etl FIRST to base questions on actual training content
2. Never generate questions about topics not found in training materials
3. Ensure correct_answer exactly matches one of the four options
4. Make distractors plausible but clearly wrong to domain experts
5. Calibrate difficulty: beginner (basic definitions), intermediate (application), advanced (edge cases/architecture)
"""


def build_quiz_agent() -> Agent:
    model = get_model(temperature=0.5)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_qa, retrieve_etl],
        conversation_manager=SummarizingConversationManager(summary_ratio=0.3),
        hooks=[LoggingThrottleHook()],
    )


@tool
def quiz_agent(technology: str, difficulty: str, num_questions: int) -> str:
    """Generate a multiple-choice quiz on a specific technology.

    Args:
        technology: Technology to quiz on (e.g. 'selenium', 'aws_glue', 'dbt', 'tosca').
        difficulty: Difficulty level: beginner, intermediate, or advanced.
        num_questions: Number of questions to generate (1-15).

    Returns:
        JSON string representing a QuizResult with questions, options, answers, and explanations.
    """
    logger.info("Quiz agent: technology=%s difficulty=%s n=%d", technology, difficulty, num_questions)
    num_questions = max(1, min(15, num_questions))

    prompt = (
        f"Generate a {num_questions}-question {difficulty} quiz on {technology}. "
        f"First retrieve relevant training content, then generate questions based only on that content. "
        f"Return a valid QuizResult JSON with technology='{technology}', "
        f"difficulty='{difficulty}', total_questions={num_questions}, passing_score=70."
    )

    agent = build_quiz_agent()
    try:
        result = agent(prompt, structured_output_model=QuizResult)
        quiz: QuizResult = result.structured_output
        return quiz.model_dump_json()
    except Exception as exc:
        logger.error("Quiz generation failed: %s", exc)
        return json.dumps({"error": str(exc), "technology": technology})
