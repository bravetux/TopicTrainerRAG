# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""SQLite-backed progress tracking tools."""
import json
import sqlite3
import logging
from strands import tool
from src.config import PROGRESS_DB

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    technology TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    score INTEGER NOT NULL,
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS topics_studied (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    technology TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str = PROGRESS_DB) -> None:
    """Create tables if they don't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    logger.debug("Database initialised at %s", db_path)


def write_quiz_result(
    db_path: str,
    session_id: str,
    technology: str,
    difficulty: str,
    score: int,
    total_questions: int,
    correct_answers: int,
) -> int:
    """Insert a quiz result row and return the new row id."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO quiz_results
               (session_id, technology, difficulty, score, total_questions, correct_answers)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, technology, difficulty, score, total_questions, correct_answers),
        )
        conn.commit()
        return cursor.lastrowid


def read_progress(db_path: str, session_id: str) -> str:
    """Read all quiz results for a session and return as JSON string."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM quiz_results WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,),
        ).fetchall()

    results = [dict(r) for r in rows]

    tech_stats: dict = {}
    for r in results:
        tech = r["technology"]
        if tech not in tech_stats:
            tech_stats[tech] = {"scores": [], "attempts": 0}
        tech_stats[tech]["scores"].append(r["score"])
        tech_stats[tech]["attempts"] += 1

    for tech, stats in tech_stats.items():
        scores = stats["scores"]
        tech_stats[tech]["avg_score"] = round(sum(scores) / len(scores), 1)
        tech_stats[tech]["best_score"] = max(scores)
        del tech_stats[tech]["scores"]

    return json.dumps({
        "session_id": session_id,
        "quiz_results": results,
        "technologies": tech_stats,
    })


@tool
def progress_reader(session_id: str) -> str:
    """Read all quiz results and study history for a user session.

    Args:
        session_id: The user's browser session identifier.

    Returns:
        JSON string with quiz_results list and per-technology stats including avg_score and best_score.
    """
    init_db()
    return read_progress(PROGRESS_DB, session_id)


@tool
def progress_writer(
    session_id: str,
    technology: str,
    difficulty: str,
    score: int,
    total_questions: int,
    correct_answers: int,
) -> str:
    """Write a quiz result to the progress database.

    Args:
        session_id: The user's browser session identifier.
        technology: Technology tested (e.g. 'selenium', 'aws_glue').
        difficulty: Quiz difficulty: beginner, intermediate, or advanced.
        score: Percentage score 0-100.
        total_questions: Total number of questions in the quiz.
        correct_answers: Number of questions answered correctly.

    Returns:
        Confirmation message with the saved record ID.
    """
    init_db()
    record_id = write_quiz_result(
        PROGRESS_DB, session_id, technology, difficulty,
        score, total_questions, correct_answers,
    )
    logger.info("Progress saved: session=%s tech=%s score=%d", session_id, technology, score)
    return f"Progress saved (record_id={record_id}): {technology} {difficulty} score={score}%"
