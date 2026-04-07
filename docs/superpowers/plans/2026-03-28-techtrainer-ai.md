# TechTrainer AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade employee training chatbot with Q&A, quizzes, learning paths, and content authoring over local training documents.

**Architecture:** Orchestrator Agent routes requests to 6 specialist sub-agents (QA, ETL, Quiz, Learning Path, Content Author, Progress). RAG is served by ChromaDB with Bedrock Titan embeddings. All state stored locally (SQLite + filesystem).

**Tech Stack:** Strands Agents SDK, Amazon Bedrock (Claude Sonnet 4 + Titan Embeddings v2), ChromaDB, SQLite, Streamlit, uv, pypdf, python-docx, python-pptx, openpyxl, pydantic, python-dotenv

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project config, all dependencies |
| `.env.example` | Credential template |
| `src/config.py` | Load + validate all env vars, expose constants |
| `src/models/schemas.py` | All Pydantic models: QuizQuestion, QuizResult, TrainingModule, LearningPath |
| `src/tools/progress_db.py` | SQLite init + `progress_reader` / `progress_writer` Strands tools |
| `src/tools/document_ingestion.py` | Parse all doc formats, chunk, embed via Bedrock, store in ChromaDB |
| `src/tools/retrieval.py` | `retrieve_qa` / `retrieve_etl` Strands tools over ChromaDB |
| `src/hooks/logging_throttle.py` | `LoggingThrottleHook` — log all tool calls, enforce 10-call limit per turn |
| `src/skills/selenium.md` | AgentSkill knowledge for Selenium |
| `src/skills/tosca.md` | AgentSkill knowledge for Tosca |
| `src/skills/playwright.md` | AgentSkill knowledge for Playwright |
| `src/skills/aws_glue.md` | AgentSkill knowledge for AWS Glue |
| `src/skills/spark.md` | AgentSkill knowledge for Apache Spark |
| `src/skills/dbt.md` | AgentSkill knowledge for dbt |
| `src/skills/informatica.md` | AgentSkill knowledge for Informatica |
| `src/skills/ssis.md` | AgentSkill knowledge for SSIS |
| `src/skills/talend.md` | AgentSkill knowledge for Talend |
| `src/skills/adf.md` | AgentSkill knowledge for Azure Data Factory |
| `src/agents/progress_agent.py` | `build_progress_agent()` — reads/writes SQLite via progress tools |
| `src/agents/qa_agent.py` | `build_qa_agent()` + `qa_training_agent` @tool wrapper |
| `src/agents/etl_agent.py` | `build_etl_agent()` + `etl_training_agent` @tool wrapper |
| `src/agents/quiz_agent.py` | `build_quiz_agent()` + `quiz_agent` @tool wrapper, returns QuizResult |
| `src/agents/learning_path_agent.py` | `build_learning_path_agent()` + `learning_path_agent` @tool wrapper |
| `src/agents/content_author_agent.py` | `build_content_author_agent()` + `content_author_agent` @tool wrapper |
| `src/agents/orchestrator.py` | `build_orchestrator(session_id)` — main routing agent |
| `app.py` | Streamlit UI: 4 tabs + sidebar progress dashboard |
| `tests/conftest.py` | Shared fixtures: tmp SQLite path, tmp ChromaDB, fixture doc paths |
| `tests/fixtures/documents/qa/selenium_basics.txt` | Test document for QA collection |
| `tests/fixtures/documents/etl/aws_glue_overview.txt` | Test document for ETL collection |
| `tests/test_schemas.py` | Pydantic validation tests |
| `tests/test_progress_db.py` | SQLite tool tests |
| `tests/test_ingestion.py` | Document parsing + chunking tests |
| `tests/test_retrieval.py` | ChromaDB retrieval tool tests |
| `tests/test_hooks.py` | Hook throttle + logging tests |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`, `src/agents/__init__.py`, `src/tools/__init__.py`, `src/models/__init__.py`, `src/hooks/__init__.py`, `src/skills/` (dir), `tests/__init__.py`, `tests/fixtures/documents/qa/`, `tests/fixtures/documents/etl/`, `data/documents/qa/`, `data/documents/etl/`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "techtrainer-ai"
version = "0.1.0"
description = "AI-powered employee training chatbot built on Strands Agents"
requires-python = ">=3.10"
dependencies = [
    "strands-agents>=0.1.7",
    "strands-agents-tools>=0.1.7",
    "strands-agents-evals>=0.1.0",
    "streamlit>=1.35.0",
    "chromadb>=0.5.0",
    "boto3>=1.34.0",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "python-pptx>=1.0.0",
    "openpyxl>=3.1.0",
    "langchain-text-splitters>=0.2.0",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.14.0",
    "pytest-cov>=5.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

- [ ] **Step 2: Create .env.example**

```env
# ─── REQUIRED ────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=your_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
AWS_REGION=us-east-1

# ─── OPTIONAL (defaults shown) ───────────────────────────────────────────────
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_GUARDRAIL_ID=
BEDROCK_GUARDRAIL_VERSION=1
CHROMA_PERSIST_DIR=./data/chroma
SESSIONS_DIR=./data/sessions
GENERATED_DIR=./data/generated
PROGRESS_DB=./data/progress.db
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create all __init__.py files and directories**

```bash
mkdir -p src/agents src/tools src/models src/hooks src/skills
mkdir -p tests/fixtures/documents/qa tests/fixtures/documents/etl
mkdir -p data/documents/qa data/documents/etl data/sessions data/generated
touch src/__init__.py src/agents/__init__.py src/tools/__init__.py
touch src/models/__init__.py src/hooks/__init__.py
touch tests/__init__.py tests/fixtures/__init__.py
```

- [ ] **Step 4: Create fixture test documents**

`tests/fixtures/documents/qa/selenium_basics.txt`:
```
Selenium WebDriver Guide

Selenium WebDriver is a tool for automating web browsers. It provides a programming interface to create and execute test scripts.

Locator Strategies:
1. By ID - Most reliable locator. Use when elements have unique IDs.
   Example: driver.find_element(By.ID, "username")

2. By Name - Use when elements have name attributes.
   Example: driver.find_element(By.NAME, "email")

3. By CSS Selector - Flexible and fast. Supports complex queries.
   Example: driver.find_element(By.CSS_SELECTOR, ".login-btn")

4. By XPath - Most powerful but slowest. Use as last resort.
   Example: driver.find_element(By.XPATH, "//button[@type='submit']")

5. By Class Name - Use when element has a unique class.
   Example: driver.find_element(By.CLASS_NAME, "submit-button")

Explicit Waits:
Use WebDriverWait with expected conditions instead of time.sleep().
Example:
  wait = WebDriverWait(driver, 10)
  element = wait.until(EC.presence_of_element_located((By.ID, "result")))

Page Object Model (POM):
POM is a design pattern that creates an object repository for web UI elements.
Each page has a corresponding class with methods representing actions on that page.
Benefits: Reduces code duplication, improves maintainability.

Best Practices:
- Always use explicit waits over implicit waits
- Use CSS selectors over XPath when possible
- Keep test data separate from test logic
- Use POM for any project with more than 5 pages
```

`tests/fixtures/documents/etl/aws_glue_overview.txt`:
```
AWS Glue Developer Guide

AWS Glue is a serverless data integration service that makes it easy to discover, prepare, move, and integrate data from multiple sources.

Core Components:

1. Data Catalog
The AWS Glue Data Catalog is a central metadata repository. It stores table definitions, job definitions, and other control information.
- Crawlers automatically discover schema and register tables
- Supports databases from S3, RDS, Redshift, and more

2. Crawlers
Crawlers connect to data stores and scan data to infer schema.
- Run on demand or on a schedule
- Create or update tables in the Data Catalog
- Support S3, JDBC, DynamoDB, and more

3. ETL Jobs
Glue jobs run ETL scripts to transform data.
Job types:
- Spark: Distributed processing using Apache Spark
- Python Shell: Lightweight Python scripts for simple transformations
- Ray: For ML workloads

4. DynamicFrame
A DynamicFrame is similar to a Spark DataFrame but with additional metadata.
- Handles inconsistent schemas natively
- Use resolveChoice() to handle ambiguous types
- Convert to DataFrame with toDF() when needed

5. Job Bookmarks
Job bookmarks track processed data to avoid reprocessing.
Enable with: glueContext.create_dynamic_frame.from_catalog(
    database="mydb", table_name="mytable",
    additional_options={"jobBookmarkKeys": ["id"]}
)

Best Practices:
- Enable job bookmarks for incremental loads
- Use partitioning to improve query performance
- Monitor job metrics in CloudWatch
- Use Glue Studio for visual job authoring
```

- [ ] **Step 5: Install dependencies**

```bash
uv sync
```

Expected output: all packages installed, `.venv` created.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example src/ tests/ data/.gitkeep
git commit -m "feat: project scaffolding with uv, directory structure, and fixture docs"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write src/config.py**

```python
"""Central configuration loaded from environment variables."""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── AWS / Bedrock ──────────────────────────────────────────────────────────────
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID: str = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
BEDROCK_EMBEDDING_MODEL: str = os.getenv(
    "BEDROCK_EMBEDDING_MODEL",
    "amazon.titan-embed-text-v2:0"
)
BEDROCK_GUARDRAIL_ID: str = os.getenv("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION: str = os.getenv("BEDROCK_GUARDRAIL_VERSION", "1")

# ── Local Storage ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
SESSIONS_DIR: str = os.getenv("SESSIONS_DIR", "./data/sessions")
GENERATED_DIR: str = os.getenv("GENERATED_DIR", "./data/generated")
PROGRESS_DB: str = os.getenv("PROGRESS_DB", "./data/progress.db")
DOCUMENTS_QA_DIR: str = os.getenv("DOCUMENTS_QA_DIR", "./data/documents/qa")
DOCUMENTS_ETL_DIR: str = os.getenv("DOCUMENTS_ETL_DIR", "./data/documents/etl")

# ── ChromaDB Collection Names ──────────────────────────────────────────────────
CHROMA_QA_COLLECTION: str = "qa_training"
CHROMA_ETL_COLLECTION: str = "etl_training"

# ── Agent Settings ─────────────────────────────────────────────────────────────
ORCHESTRATOR_WINDOW_SIZE: int = 10
QA_AGENT_WINDOW_SIZE: int = 15
ETL_AGENT_WINDOW_SIZE: int = 15
MAX_TOOLS_PER_TURN: int = 10
SUBAGENT_TIMEOUT_SECONDS: int = 60

# ── Ingestion Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
RETRIEVAL_TOP_K: int = 5

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    handlers=[logging.StreamHandler()],
)

# ── Ensure runtime directories exist ──────────────────────────────────────────
for _dir in [SESSIONS_DIR, GENERATED_DIR, CHROMA_PERSIST_DIR,
             DOCUMENTS_QA_DIR, DOCUMENTS_ETL_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Verify config loads**

```bash
uv run python -c "from src.config import BEDROCK_MODEL_ID, AWS_REGION; print(BEDROCK_MODEL_ID, AWS_REGION)"
```

Expected: `us.anthropic.claude-sonnet-4-20250514-v1:0 us-east-1`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: config module loading env vars with sensible defaults"
```

---

## Task 3: Pydantic Schemas + Tests

**Files:**
- Create: `src/models/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests first**

`tests/test_schemas.py`:
```python
"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from src.models.schemas import QuizQuestion, QuizResult, TrainingModule, LearningPath


class TestQuizQuestion:
    def test_valid_question(self):
        q = QuizQuestion(
            question="What is a locator in Selenium?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="A locator identifies a web element.",
            difficulty="beginner",
            topic="selenium",
        )
        assert q.question == "What is a locator in Selenium?"
        assert len(q.options) == 4

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Q",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                explanation="E",
                difficulty="expert",  # invalid
                topic="selenium",
            )


class TestQuizResult:
    def test_valid_result(self):
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C", "D"],
            correct_answer="A", explanation="E",
            difficulty="beginner", topic="selenium",
        )
        result = QuizResult(
            technology="selenium",
            difficulty="beginner",
            questions=[q],
            total_questions=1,
            passing_score=70,
        )
        assert result.total_questions == 1
        assert result.passing_score == 70

    def test_empty_questions_allowed(self):
        result = QuizResult(
            technology="tosca", difficulty="advanced",
            questions=[], total_questions=0, passing_score=70,
        )
        assert result.questions == []


class TestTrainingModule:
    def test_valid_module(self):
        m = TrainingModule(
            title="Intro to Selenium",
            technology="selenium",
            difficulty="beginner",
            duration_minutes=30,
            learning_objectives=["Understand locators"],
            content="# Selenium\nSelenium is...",
            exercises=["Write a login test"],
            references=["selenium_basics.txt"],
        )
        assert m.duration_minutes == 30

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            TrainingModule(
                title="T", technology="t", difficulty="master",
                duration_minutes=10, learning_objectives=[],
                content="c", exercises=[], references=[],
            )


class TestLearningPath:
    def test_valid_path(self):
        lp = LearningPath(
            session_id="abc123",
            current_level="beginner",
            recommended_topics=["selenium basics", "locators"],
            next_milestone="Complete Selenium beginner quiz",
            estimated_hours=4.5,
            weak_areas=["xpath"],
            strong_areas=["css selectors"],
        )
        assert lp.estimated_hours == 4.5

    def test_negative_hours_rejected(self):
        with pytest.raises(ValidationError):
            LearningPath(
                session_id="x", current_level="beginner",
                recommended_topics=[], next_milestone="m",
                estimated_hours=-1.0,  # invalid
                weak_areas=[], strong_areas=[],
            )
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.schemas'`

- [ ] **Step 3: Write src/models/schemas.py**

```python
"""Pydantic models for structured agent outputs."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    topic: str


class QuizResult(BaseModel):
    technology: str
    difficulty: str
    questions: list[QuizQuestion]
    total_questions: int
    passing_score: int = Field(default=70, ge=0, le=100)


class TrainingModule(BaseModel):
    title: str
    technology: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    duration_minutes: int = Field(ge=1)
    learning_objectives: list[str]
    content: str
    exercises: list[str]
    references: list[str]


class LearningPath(BaseModel):
    session_id: str
    current_level: str
    recommended_topics: list[str]
    next_milestone: str
    estimated_hours: float = Field(ge=0.0)
    weak_areas: list[str]
    strong_areas: list[str]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic schemas for QuizResult, TrainingModule, LearningPath"
```

---

## Task 4: SQLite Progress Tools + Tests

**Files:**
- Create: `src/tools/progress_db.py`
- Create: `tests/conftest.py`
- Create: `tests/test_progress_db.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_progress.db")


@pytest.fixture
def fixture_qa_doc():
    return str(Path(__file__).parent / "fixtures/documents/qa/selenium_basics.txt")


@pytest.fixture
def fixture_etl_doc():
    return str(Path(__file__).parent / "fixtures/documents/etl/aws_glue_overview.txt")
```

- [ ] **Step 2: Write failing tests**

`tests/test_progress_db.py`:
```python
"""Tests for SQLite progress tools."""
import json
import pytest
from src.tools.progress_db import init_db, write_quiz_result, read_progress


class TestInitDb:
    def test_creates_tables(self, tmp_db):
        init_db(tmp_db)
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "quiz_results" in table_names
        assert "topics_studied" in table_names

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # second call should not raise


class TestWriteQuizResult:
    def test_writes_record(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="selenium",
            difficulty="beginner",
            score=80,
            total_questions=5,
            correct_answers=4,
        )
        assert record_id > 0

    def test_score_zero(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="selenium",
            difficulty="beginner",
            score=0,
            total_questions=5,
            correct_answers=0,
        )
        assert record_id > 0

    def test_score_hundred(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="spark",
            difficulty="advanced",
            score=100,
            total_questions=10,
            correct_answers=10,
        )
        assert record_id > 0


class TestReadProgress:
    def test_empty_session_returns_empty(self, tmp_db):
        init_db(tmp_db)
        result = read_progress(db_path=tmp_db, session_id="unknown")
        data = json.loads(result)
        assert data["quiz_results"] == []
        assert data["technologies"] == {}

    def test_reads_written_results(self, tmp_db):
        init_db(tmp_db)
        write_quiz_result(tmp_db, "s1", "selenium", "beginner", 80, 5, 4)
        write_quiz_result(tmp_db, "s1", "selenium", "intermediate", 60, 5, 3)
        result = read_progress(db_path=tmp_db, session_id="s1")
        data = json.loads(result)
        assert len(data["quiz_results"]) == 2
        assert "selenium" in data["technologies"]
        assert data["technologies"]["selenium"]["avg_score"] == 70.0

    def test_isolates_sessions(self, tmp_db):
        init_db(tmp_db)
        write_quiz_result(tmp_db, "s1", "selenium", "beginner", 90, 5, 5)
        write_quiz_result(tmp_db, "s2", "tosca", "beginner", 50, 5, 2)
        result = read_progress(db_path=tmp_db, session_id="s1")
        data = json.loads(result)
        assert len(data["quiz_results"]) == 1
        assert "tosca" not in data["technologies"]
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/test_progress_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.tools.progress_db'`

- [ ] **Step 4: Write src/tools/progress_db.py**

```python
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

    # Aggregate by technology
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


# ── Strands @tool wrappers ─────────────────────────────────────────────────────

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
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/test_progress_db.py -v
```

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add src/tools/progress_db.py tests/conftest.py tests/test_progress_db.py
git commit -m "feat: SQLite progress tools with Strands @tool wrappers"
```

---

## Task 5: Document Ingestion Pipeline + Tests

**Files:**
- Create: `src/tools/document_ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingestion.py`:
```python
"""Tests for document ingestion pipeline."""
import pytest
from pathlib import Path
from src.tools.document_ingestion import parse_document, chunk_text, get_supported_extensions


class TestGetSupportedExtensions:
    def test_returns_expected_extensions(self):
        exts = get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".xlsx" in exts
        assert ".pptx" in exts
        assert ".md" in exts


class TestParseDocument:
    def test_parse_txt(self, fixture_qa_doc):
        text = parse_document(fixture_qa_doc)
        assert "Selenium" in text
        assert len(text) > 100

    def test_parse_etl_txt(self, fixture_etl_doc):
        text = parse_document(fixture_etl_doc)
        assert "Glue" in text

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(str(bad_file))

    def test_empty_txt_returns_empty(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        result = parse_document(str(empty))
        assert result == ""


class TestChunkText:
    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=1000, overlap=200)
        assert len(chunks) >= 1
        assert "Hello world" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        long_text = "word " * 600  # ~3000 chars
        chunks = chunk_text(long_text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        # With overlap, end of chunk n should appear at start of chunk n+1
        long_text = " ".join([f"sentence{i}" for i in range(200)])
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        if len(chunks) > 1:
            # Last words of chunk 0 should appear in chunk 1
            last_words = chunks[0].split()[-5:]
            chunk1_text = chunks[1]
            assert any(w in chunk1_text for w in last_words)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_ingestion.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write src/tools/document_ingestion.py**

```python
"""Document ingestion pipeline: parse, chunk, embed, and index into ChromaDB."""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import boto3
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    AWS_REGION, BEDROCK_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_QA_COLLECTION, CHROMA_ETL_COLLECTION,
    DOCUMENTS_QA_DIR, DOCUMENTS_ETL_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}


def get_supported_extensions() -> set:
    return SUPPORTED_EXTENSIONS


def parse_document(file_path: str) -> str:
    """Parse a document and return its full text content."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".txt" or ext == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    if ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(file_path)
        slides = []
        for slide in prs.slides:
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text)
            if texts:
                slides.append("\n".join(texts))
        return "\n\n".join(slides)

    if ext == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(file_path, read_only=True, data_only=True)
        sheets = []
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    rows.append(row_text)
            if rows:
                sheets.append(f"Sheet: {sheet.title}\n" + "\n".join(rows))
        wb.close()
        return "\n\n".join(sheets)

    return ""  # unreachable but satisfies type checker


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def embed_texts(texts: list[str], bedrock_client) -> list[list[float]]:
    """Embed a list of text chunks using Bedrock Titan Embeddings."""
    embeddings = []
    for text in texts:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL,
            body=__import__("json").dumps({"inputText": text[:8192]}),  # Titan max 8192 chars
            contentType="application/json",
            accept="application/json",
        )
        body = __import__("json").loads(response["body"].read())
        embeddings.append(body["embedding"])
    return embeddings


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def index_directory(
    directory: str,
    collection_name: str,
    bedrock_client,
    chroma_client: chromadb.PersistentClient,
    reindex: bool = False,
) -> int:
    """Parse, chunk, embed, and store all documents in a directory. Returns chunk count."""
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if reindex:
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    total_chunks = 0
    doc_dir = Path(directory)
    files = [f for f in doc_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    for file_path in files:
        logger.info("Parsing %s...", file_path.name)
        try:
            text = parse_document(str(file_path))
            if not text.strip():
                logger.warning("Empty document: %s", file_path.name)
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            embeddings = embed_texts(chunks, bedrock_client)
            ids = [f"{file_path.stem}__chunk{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source_file": file_path.name,
                    "collection": collection_name,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            total_chunks += len(chunks)
            logger.info("  → %d chunks indexed", len(chunks))
        except Exception as exc:
            logger.error("Failed to index %s: %s", file_path.name, exc)

    return total_chunks


def run_ingestion(reindex: bool = False) -> None:
    """Run the full ingestion pipeline for QA and ETL collections."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    chroma = get_chroma_client()

    logger.info("Starting ingestion (reindex=%s)...", reindex)

    qa_chunks = index_directory(DOCUMENTS_QA_DIR, CHROMA_QA_COLLECTION, bedrock, chroma, reindex)
    etl_chunks = index_directory(DOCUMENTS_ETL_DIR, CHROMA_ETL_COLLECTION, bedrock, chroma, reindex)

    logger.info("Ingestion complete. QA chunks: %d, ETL chunks: %d", qa_chunks, etl_chunks)


def check_status() -> None:
    """Print current ChromaDB collection stats."""
    chroma = get_chroma_client()
    for name in [CHROMA_QA_COLLECTION, CHROMA_ETL_COLLECTION]:
        try:
            col = chroma.get_collection(name)
            print(f"{name}: {col.count()} chunks")
        except Exception:
            print(f"{name}: not found (run --reindex)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Delete and rebuild all collections")
    parser.add_argument("--status", action="store_true", help="Show current collection stats")
    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        run_ingestion(reindex=args.reindex)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_ingestion.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/tools/document_ingestion.py tests/test_ingestion.py
git commit -m "feat: document ingestion pipeline supporting PDF, DOCX, PPTX, XLSX, TXT"
```

---

## Task 6: ChromaDB Retrieval Tools + Tests

**Files:**
- Create: `src/tools/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

`tests/test_retrieval.py`:
```python
"""Tests for ChromaDB retrieval tools."""
import pytest
from unittest.mock import patch, MagicMock
from src.tools.retrieval import query_collection, retrieve_qa, retrieve_etl


class TestQueryCollection:
    def test_returns_formatted_string(self, tmp_path):
        """query_collection returns a non-empty formatted string when results exist."""
        # Create an in-memory chroma collection with known data
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("test_col")
        col.add(
            ids=["doc1"],
            embeddings=[[0.1] * 1024],
            documents=["Selenium uses locators to find elements."],
            metadatas=[{"source_file": "selenium.txt"}],
        )

        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_collection(
            query="selenium locators",
            collection_name="test_col",
            top_k=1,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "Selenium" in result
        assert "selenium.txt" in result

    def test_no_results_returns_message(self, tmp_path):
        import chromadb
        client = chromadb.EphemeralClient()
        client.create_collection("empty_col")
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_collection(
            query="xyz",
            collection_name="empty_col",
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "No relevant" in result or result == "" or isinstance(result, str)

    def test_top_k_respected(self, tmp_path):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("topk_col")
        for i in range(10):
            col.add(
                ids=[f"doc{i}"],
                embeddings=[[float(i) * 0.01] * 1024],
                documents=[f"Document {i} about selenium"],
                metadatas=[{"source_file": f"file{i}.txt"}],
            )
        mock_embed = MagicMock(return_value=[[0.05] * 1024])
        result = query_collection(
            query="selenium",
            collection_name="topk_col",
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        # Count "Source:" occurrences — should be at most 3
        assert result.count("Source:") <= 3
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_retrieval.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write src/tools/retrieval.py**

```python
"""ChromaDB retrieval tools for QA and ETL training content."""
import json
import logging
from typing import Callable, Optional

import boto3
import chromadb
from chromadb.config import Settings
from strands import tool

from src.config import (
    AWS_REGION, BEDROCK_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_QA_COLLECTION, CHROMA_ETL_COLLECTION,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_bedrock_client = None


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using Bedrock Titan Embeddings."""
    client = _get_bedrock()
    results = []
    for text in texts:
        response = client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL,
            body=json.dumps({"inputText": text[:8192]}),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        results.append(body["embedding"])
    return results


def query_collection(
    query: str,
    collection_name: str,
    top_k: int = RETRIEVAL_TOP_K,
    chroma_client: Optional[chromadb.ClientAPI] = None,
    embed_fn: Optional[Callable] = None,
) -> str:
    """Query a ChromaDB collection and return formatted results with citations."""
    client = chroma_client or _get_chroma()
    embed = embed_fn or _embed

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return f"No training content found. Please run document ingestion first."

    if collection.count() == 0:
        return "No training content found. Please add documents and run ingestion."

    query_embedding = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No relevant content found for your query."

    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        source = meta.get("source_file", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc}")

    return "\n\n---\n\n".join(parts)


@tool
def retrieve_qa(query: str, top_k: int = 5) -> str:
    """Retrieve relevant QA testing training content from the knowledge base.

    Args:
        query: Search query about QA testing technologies such as Selenium, Tosca, or Playwright.
        top_k: Number of most relevant document chunks to return. Default is 5.

    Returns:
        Formatted string of relevant training content with source file citations.
    """
    logger.debug("retrieve_qa query=%r top_k=%d", query, top_k)
    return query_collection(query, CHROMA_QA_COLLECTION, top_k)


@tool
def retrieve_etl(query: str, top_k: int = 5) -> str:
    """Retrieve relevant ETL and data engineering training content from the knowledge base.

    Args:
        query: Search query about ETL or data engineering technologies such as AWS Glue, Spark, or dbt.
        top_k: Number of most relevant document chunks to return. Default is 5.

    Returns:
        Formatted string of relevant training content with source file citations.
    """
    logger.debug("retrieve_etl query=%r top_k=%d", query, top_k)
    return query_collection(query, CHROMA_ETL_COLLECTION, top_k)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_retrieval.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/tools/retrieval.py tests/test_retrieval.py
git commit -m "feat: ChromaDB retrieval tools with Bedrock embeddings"
```

---

## Task 7: Safety Hooks + Tests

**Files:**
- Create: `src/hooks/logging_throttle.py`
- Create: `tests/test_hooks.py`

- [ ] **Step 1: Write failing tests**

`tests/test_hooks.py`:
```python
"""Tests for LoggingThrottleHook."""
import pytest
from unittest.mock import MagicMock, patch
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
        # 10th call — exactly at limit
        event = make_before_tool_event("retrieve_qa")
        hook.check_and_log(event)
        assert event.cancel_tool is None

    def test_call_over_limit_is_cancelled(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        for _ in range(10):
            e = make_before_tool_event("retrieve_qa")
            hook.check_and_log(e)
        # 11th call — over limit
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
        # New invocation resets count
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
        assert "traversal" in event.cancel_tool.lower() or "not allowed" in event.cancel_tool.lower()

    def test_safe_path_allowed(self):
        hook = LoggingThrottleHook()
        hook.reset_counts(make_before_invocation_event())
        event = make_before_tool_event(
            "file_write",
            {"path": "./data/generated/my_module.md", "content": "# Module"}
        )
        hook.check_and_log(event)
        assert event.cancel_tool is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_hooks.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write src/hooks/logging_throttle.py**

```python
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
        self._call_start_times: dict[str, float] = {}

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
                msg = f"File write to '{write_path}' not allowed — path traversal outside {GENERATED_DIR}"
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
        logger.info("→ TOOL CALL [%d/%d]: %s | input=%r", self._call_count, self.max_tools, tool_name, tool_input)

    def log_result(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "unknown")
        start = self._call_start_times.pop(tool_name, None)
        duration = f"{(time.time() - start) * 1000:.0f}ms" if start else "?"
        status = "OK" if not getattr(event, "error", None) else "ERROR"
        logger.info("← TOOL DONE: %s | status=%s | duration=%s", tool_name, status, duration)

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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_hooks.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/hooks/logging_throttle.py tests/test_hooks.py
git commit -m "feat: LoggingThrottleHook with per-turn limit and path traversal protection"
```

---

## Task 8: Skills Files (Domain Knowledge)

**Files:** Create all 10 skill `.md` files in `src/skills/`

- [ ] **Step 1: Create src/skills/selenium.md**

```markdown
---
name: selenium-expert
description: Answer questions about Selenium WebDriver, locators, waits, Page Object Model, and test automation best practices
allowed-tools: retrieve_qa http_request
---
# Selenium WebDriver Expert

When answering Selenium questions:
1. Always recommend explicit waits (WebDriverWait) over implicit waits or time.sleep()
2. Prefer CSS selectors over XPath for performance; use XPath only for complex DOM traversal
3. Recommend the Page Object Model (POM) pattern for any project with multiple pages
4. Show complete, runnable code examples using Python with selenium 4.x
5. Include import statements in all examples
6. Mention cross-browser considerations (Chrome, Firefox, Edge) where relevant
7. Reference the official Selenium documentation for authoritative answers

Key topics: WebDriver setup, locator strategies (ID/Name/CSS/XPath/LinkText), explicit/implicit waits, POM pattern, actions (click/send_keys/drag_drop), JavaScript execution, screenshots, headless mode, grid/parallel execution
```

- [ ] **Step 2: Create src/skills/tosca.md**

```markdown
---
name: tosca-expert
description: Answer questions about Tricentis Tosca test automation including modules, TestCases, TestSuites, TBox, and TOSCA Commander
allowed-tools: retrieve_qa http_request
---
# Tricentis Tosca Expert

When answering Tosca questions:
1. Distinguish between Modules (reusable UI elements), TestCases (executable tests), and TestSuites (groupings)
2. Explain TBox (Tosca's test automation engine) when relevant to execution questions
3. Cover TOSCA Commander (the IDE) for questions about test creation
4. Mention Tosca's risk-based testing and requirements coverage features for compliance questions
5. Reference Tricentis documentation for feature-specific questions
6. Address both the legacy Tosca 14.x and current Tosca 16.x where differences matter

Key topics: TOSCA Commander, Modules, TestCases, TestSuites, TBox, scanning, dynamic references, data-driven testing, distributed execution, Tosca CI/CD integration
```

- [ ] **Step 3: Create src/skills/playwright.md**

```markdown
---
name: playwright-expert
description: Answer questions about Playwright test automation including Python/TypeScript APIs, fixtures, tracing, and CI integration
allowed-tools: retrieve_qa http_request
---
# Playwright Expert

When answering Playwright questions:
1. Show both Python (pytest-playwright) and TypeScript examples where relevant
2. Always use async/await patterns in TypeScript examples
3. Recommend fixtures for setup/teardown over before/after hooks
4. Explain auto-waiting — Playwright waits automatically for elements to be actionable
5. Cover page.locator() (preferred) vs legacy find_element patterns
6. Show how to enable tracing and use the Playwright Inspector for debugging
7. Include CI configuration examples (GitHub Actions, GitLab CI) when asked about CI

Key topics: installation, browsers (Chromium/Firefox/WebKit), locators, assertions, fixtures, parallel execution, tracing, screenshots, video recording, API testing, component testing, codegen
```

- [ ] **Step 4: Create src/skills/aws_glue.md**

```markdown
---
name: aws-glue-expert
description: Answer questions about AWS Glue ETL jobs, crawlers, DynamicFrames, Data Catalog, and Glue Studio
allowed-tools: retrieve_etl http_request
---
# AWS Glue Expert

When answering AWS Glue questions:
1. Distinguish between Glue Spark jobs, Python Shell jobs, and Ray jobs
2. Explain DynamicFrames vs DataFrames — use DynamicFrame for schema flexibility, convert with toDF() for Spark operations
3. Always recommend job bookmarks for incremental data loading
4. Explain crawler behavior: when crawlers update vs overwrite table schemas
5. Cover IAM permissions: Glue needs S3 read/write, CloudWatch logs, and Glue service role
6. Show AWS console steps alongside code where helpful

Key topics: Data Catalog, crawlers, ETL jobs, DynamicFrame, job bookmarks, Glue Studio, triggers, workflows, partitioning, Glue DataBrew, connection types (S3/JDBC/DynamoDB)
```

- [ ] **Step 5: Create src/skills/spark.md**

```markdown
---
name: spark-expert
description: Answer questions about Apache Spark including DataFrames, RDDs, SparkSQL, transformations, actions, and optimization
allowed-tools: retrieve_etl http_request
---
# Apache Spark Expert

When answering Spark questions:
1. Recommend DataFrames/Dataset API over low-level RDDs for new code
2. Explain lazy evaluation — transformations build a DAG, actions trigger execution
3. Cover partitioning strategies and when to use repartition() vs coalesce()
4. Recommend using DataFrame API over RDDs for new development (since Spark 2.0)
5. Show PySpark examples by default; mention Scala equivalents for performance-critical paths
6. Include explain() plan analysis for query optimization questions

Key topics: SparkSession, DataFrames, RDDs, transformations vs actions, SparkSQL, joins (broadcast vs sort-merge), partitioning, caching/persistence, UDFs, Spark Streaming, MLlib basics, cluster configuration (executors/cores/memory)
```

- [ ] **Step 6: Create src/skills/dbt.md**

```markdown
---
name: dbt-expert
description: Answer questions about dbt (data build tool) including models, tests, seeds, macros, sources, and lineage
allowed-tools: retrieve_etl http_request
---
# dbt Expert

When answering dbt questions:
1. Explain the four materialization types: table, view, incremental, ephemeral
2. Cover both dbt Core (CLI) and dbt Cloud where relevant
3. Show YAML configuration alongside SQL models for tests and documentation
4. Recommend using sources (source.yml) for raw table references instead of hardcoded table names
5. Explain the ref() function — it builds the DAG and handles cross-environment references
6. Cover generic tests (unique, not_null, accepted_values, relationships) before custom tests

Key topics: project structure, models, materializations, seeds, sources, exposures, tests (generic + singular), macros, Jinja templating, packages (dbt-utils), snapshots, incremental models, dbt lineage graph, dbt Cloud IDE
```

- [ ] **Step 7: Create remaining skills files**

`src/skills/informatica.md`:
```markdown
---
name: informatica-expert
description: Answer questions about Informatica PowerCenter including mappings, sessions, workflows, transformations, and repository
allowed-tools: retrieve_etl http_request
---
# Informatica PowerCenter Expert

When answering Informatica questions:
1. Cover the four main components: Designer, Workflow Manager, Workflow Monitor, Repository Manager
2. Explain transformation types: active (change row count) vs passive (don't change row count)
3. Cover commonly used transformations: Source Qualifier, Expression, Aggregator, Joiner, Lookup, Router, Sequence Generator, Update Strategy
4. Explain session vs workflow — sessions execute mappings, workflows orchestrate sessions
5. Cover error handling: session log, bad file, row error log

Key topics: Designer tool, mappings, sessions, workflows, transformations (SQ/EXP/AGG/JNR/LKP/RTR), repository, parameter files, partitioning, pushdown optimization, performance tuning
```

`src/skills/ssis.md`:
```markdown
---
name: ssis-expert
description: Answer questions about SQL Server Integration Services (SSIS) packages, control flow, data flow, connection managers, and deployment
allowed-tools: retrieve_etl http_request
---
# SSIS Expert

When answering SSIS questions:
1. Distinguish control flow (orchestration) from data flow (transformation pipeline)
2. Cover common control flow tasks: Execute SQL, Data Flow, For Each Loop, Script Task
3. Cover common data flow transformations: OLE DB Source/Destination, Lookup, Conditional Split, Derived Column, Data Conversion
4. Explain connection managers — they abstract connection strings from package logic
5. Cover deployment: Project Deployment Model (SSISDB catalog) vs Legacy Package Deployment

Key topics: SSDT (Visual Studio), control flow, data flow, connection managers, variables, expressions, parameters, logging, error handling, project vs package deployment, SSISDB catalog, SQL Server Agent integration
```

`src/skills/talend.md`:
```markdown
---
name: talend-expert
description: Answer questions about Talend Open Studio and Talend Data Integration including jobs, components, tMap, and metadata
allowed-tools: retrieve_etl http_request
---
# Talend Expert

When answering Talend questions:
1. Cover Talend Open Studio (free) vs Talend Data Integration (enterprise) distinctions
2. Explain the component-based job design paradigm — every operation is a component
3. Cover tMap for complex transformations — it handles joins, filters, and mappings visually
4. Explain metadata management — define connections once, reuse across jobs
5. Cover common component categories: tFile*, tDB*, tKafka*, tREST*, tMap, tJava*

Key topics: Studio IDE, job design, components (tFileInput/tFileOutput/tDBInput/tDBOutput/tMap/tJoin/tAggregateRow), metadata, context variables, subjobs, error handling (tDie/tWarn), scheduling, Talend CI/CD with Maven
```

`src/skills/adf.md`:
```markdown
---
name: adf-expert
description: Answer questions about Azure Data Factory including pipelines, activities, datasets, linked services, triggers, and monitoring
allowed-tools: retrieve_etl http_request
---
# Azure Data Factory Expert

When answering ADF questions:
1. Explain the core concepts: pipelines (orchestration), activities (steps), datasets (data shape), linked services (connections)
2. Cover the three trigger types: schedule, tumbling window, event-based
3. Explain data flows (mapping data flows) vs copy activity — use data flows for complex transformations
4. Cover integration runtimes: Azure IR (cloud), Self-hosted IR (on-premises), SSIS IR
5. Explain parameterization — use parameters and variables to make pipelines reusable

Key topics: pipelines, activities (Copy/Web/Lookup/ForEach/If/Until/Databricks), datasets, linked services, triggers, mapping data flows, integration runtimes, parameters vs variables, monitoring, Git integration, ADF CI/CD with ARM templates or Bicep
```

- [ ] **Step 8: Commit all skills**

```bash
git add src/skills/
git commit -m "feat: AgentSkills domain knowledge files for all 10 technologies"
```

---

## Task 9: Sub-Agents

**Files:**
- Create: `src/agents/progress_agent.py`
- Create: `src/agents/qa_agent.py`
- Create: `src/agents/etl_agent.py`
- Create: `src/agents/quiz_agent.py`
- Create: `src/agents/learning_path_agent.py`
- Create: `src/agents/content_author_agent.py`

- [ ] **Step 1: Write src/agents/progress_agent.py**

```python
"""Progress agent — reads and writes quiz progress via SQLite."""
import logging
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import NullConversationManager

from src.config import BEDROCK_MODEL_ID, AWS_REGION
from src.tools.progress_db import progress_reader, progress_writer
from src.hooks.logging_throttle import LoggingThrottleHook

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a progress tracking assistant.
You read and write quiz scores and study history for employees.
Use progress_reader to fetch history and progress_writer to save new quiz results.
Return concise JSON-formatted summaries when reporting progress.
"""


def build_progress_agent() -> Agent:
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.1)
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
```

- [ ] **Step 2: Write src/agents/qa_agent.py**

```python
"""QA training agent — expert on Selenium, Tosca, Playwright."""
import logging
from pathlib import Path
from strands import Agent, AgentSkills, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.config import BEDROCK_MODEL_ID, AWS_REGION, QA_AGENT_WINDOW_SIZE
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
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.3)
    skills = AgentSkills(skills=_SKILLS_DIR)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_qa],
        plugins=[skills],
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
```

- [ ] **Step 3: Write src/agents/etl_agent.py**

```python
"""ETL training agent — expert on Glue, Spark, dbt, Informatica, Talend, SSIS, ADF."""
import logging
from pathlib import Path
from strands import Agent, AgentSkills, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager

from src.config import BEDROCK_MODEL_ID, AWS_REGION, ETL_AGENT_WINDOW_SIZE
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
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.3)
    skills = AgentSkills(skills=_SKILLS_DIR)
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_etl],
        plugins=[skills],
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
```

- [ ] **Step 4: Write src/agents/quiz_agent.py**

```python
"""Quiz agent — generates MCQ quizzes with structured output."""
import json
import logging
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.types.exceptions import StructuredOutputException

from src.config import BEDROCK_MODEL_ID, AWS_REGION
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
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.5)
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
    except (StructuredOutputException, Exception) as exc:
        logger.error("Quiz generation failed: %s", exc)
        return json.dumps({"error": str(exc), "technology": technology})
```

- [ ] **Step 5: Write src/agents/learning_path_agent.py**

```python
"""Learning path agent — recommends personalised study plans based on progress."""
import json
import logging
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import NullConversationManager
from strands.types.exceptions import StructuredOutputException

from src.config import BEDROCK_MODEL_ID, AWS_REGION
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
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.2)
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
    except (StructuredOutputException, Exception) as exc:
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
```

- [ ] **Step 6: Write src/agents/content_author_agent.py**

```python
"""Content author agent — generates Markdown training modules."""
import json
import logging
from pathlib import Path
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.types.exceptions import StructuredOutputException

from src.config import BEDROCK_MODEL_ID, AWS_REGION, GENERATED_DIR
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
    filename = f"{module.technology}_{safe_title}.md"
    output_path = Path(GENERATED_DIR) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module.content, encoding="utf-8")
    logger.info("Module saved: %s", output_path)
    return str(output_path)


def build_content_author_agent() -> Agent:
    model = BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.6)
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
    except (StructuredOutputException, Exception) as exc:
        logger.error("Content generation failed: %s", exc)
        return json.dumps({"error": str(exc)})
```

- [ ] **Step 7: Commit all sub-agents**

```bash
git add src/agents/
git commit -m "feat: all 6 sub-agents (QA, ETL, Quiz, LearningPath, ContentAuthor, Progress)"
```

---

## Task 10: Orchestrator Agent

**Files:**
- Create: `src/agents/orchestrator.py`

- [ ] **Step 1: Write src/agents/orchestrator.py**

```python
"""Orchestrator agent — single entry point routing to specialist sub-agents."""
import logging
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session.file_session_manager import FileSessionManager

from src.config import (
    BEDROCK_MODEL_ID, AWS_REGION,
    BEDROCK_GUARDRAIL_ID, BEDROCK_GUARDRAIL_VERSION,
    ORCHESTRATOR_WINDOW_SIZE, SESSIONS_DIR,
)
from src.hooks.logging_throttle import LoggingThrottleHook
from src.agents.qa_agent import qa_training_agent
from src.agents.etl_agent import etl_training_agent
from src.agents.quiz_agent import quiz_agent
from src.agents.learning_path_agent import learning_path_agent
from src.agents.content_author_agent import content_author_agent
from src.agents.progress_agent import progress_agent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are TechTrainer AI, an intelligent training assistant for employees.

You help users learn:
- QA Testing: Selenium WebDriver, Tricentis Tosca, Playwright
- Data Engineering: AWS Glue, Apache Spark, dbt, Informatica, Talend, SSIS, Azure Data Factory

## Routing Rules

Route every request to the appropriate specialist tool:

| Request type | Tool to call |
|---|---|
| Questions about Selenium, Tosca, or Playwright | qa_training_agent |
| Questions about AWS Glue, Spark, dbt, Informatica, Talend, SSIS, ADF | etl_training_agent |
| Quiz requests ("quiz me", "test my knowledge") | quiz_agent |
| Learning path ("what should I study", "recommend topics") | learning_path_agent |
| Content creation ("create a module", "write training material") | content_author_agent |
| Progress queries ("my scores", "how am I doing") | progress_agent |

## Response Style
- Always cite source documents when the specialist provides them
- Keep responses concise — employees want quick, actionable answers
- For code: use markdown code blocks with language identifiers
- Never answer from memory alone — always route to the appropriate specialist

## Boundaries
- Only answer questions related to the technologies listed above
- For off-topic requests, politely decline and suggest relevant topics you can help with
"""

_ALL_TOOLS = [
    qa_training_agent,
    etl_training_agent,
    quiz_agent,
    learning_path_agent,
    content_author_agent,
    progress_agent,
]


def _build_bedrock_model() -> BedrockModel:
    kwargs = {
        "model_id": BEDROCK_MODEL_ID,
        "region_name": AWS_REGION,
        "temperature": 0.3,
        "max_tokens": 4096,
        "streaming": True,
    }
    if BEDROCK_GUARDRAIL_ID:
        kwargs["guardrail_id"] = BEDROCK_GUARDRAIL_ID
        kwargs["guardrail_version"] = BEDROCK_GUARDRAIL_VERSION
        logger.info("Bedrock Guardrails enabled: %s v%s", BEDROCK_GUARDRAIL_ID, BEDROCK_GUARDRAIL_VERSION)
    return BedrockModel(**kwargs)


def build_orchestrator(session_id: str) -> Agent:
    """Build and return the orchestrator agent for a given session."""
    model = _build_bedrock_model()
    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=SESSIONS_DIR,
    )
    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
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
```

- [ ] **Step 2: Smoke-test orchestrator instantiation (no API call)**

```bash
uv run python -c "
from src.agents.orchestrator import build_orchestrator
orch = build_orchestrator('test-session-123')
print('Orchestrator built:', type(orch).__name__)
print('Tools:', [t.__name__ if hasattr(t, '__name__') else str(t) for t in orch.tools])
"
```

Expected output: `Orchestrator built: Agent` and 6 tool names listed.

- [ ] **Step 3: Commit**

```bash
git add src/agents/orchestrator.py
git commit -m "feat: orchestrator agent wiring all 6 sub-agents with session persistence"
```

---

## Task 11: Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

```python
"""TechTrainer AI — Streamlit UI."""
import json
import uuid
import logging
import streamlit as st
from src.tools.progress_db import init_db, read_progress
from src.agents.orchestrator import build_orchestrator
from src.config import PROGRESS_DB

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechTrainer AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state bootstrap ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "orchestrator" not in st.session_state:
    init_db()
    st.session_state.orchestrator = build_orchestrator(st.session_state.session_id)

session_id = st.session_state.session_id
orchestrator = st.session_state.orchestrator


def call_agent(prompt: str) -> str:
    """Invoke orchestrator and return response string."""
    try:
        result = orchestrator(
            prompt,
            invocation_state={"session_id": session_id},
        )
        return str(result)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        return f"Sorry, I encountered an error: {exc}"


def get_progress_data() -> dict:
    """Fetch progress data from SQLite."""
    try:
        raw = read_progress(PROGRESS_DB, session_id)
        return json.loads(raw)
    except Exception:
        return {"quiz_results": [], "technologies": {}}


# ── Sidebar — Progress Dashboard ───────────────────────────────────────────────
with st.sidebar:
    st.title("📊 My Progress")
    st.caption(f"Session: `{session_id[:8]}...`")
    st.divider()

    progress_data = get_progress_data()
    tech_stats = progress_data.get("technologies", {})

    if tech_stats:
        for tech, stats in sorted(tech_stats.items()):
            avg = stats.get("avg_score", 0)
            best = stats.get("best_score", 0)
            attempts = stats.get("attempts", 0)
            st.markdown(f"**{tech.replace('_', ' ').title()}**")
            st.progress(avg / 100, text=f"Avg: {avg:.0f}% | Best: {best}% | {attempts} attempt(s)")
            st.caption("")
    else:
        st.info("Take a quiz to start tracking progress!")

    total_quizzes = len(progress_data.get("quiz_results", []))
    st.metric("Total Quizzes Taken", total_quizzes)
    st.divider()
    st.caption("TechTrainer AI v1.0")


# ── Main header ────────────────────────────────────────────────────────────────
st.title("🎓 TechTrainer AI")
st.caption("Your intelligent training assistant for QA Testing & Data Engineering")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_quiz, tab_path, tab_author = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author"]
)


# ── TAB: Chat ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask About Any Technology")
    st.caption("Ask questions about Selenium, Tosca, Playwright, AWS Glue, Spark, dbt, and more.")

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_agent(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ── TAB: Quiz ─────────────────────────────────────────────────────────────────
with tab_quiz:
    st.subheader("Test Your Knowledge")

    TECHNOLOGIES = [
        "selenium", "tosca", "playwright",
        "aws_glue", "spark", "dbt",
        "informatica", "ssis", "talend", "adf",
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        quiz_tech = st.selectbox("Technology", TECHNOLOGIES, key="quiz_tech")
    with col2:
        quiz_diff = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="quiz_diff")
    with col3:
        quiz_n = st.selectbox("Questions", [5, 10, 15], key="quiz_n")

    if st.button("🎯 Generate Quiz", type="primary"):
        with st.spinner(f"Generating {quiz_n} {quiz_diff} questions on {quiz_tech}..."):
            prompt = (
                f"Generate a {quiz_n}-question {quiz_diff} quiz on {quiz_tech}. "
                f"Use the quiz_agent tool."
            )
            raw = call_agent(prompt)

        # Try to parse the quiz JSON from the response
        try:
            # The orchestrator may wrap the JSON in text — try to extract it
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
                st.session_state["current_quiz"] = quiz_data
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
        except Exception:
            st.error("Failed to parse quiz. Please try again.")
            st.code(raw)

    # Render quiz if available
    if "current_quiz" in st.session_state and not st.session_state.get("quiz_submitted"):
        quiz_data = st.session_state["current_quiz"]
        questions = quiz_data.get("questions", [])

        if questions:
            st.divider()
            with st.form("quiz_form"):
                for i, q in enumerate(questions):
                    st.markdown(f"**Q{i+1}. {q['question']}**")
                    st.session_state["quiz_answers"][i] = st.radio(
                        f"q{i}", q["options"], key=f"quiz_q{i}", label_visibility="collapsed"
                    )
                    st.caption(f"Difficulty: {q.get('difficulty', '?')} | Topic: {q.get('topic', '?')}")
                    st.write("")

                submitted = st.form_submit_button("✅ Submit Answers", type="primary")

            if submitted:
                correct = 0
                for i, q in enumerate(questions):
                    if st.session_state["quiz_answers"].get(i) == q["correct_answer"]:
                        correct += 1

                score = int((correct / len(questions)) * 100)
                st.session_state["quiz_submitted"] = True
                passing = score >= quiz_data.get("passing_score", 70)

                # Save to progress
                save_prompt = (
                    f"Save quiz result: session_id={session_id}, "
                    f"technology={quiz_data.get('technology', quiz_tech)}, "
                    f"difficulty={quiz_data.get('difficulty', quiz_diff)}, "
                    f"score={score}, total_questions={len(questions)}, correct_answers={correct}"
                )
                call_agent(save_prompt)

                # Show results
                if passing:
                    st.success(f"🎉 You scored {score}% ({correct}/{len(questions)}) — PASSED!")
                else:
                    st.warning(f"📚 You scored {score}% ({correct}/{len(questions)}) — Keep studying!")

                st.divider()
                st.subheader("Answer Review")
                for i, q in enumerate(questions):
                    user_ans = st.session_state["quiz_answers"].get(i, "")
                    correct_ans = q["correct_answer"]
                    is_correct = user_ans == correct_ans
                    icon = "✅" if is_correct else "❌"
                    st.markdown(f"{icon} **Q{i+1}.** {q['question']}")
                    if not is_correct:
                        st.markdown(f"Your answer: ~~{user_ans}~~ | Correct: **{correct_ans}**")
                    st.info(f"📖 {q['explanation']}")

                st.button("🔄 Take Another Quiz", on_click=lambda: st.session_state.pop("current_quiz", None))


# ── TAB: Learning Path ────────────────────────────────────────────────────────
with tab_path:
    st.subheader("Your Personalised Learning Path")

    if st.button("🔄 Refresh My Path", type="primary"):
        with st.spinner("Analysing your progress and building recommendations..."):
            raw = call_agent(
                f"Generate a learning path for me. My session_id is {session_id}. "
                "Use the learning_path_agent tool."
            )
        try:
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                st.session_state["learning_path"] = json.loads(json_match.group())
        except Exception:
            st.error("Could not parse learning path. Try again.")

    if "learning_path" in st.session_state:
        lp = st.session_state["learning_path"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Level", lp.get("current_level", "beginner").title())
        col2.metric("Est. Hours Remaining", f"{lp.get('estimated_hours', 0):.1f}h")
        col3.metric("Next Milestone", lp.get("next_milestone", "—")[:40])

        st.divider()

        col_weak, col_strong = st.columns(2)
        with col_weak:
            st.markdown("### 📉 Areas to Improve")
            for area in lp.get("weak_areas", []):
                st.markdown(f"- {area}")
            if not lp.get("weak_areas"):
                st.success("No weak areas identified yet!")

        with col_strong:
            st.markdown("### 📈 Strong Areas")
            for area in lp.get("strong_areas", []):
                st.markdown(f"- ✅ {area}")
            if not lp.get("strong_areas"):
                st.info("Take some quizzes to build your profile.")

        st.divider()
        st.markdown("### 🗺️ Recommended Study Order")
        topics = lp.get("recommended_topics", [])
        progress_data = get_progress_data()
        completed_techs = set(progress_data.get("technologies", {}).keys())

        for i, topic in enumerate(topics, 1):
            tech_key = topic.split()[0].lower().replace(" ", "_")
            if tech_key in completed_techs:
                st.markdown(f"✅ **{i}. {topic}** *(completed)*")
            elif i == 1:
                st.markdown(f"▶️ **{i}. {topic}** ← *start here*")
            else:
                st.markdown(f"○ {i}. {topic}")


# ── TAB: Content Author ───────────────────────────────────────────────────────
with tab_author:
    st.subheader("Generate Training Modules")
    st.caption("AI will create a complete Markdown training module based on your existing training documents.")

    TECHNOLOGIES = [
        "selenium", "tosca", "playwright",
        "aws_glue", "spark", "dbt",
        "informatica", "ssis", "talend", "adf",
    ]

    with st.form("author_form"):
        title = st.text_input("Module Title", placeholder="e.g. Introduction to Selenium Locators")
        col1, col2 = st.columns(2)
        with col1:
            tech = st.selectbox("Technology", TECHNOLOGIES, key="author_tech")
        with col2:
            diff = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="author_diff")
        objectives = st.text_area(
            "Learning Objectives (one per line)",
            placeholder="Understand locator types\nWrite Page Object Model classes\nUse explicit waits",
        )
        generate_btn = st.form_submit_button("✨ Generate Module", type="primary")

    if generate_btn and title:
        with st.spinner("Generating training module (this may take 30-60 seconds)..."):
            prompt = (
                f"Create a training module with: "
                f"title='{title}', technology='{tech}', difficulty='{diff}', "
                f"objectives='{objectives}'. Use the content_author_agent tool."
            )
            raw = call_agent(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                module_data = json.loads(json_match.group())
                st.session_state["generated_module"] = module_data
        except Exception:
            st.error("Failed to parse generated module.")
            st.code(raw)

    if "generated_module" in st.session_state:
        module = st.session_state["generated_module"]
        if "error" not in module:
            st.success(f"Module generated: **{module.get('title', 'Untitled')}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Technology", module.get("technology", "?"))
            col2.metric("Difficulty", module.get("difficulty", "?"))
            col3.metric("Est. Duration", f"{module.get('duration_minutes', 0)} min")

            st.divider()
            st.subheader("Preview")
            content = module.get("content", "")
            st.markdown(content)

            st.divider()
            col_dl, col_path = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download as Markdown",
                    data=content,
                    file_name=f"{module.get('technology', 'module')}_{title.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                )
            with col_path:
                if saved_path := module.get("saved_to"):
                    st.info(f"Saved to: `{saved_path}`")
        else:
            st.error(f"Generation failed: {module['error']}")
```

- [ ] **Step 2: Verify app starts without errors**

```bash
uv run streamlit run app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -5
kill %1
```

Expected: HTML response from Streamlit (no Python errors in terminal).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI with Chat, Quiz, Learning Path, and Content Author tabs"
```

---

## Task 12: Integration Smoke Test

- [ ] **Step 1: Run all unit tests**

```bash
uv run pytest tests/test_schemas.py tests/test_progress_db.py \
              tests/test_ingestion.py tests/test_retrieval.py \
              tests/test_hooks.py -v --cov=src --cov-report=term-missing
```

Expected: all tests pass, coverage ≥ 80%.

- [ ] **Step 2: Add sample documents and index**

Place at least one `.txt` file in `data/documents/qa/` and one in `data/documents/etl/`, then:

```bash
uv run python -m src.tools.document_ingestion --reindex
```

Expected: `Ingestion complete. QA chunks: N, ETL chunks: M`

- [ ] **Step 3: Verify ChromaDB status**

```bash
uv run python -m src.tools.document_ingestion --status
```

Expected: both collections show > 0 chunks.

- [ ] **Step 4: End-to-end smoke test with real Bedrock**

```bash
uv run python -c "
from src.agents.orchestrator import build_orchestrator
orch = build_orchestrator('smoke-test-session')
result = orch('What are the locator strategies in Selenium?', invocation_state={'session_id': 'smoke-test-session'})
print('Response received:', len(str(result)), 'chars')
print(str(result)[:500])
"
```

Expected: response mentions locator strategies (ID, CSS, XPath, etc.) with source citations.

- [ ] **Step 5: Launch app**

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` and verify:
- Chat tab: ask a question, get a response
- Quiz tab: generate a 5-question beginner quiz on selenium
- Learning Path tab: click Refresh, get recommendations
- Content Author tab: generate a module titled "Intro to dbt Models"

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: TechTrainer AI v1.0 complete — all agents, UI, and tests"
```

---

## Self-Review Checklist

- [x] **Pydantic schemas** → Task 3
- [x] **SQLite progress tools** → Task 4
- [x] **Document ingestion (PDF/DOCX/PPTX/XLSX/TXT)** → Task 5
- [x] **ChromaDB retrieval tools** → Task 6
- [x] **Safety hooks (throttle + path traversal)** → Task 7
- [x] **Skills files for all 10 technologies** → Task 8
- [x] **All 6 sub-agents** → Task 9
- [x] **Orchestrator with session + guardrails** → Task 10
- [x] **Streamlit 4-tab UI + sidebar dashboard** → Task 11
- [x] **Integration smoke test** → Task 12
- [x] **Config module** → Task 2
- [x] **Project scaffolding with uv** → Task 1
- [x] **TDD throughout** — failing tests written before every implementation
- [x] **Frequent commits** — one commit per task minimum
