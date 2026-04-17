<!--
  Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
  Date   : 26 March 2026
-->
# TechTrainer AI — Design Specification
**Date:** 2026-03-28
**Project:** TechTrainer AI | Madurai | Applied AI
**Author:** B.Vignesh Kumar aka bravetux <ic19939@gmail.com>
**Status:** Approved

---

## 1. Overview

TechTrainer AI is a production-grade employee training chatbot built on the **Strands Agents SDK** with **Amazon Bedrock** as the LLM backend and **Streamlit** as the frontend. It enables employees to access, navigate, and interact with training materials across QA testing technologies (Tosca, Selenium, Playwright) and data engineering tools (Informatica, Talend, SSIS, AWS Glue, Azure Data Factory, Apache Spark, dbt).

### Goals
- Instant Q&A over all training documents (PDF, DOCX, XLSX, PPTX, TXT)
- Interactive quizzes with scoring and explanations
- Personalized learning path recommendations based on progress
- AI-generated training module authoring
- Progress tracking with a visual dashboard

### Non-Goals
- User authentication (open access on internal network)
- Cloud file storage (local filesystem only)
- Video content processing (text documents only in v1)
- Mobile app (web UI only)

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │   Chat   │ │   Quiz   │ │Learning Path │ │Content Author │  │
│  │   Tab    │ │   Tab    │ │    Tab       │ │    Tab        │  │
│  └──────────┘ └──────────┘ └──────────────┘ └───────────────┘  │
│                    Progress Dashboard (sidebar)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ stream_async()
┌────────────────────────────▼────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                             │
│  model: BedrockModel (Claude Sonnet 4)                          │
│  conversation: SlidingWindowConversationManager(window=10)       │
│  session: FileSessionManager (./data/sessions/)                 │
│  hooks: LoggingThrottleHook (logging + per-turn throttle)        │
│  retry: ModelRetryStrategy(max_attempts=3)                       │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
@tool      @tool      @tool      @tool      @tool
QA Agent  ETL Agent  Quiz Agent  LP Agent  Content Agent
   │          │          │          │          │
   └────┬─────┘          └────┬─────┘          ▼
        ▼                     ▼           ./data/generated/
  ChromaDB (qa)          SQLite DB
  ChromaDB (etl)         ./data/progress.db
        ▲
        │ ingestion pipeline
  ./data/documents/
  (PDF, DOCX, XLSX, PPTX, TXT)
```

### 2.2 Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Agent Framework | Strands Agents SDK (strands-agents) |
| LLM | Configurable via `provider_manager.py` — AWS Bedrock (default), Ollama, LM Studio, OpenRouter, Google Gemini, OpenAI, Custom |
| LLM Adapter | `BedrockModel` (Bedrock) / `LiteLLMModel` via `litellm` package (all others) |
| Embeddings | Configurable via `embedding_manager.py` — AWS Bedrock Titan (default), Ollama, OpenAI, Custom |
| Vector Store | ChromaDB (local, persistent) |
| Chat History | ChromaDB `chat_history` collection — last 50 Q&A exchanges, persisted across browser refreshes |
| Document Parsing | pypdf, python-docx, python-pptx, openpyxl |
| Progress Storage | SQLite (`./data/progress.db`) |
| Session Storage | Strands FileSessionManager (`./data/sessions/`) |
| Safety | Amazon Bedrock Guardrails (optional) + Hook-based throttling |
| Observability | Python logging + OpenTelemetry trace_attributes |
| Package Manager | uv |

---

## 3. Agent Design

### 3.1 Orchestrator Agent

**File:** `src/agents/orchestrator.py`

**Responsibility:** Single entry point for all user requests. Determines intent, routes to the appropriate specialist sub-agent, and streams the synthesized response back to the UI.

**Configuration:**
```python
Agent(
    model=BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        temperature=0.3,
        max_tokens=2048,
        guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),  # optional
        guardrail_version="1",
    ),
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    tools=[qa_training_agent, etl_training_agent, quiz_agent,
           learning_path_agent, content_author_agent, progress_agent],
    conversation_manager=SlidingWindowConversationManager(window_size=10),
    session_manager=FileSessionManager(session_id=session_id, storage_dir="./data/sessions"),
    hooks=[LoggingThrottleHook()],  # Guardrails configured on BedrockModel, not as a hook
    retry_strategy=ModelRetryStrategy(max_attempts=3),
    trace_attributes={"session.id": session_id, "app": "techtrainer"}
)
```

**System Prompt excerpt:**
```
You are TechTrainer AI, an intelligent training assistant for employees.
You help users learn QA testing tools (Tosca, Selenium, Playwright) and
data engineering technologies (Informatica, Talend, SSIS, AWS Glue,
Azure Data Factory, Apache Spark, dbt).

Route requests as follows:
- Training questions about QA/testing → qa_training_agent
- Training questions about ETL/data engineering → etl_training_agent
- Quiz requests → quiz_agent
- Learning path or progress questions → learning_path_agent
- Content creation requests → content_author_agent
- Score or progress queries → progress_agent

Always cite sources. Never fabricate content not found in training materials.
```

---

### 3.2 QA Training Agent

**File:** `src/agents/qa_agent.py`

**Responsibility:** Expert on Tosca, Selenium, Playwright, and general test automation best practices. Retrieves content from the `qa_training` ChromaDB collection.

**Strands Skills (AgentSkills plugin):**
- `skills/selenium.md` — WebDriver API, locators, waits, Page Object Model
- `skills/tosca.md` — Tricentis Tosca modules, TestCases, TestSuites, TBox
- `skills/playwright.md` — Playwright Python/JS, fixtures, tracing, CI integration

**Tools:** `retrieve_qa`, `http_request`
**Conversation Manager:** `SlidingWindowConversationManager(window_size=15)`

---

### 3.3 ETL Training Agent

**File:** `src/agents/etl_agent.py`

**Responsibility:** Expert on Informatica PowerCenter, Talend, SSIS, AWS Glue, Azure Data Factory, Apache Spark, and dbt.

**Strands Skills:**
- `skills/aws_glue.md` — Crawlers, Jobs, DynamicFrames, Bookmarks
- `skills/spark.md` — RDDs, DataFrames, transformations, SparkSQL
- `skills/dbt.md` — Models, tests, seeds, macros, lineage
- `skills/informatica.md` — Mappings, sessions, workflows, transformations
- `skills/ssis.md` — Control flow, data flow, packages, connection managers
- `skills/talend.md` — Jobs, components, tMap, metadata, repository
- `skills/adf.md` — Pipelines, activities, datasets, linked services, triggers

**Tools:** `retrieve_etl`, `http_request`
**Conversation Manager:** `SlidingWindowConversationManager(window_size=15)`

---

### 3.4 Quiz Agent

**File:** `src/agents/quiz_agent.py`

**Responsibility:** Generates multiple-choice quizzes on any technology, evaluates user answers, and returns structured `QuizResult` objects.

**Structured Output:**
```python
class QuizQuestion(BaseModel):
    question: str
    options: list[str]          # exactly 4 options
    correct_answer: str         # matches one of options exactly
    explanation: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    topic: str

class QuizResult(BaseModel):
    technology: str
    difficulty: str
    questions: list[QuizQuestion]
    total_questions: int
    passing_score: int          # percentage, default 70
```

**Tools:** `retrieve_qa`, `retrieve_etl`
**Conversation Manager:** `SummarizingConversationManager(summary_ratio=0.3)`

---

### 3.5 Learning Path Agent

**File:** `src/agents/learning_path_agent.py`

**Responsibility:** Reads user progress from SQLite and generates a personalized learning path recommendation.

**Structured Output:**
```python
class LearningPath(BaseModel):
    session_id: str
    current_level: str
    recommended_topics: list[str]   # ordered list
    next_milestone: str
    estimated_hours: float
    weak_areas: list[str]
    strong_areas: list[str]
```

**Tools:** `progress_reader`

---

### 3.6 Content Author Agent

**File:** `src/agents/content_author_agent.py`

**Responsibility:** Drafts new training modules in Markdown format. Uses existing training materials as context and reference.

**Structured Output:**
```python
class TrainingModule(BaseModel):
    title: str
    technology: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    duration_minutes: int
    learning_objectives: list[str]
    content: str                    # full Markdown body
    exercises: list[str]            # hands-on exercises
    references: list[str]           # source citations
```

**Tools:** `retrieve_qa`, `retrieve_etl`, `file_write`
**Conversation Manager:** `SummarizingConversationManager(summary_ratio=0.4)`

---

### 3.8 Sub-Agent Lifecycle Note

All sub-agents (QA, ETL, Quiz, LP, Content Author, Progress) are **stateless**. They are instantiated fresh inside their respective `@tool` wrapper function on each invocation. No `FileSessionManager` is attached to sub-agents — only the Orchestrator maintains cross-request session state. Sub-agents use `NullConversationManager` or a window manager scoped to that single invocation.

Session IDs are generated as `uuid4()` strings, stored in `st.session_state` on first load, and passed to the Orchestrator via `invocation_state={"session_id": session_id}`.

---

### 3.9 Progress Agent

**File:** `src/agents/progress_agent.py`

**Responsibility:** Reads and writes user quiz scores, topics studied, and learning streaks to SQLite.

**Tools:** `progress_reader`, `progress_writer`

**SQLite Schema:**
```sql
CREATE TABLE quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    technology TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    score INTEGER NOT NULL,        -- percentage 0-100
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topics_studied (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    technology TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Tools

### 4.1 Retrieval Tools

**File:** `src/tools/retrieval.py`

```python
@tool
def retrieve_qa(query: str, top_k: int = 5) -> str:
    """Retrieve relevant QA training content from the knowledge base.

    Args:
        query: The search query about QA testing technologies.
        top_k: Number of most relevant document chunks to retrieve.
    Returns:
        Formatted string of relevant content with source file citations.
    """

@tool
def retrieve_etl(query: str, top_k: int = 5) -> str:
    """Retrieve relevant ETL/data engineering training content from the knowledge base.

    Args:
        query: The search query about ETL or data engineering technologies.
        top_k: Number of most relevant document chunks to retrieve.
    Returns:
        Formatted string of relevant content with source file citations.
    """
```

### 4.2 Progress Tools

**File:** `src/tools/progress_db.py`

```python
@tool
def progress_reader(session_id: str) -> str:
    """Read all quiz results and study history for a session.

    Args:
        session_id: The user's browser session identifier.
    Returns:
        JSON string with quiz history, scores by technology, and study topics.
    """

@tool
def progress_writer(session_id: str, technology: str, difficulty: str,
                    score: int, total_questions: int, correct_answers: int) -> str:
    """Write a quiz result to the progress database.

    Args:
        session_id: The user's browser session identifier.
        technology: The technology tested (e.g. 'selenium', 'aws_glue').
        difficulty: Quiz difficulty level.
        score: Percentage score (0-100).
        total_questions: Total number of questions.
        correct_answers: Number of correct answers.
    Returns:
        Confirmation message with the saved record ID.
    """
```

---

## 5. Document Ingestion Pipeline

**File:** `src/tools/document_ingestion.py`

### Supported Formats
| Extension | Library | Notes |
|---|---|---|
| `.pdf` | pypdf | Extracts text per page |
| `.docx` | python-docx | Extracts paragraphs and tables |
| `.pptx` | python-pptx | Extracts slide text and notes |
| `.xlsx` | openpyxl | Extracts all sheets as text |
| `.txt`, `.md` | built-in | Direct read |

### Chunking Strategy
- Chunk size: **1000 tokens**
- Overlap: **200 tokens**
- Splitter: `RecursiveCharacterTextSplitter` (LangChain)
- Metadata per chunk: `source_file`, `page_number`, `collection`, `ingested_at`

### Embedding
- Model: `amazon.titan-embed-text-v2:0` via Bedrock `embed_text` API
- Dimension: 1024
- Called via `boto3` directly (not Strands tool — offline batch process)

### ChromaDB Collections
- `qa_training` — all files under `./data/documents/qa/`
- `etl_training` — all files under `./data/documents/etl/`

### Invocation
```bash
# Manual reindex
uv run python -m src.tools.document_ingestion --reindex

# Startup check (only indexes if collection is empty)
# Called automatically by app.py on startup
```

---

## 6. Safety and Security

### 6.1 Bedrock Guardrails (Optional)
When `BEDROCK_GUARDRAIL_ID` is set in `.env`:
- Content filtering: blocks toxicity, hate speech, off-topic queries
- PII detection: logs PII mentions (no employee PII expected but defense-in-depth)
- Topic policy: restricts agent to training/education topics only

### 6.2 Hook-Based Safety

**File:** `src/hooks/logging_throttle.py`

```python
class LoggingThrottleHook(HookProvider):
    """Logs all tool calls and enforces per-turn tool call limits."""

    MAX_TOOLS_PER_TURN = 10

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.check_and_log)
        registry.add_callback(AfterToolCallEvent, self.log_result)

    def check_and_log(self, event: BeforeToolCallEvent) -> None:
        # Throttle: cancel if over limit
        # Log: tool name, inputs (sanitized)
        # Validate: sub-agent file writes stay within ./data/generated/

    def log_result(self, event: AfterToolCallEvent) -> None:
        # Log: tool name, success/failure, duration
```

### 6.3 Timeout Safety
All sub-agent calls wrapped in a 60-second watchdog thread using `agent.cancel()`.

### 6.4 File Write Sandboxing
`content_author_agent` file writes are validated to stay within `./data/generated/` — path traversal attempts are blocked in the `file_write` tool wrapper.

---

## 7. Streamlit UI Design

**File:** `app.py`

### Layout
- **Sidebar:** Progress dashboard (per-technology scores as progress bars, study streak, session info)
- **Top:** App title + global technology filter dropdown
- **Main:** 4-tab interface

### Tab: Chat
- `st.chat_message` components for conversation history
- `st.chat_input` for user input
- Streaming via `stream_async()` with `st.write_stream`
- Expandable "Sources" section below each assistant message

### Tab: Quiz
- Form: Technology selector + Difficulty selector + Question count (5/10/15)
- "Generate Quiz" button → calls quiz agent
- Renders MCQ as radio buttons per question
- "Submit" button → scores and shows results with explanations
- Saves score to SQLite via progress agent

### Tab: Learning Path
- "Refresh My Path" button → calls learning path agent
- Renders: Current Level badge, Recommended Topics list, Progress timeline
- Visual roadmap: ✓ completed / → in-progress / ○ upcoming

### Tab: Content Author
- Form: Title, Technology, Difficulty, Learning Objectives (text area)
- "Generate Module" button → calls content author agent
- Live `st.markdown` preview of generated module
- Download button (`.md` file) + "Save to Library" button

---

## 8. Configuration

**File:** `src/config.py`
**Source:** `.env` file

```python
# Required
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_GUARDRAIL_ID=          # leave empty to disable guardrails
BEDROCK_GUARDRAIL_VERSION=1
CHROMA_PERSIST_DIR=./data/chroma
SESSIONS_DIR=./data/sessions
GENERATED_DIR=./data/generated
PROGRESS_DB=./data/progress.db
LOG_LEVEL=INFO
```

---

## 9. Observability

All agents initialized with `trace_attributes`:
```python
trace_attributes={
    "session.id": session_id,
    "app.name": "techtrainer-ai",
    "app.version": "1.0.0"
}
```

Python logging at `INFO` level covers:
- Document ingestion progress (files indexed, chunk counts)
- Agent invocations (which sub-agent, duration)
- Tool calls (name, inputs, outputs, duration)
- Errors and retries

---

## 10. Strands Features Coverage Matrix

| Strands Feature | Used In | Purpose |
|---|---|---|
| `Agent` | All 7 agents | Core agent loop |
| `@tool` decorator | 6 sub-agents + 4 custom tools | Tool definition |
| `BedrockModel` | All agents | LLM provider |
| `AgentSkills` plugin | QA Agent, ETL Agent | Domain skill loading |
| `SlidingWindowConversationManager` | Orchestrator, QA, ETL | Context management |
| `SummarizingConversationManager` | Quiz, Content Author | Long session compression |
| `FileSessionManager` | Orchestrator | Cross-request persistence |
| `structured_output_model` | Quiz, LP, Content Author agents | Typed outputs |
| `HookProvider` | Orchestrator | Logging, throttling, safety |
| `ModelRetryStrategy` | Orchestrator | Bedrock throttle handling |
| `stream_async()` | app.py → Orchestrator | Streaming UI responses |
| `agent.cancel()` | Sub-agent wrappers | Timeout safety |
| `trace_attributes` | All agents | Observability tagging |
| `invocation_state` | Orchestrator → all | Session ID propagation |
| `BeforeToolCallEvent` | LoggingThrottleHook | Pre-call validation |
| `AfterToolCallEvent` | LoggingThrottleHook | Post-call logging |
