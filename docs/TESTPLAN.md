# TechTrainer AI — Test Plan
**Date:** 2026-03-28
**Project:** AG-UC-0887 | Madurai | Applied AI
**Version:** 1.0

---

## 1. Test Objectives

- Verify all 7 agents respond correctly to their intended inputs
- Validate document ingestion pipeline handles all supported formats
- Confirm ChromaDB retrieval returns relevant results
- Verify SQLite progress tracking stores and reads correctly
- Validate Pydantic structured outputs are well-formed
- Confirm Streamlit UI renders all tabs without errors
- Verify safety hooks throttle tool calls as expected
- Validate session persistence across page refreshes

---

## 2. Test Scope

### In Scope
- All Strands agents (unit and integration)
- Document ingestion pipeline
- ChromaDB retrieval tools
- SQLite progress tools
- Pydantic schema validation
- Streamlit tab rendering
- Hook behavior (logging, throttling)
- Session management (FileSessionManager)
- End-to-end user flows per tab

### Out of Scope
- AWS Bedrock API internals
- ChromaDB internal vector math
- Streamlit framework internals
- Network-level performance testing

---

## 3. Test Environment

| Component | Value |
|---|---|
| Python | 3.10+ |
| Test Framework | pytest |
| Mocking | pytest-mock, unittest.mock |
| Coverage | pytest-cov (target: 80%+) |
| Async tests | pytest-asyncio |
| Eval SDK | strands-agents-evals |
| Test data | `tests/fixtures/` |
| AWS mocking | `moto` library for Bedrock calls |

---

## 4. Test Data

### 4.1 Sample Training Documents
Place in `tests/fixtures/documents/`:
```
tests/fixtures/documents/
├── qa/
│   ├── selenium_basics.txt       # ~500 words on Selenium WebDriver
│   ├── tosca_intro.pdf           # simple test PDF (use fpdf to generate)
│   └── playwright_guide.docx     # simple test DOCX
└── etl/
    ├── aws_glue_overview.txt     # ~500 words on AWS Glue
    ├── spark_dataframes.txt      # ~500 words on Spark
    └── dbt_models.txt            # ~500 words on dbt
```

### 4.2 Mock Bedrock Responses
`tests/fixtures/mock_responses.py` — pre-canned LLM responses for deterministic testing.

---

## 5. Unit Tests

### 5.1 Document Ingestion (`tests/test_ingestion.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| ING-001 | Parse `.txt` file | Returns chunked text with metadata |
| ING-002 | Parse `.pdf` file | Extracts text from all pages |
| ING-003 | Parse `.docx` file | Extracts all paragraphs |
| ING-004 | Parse `.pptx` file | Extracts all slide text |
| ING-005 | Parse `.xlsx` file | Extracts all sheet content |
| ING-006 | Parse unsupported format | Raises `UnsupportedFormatError` |
| ING-007 | Chunk text (1000 tokens, 200 overlap) | Chunks have correct size and overlap |
| ING-008 | Empty document | Returns empty list, no crash |
| ING-009 | Large document (100+ pages) | Processes without OOM error |
| ING-010 | Duplicate file ingestion | No duplicate chunks in ChromaDB |

### 5.2 Retrieval Tools (`tests/test_retrieval.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| RET-001 | `retrieve_qa("selenium locator strategies")` | Returns ≥1 relevant chunk |
| RET-002 | `retrieve_etl("aws glue crawler")` | Returns ≥1 relevant chunk |
| RET-003 | Query with no matching content | Returns empty result message (no hallucination) |
| RET-004 | `top_k=3` respected | Returns exactly 3 chunks |
| RET-005 | Results include source file names | Metadata `source_file` present in output |
| RET-006 | Cross-collection query isolation | `retrieve_qa` never returns ETL content |

### 5.3 Progress Tools (`tests/test_progress_db.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| PRG-001 | Write quiz result | Record saved to SQLite |
| PRG-002 | Read results for session | Returns correct records |
| PRG-003 | Read results for unknown session | Returns empty result (not error) |
| PRG-004 | Write multiple results | All records present on read |
| PRG-005 | Score boundary: 0% | Saved correctly |
| PRG-006 | Score boundary: 100% | Saved correctly |
| PRG-007 | Concurrent writes | No SQLite lock errors |

### 5.4 Pydantic Schemas (`tests/test_schemas.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| SCH-001 | Valid `QuizResult` | Parses without error |
| SCH-002 | `QuizResult` with wrong options count | Validation error raised |
| SCH-003 | Valid `TrainingModule` | Parses without error |
| SCH-004 | `TrainingModule` invalid difficulty | Validation error raised |
| SCH-005 | Valid `LearningPath` | Parses without error |
| SCH-006 | `LearningPath` negative hours | Validation error raised |

### 5.5 Hooks (`tests/test_hooks.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| HK-001 | Tool call under limit (5/10) | Call proceeds normally |
| HK-002 | Tool call at limit (10/10) | Call cancelled with message |
| HK-003 | Tool call over limit (11/10) | Call cancelled with message |
| HK-004 | File write to safe path | Allowed |
| HK-005 | File write with path traversal `../../etc/passwd` | Blocked, error returned |
| HK-006 | New turn resets tool count | Count resets to 0 |
| HK-007 | All tool calls logged | Log entries present for each call |

---

## 6. Integration Tests

### 6.1 Agent Integration (`tests/test_agents_integration.py`)

Uses mocked Bedrock responses to test agent wiring without real API calls.

| Test ID | Test Case | Expected Result |
|---|---|---|
| AGT-001 | QA Agent answers Selenium question | Answer references selenium content |
| AGT-002 | ETL Agent answers AWS Glue question | Answer references glue content |
| AGT-003 | Quiz Agent generates QuizResult | Valid `QuizResult` object returned |
| AGT-004 | Quiz Agent: 5 questions requested | Exactly 5 questions in result |
| AGT-005 | Learning Path Agent reads empty progress | Returns beginner-level path |
| AGT-006 | Learning Path Agent reads existing progress | Path reflects completed topics |
| AGT-007 | Content Author generates TrainingModule | Valid `TrainingModule` returned |
| AGT-008 | Content Author saves file | File exists in `./data/generated/` |
| AGT-009 | Progress Agent writes then reads | Data round-trips correctly |
| AGT-010 | Orchestrator routes QA query to QA Agent | `qa_training_agent` tool called |
| AGT-011 | Orchestrator routes ETL query to ETL Agent | `etl_training_agent` tool called |
| AGT-012 | Orchestrator routes quiz request to Quiz Agent | `quiz_agent` tool called |
| AGT-013 | Sub-agent timeout (>60s) | `agent.cancel()` called, graceful error |

### 6.2 Session Persistence (`tests/test_sessions.py`)

| Test ID | Test Case | Expected Result |
|---|---|---|
| SES-001 | Conversation saved after first message | Session file exists |
| SES-002 | Conversation restored on second message | History present in context |
| SES-003 | Different session IDs = isolated histories | No cross-contamination |
| SES-004 | Session file survives process restart | History still available |

---

## 7. Strands Evals SDK Tests

**File:** `tests/test_evals.py`

Uses `strands-agents-evals` for LLM-as-judge quality evaluation.

### 7.1 Test Cases

```python
test_cases = [
    Case(
        name="selenium-locator-question",
        input="What are the different types of locator strategies in Selenium?",
        expected_output="id, name, class, xpath, css selector, link text",
        metadata={"technology": "selenium", "difficulty": "beginner"}
    ),
    Case(
        name="tosca-testcase-question",
        input="How do you create a TestCase in Tricentis Tosca?",
        metadata={"technology": "tosca", "difficulty": "intermediate"}
    ),
    Case(
        name="aws-glue-crawler",
        input="What is an AWS Glue Crawler and when should I use one?",
        metadata={"technology": "aws_glue", "difficulty": "beginner"}
    ),
    Case(
        name="dbt-model-materializations",
        input="Explain the different dbt model materialization types",
        metadata={"technology": "dbt", "difficulty": "intermediate"}
    ),
    Case(
        name="spark-dataframe-transformation",
        input="How do I filter and select columns in a Spark DataFrame?",
        metadata={"technology": "spark", "difficulty": "beginner"}
    ),
    Case(
        name="quiz-generation",
        input="Generate a 3-question beginner quiz on Playwright",
        metadata={"feature": "quiz", "technology": "playwright"}
    ),
    Case(
        name="learning-path-empty",
        input="What should I study first as a complete beginner?",
        metadata={"feature": "learning_path"}
    ),
    Case(
        name="off-topic-rejection",
        input="What is the recipe for chocolate cake?",
        expected_output="I can only help with training materials",
        metadata={"safety": "off_topic"}
    ),
]
```

### 7.2 Evaluators

| Evaluator | Cases | Pass Threshold |
|---|---|---|
| `OutputEvaluator` | All Q&A cases | Score ≥ 0.7 |
| `HelpfulnessEvaluator` | All Q&A cases | Level ≥ 4 (of 7) |
| `TrajectoryEvaluator` | Tool-using cases | Correct tools used |
| `ToolSelectionAccuracyEvaluator` | Routing cases | Correct sub-agent called |
| `GoalSuccessRateEvaluator` | All cases | ≥ 80% pass rate |
| `FaithfulnessEvaluator` | Q&A cases | No hallucination vs. retrieved content |

### 7.3 Running Evals

```bash
# Run full eval suite
uv run pytest tests/test_evals.py -v --eval-report

# Run specific evaluator
uv run python -m tests.run_evals --evaluator output_quality

# Generate HTML report
uv run python -m tests.run_evals --report html
```

---

## 8. End-to-End Tests

**File:** `tests/test_e2e.py`

Uses Streamlit testing utilities (`streamlit.testing.v1`) and a real Bedrock connection (requires `.env`).

| Test ID | User Flow | Expected Result |
|---|---|---|
| E2E-001 | Open app → Chat tab loads | No errors, input visible |
| E2E-002 | Ask "What is Selenium WebDriver?" | Answer streamed, sources shown |
| E2E-003 | Open Quiz tab → Generate 5-question Selenium beginner quiz | 5 MCQ questions rendered |
| E2E-004 | Answer quiz → Submit | Score shown, explanations visible |
| E2E-005 | Open Learning Path tab → Refresh | Recommendations shown |
| E2E-006 | Open Content Author tab → Generate module | Markdown preview rendered |
| E2E-007 | Download generated module | `.md` file downloaded |
| E2E-008 | Sidebar shows quiz score after taking quiz | Score bar updated |
| E2E-009 | Refresh browser → Chat history persists | Previous messages visible |
| E2E-010 | Ask completely off-topic question | Polite refusal, no crash |

---

## 9. Performance Tests

| Test ID | Scenario | Target |
|---|---|---|
| PERF-001 | Chat Q&A response time (P95) | < 8 seconds |
| PERF-002 | Quiz generation (5 questions) | < 10 seconds |
| PERF-003 | Document ingestion (100 files) | < 5 minutes |
| PERF-004 | ChromaDB retrieval latency | < 500ms |
| PERF-005 | Concurrent users (5 simultaneous) | No errors |
| PERF-006 | Session file read/write | < 100ms |

---

## 10. Test Execution Plan

### Phase 1 — Unit Tests (pre-integration)
```bash
uv run pytest tests/test_ingestion.py tests/test_retrieval.py \
              tests/test_progress_db.py tests/test_schemas.py \
              tests/test_hooks.py -v --cov=src --cov-report=html
```

### Phase 2 — Integration Tests (mocked Bedrock)
```bash
uv run pytest tests/test_agents_integration.py tests/test_sessions.py \
              -v --mock-bedrock
```

### Phase 3 — Evals (requires real Bedrock)
```bash
uv run pytest tests/test_evals.py -v -s
```

### Phase 4 — End-to-End (requires real Bedrock + documents)
```bash
uv run pytest tests/test_e2e.py -v --slow
```

### Coverage Target
- Unit tests: **90%** coverage
- Integration tests: **80%** coverage
- Overall: **85%** coverage

---

## 11. Defect Tracking

| Severity | Definition | Response Time |
|---|---|---|
| Critical | App crashes, data loss, security breach | Fix immediately |
| High | Feature broken, wrong answers, data corruption | Fix before release |
| Medium | Degraded UX, slow responses, minor errors | Fix in next iteration |
| Low | Cosmetic issues, minor wording | Fix when convenient |
