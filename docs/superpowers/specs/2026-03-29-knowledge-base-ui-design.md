# Knowledge Base UI & Per-Technology RAG Design

> **For agentic workers:** This spec is ready for implementation. Use superpowers:writing-plans to produce the implementation plan.

**Goal:** Add a Knowledge Base management tab to TechTrainer AI that lets users upload training documents per technology topic, trigger ChromaDB indexing, and see live availability status — with availability flowing through to Chat, Quiz, Learning Path, and Content Author tabs.

**Architecture:** Replace two broad ChromaDB collections (`qa_training`, `etl_training`) with 10 per-technology collections plus dynamic support for user-defined custom topics persisted in a JSON registry. A new `kb_manager.py` tool handles all registry and ingestion orchestration. All dropdowns and agent system prompts are built dynamically from the registry at runtime.

**Tech Stack:** Streamlit `st.file_uploader`, ChromaDB PersistentClient (existing), `data/topics_registry.json` (new), existing Bedrock embedding pipeline.

---

## 1. Topic Model

### 1.1 Built-in Topics (hardcoded in `config.py`)

Ten fixed topics that cannot be deleted:

| id | display_name | collection | doc_dir |
|----|-------------|------------|---------|
| selenium | Selenium | tech_selenium | data/documents/selenium |
| tosca | Tosca | tech_tosca | data/documents/tosca |
| playwright | Playwright | tech_playwright | data/documents/playwright |
| aws_glue | AWS Glue | tech_aws_glue | data/documents/aws_glue |
| spark | Spark | tech_spark | data/documents/spark |
| dbt | dbt | tech_dbt | data/documents/dbt |
| informatica | Informatica | tech_informatica | data/documents/informatica |
| ssis | SSIS | tech_ssis | data/documents/ssis |
| talend | Talend | tech_talend | data/documents/talend |
| adf | Azure Data Factory | tech_adf | data/documents/adf |

### 1.2 Custom Topics (persisted in `data/topics_registry.json`)

```json
{
  "custom": [
    {
      "id": "kubernetes",
      "display_name": "Kubernetes",
      "description": "Container orchestration platform training material",
      "collection": "tech_kubernetes",
      "doc_dir": "data/documents/kubernetes",
      "created_at": "2026-03-29"
    }
  ]
}
```

- `id` is the lowercase, alphanumeric-only sanitised form of the display name (spaces → underscores).
- `collection` is always `tech_<id>`.
- `doc_dir` is always `data/documents/<id>`.
- Built-in topics are never written to the registry file.

### 1.3 Topic Status

Status is computed at runtime by querying ChromaDB — it is never stored:

| Status | Condition |
|--------|-----------|
| **AVAILABLE** | ChromaDB collection exists and `count() > 0` |
| **PENDING** | Files exist in `doc_dir` but collection is missing or `count() == 0` |
| **DISABLED** | No files in `doc_dir` AND collection missing or empty |

---

## 2. New & Modified Files

### 2.1 `src/tools/kb_manager.py` (new)

Responsibilities:
- `load_all_topics() -> list[dict]` — merges built-in list with registry file; returns unified list with runtime status attached.
- `create_custom_topic(display_name, description) -> dict` — validates name, sanitises id, creates doc_dir, appends to registry, returns the new topic dict. Raises `ValueError` if name collides with existing topic.
- `delete_custom_topic(topic_id)` — removes from registry and deletes the ChromaDB collection. Does not delete files from disk (user data). Raises `ValueError` for built-in topics.
- `save_uploaded_file(topic_id, filename, file_bytes) -> str` — writes bytes to `data/documents/<topic_id>/<filename>`, returns absolute path.
- `get_topic_status(topic_id) -> dict` — returns `{status, chunk_count, file_count, last_indexed}` for one topic.
- `list_topic_files(topic_id) -> list[str]` — returns filenames in the topic's doc_dir.

No `@tool` decorator — this module is called directly by `app.py`, not by agents.

### 2.2 `src/tools/document_ingestion.py` (modified)

Add:
```python
def index_technology(topic_id: str, reindex: bool = False) -> int:
    """Index a single technology's documents. Returns chunk count."""
```

This function looks up the topic's `collection` and `doc_dir` from config/registry, then calls the existing `index_directory(...)`. The existing `run_ingestion()` is updated to iterate over all built-in topics using `index_technology`.

### 2.3 `src/tools/retrieval.py` (modified)

Add:
```python
@tool
def retrieve_topic(query: str, topic_id: str, top_k: int = 5) -> str:
    """Retrieve training content from a specific topic collection.

    Args:
        query: Search query.
        topic_id: Topic identifier (e.g. 'kubernetes', 'selenium').
        top_k: Number of results to return.

    Returns:
        Formatted source blocks, or a 'no material' message if topic is disabled.
    """
```

Update `retrieve_qa` to query `tech_selenium`, `tech_tosca`, `tech_playwright` (merged, deduped by source_file).
Update `retrieve_etl` to query `tech_aws_glue`, `tech_spark`, `tech_dbt`, `tech_informatica`, `tech_ssis`, `tech_talend`, `tech_adf`.

### 2.4 `src/config.py` (modified)

Replace:
```python
CHROMA_QA_COLLECTION = "qa_training"
CHROMA_ETL_COLLECTION = "etl_training"
DOCUMENTS_QA_DIR = ...
DOCUMENTS_ETL_DIR = ...
```

With:
```python
BUILTIN_TOPICS: list[dict] = [
    {"id": "selenium",     "display_name": "Selenium",            "collection": "tech_selenium",    "doc_dir": "./data/documents/selenium"},
    {"id": "tosca",        "display_name": "Tosca",               "collection": "tech_tosca",       "doc_dir": "./data/documents/tosca"},
    {"id": "playwright",   "display_name": "Playwright",          "collection": "tech_playwright",  "doc_dir": "./data/documents/playwright"},
    {"id": "aws_glue",     "display_name": "AWS Glue",            "collection": "tech_aws_glue",    "doc_dir": "./data/documents/aws_glue"},
    {"id": "spark",        "display_name": "Spark",               "collection": "tech_spark",       "doc_dir": "./data/documents/spark"},
    {"id": "dbt",          "display_name": "dbt",                 "collection": "tech_dbt",         "doc_dir": "./data/documents/dbt"},
    {"id": "informatica",  "display_name": "Informatica",         "collection": "tech_informatica", "doc_dir": "./data/documents/informatica"},
    {"id": "ssis",         "display_name": "SSIS",                "collection": "tech_ssis",        "doc_dir": "./data/documents/ssis"},
    {"id": "talend",       "display_name": "Talend",              "collection": "tech_talend",      "doc_dir": "./data/documents/talend"},
    {"id": "adf",          "display_name": "Azure Data Factory",  "collection": "tech_adf",         "doc_dir": "./data/documents/adf"},
]
TOPICS_REGISTRY_PATH: str = os.getenv("TOPICS_REGISTRY_PATH", "./data/topics_registry.json")
QA_TOPIC_IDS: list[str] = ["selenium", "tosca", "playwright"]
ETL_TOPIC_IDS: list[str] = ["aws_glue", "spark", "dbt", "informatica", "ssis", "talend", "adf"]
```

All 10 doc dirs are created on import (replacing the old two-dir loop).

### 2.5 `src/agents/orchestrator.py` (modified)

`build_orchestrator(session_id)` calls `kb_manager.load_all_topics()` to get the available-topic list and injects it into the system prompt:

```
## Available Topics
The following topics have indexed training material and can be queried:
- Selenium (QA)
- AWS Glue (ETL)
- Kubernetes (custom)

Topics NOT available (no material loaded): Tosca, Playwright, dbt, ...
For unavailable topics, tell the user to visit the Knowledge Base tab.
```

`retrieve_topic` is added to `_ALL_TOOLS`.

### 2.6 `app.py` (modified)

- Add `📚 Knowledge Base` as the 5th tab.
- Quiz and Content Author technology dropdowns replaced with `get_available_topic_ids()` (Available only, loaded from `kb_manager`).
- Learning path prompt includes available topic list.

---

## 3. Knowledge Base Tab — UI Specification

### 3.1 Layout

```
[Manage Training Documents]                    [🔄 Refresh Status] [⚡ Index All]
Status chips: selenium ✓245  aws_glue ✓183  tosca ✗  playwright ⏳  ...

── Add Custom Topic ─────────────────────────────────────────────────
  Topic Name: [___________]  Description: [___________________]  [➕ Create]

── Built-in Technologies (10) ───────────────────────────────────────
  2-column grid of topic cards (see §3.2)

── Custom Topics (N) ────────────────────────────────────────────────
  2-column grid of topic cards + delete button per card (see §3.2)
```

### 3.2 Topic Card — states

**AVAILABLE card:**
- Green border, "✓ AVAILABLE" badge, chunk count + file count + last indexed date
- File list (filenames only)
- Upload zone (`st.file_uploader`, multiple files, all supported extensions)
- Auto-index toggle (`st.checkbox`, default ON)
- "⬆ Upload more" + "🔄 Re-index" buttons
- Delete button (custom topics only, red 🗑 icon)

**PENDING card:**
- Amber border, "⏳ PENDING" badge
- Warning: "Files found but not indexed. Click Index to make this topic available."
- Upload zone
- "⚡ Index now" button

**DISABLED card:**
- Red border, "✗ DISABLED" badge
- Warning: "No material uploaded. Upload documents to enable this topic."
- Upload zone (highlighted with amber dashes)
- Delete button (custom topics only)

### 3.3 Upload Behaviour

1. User drops files onto the upload zone for topic X.
2. `kb_manager.save_uploaded_file(topic_id, filename, bytes)` saves each file to `data/documents/<topic_id>/`.
3. If auto-index is ON: `index_technology(topic_id)` is called immediately with a `st.spinner`. On completion, `st.success("Indexed N chunks")` and status chip refreshes.
4. If auto-index is OFF: files are saved, card status updates to PENDING, "Index now" button appears.

### 3.4 Index All

Calls `index_technology(topic_id)` sequentially for every topic whose status is PENDING or AVAILABLE (re-indexes all that have files). Shows a progress bar with `st.progress`.

### 3.5 Delete Custom Topic

Confirmation dialog (`st.warning` + confirm button). On confirm:
- `kb_manager.delete_custom_topic(topic_id)` drops the ChromaDB collection.
- Files on disk are **not** deleted — user keeps their source documents.
- Topic card disappears from the UI on next refresh.

---

## 4. Status Propagation to Other Tabs

### 4.1 Quiz Tab
Technology dropdown: `st.selectbox("Technology", get_available_topic_ids())` — Available topics only. If no topics are available, show an info message directing to Knowledge Base tab.

### 4.2 Content Author Tab
Same dynamic dropdown as Quiz.

### 4.3 Learning Path Tab
`learning_path_agent` prompt includes: `"Available topics: {available_ids}. Only recommend topics from this list."`

### 4.4 Chat Tab
Orchestrator system prompt (built at startup) lists available and unavailable topics. For unavailable topic queries, the agent responds:
> "⚠ [Topic] training material hasn't been loaded yet. Go to the 📚 Knowledge Base tab and upload [Topic] documents to enable this topic."

---

## 5. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Duplicate custom topic name | `st.error("Topic '[name]' already exists.")` |
| Invalid topic name characters | `st.error("Only letters, numbers and spaces allowed.")` |
| Upload of unsupported file type | `st.warning("Skipped [file]: unsupported format.")` |
| Indexing fails for one file | Log error, continue with remaining files, show warning summary |
| ChromaDB unavailable | `st.error("Knowledge base unavailable. Check data/chroma/ directory.")` |
| Delete built-in topic attempted | Button not shown for built-in topics (UI-level guard only) |

---

## 6. Files Not Changed

- `src/agents/qa_agent.py` — tools updated via retrieval.py changes, no agent code changes
- `src/agents/etl_agent.py` — same
- `src/agents/quiz_agent.py` — technology comes from orchestrator call, no changes
- `src/agents/learning_path_agent.py` — prompt updated at call time, no agent code changes
- `src/agents/content_author_agent.py` — no changes
- `src/agents/progress_agent.py` — no changes
- `src/hooks/logging_throttle.py` — no changes
- `src/models/schemas.py` — no changes
- All test files (new tests added, none removed)
