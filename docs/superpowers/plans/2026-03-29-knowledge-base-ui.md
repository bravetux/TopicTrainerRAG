# Knowledge Base UI & Per-Technology RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Knowledge Base management tab to TechTrainer AI that lets users upload documents per-technology topic, trigger ChromaDB indexing, manage custom topics, and see live availability status propagated to Chat, Quiz, Learning Path, and Content Author tabs.

**Architecture:** Replace two broad ChromaDB collections with 10 per-technology collections plus a JSON registry for custom topics. A new `kb_manager.py` module handles all registry CRUD and status checks. All agent-facing dropdowns and the orchestrator system prompt are built dynamically from the registry at runtime.

**Tech Stack:** Streamlit `st.file_uploader`, ChromaDB PersistentClient (existing), `data/topics_registry.json` (new), Bedrock Titan embeddings (existing).

---

## File Map

| File | Change |
|------|--------|
| `src/config.py` | Remove 4 old constants; add `BUILTIN_TOPICS`, `QA_TOPIC_IDS`, `ETL_TOPIC_IDS`, `TOPICS_REGISTRY_PATH` |
| `src/tools/kb_manager.py` | **New** — topic registry CRUD, status checks, file-save helper |
| `src/tools/document_ingestion.py` | Add `index_technology(topic_id, reindex)`; update `run_ingestion` and `check_status` |
| `src/tools/retrieval.py` | Add `query_multi_collections`; update `retrieve_qa`/`retrieve_etl`; add `retrieve_topic` tool |
| `src/agents/orchestrator.py` | Dynamic system prompt; add `retrieve_topic` to tools |
| `app.py` | Add 5th `📚 Knowledge Base` tab; helper functions; dynamic dropdowns in Quiz/Content Author/Learning Path |
| `tests/test_kb_manager.py` | **New** — 9 unit tests for kb_manager |
| `tests/test_retrieval.py` | Add 3 tests for `query_multi_collections` and `retrieve_topic` |

---

### Task 1: Update config.py — BUILTIN_TOPICS and topic constants

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Replace the old collection constants and directory setup**

Replace the entire `src/config.py` with:

```python
"""Central configuration loaded from environment variables."""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── AWS / Bedrock ──────────────────────────────────────────────────────────────
AWS_REGION: str = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
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
TOPICS_REGISTRY_PATH: str = os.getenv("TOPICS_REGISTRY_PATH", "./data/topics_registry.json")

# ── Built-in Topics ────────────────────────────────────────────────────────────
BUILTIN_TOPICS: list[dict] = [
    {"id": "selenium",    "display_name": "Selenium",           "collection": "tech_selenium",    "doc_dir": "./data/documents/selenium"},
    {"id": "tosca",       "display_name": "Tosca",              "collection": "tech_tosca",       "doc_dir": "./data/documents/tosca"},
    {"id": "playwright",  "display_name": "Playwright",         "collection": "tech_playwright",  "doc_dir": "./data/documents/playwright"},
    {"id": "aws_glue",    "display_name": "AWS Glue",           "collection": "tech_aws_glue",    "doc_dir": "./data/documents/aws_glue"},
    {"id": "spark",       "display_name": "Spark",              "collection": "tech_spark",       "doc_dir": "./data/documents/spark"},
    {"id": "dbt",         "display_name": "dbt",                "collection": "tech_dbt",         "doc_dir": "./data/documents/dbt"},
    {"id": "informatica", "display_name": "Informatica",        "collection": "tech_informatica", "doc_dir": "./data/documents/informatica"},
    {"id": "ssis",        "display_name": "SSIS",               "collection": "tech_ssis",        "doc_dir": "./data/documents/ssis"},
    {"id": "talend",      "display_name": "Talend",             "collection": "tech_talend",      "doc_dir": "./data/documents/talend"},
    {"id": "adf",         "display_name": "Azure Data Factory", "collection": "tech_adf",         "doc_dir": "./data/documents/adf"},
]

QA_TOPIC_IDS: list[str] = ["selenium", "tosca", "playwright"]
ETL_TOPIC_IDS: list[str] = ["aws_glue", "spark", "dbt", "informatica", "ssis", "talend", "adf"]

# ── Agent Settings ─────────────────────────────────────────────────────────────
ORCHESTRATOR_WINDOW_SIZE: int = 10
QA_AGENT_WINDOW_SIZE: int = 15
ETL_AGENT_WINDOW_SIZE: int = 15
MAX_TOOLS_PER_TURN: int = 10
SUBAGENT_TIMEOUT_SECONDS: int = 60
AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
AGENT_TOP_P: float = float(os.getenv("AGENT_TOP_P", "0.9"))
AGENT_MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))

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
for _dir in [SESSIONS_DIR, GENERATED_DIR, CHROMA_PERSIST_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)

for _topic in BUILTIN_TOPICS:
    Path(_topic["doc_dir"]).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Verify imports still work**

Run: `uv run python -c "from src.config import BUILTIN_TOPICS, QA_TOPIC_IDS, ETL_TOPIC_IDS, TOPICS_REGISTRY_PATH; print(len(BUILTIN_TOPICS), 'topics loaded')"`
Expected: `10 topics loaded`

- [ ] **Step 3: Run existing test suite to catch any breakage from removed constants**

Run: `uv run pytest tests/ -x -q`
Expected: Some tests may fail if they import the removed constants (`CHROMA_QA_COLLECTION`, `CHROMA_ETL_COLLECTION`, `DOCUMENTS_QA_DIR`, `DOCUMENTS_ETL_DIR`). Fix any such imports in test files to use `BUILTIN_TOPICS` instead before proceeding.

- [ ] **Step 4: Commit**

```bash
git add src/config.py
git commit -m "refactor(config): replace 2 broad collections with BUILTIN_TOPICS list for per-tech RAG"
```

---

### Task 2: Write failing tests for kb_manager

**Files:**
- Create: `tests/test_kb_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kb_manager.py`:

```python
"""Tests for Knowledge Base manager."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSanitiseId:
    def test_spaces_become_underscores(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("My Topic") == "my_topic"

    def test_special_chars_removed(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("C# & .NET") == "c_net"

    def test_already_clean(self):
        from src.tools.kb_manager import _sanitise_id
        assert _sanitise_id("kubernetes") == "kubernetes"


class TestRegistryIO:
    def test_load_empty_registry(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        result = kb_manager._load_registry()
        assert result == {"custom": []}

    def test_save_and_reload(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        reg_path = str(tmp_path / "reg.json")
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", reg_path)
        data = {"custom": [{"id": "test", "display_name": "Test"}]}
        kb_manager._save_registry(data)
        assert kb_manager._load_registry() == data


class TestCreateCustomTopic:
    def test_creates_topic_in_registry(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        topic = kb_manager.create_custom_topic("Kubernetes", "K8s training")
        assert topic["id"] == "kubernetes"
        assert topic["collection"] == "tech_kubernetes"
        assert topic["display_name"] == "Kubernetes"
        registry = kb_manager._load_registry()
        assert any(t["id"] == "kubernetes" for t in registry["custom"])

    def test_duplicate_raises(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        kb_manager.create_custom_topic("Kubernetes")
        with pytest.raises(ValueError, match="already exists"):
            kb_manager.create_custom_topic("Kubernetes")

    def test_builtin_name_collision_raises(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            kb_manager.create_custom_topic("Selenium")


class TestDeleteCustomTopic:
    def test_delete_removes_from_registry(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.chdir(tmp_path)
        # Patch chroma so no real DB is hit
        mock_chroma = MagicMock()
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_custom_topic("jenkins")
        registry = kb_manager._load_registry()
        assert not any(t["id"] == "jenkins" for t in registry["custom"])

    def test_delete_builtin_raises(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        with pytest.raises(ValueError, match="built-in"):
            kb_manager.delete_custom_topic("selenium")


class TestSaveUploadedFile:
    def test_saves_file_to_topic_dir(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        # Patch BUILTIN_TOPICS to use tmp_path dir
        fake_topic = {
            "id": "selenium", "display_name": "Selenium",
            "collection": "tech_selenium",
            "doc_dir": str(tmp_path / "selenium"),
        }
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [fake_topic])
        saved = kb_manager.save_uploaded_file("selenium", "guide.pdf", b"pdf content")
        assert Path(saved).exists()
        assert Path(saved).read_bytes() == b"pdf content"

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        fake_topic = {
            "id": "selenium", "display_name": "Selenium",
            "collection": "tech_selenium",
            "doc_dir": str(tmp_path / "selenium"),
        }
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [fake_topic])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        # Path traversal attempt — should save as literal filename, not navigate up
        saved = kb_manager.save_uploaded_file("selenium", "../../evil.pdf", b"bad")
        assert ".." not in saved
        assert Path(saved).parent == (tmp_path / "selenium").resolve() or \
               str(tmp_path / "selenium") in saved
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_kb_manager.py -v`
Expected: `ImportError: cannot import name '_sanitise_id' from 'src.tools.kb_manager'` — file doesn't exist yet.

---

### Task 3: Implement src/tools/kb_manager.py

**Files:**
- Create: `src/tools/kb_manager.py`

- [ ] **Step 1: Write the implementation**

Create `src/tools/kb_manager.py`:

```python
"""Knowledge base manager — topic registry CRUD, status checks, file persistence."""
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.config import BUILTIN_TOPICS, TOPICS_REGISTRY_PATH, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _sanitise_id(name: str) -> str:
    """Convert display name to a valid topic id (lowercase, alphanumeric, underscores)."""
    sanitised = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    return sanitised.strip("_")


def _load_registry() -> dict:
    """Load custom topics from the registry JSON file."""
    path = Path(TOPICS_REGISTRY_PATH)
    if not path.exists():
        return {"custom": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read topics registry; returning empty.")
        return {"custom": []}


def _save_registry(data: dict) -> None:
    """Persist the custom topics registry to disk."""
    path = Path(TOPICS_REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _find_topic(topic_id: str) -> Optional[dict]:
    """Return the topic dict for a given id, searching built-ins then custom."""
    for t in BUILTIN_TOPICS:
        if t["id"] == topic_id:
            return t
    registry = _load_registry()
    for t in registry.get("custom", []):
        if t["id"] == topic_id:
            return t
    return None


def get_topic_status(topic_id: str) -> dict:
    """Return runtime status for one topic: AVAILABLE, PENDING, or DISABLED."""
    topic = _find_topic(topic_id)
    if topic is None:
        return {"status": "DISABLED", "chunk_count": 0, "file_count": 0}

    doc_dir = Path(topic["doc_dir"])
    file_count = sum(1 for f in doc_dir.iterdir() if f.is_file()) if doc_dir.exists() else 0

    chunk_count = 0
    try:
        col = _get_chroma().get_collection(topic["collection"])
        chunk_count = col.count()
    except Exception:
        pass

    if chunk_count > 0:
        status = "AVAILABLE"
    elif file_count > 0:
        status = "PENDING"
    else:
        status = "DISABLED"

    return {"status": status, "chunk_count": chunk_count, "file_count": file_count}


def load_all_topics() -> list[dict]:
    """Return all topics (built-in + custom) with runtime status attached."""
    registry = _load_registry()
    custom = registry.get("custom", [])

    all_topics = []
    for t in BUILTIN_TOPICS:
        topic = dict(t)
        topic["is_builtin"] = True
        topic.update(get_topic_status(t["id"]))
        all_topics.append(topic)

    for t in custom:
        topic = dict(t)
        topic["is_builtin"] = False
        topic.update(get_topic_status(t["id"]))
        all_topics.append(topic)

    return all_topics


def get_available_topic_ids() -> list[str]:
    """Return topic ids whose status is AVAILABLE."""
    return [t["id"] for t in load_all_topics() if t["status"] == "AVAILABLE"]


def get_available_topics() -> list[dict]:
    """Return full topic dicts for topics whose status is AVAILABLE."""
    return [t for t in load_all_topics() if t["status"] == "AVAILABLE"]


def create_custom_topic(display_name: str, description: str = "") -> dict:
    """Create a new custom topic. Returns the new topic dict.

    Raises ValueError if the name collides with an existing built-in or custom topic.
    """
    topic_id = _sanitise_id(display_name)
    if not topic_id:
        raise ValueError("Topic name must contain at least one alphanumeric character.")

    existing_ids = {t["id"] for t in BUILTIN_TOPICS}
    registry = _load_registry()
    existing_ids.update(t["id"] for t in registry.get("custom", []))

    if topic_id in existing_ids:
        raise ValueError(f"Topic '{topic_id}' already exists.")

    doc_dir = f"./data/documents/{topic_id}"
    Path(doc_dir).mkdir(parents=True, exist_ok=True)

    new_topic = {
        "id": topic_id,
        "display_name": display_name.strip(),
        "description": description.strip(),
        "collection": f"tech_{topic_id}",
        "doc_dir": doc_dir,
        "created_at": str(date.today()),
    }

    registry.setdefault("custom", []).append(new_topic)
    _save_registry(registry)
    logger.info("Created custom topic: %s", topic_id)
    return new_topic


def delete_custom_topic(topic_id: str) -> None:
    """Delete a custom topic from registry and drop its ChromaDB collection.

    Files on disk are NOT deleted. Raises ValueError for built-in topics.
    """
    builtin_ids = {t["id"] for t in BUILTIN_TOPICS}
    if topic_id in builtin_ids:
        raise ValueError(f"Cannot delete built-in topic '{topic_id}'.")

    registry = _load_registry()
    registry["custom"] = [t for t in registry.get("custom", []) if t["id"] != topic_id]
    _save_registry(registry)

    try:
        _get_chroma().delete_collection(f"tech_{topic_id}")
    except Exception:
        pass

    logger.info("Deleted custom topic: %s", topic_id)


def save_uploaded_file(topic_id: str, filename: str, file_bytes: bytes) -> str:
    """Save uploaded file bytes to the topic's document directory.

    Returns the absolute path to the saved file. Raises ValueError for unknown topics.
    Strips directory components from filename to prevent path traversal.
    """
    topic = _find_topic(topic_id)
    if topic is None:
        raise ValueError(f"Unknown topic: {topic_id}")

    doc_dir = Path(topic["doc_dir"])
    doc_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name  # strips any directory prefix
    output_path = doc_dir / safe_filename
    output_path.write_bytes(file_bytes)

    logger.info("Saved uploaded file: %s", output_path)
    return str(output_path)


def list_topic_files(topic_id: str) -> list[str]:
    """Return sorted list of filenames in the topic's document directory."""
    topic = _find_topic(topic_id)
    if topic is None:
        return []
    doc_dir = Path(topic["doc_dir"])
    if not doc_dir.exists():
        return []
    return sorted(f.name for f in doc_dir.iterdir() if f.is_file())
```

- [ ] **Step 2: Run the tests and verify they pass**

Run: `uv run pytest tests/test_kb_manager.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: All tests pass (or same failures as before this task).

- [ ] **Step 4: Commit**

```bash
git add src/tools/kb_manager.py tests/test_kb_manager.py
git commit -m "feat: add kb_manager — topic registry CRUD, status checks, file persistence"
```

---

### Task 4: Update document_ingestion.py — add index_technology()

**Files:**
- Modify: `src/tools/document_ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write a failing test for index_technology**

Open `tests/test_ingestion.py` and add this test class at the end:

```python
class TestIndexTechnology:
    def test_index_technology_returns_int(self, tmp_path, monkeypatch):
        """index_technology returns 0 when directory is empty."""
        import chromadb
        from src.tools.document_ingestion import index_technology

        fake_topic = {
            "id": "test_tech",
            "display_name": "Test Tech",
            "collection": "tech_test_tech",
            "doc_dir": str(tmp_path / "test_tech"),
        }
        (tmp_path / "test_tech").mkdir()

        mock_bedrock = MagicMock()
        chroma_client = chromadb.EphemeralClient()

        with patch("src.tools.document_ingestion.boto3") as mock_boto3, \
             patch("src.tools.document_ingestion.get_chroma_client", return_value=chroma_client), \
             patch("src.tools.document_ingestion._find_topic", return_value=fake_topic):
            mock_boto3.client.return_value = mock_bedrock
            result = index_technology("test_tech")
        assert isinstance(result, int)
        assert result == 0  # empty directory → 0 chunks
```

Note: `from unittest.mock import patch, MagicMock` is already imported at the top of `test_ingestion.py`. If not, add it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingestion.py::TestIndexTechnology -v`
Expected: FAIL — `ImportError: cannot import name 'index_technology'`

- [ ] **Step 3: Update document_ingestion.py**

Replace the imports section and add the new function. The full updated file:

```python
"""Document ingestion pipeline: parse, chunk, embed, and index into ChromaDB."""
import os
import json
import logging
import argparse
from pathlib import Path

import boto3
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    AWS_REGION, BEDROCK_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, BUILTIN_TOPICS,
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

    if ext in (".txt", ".md"):
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

    return ""


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
            body=json.dumps({"inputText": text[:8192]}),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
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


def _find_topic(topic_id: str) -> dict | None:
    """Find a topic by id — built-ins first, then custom registry."""
    for t in BUILTIN_TOPICS:
        if t["id"] == topic_id:
            return t
    try:
        from src.tools.kb_manager import _load_registry
        registry = _load_registry()
        for t in registry.get("custom", []):
            if t["id"] == topic_id:
                return t
    except Exception:
        pass
    return None


def index_technology(topic_id: str, reindex: bool = False) -> int:
    """Index documents for a single technology topic. Returns chunk count.

    Args:
        topic_id: Topic identifier (e.g. 'selenium', 'kubernetes').
        reindex: If True, drops the existing collection and rebuilds from scratch.

    Returns:
        Number of chunks indexed. Returns 0 if topic not found or directory is empty.
    """
    topic = _find_topic(topic_id)
    if topic is None:
        logger.error("Unknown topic: %s", topic_id)
        return 0

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    chroma = get_chroma_client()
    logger.info("Indexing topic '%s' (reindex=%s)...", topic_id, reindex)
    chunks = index_directory(
        directory=topic["doc_dir"],
        collection_name=topic["collection"],
        bedrock_client=bedrock,
        chroma_client=chroma,
        reindex=reindex,
    )
    logger.info("Topic '%s' indexed: %d chunks", topic_id, chunks)
    return chunks


def run_ingestion(reindex: bool = False) -> None:
    """Run the full ingestion pipeline for all built-in topics."""
    logger.info("Starting ingestion for all built-in topics (reindex=%s)...", reindex)
    total = 0
    for topic in BUILTIN_TOPICS:
        chunks = index_technology(topic["id"], reindex=reindex)
        total += chunks
    logger.info("Ingestion complete. Total chunks indexed: %d", total)


def check_status() -> None:
    """Print current ChromaDB collection stats for all built-in topics."""
    chroma = get_chroma_client()
    for topic in BUILTIN_TOPICS:
        try:
            col = chroma.get_collection(topic["collection"])
            print(f"{topic['id']}: {col.count()} chunks")
        except Exception:
            print(f"{topic['id']}: not indexed (run ingestion)")


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

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_ingestion.py::TestIndexTechnology -v`
Expected: PASS

- [ ] **Step 5: Run the full ingestion test suite**

Run: `uv run pytest tests/test_ingestion.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tools/document_ingestion.py tests/test_ingestion.py
git commit -m "feat(ingestion): add index_technology() for per-topic indexing; update run_ingestion to use BUILTIN_TOPICS"
```

---

### Task 5: Update retrieval.py — per-tech queries and retrieve_topic tool

**Files:**
- Modify: `src/tools/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests for the new functions**

Add these test classes to `tests/test_retrieval.py`:

```python
class TestQueryMultiCollections:
    def test_merges_results_from_two_collections(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        col1 = client.create_collection("col_a")
        col1.add(ids=["a1"], embeddings=[[0.1] * 1024],
                 documents=["Selenium locators guide"], metadatas=[{"source_file": "sel.txt"}])
        col2 = client.create_collection("col_b")
        col2.add(ids=["b1"], embeddings=[[0.2] * 1024],
                 documents=["Tosca test case creation"], metadatas=[{"source_file": "tosca.txt"}])

        mock_embed = MagicMock(return_value=[[0.15] * 1024])
        result = query_multi_collections(
            query="test",
            collection_names=["col_a", "col_b"],
            top_k=5,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "sel.txt" in result or "tosca.txt" in result

    def test_skips_missing_collections_gracefully(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        col = client.create_collection("exists")
        col.add(ids=["x1"], embeddings=[[0.1] * 1024],
                documents=["Some content"], metadatas=[{"source_file": "doc.txt"}])

        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_multi_collections(
            query="content",
            collection_names=["exists", "does_not_exist"],
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "Source:" in result

    def test_all_empty_returns_message(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_multi_collections(
            query="anything",
            collection_names=["missing1", "missing2"],
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert isinstance(result, str)
        assert "No training content" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval.py::TestQueryMultiCollections -v`
Expected: FAIL — `ImportError: cannot import name 'query_multi_collections'`

- [ ] **Step 3: Rewrite src/tools/retrieval.py**

```python
"""ChromaDB retrieval tools — per-technology and multi-collection queries."""
import json
import logging
from typing import Callable, Optional

import boto3
import chromadb
from chromadb.config import Settings
from strands import tool

from src.config import (
    AWS_REGION, BEDROCK_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K,
    QA_TOPIC_IDS, ETL_TOPIC_IDS, BUILTIN_TOPICS,
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
    chroma_client=None,
    embed_fn: Optional[Callable] = None,
) -> str:
    """Query a single ChromaDB collection and return formatted results with citations."""
    client = chroma_client or _get_chroma()
    embed = embed_fn or _embed

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return "No training content found. Please run document ingestion first."

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


def query_multi_collections(
    query: str,
    collection_names: list[str],
    top_k: int = RETRIEVAL_TOP_K,
    chroma_client=None,
    embed_fn: Optional[Callable] = None,
) -> str:
    """Query multiple collections, merge results by relevance, deduplicate by source file."""
    client = chroma_client or _get_chroma()
    embed = embed_fn or _embed

    query_embedding = embed([query])[0]
    all_results: list[tuple[float, str, dict]] = []
    seen_sources: set[str] = set()

    for collection_name in collection_names:
        try:
            collection = client.get_collection(collection_name)
            if collection.count() == 0:
                continue
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, distances):
                source = meta.get("source_file", "unknown")
                if source not in seen_sources:
                    seen_sources.add(source)
                    all_results.append((dist, doc, meta))
        except Exception:
            continue

    if not all_results:
        return "No training content found. Please run document ingestion first."

    all_results.sort(key=lambda x: x[0])
    top_results = all_results[:top_k]

    parts = []
    for i, (_, doc, meta) in enumerate(top_results, 1):
        source = meta.get("source_file", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc}")

    return "\n\n---\n\n".join(parts)


def _collection_names_for_ids(topic_ids: list[str]) -> list[str]:
    """Map topic ids to their ChromaDB collection names via BUILTIN_TOPICS."""
    id_to_col = {t["id"]: t["collection"] for t in BUILTIN_TOPICS}
    return [id_to_col[tid] for tid in topic_ids if tid in id_to_col]


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
    collections = _collection_names_for_ids(QA_TOPIC_IDS)
    return query_multi_collections(query, collections, top_k)


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
    collections = _collection_names_for_ids(ETL_TOPIC_IDS)
    return query_multi_collections(query, collections, top_k)


@tool
def retrieve_topic(query: str, topic_id: str, top_k: int = 5) -> str:
    """Retrieve training content from a specific topic collection (built-in or custom).

    Args:
        query: Search query.
        topic_id: Topic identifier (e.g. 'selenium', 'kubernetes').
        top_k: Number of results to return.

    Returns:
        Formatted source blocks, or a message directing the user to upload material.
    """
    logger.debug("retrieve_topic query=%r topic_id=%r top_k=%d", query, topic_id, top_k)
    from src.tools.kb_manager import _find_topic, get_topic_status
    topic = _find_topic(topic_id)
    if topic is None:
        return f"Unknown topic '{topic_id}'. Check the Knowledge Base tab for available topics."
    status_info = get_topic_status(topic_id)
    if status_info["status"] != "AVAILABLE":
        return (
            f"No training material is available for '{topic_id}'. "
            f"Please go to the 📚 Knowledge Base tab and upload documents for this topic."
        )
    return query_collection(query, topic["collection"], top_k)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: All tests PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/tools/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): add query_multi_collections and retrieve_topic; update retrieve_qa/etl for per-tech collections"
```

---

### Task 6: Update orchestrator.py — dynamic system prompt

**Files:**
- Modify: `src/agents/orchestrator.py`

- [ ] **Step 1: Rewrite orchestrator.py**

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
    AGENT_TEMPERATURE, AGENT_TOP_P, AGENT_MAX_TOKENS,
)
from src.hooks.logging_throttle import LoggingThrottleHook
from src.agents.qa_agent import qa_training_agent
from src.agents.etl_agent import etl_training_agent
from src.agents.quiz_agent import quiz_agent
from src.agents.learning_path_agent import learning_path_agent
from src.agents.content_author_agent import content_author_agent
from src.agents.progress_agent import progress_agent
from src.tools.retrieval import retrieve_topic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = """You are TechTrainer AI, an intelligent training assistant for employees.

## Available Topics
{available_section}

## Routing Rules

Route every request to the appropriate specialist tool:

| Request type | Tool to call |
|---|---|
| Questions about Selenium, Tosca, or Playwright | qa_training_agent |
| Questions about AWS Glue, Spark, dbt, Informatica, Talend, SSIS, ADF | etl_training_agent |
| Questions about a custom topic | retrieve_topic with the correct topic_id |
| Quiz requests ("quiz me", "test my knowledge") | quiz_agent |
| Learning path ("what should I study", "recommend topics") | learning_path_agent |
| Content creation ("create a module", "write training material") | content_author_agent |
| Progress queries ("my scores", "how am I doing") | progress_agent |

## Response Style
- Always cite source documents when the specialist provides them
- Keep responses concise — employees want quick, actionable answers
- For code: use markdown code blocks with language identifiers
- Never answer from memory alone — always route to the appropriate specialist

## CRITICAL: Structured Data Pass-Through
When quiz_agent, learning_path_agent, or content_author_agent return a JSON string,
you MUST return that JSON string EXACTLY as-is — no paraphrasing, no summarising,
no wrapping in markdown code fences. The UI parses the raw JSON directly.
Only add a short plain-text line AFTER the JSON if you wish to comment on it.

## Boundaries
- Only answer questions related to the available topics listed above
- For topics with no material, tell the user to visit the 📚 Knowledge Base tab to upload documents
- For completely off-topic requests, politely decline and suggest relevant topics you can help with
"""


def _build_system_prompt() -> str:
    """Build the system prompt with current available/unavailable topic lists."""
    try:
        from src.tools.kb_manager import load_all_topics
        all_topics = load_all_topics()
        available = [t for t in all_topics if t["status"] == "AVAILABLE"]
        unavailable = [t for t in all_topics if t["status"] != "AVAILABLE"]

        if available:
            avail_lines = "\n".join(
                f"- {t['display_name']} (topic_id: {t['id']})"
                for t in available
            )
        else:
            avail_lines = "- None yet. Ask the user to upload documents in the 📚 Knowledge Base tab."

        if unavailable:
            unavail_names = ", ".join(t["display_name"] for t in unavailable)
            unavail_section = (
                f"\nTopics with no material (direct users to Knowledge Base tab): {unavail_names}"
            )
        else:
            unavail_section = ""

        available_section = avail_lines + unavail_section
    except Exception as exc:
        logger.warning("Could not load topic status for system prompt: %s", exc)
        available_section = "- Topic status unavailable. Route normally."

    return _SYSTEM_PROMPT_BASE.format(available_section=available_section)


_ALL_TOOLS = [
    qa_training_agent,
    etl_training_agent,
    quiz_agent,
    learning_path_agent,
    content_author_agent,
    progress_agent,
    retrieve_topic,
]


def _build_bedrock_model() -> BedrockModel:
    kwargs = {
        "model_id": BEDROCK_MODEL_ID,
        "region_name": AWS_REGION,
        "temperature": AGENT_TEMPERATURE,
        "top_p": AGENT_TOP_P,
        "max_tokens": AGENT_MAX_TOKENS,
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
        system_prompt=_build_system_prompt(),
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

- [ ] **Step 2: Verify the orchestrator imports cleanly**

Run: `uv run python -c "from src.agents.orchestrator import build_orchestrator; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/agents/orchestrator.py
git commit -m "feat(orchestrator): dynamic system prompt with available topics; add retrieve_topic tool"
```

---

### Task 7: Add Knowledge Base tab to app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add imports at the top of app.py**

After the existing imports (after `from src.config import PROGRESS_DB`), add:

```python
from src.tools.kb_manager import (
    load_all_topics,
    create_custom_topic,
    delete_custom_topic,
    save_uploaded_file,
    list_topic_files,
    get_available_topic_ids,
    get_available_topics,
)
from src.tools.document_ingestion import index_technology
```

- [ ] **Step 2: Add helper functions before the sidebar block**

Add these two functions after `_extract_json` and before `call_agent`:

```python
def _render_topic_card(topic: dict) -> None:
    """Render a single topic card with status, file list, upload zone, and index controls."""
    tid = topic["id"]
    status = topic["status"]
    is_builtin = topic.get("is_builtin", True)

    # Status header
    if status == "AVAILABLE":
        st.success(
            f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ✓ {topic['chunk_count']} chunks",
        )
    elif status == "PENDING":
        st.warning(f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ⏳ Files not indexed")
    else:
        st.error(f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ✗ No material")

    # File list
    files = list_topic_files(tid)
    if files:
        with st.expander(f"📂 {len(files)} file(s) in folder"):
            for fname in files:
                st.text(f"  📄 {fname}")

    if status == "DISABLED" and not files:
        st.caption("Upload documents or copy files to data/documents/" + tid + "/")

    # Upload widget
    auto_idx = st.checkbox("Auto-index on upload", value=True, key=f"auto_{tid}")
    uploaded = st.file_uploader(
        f"Upload for {topic['display_name']}",
        accept_multiple_files=True,
        key=f"upload_{tid}",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md"],
        label_visibility="collapsed",
    )

    if uploaded:
        saved_names = []
        for uf in uploaded:
            try:
                save_uploaded_file(tid, uf.name, uf.getvalue())
                saved_names.append(uf.name)
            except Exception as e:
                st.warning(f"Skipped {uf.name}: {e}")

        if saved_names and auto_idx:
            with st.spinner(f"Indexing {topic['display_name']}..."):
                try:
                    n = index_technology(tid)
                    st.success(f"Indexed {n} chunks from {len(saved_names)} file(s).")
                except Exception as e:
                    st.warning(f"Indexing error: {e}")
            st.rerun()
        elif saved_names:
            st.info(f"Saved {len(saved_names)} file(s). Click 'Index' to make searchable.")
            st.rerun()

    # Action buttons
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        if status == "AVAILABLE":
            if st.button("🔄 Re-index", key=f"reindex_{tid}"):
                with st.spinner(f"Re-indexing {topic['display_name']}..."):
                    try:
                        n = index_technology(tid, reindex=True)
                        st.success(f"Re-indexed: {n} chunks")
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()
    with b2:
        if status in ("PENDING", "DISABLED") and topic["file_count"] > 0:
            if st.button("⚡ Index", key=f"index_{tid}"):
                with st.spinner(f"Indexing {topic['display_name']}..."):
                    try:
                        n = index_technology(tid)
                        if n > 0:
                            st.success(f"Indexed: {n} chunks")
                        else:
                            st.warning("No indexable documents found.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()
    with b3:
        if not is_builtin:
            if st.button("🗑", key=f"del_{tid}", help=f"Delete {topic['display_name']}"):
                st.session_state[f"confirm_del_{tid}"] = True

    # Delete confirmation (custom topics only)
    if not is_builtin and st.session_state.get(f"confirm_del_{tid}"):
        st.warning(f"Delete '{topic['display_name']}'? Files on disk are kept.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", key=f"yes_del_{tid}", type="primary"):
                try:
                    delete_custom_topic(tid)
                    st.session_state.pop(f"confirm_del_{tid}", None)
                    st.success(f"Deleted: {topic['display_name']}")
                except ValueError as e:
                    st.error(str(e))
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"no_del_{tid}"):
                st.session_state.pop(f"confirm_del_{tid}", None)
                st.rerun()


def _render_topic_grid(topics: list) -> None:
    """Render topics in a 2-column grid of cards."""
    for i in range(0, len(topics), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(topics):
                with col:
                    with st.container(border=True):
                        _render_topic_card(topics[i + j])
```

- [ ] **Step 3: Add the 5th tab to the tab bar**

Find this line in app.py:
```python
tab_chat, tab_quiz, tab_path, tab_author = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author"]
)
```

Replace with:
```python
tab_chat, tab_quiz, tab_path, tab_author, tab_kb = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author", "📚 Knowledge Base"]
)
```

- [ ] **Step 4: Add the Knowledge Base tab content at the end of app.py**

Append after the Content Author tab block:

```python
# ── TAB: Knowledge Base ───────────────────────────────────────────────────────
with tab_kb:
    st.subheader("Manage Training Documents")
    st.caption(
        "Upload documents or copy files into data/documents/<technology>/ — "
        "then index to make them searchable in Chat, Quiz, and Learning Path."
    )

    # ── Top action bar ──────────────────────────────────────────────────────
    col_refresh, col_index_all = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Status", key="kb_refresh"):
            st.rerun()
    with col_index_all:
        if st.button("⚡ Index All", key="kb_index_all", type="primary"):
            all_topics_now = load_all_topics()
            to_index = [
                t for t in all_topics_now
                if t["status"] in ("PENDING", "AVAILABLE") and t["file_count"] > 0
            ]
            if to_index:
                bar = st.progress(0, text="Starting...")
                for i, t in enumerate(to_index):
                    bar.progress((i) / len(to_index), text=f"Indexing {t['display_name']}...")
                    try:
                        index_technology(t["id"])
                    except Exception as e:
                        st.warning(f"Error indexing {t['display_name']}: {e}")
                bar.progress(1.0, text="Done!")
                st.success(f"Indexed {len(to_index)} topic(s).")
                st.rerun()
            else:
                st.info("No topics with files to index.")

    # ── Summary metrics ─────────────────────────────────────────────────────
    all_topics = load_all_topics()
    available_count = sum(1 for t in all_topics if t["status"] == "AVAILABLE")
    pending_count = sum(1 for t in all_topics if t["status"] == "PENDING")
    disabled_count = sum(1 for t in all_topics if t["status"] == "DISABLED")
    m1, m2, m3 = st.columns(3)
    m1.metric("✓ Available", available_count)
    m2.metric("⏳ Pending", pending_count)
    m3.metric("✗ Disabled", disabled_count)

    st.divider()

    # ── Add Custom Topic ─────────────────────────────────────────────────────
    with st.expander("➕ Add Custom Topic"):
        with st.form("add_topic_form"):
            new_name = st.text_input("Topic Name *", placeholder="e.g. Kubernetes")
            new_desc = st.text_area(
                "Description (optional)",
                placeholder="Container orchestration platform training material",
            )
            if st.form_submit_button("Create Topic"):
                if new_name.strip():
                    try:
                        create_custom_topic(new_name.strip(), new_desc.strip())
                        st.success(f"Topic '{new_name}' created successfully.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Topic name is required.")

    st.divider()

    # ── Built-in Technologies ────────────────────────────────────────────────
    builtin = [t for t in all_topics if t.get("is_builtin", True)]
    custom = [t for t in all_topics if not t.get("is_builtin", True)]

    st.markdown("### 🔒 Built-in Technologies")
    _render_topic_grid(builtin)

    if custom:
        st.markdown("### ✏️ Custom Topics")
        _render_topic_grid(custom)
```

- [ ] **Step 5: Verify the app starts without errors**

Run: `uv run streamlit run app.py --server.headless true &`
Then check the terminal for errors. Stop the server with Ctrl+C.

If it starts cleanly with no import errors, proceed.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(ui): add Knowledge Base tab with per-topic upload, indexing, status, and custom topic management"
```

---

### Task 8: Update Quiz, Content Author, and Learning Path tabs for dynamic topics

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update the Quiz tab technology selectbox**

Find the Quiz tab section. Replace the hardcoded `TECHNOLOGIES` list and selectbox:

```python
# REMOVE this block:
TECHNOLOGIES = [
    "selenium", "tosca", "playwright",
    "aws_glue", "spark", "dbt",
    "informatica", "ssis", "talend", "adf",
]

col1, col2, col3 = st.columns(3)
with col1:
    quiz_tech = st.selectbox("Technology", TECHNOLOGIES, key="quiz_tech")
```

Replace with:

```python
available_quiz_topics = get_available_topics()
available_quiz_ids = [t["id"] for t in available_quiz_topics]
available_quiz_names = {t["id"]: t["display_name"] for t in available_quiz_topics}

col1, col2, col3 = st.columns(3)
with col1:
    if not available_quiz_ids:
        st.warning("No topics available yet. Go to 📚 Knowledge Base to upload training documents.")
        quiz_tech = None
    else:
        quiz_tech = st.selectbox(
            "Technology",
            options=available_quiz_ids,
            format_func=lambda x: available_quiz_names.get(x, x),
            key="quiz_tech",
        )
```

Also update the quiz button guard — wrap the `if st.button("🎯 Generate Quiz"...)` block:

```python
if quiz_tech and st.button("🎯 Generate Quiz", type="primary"):
```

- [ ] **Step 2: Update the Content Author tab technology selectbox**

Find the Content Author tab. Replace the hardcoded `AUTHOR_TECHNOLOGIES` list and selectbox:

```python
# REMOVE this block:
AUTHOR_TECHNOLOGIES = [
    "selenium", "tosca", "playwright",
    "aws_glue", "spark", "dbt",
    "informatica", "ssis", "talend", "adf",
]

with st.form("author_form"):
    title = st.text_input("Module Title", placeholder="e.g. Introduction to Selenium Locators")
    col1, col2 = st.columns(2)
    with col1:
        tech = st.selectbox("Technology", AUTHOR_TECHNOLOGIES, key="author_tech")
```

Replace with:

```python
available_author_topics = get_available_topics()
available_author_ids = [t["id"] for t in available_author_topics]
available_author_names = {t["id"]: t["display_name"] for t in available_author_topics}

with st.form("author_form"):
    title = st.text_input("Module Title", placeholder="e.g. Introduction to Selenium Locators")
    col1, col2 = st.columns(2)
    with col1:
        if not available_author_ids:
            st.warning("No topics available. Go to 📚 Knowledge Base first.")
            tech = None
        else:
            tech = st.selectbox(
                "Technology",
                options=available_author_ids,
                format_func=lambda x: available_author_names.get(x, x),
                key="author_tech",
            )
```

Also guard the generate button:
```python
if generate_btn and title and tech:
```

- [ ] **Step 3: Update the Chat tab caption**

Find:
```python
st.caption("Ask questions about Selenium, Tosca, Playwright, AWS Glue, Spark, dbt, and more.")
```

Replace with:

```python
available_names = [t["display_name"] for t in get_available_topics()]
if available_names:
    st.caption(f"Ask questions about: {', '.join(available_names[:6])}" +
               (" and more." if len(available_names) > 6 else "."))
else:
    st.caption("No topics indexed yet. Go to 📚 Knowledge Base to upload training documents.")
```

- [ ] **Step 4: Update the Learning Path prompt to include available topics**

Find the learning path prompt inside the `tab_path` block:

```python
raw = call_agent(
    f"Generate a learning path for me. My session_id is {session_id}. "
    "Use the learning_path_agent tool."
)
```

Replace with:

```python
available_lp = [t["display_name"] for t in get_available_topics()]
available_lp_str = ", ".join(available_lp) if available_lp else "none yet"
raw = call_agent(
    f"Generate a learning path for me. My session_id is {session_id}. "
    f"Only recommend topics from this available list: {available_lp_str}. "
    "Use the learning_path_agent tool."
)
```

- [ ] **Step 5: Verify the app runs correctly end-to-end**

Run: `uv run streamlit run app.py`

Manual checks:
1. Open http://localhost:8501
2. Confirm 5 tabs appear: Chat, Quiz, Learning Path, Content Author, Knowledge Base
3. Click Knowledge Base — summary metrics show, built-in grid renders, "Add Custom Topic" expander works
4. Quiz tab — technology dropdown only shows available topics (empty if none indexed yet)
5. Content Author tab — same dynamic dropdown

- [ ] **Step 6: Run the full test suite one final time**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat(ui): dynamic topic dropdowns in Quiz, Content Author, Chat; learning path uses available topics"
```
