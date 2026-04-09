# Wikipedia ZIM Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Wikipedia ZIM file support as a knowledge base source using the hybrid approach — libzim's Xapian index for fast article filtering, then on-the-fly embedding + semantic ranking of matched articles.

**Architecture:** A new `zim_reader.py` module handles ZIM file I/O and Xapian search. A new `retrieve_wikipedia` Strands tool in `retrieval.py` uses zim_reader to find articles, chunks them, embeds on-the-fly, and ranks by cosine similarity. The orchestrator routes general-knowledge queries to this tool. A new "Wikipedia ZIM" section in the KB tab lets users configure ZIM file paths. Settings are persisted in `settings.json`.

**Tech Stack:** `libzim>=3.6.0`, `beautifulsoup4` (HTML→text), existing `embedding_manager`, existing `chromadb` for optional caching.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tools/zim_reader.py` | Create | ZIM file I/O: open archive, Xapian search, HTML→text extraction |
| `src/tools/retrieval.py` | Modify | Add `retrieve_wikipedia` Strands tool |
| `src/config.py` | Modify | Add `WIKIPEDIA_ZIM_PATHS`, `WIKIPEDIA_ENABLED`, `WIKIPEDIA_RESULTS` settings |
| `src/agents/orchestrator.py` | Modify | Add `retrieve_wikipedia` to tool list + routing rule |
| `app.py` | Modify | Add "Wikipedia ZIM" section in KB tab |
| `pyproject.toml` | Modify | Add `libzim>=3.6.0`, `beautifulsoup4>=4.12.0` dependencies |
| `tests/test_zim_reader.py` | Create | Unit tests for zim_reader module |
| `tests/test_retrieval_wikipedia.py` | Create | Tests for retrieve_wikipedia tool |

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml:10-25`

- [ ] **Step 1: Add libzim and beautifulsoup4 to dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
    "libzim>=3.6.0",
    "beautifulsoup4>=4.12.0",
```

- [ ] **Step 2: Install dependencies**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv sync`
Expected: Dependencies install successfully.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add libzim and beautifulsoup4 dependencies for Wikipedia ZIM support"
```

---

### Task 2: Add Wikipedia config settings

**Files:**
- Modify: `src/config.py:26-31`

- [ ] **Step 1: Write failing test**

Create `tests/test_zim_config.py`:

```python
"""Tests for Wikipedia ZIM configuration."""
from src.config import WIKIPEDIA_ZIM_PATHS, WIKIPEDIA_ENABLED, WIKIPEDIA_RESULTS


def test_wikipedia_defaults():
    assert WIKIPEDIA_ENABLED is False
    assert WIKIPEDIA_RESULTS == 5
    assert isinstance(WIKIPEDIA_ZIM_PATHS, list)
    assert len(WIKIPEDIA_ZIM_PATHS) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_zim_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'WIKIPEDIA_ZIM_PATHS'`

- [ ] **Step 3: Add config constants to src/config.py**

Add after the `TOPICS_REGISTRY_PATH` line (line 31):

```python
# ── Wikipedia ZIM ─────────────────────────────────────────────────────────────
WIKIPEDIA_ENABLED: bool = os.getenv("WIKIPEDIA_ENABLED", "false").lower() == "true"
WIKIPEDIA_ZIM_PATHS: list[str] = [
    p.strip() for p in os.getenv("WIKIPEDIA_ZIM_PATHS", "").split(",") if p.strip()
]
WIKIPEDIA_RESULTS: int = int(os.getenv("WIKIPEDIA_RESULTS", "5"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_zim_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_zim_config.py
git commit -m "feat: add Wikipedia ZIM configuration constants"
```

---

### Task 3: Create zim_reader module

**Files:**
- Create: `src/tools/zim_reader.py`
- Create: `tests/test_zim_reader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_zim_reader.py`:

```python
"""Tests for ZIM reader module."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.tools.zim_reader import search_zim, extract_text_from_html, open_zim_archive


class TestExtractTextFromHtml:
    def test_strips_html_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = extract_text_from_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_removes_script_and_style(self):
        html = "<html><style>body{color:red}</style><script>alert(1)</script><p>Content</p></html>"
        result = extract_text_from_html(html)
        assert "Content" in result
        assert "color:red" not in result
        assert "alert" not in result

    def test_empty_html_returns_empty(self):
        assert extract_text_from_html("") == ""

    def test_collapses_whitespace(self):
        html = "<p>Hello</p>\n\n\n<p>World</p>"
        result = extract_text_from_html(html)
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result


class TestOpenZimArchive:
    @patch("src.tools.zim_reader.Archive")
    def test_returns_archive_for_valid_path(self, mock_archive_cls):
        mock_archive = MagicMock()
        mock_archive_cls.return_value = mock_archive
        result = open_zim_archive("/path/to/file.zim")
        assert result is mock_archive
        mock_archive_cls.assert_called_once_with("/path/to/file.zim")

    def test_returns_none_for_missing_file(self):
        result = open_zim_archive("/nonexistent/path.zim")
        assert result is None


class TestSearchZim:
    @patch("src.tools.zim_reader.Archive")
    @patch("src.tools.zim_reader.Searcher")
    @patch("src.tools.zim_reader.Query")
    def test_returns_results_list(self, mock_query_cls, mock_searcher_cls, mock_archive_cls):
        # Set up mock archive
        mock_archive = MagicMock()
        type(mock_archive).has_fulltext_index = PropertyMock(return_value=True)
        mock_archive_cls.return_value = mock_archive

        # Set up mock entry
        mock_entry = MagicMock()
        mock_entry.title = "Python (programming language)"
        mock_entry.is_redirect = False
        mock_item = MagicMock()
        mock_item.mimetype = "text/html"
        mock_item.content = memoryview(b"<p>Python is a programming language.</p>")
        mock_entry.get_item.return_value = mock_item
        mock_archive.get_entry_by_path.return_value = mock_entry

        # Set up mock search
        mock_query = MagicMock()
        mock_query.set_query.return_value = mock_query
        mock_query_cls.return_value = mock_query

        mock_search = MagicMock()
        mock_search.getResults.return_value = ["A/Python"]
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = mock_search
        mock_searcher_cls.return_value = mock_searcher

        results = search_zim("python", "/path/to/wiki.zim", top_k=5)

        assert len(results) == 1
        assert results[0]["title"] == "Python (programming language)"
        assert "Python is a programming language" in results[0]["content"]
        assert results[0]["path"] == "A/Python"

    @patch("src.tools.zim_reader.Archive")
    def test_returns_empty_when_no_fulltext_index(self, mock_archive_cls):
        mock_archive = MagicMock()
        type(mock_archive).has_fulltext_index = PropertyMock(return_value=False)
        mock_archive_cls.return_value = mock_archive

        results = search_zim("python", "/path/to/wiki.zim")
        assert results == []

    def test_returns_empty_for_missing_file(self):
        results = search_zim("python", "/nonexistent/path.zim")
        assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_zim_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.zim_reader'`

- [ ] **Step 3: Implement zim_reader module**

Create `src/tools/zim_reader.py`:

```python
"""Wikipedia ZIM file reader — search and extract article content."""
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

try:
    from libzim.reader import Archive
    from libzim.search import Searcher, Query
    LIBZIM_AVAILABLE = True
except ImportError:
    LIBZIM_AVAILABLE = False
    Archive = None
    Searcher = None
    Query = None
    logger.warning("libzim not installed. Wikipedia ZIM support disabled.")


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return clean text content."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "link", "meta"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def open_zim_archive(zim_path: str) -> Optional[object]:
    """Open a ZIM archive. Returns Archive object or None if file missing or libzim unavailable."""
    if not LIBZIM_AVAILABLE:
        logger.error("libzim is not installed.")
        return None
    try:
        return Archive(zim_path)
    except Exception as exc:
        logger.error("Failed to open ZIM file '%s': %s", zim_path, exc)
        return None


def search_zim(
    query: str,
    zim_path: str,
    top_k: int = 5,
    max_content_chars: int = 3000,
) -> list[dict]:
    """Search a ZIM file using its built-in Xapian full-text index.

    Args:
        query: Search query string.
        zim_path: Path to the .zim file.
        top_k: Maximum number of results to return.
        max_content_chars: Truncate article text to this many characters.

    Returns:
        List of dicts with keys: title, content, path, zim_file.
        Empty list if file is missing, has no index, or libzim is unavailable.
    """
    archive = open_zim_archive(zim_path)
    if archive is None:
        return []

    if not archive.has_fulltext_index:
        logger.warning("ZIM file '%s' has no full-text index.", zim_path)
        return []

    searcher = Searcher(archive)
    search_query = Query().set_query(query)
    search = searcher.search(search_query)

    results = []
    for path in search.getResults(0, top_k):
        try:
            entry = archive.get_entry_by_path(path)
            if entry.is_redirect:
                entry = entry.get_redirect_entry()

            item = entry.get_item()
            if not item.mimetype.startswith("text/html"):
                continue

            html = bytes(item.content).decode("utf-8", errors="replace")
            text = extract_text_from_html(html)
            if not text:
                continue

            results.append({
                "title": entry.title,
                "content": text[:max_content_chars],
                "path": path,
                "zim_file": zim_path,
            })
        except Exception as exc:
            logger.warning("Failed to read ZIM entry '%s': %s", path, exc)
            continue

    return results


def search_multiple_zim(
    query: str,
    zim_paths: list[str],
    top_k: int = 5,
    max_content_chars: int = 3000,
) -> list[dict]:
    """Search across multiple ZIM files and merge results.

    Args:
        query: Search query string.
        zim_paths: List of paths to .zim files.
        top_k: Maximum total results to return.
        max_content_chars: Truncate article text to this many characters.

    Returns:
        Combined list of results from all ZIM files, truncated to top_k.
    """
    all_results = []
    seen_titles = set()
    for zim_path in zim_paths:
        results = search_zim(query, zim_path, top_k=top_k, max_content_chars=max_content_chars)
        for r in results:
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                all_results.append(r)
    return all_results[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_zim_reader.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/zim_reader.py tests/test_zim_reader.py
git commit -m "feat: add zim_reader module for Wikipedia ZIM file search"
```

---

### Task 4: Add retrieve_wikipedia tool with hybrid semantic ranking

**Files:**
- Modify: `src/tools/retrieval.py:1-195`
- Create: `tests/test_retrieval_wikipedia.py`

This is the core hybrid approach: use Xapian to find candidate articles, chunk them, embed on-the-fly, and rank by cosine similarity against the query embedding.

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval_wikipedia.py`:

```python
"""Tests for the retrieve_wikipedia Strands tool."""
import pytest
from unittest.mock import patch, MagicMock


class TestRetrieveWikipedia:
    @patch("src.tools.retrieval._get_wikipedia_config")
    @patch("src.tools.retrieval._search_and_rank_zim")
    def test_returns_formatted_citations(self, mock_search, mock_config):
        mock_config.return_value = {
            "enabled": True,
            "zim_paths": ["/path/to/wiki.zim"],
            "top_k": 5,
        }
        mock_search.return_value = [
            {"title": "Python (language)", "content": "Python is a language.", "source": "wikipedia.zim"},
        ]

        from src.tools.retrieval import retrieve_wikipedia
        # Call the underlying function (unwrap Strands @tool)
        result = retrieve_wikipedia.tool_handler(
            tool=MagicMock(),
            query="what is python",
            top_k=3,
        )
        assert "Python" in result
        assert "Source:" in result

    @patch("src.tools.retrieval._get_wikipedia_config")
    def test_returns_message_when_disabled(self, mock_config):
        mock_config.return_value = {
            "enabled": False,
            "zim_paths": [],
            "top_k": 5,
        }
        from src.tools.retrieval import retrieve_wikipedia
        result = retrieve_wikipedia.tool_handler(
            tool=MagicMock(),
            query="anything",
            top_k=3,
        )
        assert "not configured" in result.lower() or "not enabled" in result.lower()

    @patch("src.tools.retrieval._get_wikipedia_config")
    def test_returns_message_when_no_zim_paths(self, mock_config):
        mock_config.return_value = {
            "enabled": True,
            "zim_paths": [],
            "top_k": 5,
        }
        from src.tools.retrieval import retrieve_wikipedia
        result = retrieve_wikipedia.tool_handler(
            tool=MagicMock(),
            query="anything",
            top_k=3,
        )
        assert "no zim" in result.lower() or "not configured" in result.lower()


class TestSearchAndRankZim:
    @patch("src.tools.retrieval._embed")
    @patch("src.tools.zim_reader.search_multiple_zim")
    def test_chunks_and_ranks_by_similarity(self, mock_search_zim, mock_embed):
        mock_search_zim.return_value = [
            {"title": "Article A", "content": "Content about topic A which is relevant.", "path": "A/A", "zim_file": "wiki.zim"},
            {"title": "Article B", "content": "Content about topic B which is less relevant.", "path": "A/B", "zim_file": "wiki.zim"},
        ]
        # Mock embeddings: query closer to Article A
        mock_embed.side_effect = [
            [[1.0, 0.0, 0.0]],       # query embedding
            [[0.9, 0.1, 0.0],         # chunk from Article A
             [0.1, 0.9, 0.0]],        # chunk from Article B
        ]

        from src.tools.retrieval import _search_and_rank_zim
        results = _search_and_rank_zim(
            query="topic A",
            zim_paths=["wiki.zim"],
            top_k=2,
        )
        assert len(results) >= 1
        # Article A should rank first (closer embedding)
        assert results[0]["title"] == "Article A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_retrieval_wikipedia.py -v`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Add Wikipedia retrieval functions to retrieval.py**

Add the following imports at the top of `src/tools/retrieval.py` (after existing imports):

```python
import math
from src.tools.zim_reader import search_multiple_zim
from src.config import WIKIPEDIA_ENABLED, WIKIPEDIA_ZIM_PATHS, WIKIPEDIA_RESULTS
```

Add the following functions at the end of `src/tools/retrieval.py` (before the file ends):

```python
def _get_wikipedia_config() -> dict:
    """Load Wikipedia settings, merging settings.json with env/config defaults."""
    try:
        from src.tools.provider_manager import load_settings
        settings = load_settings()
    except Exception:
        settings = {}

    enabled = settings.get("wikipedia_enabled", WIKIPEDIA_ENABLED)
    zim_paths = settings.get("wikipedia_zim_paths", WIKIPEDIA_ZIM_PATHS)
    top_k = settings.get("wikipedia_results", WIKIPEDIA_RESULTS)

    return {"enabled": enabled, "zim_paths": zim_paths, "top_k": top_k}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _search_and_rank_zim(
    query: str,
    zim_paths: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Hybrid search: Xapian keyword search → chunk → embed → rank by cosine similarity.

    Returns list of dicts with keys: title, content, source, similarity.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP

    # Step 1: Xapian search to find candidate articles (fetch more than top_k for re-ranking)
    candidates = search_multiple_zim(query, zim_paths, top_k=top_k * 2)
    if not candidates:
        return []

    # Step 2: Chunk candidate articles
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
    )
    chunks = []  # list of (chunk_text, title, zim_file)
    for article in candidates:
        article_chunks = splitter.split_text(article["content"])
        for chunk in article_chunks:
            chunks.append((chunk, article["title"], article["zim_file"]))

    if not chunks:
        return []

    # Step 3: Embed query and all chunks
    query_embedding = _embed([query])[0]
    chunk_texts = [c[0] for c in chunks]
    chunk_embeddings = _embed(chunk_texts)

    # Step 4: Rank chunks by cosine similarity
    scored = []
    for i, (text, title, zim_file) in enumerate(chunks):
        sim = _cosine_similarity(query_embedding, chunk_embeddings[i])
        scored.append({
            "title": title,
            "content": text,
            "source": f"Wikipedia ({Path(zim_file).stem})",
            "similarity": sim,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # Deduplicate by title — keep best chunk per article
    seen_titles = set()
    deduped = []
    for item in scored:
        if item["title"] not in seen_titles:
            seen_titles.add(item["title"])
            deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


@tool
def retrieve_wikipedia(query: str, top_k: int = 5) -> str:
    """Search Wikipedia ZIM knowledge base for general reference content.

    Args:
        query: Search query for Wikipedia articles.
        top_k: Number of most relevant article chunks to return. Default is 5.

    Returns:
        Formatted string of relevant Wikipedia content with source citations.
    """
    logger.debug("retrieve_wikipedia query=%r top_k=%d", query, top_k)
    config = _get_wikipedia_config()

    if not config["enabled"]:
        return "Wikipedia search is not enabled. Configure a ZIM file in the Knowledge Base tab."

    if not config["zim_paths"]:
        return "No ZIM files configured. Add a Wikipedia ZIM file path in the Knowledge Base tab."

    results = _search_and_rank_zim(query, config["zim_paths"], top_k=top_k)
    if not results:
        return "No relevant Wikipedia articles found for your query."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] Source: {r['source']} — {r['title']}\n{r['content']}")

    return "\n\n---\n\n".join(parts)
```

Also add `from pathlib import Path` to the imports at top of `retrieval.py` if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_retrieval_wikipedia.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing retrieval tests to ensure no regression**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_retrieval.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools/retrieval.py tests/test_retrieval_wikipedia.py
git commit -m "feat: add retrieve_wikipedia tool with hybrid Xapian + semantic ranking"
```

---

### Task 5: Integrate retrieve_wikipedia into orchestrator

**Files:**
- Modify: `src/agents/orchestrator.py:1-125`

- [ ] **Step 1: Add import for retrieve_wikipedia**

Add to orchestrator.py imports (after `from src.tools.retrieval import retrieve_topic`):

```python
from src.tools.retrieval import retrieve_wikipedia
```

- [ ] **Step 2: Add Wikipedia routing rule to system prompt**

In the `_SYSTEM_PROMPT_BASE` string, add a new row to the routing rules table:

```
| General knowledge questions (when Wikipedia is enabled) | retrieve_wikipedia |
```

- [ ] **Step 3: Add retrieve_wikipedia to _ALL_TOOLS list**

```python
_ALL_TOOLS = [
    qa_training_agent,
    etl_training_agent,
    quiz_agent,
    learning_path_agent,
    content_author_agent,
    progress_agent,
    retrieve_topic,
    retrieve_wikipedia,
]
```

- [ ] **Step 4: Add Wikipedia status to system prompt builder**

In `_build_system_prompt()`, after the available/unavailable topic logic, add:

```python
    # Wikipedia ZIM status
    try:
        from src.tools.retrieval import _get_wikipedia_config
        wiki_cfg = _get_wikipedia_config()
        if wiki_cfg["enabled"] and wiki_cfg["zim_paths"]:
            available_section += "\n- Wikipedia (via ZIM files) — use retrieve_wikipedia for general reference"
    except Exception:
        pass
```

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/orchestrator.py
git commit -m "feat: integrate retrieve_wikipedia into orchestrator routing"
```

---

### Task 6: Add Wikipedia ZIM section to Knowledge Base UI

**Files:**
- Modify: `app.py:679-790`

- [ ] **Step 1: Add Wikipedia ZIM UI section**

In `app.py`, inside the `with tab_kb:` block, after the `st.divider()` on line 778 and before the "Built-in Technologies" section (line 780), add:

```python
    # ── Wikipedia ZIM Files ─────────────────────────────────────────────────
    with st.expander("📖 Wikipedia ZIM Files"):
        st.caption(
            "Point to Wikipedia ZIM files for offline reference. "
            "Download topic-specific ZIM dumps from download.kiwix.org."
        )

        from src.tools.provider_manager import load_settings, save_settings

        current_settings = load_settings()
        wiki_enabled = current_settings.get("wikipedia_enabled", False)
        wiki_paths = current_settings.get("wikipedia_zim_paths", [])

        new_enabled = st.toggle("Enable Wikipedia search", value=wiki_enabled, key="wiki_enabled")

        # Display current ZIM paths
        if wiki_paths:
            st.markdown("**Configured ZIM files:**")
            paths_to_remove = []
            for i, p in enumerate(wiki_paths):
                col_path, col_del = st.columns([4, 1])
                with col_path:
                    st.text(p)
                with col_del:
                    if st.button("✕", key=f"wiki_del_{i}"):
                        paths_to_remove.append(i)
            if paths_to_remove:
                wiki_paths = [p for i, p in enumerate(wiki_paths) if i not in paths_to_remove]
                current_settings["wikipedia_zim_paths"] = wiki_paths
                save_settings(current_settings)
                st.rerun()
        else:
            st.info("No ZIM files configured yet.")

        # Add new ZIM path
        new_path = st.text_input(
            "ZIM file path",
            placeholder="e.g. D:/data/wikipedia_en_computer.zim",
            key="wiki_new_path",
        )
        if st.button("Add ZIM File", key="wiki_add"):
            if new_path.strip():
                from pathlib import Path as P
                if P(new_path.strip()).exists():
                    wiki_paths.append(new_path.strip())
                    current_settings["wikipedia_zim_paths"] = wiki_paths
                    current_settings["wikipedia_enabled"] = new_enabled
                    save_settings(current_settings)
                    st.success(f"Added: {new_path.strip()}")
                    st.rerun()
                else:
                    st.error(f"File not found: {new_path.strip()}")
            else:
                st.error("Please enter a file path.")

        # Save enable/disable toggle
        if new_enabled != wiki_enabled:
            current_settings["wikipedia_enabled"] = new_enabled
            save_settings(current_settings)
            st.rerun()

    st.divider()
```

- [ ] **Step 2: Manually test the UI**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run streamlit run app.py`

Verify:
1. Navigate to Knowledge Base tab
2. See "Wikipedia ZIM Files" expander
3. Toggle enable/disable works
4. Adding a valid ZIM file path works
5. Removing a ZIM path works
6. Invalid path shows error

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Wikipedia ZIM configuration UI in Knowledge Base tab"
```

---

### Task 7: Settings persistence for Wikipedia config

**Files:**
- Modify: `src/tools/retrieval.py` (already done in Task 4 — `_get_wikipedia_config` reads from settings.json)

This task verifies that the settings written by the UI in Task 6 are correctly read by the retrieval logic in Task 4.

- [ ] **Step 1: Write integration test**

Add to `tests/test_retrieval_wikipedia.py`:

```python
class TestWikipediaConfigPersistence:
    @patch("src.tools.retrieval.load_settings")
    def test_config_reads_from_settings_json(self, mock_load):
        mock_load.return_value = {
            "wikipedia_enabled": True,
            "wikipedia_zim_paths": ["/data/wiki.zim"],
            "wikipedia_results": 10,
        }
        from src.tools.retrieval import _get_wikipedia_config
        config = _get_wikipedia_config()
        assert config["enabled"] is True
        assert config["zim_paths"] == ["/data/wiki.zim"]
        assert config["top_k"] == 10

    @patch("src.tools.retrieval.load_settings")
    def test_config_falls_back_to_env_defaults(self, mock_load):
        mock_load.return_value = {}
        from src.tools.retrieval import _get_wikipedia_config
        config = _get_wikipedia_config()
        assert config["enabled"] is False
        assert config["zim_paths"] == []
        assert config["top_k"] == 5
```

- [ ] **Step 2: Run tests**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/test_retrieval_wikipedia.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_wikipedia.py
git commit -m "test: add Wikipedia config persistence integration tests"
```

---

### Task 8: Final integration test and cleanup

- [ ] **Step 1: Run full test suite**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify import chain works end-to-end**

Run: `cd D:/Downloads/Projects/ai_arena/887 && uv run python -c "from src.tools.retrieval import retrieve_wikipedia; print('OK')"` 
Expected: `OK`

- [ ] **Step 3: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: complete Wikipedia ZIM integration (hybrid search)"
```
