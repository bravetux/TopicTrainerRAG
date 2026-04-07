# Topic Classification Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users reclassify topics between Built-in Technologies and Custom Topics from the Settings tab, with deletion rights following the resolved classification.

**Architecture:** Add `topic_classifications` dict to `settings.json` (overrides only). `kb_manager.load_all_topics()` reads overrides and sets `is_builtin` accordingly. `delete_custom_topic()` is replaced by `delete_topic()` which checks resolved classification. Settings tab gets a new Topic Classification section.

**Tech Stack:** Python, Streamlit, existing `provider_manager.save_settings()` / `load_settings()`, `kb_manager.load_all_topics()`, `pytest` + `monkeypatch`.

---

## Files

| File | Change |
|---|---|
| `src/tools/provider_manager.py` | Add `"topic_classifications": {}` to `DEFAULTS` |
| `src/tools/kb_manager.py` | Update `load_all_topics()`; replace `delete_custom_topic()` with `delete_topic()` |
| `app.py` | Update import; update delete call-site; add Topic Classification UI section |
| `tests/test_provider_manager.py` | Add test: DEFAULTS contains `topic_classifications` |
| `tests/test_kb_manager.py` | Add tests for classification overrides and `delete_topic()` |

---

## Task 1: Add `topic_classifications` to provider_manager DEFAULTS

**Files:**
- Modify: `src/tools/provider_manager.py`
- Test: `tests/test_provider_manager.py`

- [ ] **Step 1: Write the failing test**

Add to the `TestLoadSettings` class in `tests/test_provider_manager.py`:

```python
def test_defaults_contain_topic_classifications(self, tmp_path, monkeypatch):
    import src.tools.provider_manager as pm
    monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    result = pm.load_settings()
    assert "topic_classifications" in result
    assert result["topic_classifications"] == {}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd D:/Downloads/Projects/ai_arena/887
uv run pytest tests/test_provider_manager.py::TestLoadSettings::test_defaults_contain_topic_classifications -v
```

Expected: `FAILED` — `KeyError: 'topic_classifications'`

- [ ] **Step 3: Add `topic_classifications` to DEFAULTS in `src/tools/provider_manager.py`**

Find the `DEFAULTS` dict (line ~19) and add the new key:

```python
DEFAULTS: dict = {
    "active_provider": "bedrock",
    "model_name": "",
    "base_url": "",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 32768,
    "topic_classifications": {},   # NEW
}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest tests/test_provider_manager.py::TestLoadSettings::test_defaults_contain_topic_classifications -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full provider_manager test suite — no regressions**

```bash
uv run pytest tests/test_provider_manager.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/tools/provider_manager.py tests/test_provider_manager.py
git commit -m "feat(settings): add topic_classifications to settings defaults"
```

---

## Task 2: Update `load_all_topics()` to apply classification overrides

**Files:**
- Modify: `src/tools/kb_manager.py`
- Test: `tests/test_kb_manager.py`

- [ ] **Step 1: Write the failing tests**

Add a new `TestTopicClassificationOverrides` class to `tests/test_kb_manager.py`:

```python
class TestTopicClassificationOverrides:
    def _make_fake_builtin(self, tmp_path):
        return [{"id": "selenium", "display_name": "Selenium",
                 "collection": "tech_selenium",
                 "doc_dir": str(tmp_path / "selenium")}]

    def test_builtin_default_is_builtin(self, tmp_path, monkeypatch):
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", self._make_fake_builtin(tmp_path))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        topics = kb_manager.load_all_topics()
        selenium = next(t for t in topics if t["id"] == "selenium")
        assert selenium["is_builtin"] is True

    def test_demoted_builtin_is_not_builtin(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", self._make_fake_builtin(tmp_path))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"selenium": "custom"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        topics = kb_manager.load_all_topics()
        selenium = next(t for t in topics if t["id"] == "selenium")
        assert selenium["is_builtin"] is False

    def test_promoted_custom_is_builtin(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        reg_path = tmp_path / "reg.json"
        reg_path.write_text(json.dumps({"custom": [
            {"id": "vxworks", "display_name": "VxWorks",
             "collection": "tech_vxworks", "doc_dir": str(tmp_path / "vxworks")}
        ]}))
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(reg_path))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"vxworks": "builtin"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        topics = kb_manager.load_all_topics()
        vxworks = next(t for t in topics if t["id"] == "vxworks")
        assert vxworks["is_builtin"] is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_kb_manager.py::TestTopicClassificationOverrides -v
```

Expected: `FAILED` — `load_all_topics` ignores overrides.

- [ ] **Step 3: Update `load_all_topics()` in `src/tools/kb_manager.py`**

Replace the existing `load_all_topics()` function (lines ~92–110):

```python
def load_all_topics() -> list[dict]:
    """Return all topics (built-in + custom) with runtime status and classification applied."""
    from src.tools.provider_manager import load_settings
    overrides: dict = load_settings().get("topic_classifications", {})

    registry = _load_registry()
    all_topics = []

    for t in BUILTIN_TOPICS:
        topic = dict(t)
        resolved = overrides.get(t["id"], "builtin")
        topic["is_builtin"] = (resolved == "builtin")
        topic.update(get_topic_status(t["id"]))
        all_topics.append(topic)

    for t in registry.get("custom", []):
        topic = dict(t)
        resolved = overrides.get(t["id"], "custom")
        topic["is_builtin"] = (resolved == "builtin")
        topic.update(get_topic_status(t["id"]))
        all_topics.append(topic)

    return all_topics
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_kb_manager.py::TestTopicClassificationOverrides -v
```

Expected: all 3 `PASSED`.

- [ ] **Step 5: Run full kb_manager test suite — no regressions**

```bash
uv run pytest tests/test_kb_manager.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/tools/kb_manager.py tests/test_kb_manager.py
git commit -m "feat(kb): apply topic_classifications overrides in load_all_topics"
```

---

## Task 3: Replace `delete_custom_topic()` with `delete_topic()`

**Files:**
- Modify: `src/tools/kb_manager.py`
- Test: `tests/test_kb_manager.py`

- [ ] **Step 1: Write the failing tests**

Add a `TestDeleteTopic` class to `tests/test_kb_manager.py`:

```python
class TestDeleteTopic:
    def test_delete_user_created_custom_removes_from_registry(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.chdir(tmp_path)
        mock_chroma = MagicMock()
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_topic("jenkins")
        registry = kb_manager._load_registry()
        assert not any(t["id"] == "jenkins" for t in registry["custom"])

    def test_delete_user_created_custom_drops_collection(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", [])
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.chdir(tmp_path)
        mock_chroma = MagicMock()
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.create_custom_topic("Jenkins")
        kb_manager.delete_topic("jenkins")
        mock_chroma.delete_collection.assert_called_once_with("tech_jenkins")

    def test_delete_demoted_builtin_drops_collection_only(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        fake_builtin = [{"id": "selenium", "display_name": "Selenium",
                         "collection": "tech_selenium",
                         "doc_dir": str(tmp_path / "selenium")}]
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", fake_builtin)
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"topic_classifications": {"selenium": "custom"}}))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(settings_file))
        mock_chroma = MagicMock()
        monkeypatch.setattr(kb_manager, "_chroma_client", mock_chroma)
        kb_manager.delete_topic("selenium")
        # collection dropped
        mock_chroma.delete_collection.assert_called_once_with("tech_selenium")
        # NOT removed from registry (was never there)
        registry = kb_manager._load_registry()
        assert registry == {"custom": []}

    def test_delete_builtin_classified_topic_raises(self, tmp_path, monkeypatch):
        import json
        from src.tools import kb_manager
        import src.tools.provider_manager as pm
        fake_builtin = [{"id": "selenium", "display_name": "Selenium",
                         "collection": "tech_selenium",
                         "doc_dir": str(tmp_path / "selenium")}]
        monkeypatch.setattr(kb_manager, "BUILTIN_TOPICS", fake_builtin)
        monkeypatch.setattr(kb_manager, "TOPICS_REGISTRY_PATH", str(tmp_path / "reg.json"))
        monkeypatch.setattr(pm, "SETTINGS_PATH", str(tmp_path / "settings.json"))
        monkeypatch.setattr(kb_manager, "_chroma_client", MagicMock(get_collection=MagicMock(side_effect=Exception())))
        with pytest.raises(ValueError, match="Built-in"):
            kb_manager.delete_topic("selenium")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_kb_manager.py::TestDeleteTopic -v
```

Expected: `FAILED` — `delete_topic` not defined.

- [ ] **Step 3: Add `delete_topic()` to `src/tools/kb_manager.py` and keep `delete_custom_topic()` as a deprecated alias**

Add after the existing `delete_custom_topic()` function:

```python
def delete_topic(topic_id: str) -> None:
    """Delete a topic that is currently classified as Custom.

    - User-created custom topics: removed from registry + ChromaDB collection dropped.
    - Demoted built-in topics: ChromaDB collection dropped only; files kept; topic stays.
    Raises ValueError if the topic's resolved classification is Built-in.
    """
    topic_map = {t["id"]: t for t in load_all_topics()}
    topic = topic_map.get(topic_id)

    if topic is None:
        raise ValueError(f"Unknown topic: {topic_id}")
    if topic.get("is_builtin", True):
        raise ValueError(
            f"'{topic_id}' is classified as Built-in. Demote it to Custom in Settings first."
        )

    # Remove from registry only if user-created (not in hardcoded BUILTIN_TOPICS)
    builtin_ids = {t["id"] for t in BUILTIN_TOPICS}
    if topic_id not in builtin_ids:
        registry = _load_registry()
        registry["custom"] = [t for t in registry.get("custom", []) if t["id"] != topic_id]
        _save_registry(registry)

    # Drop ChromaDB collection (document files are kept)
    try:
        _get_chroma().delete_collection(f"tech_{topic_id}")
    except Exception:
        pass

    logger.info("Deleted topic: %s (builtin_origin=%s)", topic_id, topic_id in builtin_ids)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_kb_manager.py::TestDeleteTopic -v
```

Expected: all 4 `PASSED`.

- [ ] **Step 5: Run full kb_manager suite — no regressions**

```bash
uv run pytest tests/test_kb_manager.py -v
```

Expected: all green. (The existing `TestDeleteCustomTopic` tests still pass because `delete_custom_topic()` still exists.)

- [ ] **Step 6: Commit**

```bash
git add src/tools/kb_manager.py tests/test_kb_manager.py
git commit -m "feat(kb): add delete_topic() with resolved-classification guard"
```

---

## Task 4: Update `app.py` — import + delete call-site

**Files:**
- Modify: `app.py` (imports block ~line 13–21, delete call-site ~line 258)

- [ ] **Step 1: Update the `kb_manager` import block**

Find lines 13–21 in `app.py`:

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
```

Replace `delete_custom_topic` with `delete_topic`:

```python
from src.tools.kb_manager import (
    load_all_topics,
    create_custom_topic,
    delete_topic,
    save_uploaded_file,
    list_topic_files,
    get_available_topic_ids,
    get_available_topics,
)
```

- [ ] **Step 2: Update the delete call-site (~line 258)**

Find:
```python
                    delete_custom_topic(tid)
```

Replace with:
```python
                    delete_topic(tid)
```

- [ ] **Step 3: Verify app starts without import errors**

```bash
uv run python -c "import app" 2>&1 | head -5
```

Expected: no output (clean import).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor(app): use delete_topic() replacing delete_custom_topic()"
```

---

## Task 5: Add Topic Classification UI to the Settings tab

**Files:**
- Modify: `app.py` (~after line 1041, inside `with tab_settings:`)

- [ ] **Step 1: Add the Topic Classification section**

Find the end of the Settings tab content at line ~1036–1041:

```python
    st.caption(
        "Credentials are written to .env (gitignored). "
        "Non-sensitive config is written to data/settings.json. "
        "Values already in .env are pre-loaded in fields above. "
        "Password fields are always blank — re-enter only if you want to change them."
    )
```

Insert the following **after** that `st.caption(...)` block (still inside `with tab_settings:`):

```python
    # ── Topic Classification ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Topic Classification")
    st.caption(
        "Control which topics appear under Built-in Technologies or Custom Topics "
        "in the Knowledge Base tab. Demoted built-in topics become deletable."
    )

    _all_topics_cls = load_all_topics()
    _saved_overrides = load_settings().get("topic_classifications", {})
    _builtin_ids = {t["id"] for t in __import__("src.config", fromlist=["BUILTIN_TOPICS"]).BUILTIN_TOPICS}
    _cls_options = ["Built-in", "Custom"]
    _new_overrides: dict = {}

    for _topic in _all_topics_cls:
        _tid = _topic["id"]
        _default_class = "builtin" if _tid in _builtin_ids else "custom"
        _current_class = _saved_overrides.get(_tid, _default_class)
        _current_label = "Built-in" if _current_class == "builtin" else "Custom"

        _col_name, _col_select = st.columns([3, 1])
        with _col_name:
            st.write(_topic["display_name"])
        with _col_select:
            _selected = st.selectbox(
                label="classification",
                options=_cls_options,
                index=_cls_options.index(_current_label),
                key=f"cls_{_tid}",
                label_visibility="collapsed",
            )
        _resolved = "builtin" if _selected == "Built-in" else "custom"
        if _resolved != _default_class:
            _new_overrides[_tid] = _resolved

    if st.button("💾 Save Classifications", key="save_classifications"):
        try:
            _cls_settings = load_settings()
            _cls_settings["topic_classifications"] = _new_overrides
            save_settings(_cls_settings)
            st.success("Topic classifications saved.")
            st.rerun()
        except Exception as _e:
            st.error(f"Could not save classifications: {_e}")
```

- [ ] **Step 2: Verify app starts without errors**

```bash
uv run python -c "import app" 2>&1 | head -5
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(settings): add Topic Classification section to Settings tab"
```

---

## Task 6: End-to-end smoke test

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_evals.py
```

Expected: all existing tests green. New tests in `TestTopicClassificationOverrides` and `TestDeleteTopic` green.

- [ ] **Step 2: Manual smoke test**

```bash
uv run streamlit run app.py
```

1. Open `http://localhost:8501` → **Settings tab** → scroll to **Topic Classification**
2. Change any topic from "Built-in" to "Custom" → click **Save Classifications**
3. Switch to **Knowledge Base tab** — the topic now appears under **Custom Topics** with a 🗑 delete button
4. Delete it — confirm the ChromaDB collection is dropped but the document files remain in `data/documents/<topic_id>/`
5. Change it back to "Built-in" in Settings → save → KB tab shows it under **Built-in Technologies** again with no delete button

- [ ] **Step 3: Final commit if any fixups were made**

```bash
git add -p
git commit -m "fix: topic classification smoke test fixups"
```
