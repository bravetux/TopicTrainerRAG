<!--
  Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
  Date   : 26 March 2026
-->
# Topic Classification Control — Design Specification

**Date:** 2026-04-07
**Project:** AG-UC-0887 | TechTrainer AI
**Status:** Approved

---

## 1. Overview

Users can control, from the **Settings tab**, whether each topic appears under **"Built-in Technologies"** or **"Custom Topics"** in the Knowledge Base tab.

Classification also governs deletion rights:
- `Built-in` → protected (no delete button shown)
- `Custom` → deletable (delete button shown)

A built-in topic demoted to Custom and then deleted has its ChromaDB collection dropped and document files kept. A user-created custom topic promoted to Built-in becomes protected.

### Non-Goals
- Renaming sections
- Creating more than two groups
- Moving document files on reclassification

---

## 2. Data Model

`data/settings.json` gains a `topic_classifications` key storing only **overrides** from the defaults. Topics not listed keep their default classification.

```json
{
  "active_provider": "bedrock",
  "topic_classifications": {
    "selenium": "custom",
    "my_topic": "builtin"
  }
}
```

**Default classification rules:**
| Topic origin | Default |
|---|---|
| In `BUILTIN_TOPICS` (config.py) | `"builtin"` |
| In `topics_registry.json` (user-created) | `"custom"` |

**Resolution:** `overrides.get(topic_id, default)`

---

## 3. Component Changes

### 3.1 `src/tools/provider_manager.py`

Add `"topic_classifications": {}` to `DEFAULTS`:

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

No other changes. `load_settings()` and `save_settings()` work unchanged.

---

### 3.2 `src/tools/kb_manager.py`

**`load_all_topics()` — apply classification overrides:**

```python
def load_all_topics() -> list[dict]:
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

**`delete_custom_topic()` — check resolved classification, not hardcoded list:**

Rename to `delete_topic()` and change the guard:

```python
def delete_topic(topic_id: str) -> None:
    """Delete a topic that is currently classified as Custom.

    - User-created custom topics: removed from registry + ChromaDB collection dropped.
    - Demoted built-in topics: ChromaDB collection dropped only (files kept, topic stays).
    Raises ValueError if topic is classified as Built-in.
    """
    resolved = load_all_topics()
    topic_map = {t["id"]: t for t in resolved}
    topic = topic_map.get(topic_id)

    if topic is None:
        raise ValueError(f"Unknown topic: {topic_id}")
    if topic.get("is_builtin", True):
        raise ValueError(
            f"'{topic_id}' is classified as Built-in. Demote it to Custom in Settings first."
        )

    # Remove from registry only if user-created (not in BUILTIN_TOPICS)
    builtin_ids = {t["id"] for t in BUILTIN_TOPICS}
    if topic_id not in builtin_ids:
        registry = _load_registry()
        registry["custom"] = [t for t in registry.get("custom", []) if t["id"] != topic_id]
        _save_registry(registry)

    # Drop ChromaDB collection (keep document files)
    try:
        _get_chroma().delete_collection(f"tech_{topic_id}")
    except Exception:
        pass

    logger.info("Deleted topic: %s (builtin_origin=%s)", topic_id, topic_id in builtin_ids)
```

**Call-site in `app.py`:** update `delete_custom_topic(tid)` → `delete_topic(tid)`.

---

### 3.3 `app.py` — Settings Tab

Add a **Topic Classification** section at the bottom of the Settings tab, after the existing Save button:

```python
st.divider()
st.subheader("🗂️ Topic Classification")
st.caption(
    "Control which topics appear under Built-in Technologies or Custom Topics "
    "in the Knowledge Base tab. Demoted built-in topics become deletable."
)

all_topics = load_all_topics()
saved_overrides = load_settings().get("topic_classifications", {})
OPTIONS = ["Built-in", "Custom"]

new_overrides = {}
for topic in all_topics:
    tid = topic["id"]
    default_class = "builtin" if topic["id"] in {t["id"] for t in BUILTIN_TOPICS} else "custom"
    current_class = saved_overrides.get(tid, default_class)
    current_label = "Built-in" if current_class == "builtin" else "Custom"

    col_name, col_select = st.columns([2, 1])
    with col_name:
        st.write(topic["display_name"])
    with col_select:
        selected = st.selectbox(
            label="",
            options=OPTIONS,
            index=OPTIONS.index(current_label),
            key=f"cls_{tid}",
            label_visibility="collapsed",
        )
    # Only store if different from default
    resolved = "builtin" if selected == "Built-in" else "custom"
    if resolved != default_class:
        new_overrides[tid] = resolved

if st.button("Save Classifications", type="primary"):
    settings = load_settings()
    settings["topic_classifications"] = new_overrides
    save_settings(settings)
    st.success("Topic classifications saved.")
    st.rerun()
```

---

## 4. Deletion Behaviour Matrix

| Topic type | Classification | Delete button shown? | On delete |
|---|---|---|---|
| Built-in (hardcoded) | Built-in (default) | No | — |
| Built-in (hardcoded) | Custom (demoted) | Yes | Drop ChromaDB collection; keep files; topic stays in Custom section at NOT INDEXED |
| User-created | Custom (default) | Yes | Remove from registry + drop ChromaDB collection |
| User-created | Built-in (promoted) | No | — |

---

## 5. Error Handling

- `delete_topic()` raises `ValueError` if called on a Built-in-classified topic — app catches and shows `st.error()`.
- If `save_settings()` fails, show `st.error("Could not save classifications.")`.
- `load_all_topics()` import of `provider_manager` is local (inside function) to avoid circular imports between `kb_manager` and `provider_manager`.

---

## 6. Testing

| Test | File |
|---|---|
| `load_all_topics()` returns `is_builtin=False` for a demoted built-in | `test_kb_manager.py` |
| `load_all_topics()` returns `is_builtin=True` for a promoted custom | `test_kb_manager.py` |
| `delete_topic()` raises on Built-in-classified topic | `test_kb_manager.py` |
| `delete_topic()` drops collection but not files for demoted built-in | `test_kb_manager.py` |
| `delete_topic()` removes registry entry for user-created custom | `test_kb_manager.py` |
| `DEFAULTS` contains `topic_classifications: {}` | `test_provider_manager.py` |
