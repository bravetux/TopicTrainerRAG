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
    """Deprecated alias for delete_topic(). Use delete_topic() instead."""
    import warnings
    warnings.warn(
        "delete_custom_topic() is deprecated; use delete_topic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    delete_topic(topic_id)


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
