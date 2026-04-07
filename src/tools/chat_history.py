# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Persistent chat history backed by ChromaDB.

Stores Q&A pairs in a dedicated 'chat_history' collection.
Embeddings are stored explicitly as zero vectors — no semantic search needed.
"""
import time
import logging
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chat_history"
MAX_HISTORY = 50
_EMB_DIM = 4  # minimal dimension for the dummy embedding

from typing import Optional
_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        COLLECTION_NAME,
        metadata={"description": "TechTrainer chat history"},
    )


def save_exchange(question: str, answer: str) -> None:
    """Persist a single Q&A exchange. Trims to MAX_HISTORY after saving."""
    try:
        col = _get_collection()
        ts = int(time.time() * 1000)
        col.upsert(
            ids=[f"chat_{ts}"],
            documents=[question],
            metadatas=[{"question": question, "answer": answer, "ts": ts}],
            embeddings=[[0.0] * _EMB_DIM],
        )
        _trim(col)
        logger.debug("Saved chat exchange ts=%d", ts)
    except Exception as exc:
        logger.error("Could not save chat exchange: %s", exc)
        raise


def load_history() -> list[dict]:
    """Return last MAX_HISTORY exchanges as a flat messages list.

    Returns:
        List of {"role": "user"|"assistant", "content": str} dicts,
        oldest first (ready to append to st.session_state.messages).
    """
    try:
        col = _get_collection()
        count = col.count()
        if count == 0:
            return []
        result = col.get(include=["metadatas"])
        pairs = sorted(result["metadatas"], key=lambda m: m.get("ts", 0))
        messages: list[dict] = []
        for pair in pairs[-MAX_HISTORY:]:
            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})
        logger.debug("Loaded %d exchanges from chat history", len(messages) // 2)
        return messages
    except Exception as exc:
        logger.error("Could not load chat history: %s", exc)
        return []


def clear_history() -> None:
    """Delete the entire chat history collection."""
    global _client
    try:
        client = _get_client()
        client.delete_collection(COLLECTION_NAME)
        logger.info("Chat history cleared.")
    except Exception as exc:
        logger.warning("Could not clear chat history: %s", exc)


def _trim(col: chromadb.Collection) -> None:
    """Remove oldest entries so the collection stays within MAX_HISTORY."""
    count = col.count()
    if count <= MAX_HISTORY:
        return
    result = col.get(include=["metadatas"])
    ordered = sorted(
        zip(result["ids"], result["metadatas"]),
        key=lambda x: x[1].get("ts", 0),
    )
    to_delete = [doc_id for doc_id, _ in ordered[: count - MAX_HISTORY]]
    if to_delete:
        col.delete(ids=to_delete)
