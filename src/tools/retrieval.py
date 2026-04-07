# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""ChromaDB retrieval tools — per-technology and multi-collection queries."""
import logging
from typing import Callable, Optional

import chromadb
from chromadb.config import Settings
from strands import tool

from src.config import (
    CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K,
    QA_TOPIC_IDS, ETL_TOPIC_IDS, BUILTIN_TOPICS,
)
from src.tools.embedding_manager import embed_texts as _embed_texts

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


def _embed(texts: list) -> list:
    """Embed texts using the configured embedding provider."""
    return _embed_texts(texts)


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

    if not collection_names:
        return "No training content found. Please run document ingestion first."

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
        except Exception as exc:
            logger.warning("Failed to query collection %s: %s", collection_name, exc)
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
            f"Please go to the \U0001f4da Knowledge Base tab and upload documents for this topic."
        )
    return query_collection(query, topic["collection"], top_k)
