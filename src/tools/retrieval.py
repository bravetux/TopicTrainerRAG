# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""ChromaDB retrieval tools — per-technology and multi-collection queries."""
import logging
import math
from pathlib import Path
from typing import Callable, Optional

import chromadb
from chromadb.config import Settings
from strands import tool

from src.config import (
    CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K,
    QA_TOPIC_IDS, ETL_TOPIC_IDS, BUILTIN_TOPICS,
    WIKIPEDIA_ENABLED, WIKIPEDIA_ZIM_PATHS, WIKIPEDIA_RESULTS,
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
    from src.tools.zim_reader import search_multiple_zim

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
