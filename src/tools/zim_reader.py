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
