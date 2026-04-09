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
            {"title": "Python (language)", "content": "Python is a language.", "source": "Wikipedia (wiki)"},
        ]

        from src.tools.retrieval import retrieve_wikipedia
        # Call the underlying function (unwrap Strands @tool)
        result = retrieve_wikipedia._tool_func(
            query="what is python",
            top_k=3,
        )
        assert "[1] Source: Wikipedia (wiki)" in result
        assert "Python (language)" in result
        assert "Python is a language." in result

    @patch("src.tools.retrieval._get_wikipedia_config")
    def test_returns_message_when_disabled(self, mock_config):
        mock_config.return_value = {
            "enabled": False,
            "zim_paths": [],
            "top_k": 5,
        }
        from src.tools.retrieval import retrieve_wikipedia
        result = retrieve_wikipedia._tool_func(
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
        result = retrieve_wikipedia._tool_func(
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
