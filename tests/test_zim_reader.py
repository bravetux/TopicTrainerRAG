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
