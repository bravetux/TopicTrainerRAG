"""Tests for document ingestion pipeline."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.tools.document_ingestion import parse_document, chunk_text, get_supported_extensions


class TestGetSupportedExtensions:
    def test_returns_expected_extensions(self):
        exts = get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".xlsx" in exts
        assert ".pptx" in exts
        assert ".md" in exts


class TestParseDocument:
    def test_parse_txt(self, fixture_qa_doc):
        text = parse_document(fixture_qa_doc)
        assert "Selenium" in text
        assert len(text) > 100

    def test_parse_etl_txt(self, fixture_etl_doc):
        text = parse_document(fixture_etl_doc)
        assert "Glue" in text

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(str(bad_file))

    def test_empty_txt_returns_empty(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        result = parse_document(str(empty))
        assert result == ""


class TestChunkText:
    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=1000, overlap=200)
        assert len(chunks) >= 1
        assert "Hello world" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        long_text = "word " * 600  # ~3000 chars
        chunks = chunk_text(long_text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        long_text = " ".join([f"sentence{i}" for i in range(200)])
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        if len(chunks) > 1:
            last_words = chunks[0].split()[-5:]
            chunk1_text = chunks[1]
            assert any(w in chunk1_text for w in last_words)


class TestIndexTechnology:
    def test_index_technology_returns_int(self, tmp_path, monkeypatch):
        """index_technology returns 0 when directory is empty."""
        import chromadb
        from src.tools.document_ingestion import index_technology

        fake_topic = {
            "id": "test_tech",
            "display_name": "Test Tech",
            "collection": "tech_test_tech",
            "doc_dir": str(tmp_path / "test_tech"),
        }
        (tmp_path / "test_tech").mkdir()

        mock_bedrock = MagicMock()
        chroma_client = chromadb.EphemeralClient()

        with patch("src.tools.document_ingestion.boto3") as mock_boto3, \
             patch("src.tools.document_ingestion.get_chroma_client", return_value=chroma_client), \
             patch("src.tools.document_ingestion._find_topic", return_value=fake_topic):
            mock_boto3.client.return_value = mock_bedrock
            result = index_technology("test_tech")
        assert isinstance(result, int)
        assert result == 0  # empty directory → 0 chunks
