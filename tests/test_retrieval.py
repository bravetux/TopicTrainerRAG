"""Tests for ChromaDB retrieval tools."""
import pytest
from unittest.mock import MagicMock
from src.tools.retrieval import query_collection


class TestQueryCollection:
    def test_returns_formatted_string(self):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("test_col")
        col.add(
            ids=["doc1"],
            embeddings=[[0.1] * 1024],
            documents=["Selenium uses locators to find elements."],
            metadatas=[{"source_file": "selenium.txt"}],
        )
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_collection(
            query="selenium locators",
            collection_name="test_col",
            top_k=1,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "Selenium" in result
        assert "selenium.txt" in result

    def test_empty_collection_returns_message(self):
        import chromadb
        client = chromadb.EphemeralClient()
        client.create_collection("empty_col")
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_collection(
            query="xyz",
            collection_name="empty_col",
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert isinstance(result, str)

    def test_top_k_respected(self):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("topk_col")
        for i in range(10):
            col.add(
                ids=[f"doc{i}"],
                embeddings=[[float(i) * 0.01 + 0.001] * 1024],
                documents=[f"Document {i} about selenium testing"],
                metadatas=[{"source_file": f"file{i}.txt"}],
            )
        mock_embed = MagicMock(return_value=[[0.05] * 1024])
        result = query_collection(
            query="selenium",
            collection_name="topk_col",
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert result.count("Source:") <= 3

    def test_missing_collection_returns_message(self):
        import chromadb
        client = chromadb.EphemeralClient()
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_collection(
            query="anything",
            collection_name="nonexistent_col",
            top_k=5,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestQueryMultiCollections:
    def test_merges_results_from_two_collections(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        col1 = client.create_collection("col_a")
        col1.add(ids=["a1"], embeddings=[[0.1] * 1024],
                 documents=["Selenium locators guide"], metadatas=[{"source_file": "sel.txt"}])
        col2 = client.create_collection("col_b")
        col2.add(ids=["b1"], embeddings=[[0.2] * 1024],
                 documents=["Tosca test case creation"], metadatas=[{"source_file": "tosca.txt"}])

        mock_embed = MagicMock(return_value=[[0.15] * 1024])
        result = query_multi_collections(
            query="test",
            collection_names=["col_a", "col_b"],
            top_k=5,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "sel.txt" in result or "tosca.txt" in result

    def test_skips_missing_collections_gracefully(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        col = client.create_collection("exists")
        col.add(ids=["x1"], embeddings=[[0.1] * 1024],
                documents=["Some content"], metadatas=[{"source_file": "doc.txt"}])

        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_multi_collections(
            query="content",
            collection_names=["exists", "does_not_exist"],
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert "Source:" in result

    def test_all_empty_returns_message(self):
        import chromadb
        from src.tools.retrieval import query_multi_collections
        from unittest.mock import MagicMock

        client = chromadb.EphemeralClient()
        mock_embed = MagicMock(return_value=[[0.1] * 1024])
        result = query_multi_collections(
            query="anything",
            collection_names=["missing1", "missing2"],
            top_k=3,
            chroma_client=client,
            embed_fn=mock_embed,
        )
        assert isinstance(result, str)
        assert "No training content" in result
