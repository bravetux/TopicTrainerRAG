# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Shared pytest fixtures."""
import sys
import types
from unittest.mock import MagicMock
import pytest
from pathlib import Path


def _mock_chromadb():
    """Insert a fake chromadb package into sys.modules to avoid opentelemetry conflicts."""
    if "chromadb" in sys.modules:
        return

    chromadb_mod = types.ModuleType("chromadb")
    chromadb_config = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs):
            pass

    class PersistentClient(MagicMock):
        pass

    chromadb_config.Settings = Settings
    chromadb_mod.Settings = Settings
    chromadb_mod.PersistentClient = PersistentClient

    sys.modules["chromadb"] = chromadb_mod
    sys.modules["chromadb.config"] = chromadb_config


_mock_chromadb()


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_progress.db")


@pytest.fixture
def fixture_qa_doc():
    return str(Path(__file__).parent / "fixtures/documents/qa/selenium_basics.txt")


@pytest.fixture
def fixture_etl_doc():
    return str(Path(__file__).parent / "fixtures/documents/etl/aws_glue_overview.txt")
