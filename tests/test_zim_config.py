"""Tests for Wikipedia ZIM configuration."""
from src.config import WIKIPEDIA_ZIM_PATHS, WIKIPEDIA_ENABLED, WIKIPEDIA_RESULTS


def test_wikipedia_defaults():
    assert WIKIPEDIA_ENABLED is False
    assert WIKIPEDIA_RESULTS == 5
    assert isinstance(WIKIPEDIA_ZIM_PATHS, list)
    assert len(WIKIPEDIA_ZIM_PATHS) == 0
