"""Tests for SQLite progress tools."""
import json
import pytest
from src.tools.progress_db import init_db, write_quiz_result, read_progress


class TestInitDb:
    def test_creates_tables(self, tmp_db):
        init_db(tmp_db)
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "quiz_results" in table_names
        assert "topics_studied" in table_names

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # second call should not raise


class TestWriteQuizResult:
    def test_writes_record(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="selenium",
            difficulty="beginner",
            score=80,
            total_questions=5,
            correct_answers=4,
        )
        assert record_id > 0

    def test_score_zero(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="selenium",
            difficulty="beginner",
            score=0,
            total_questions=5,
            correct_answers=0,
        )
        assert record_id > 0

    def test_score_hundred(self, tmp_db):
        init_db(tmp_db)
        record_id = write_quiz_result(
            db_path=tmp_db,
            session_id="sess1",
            technology="spark",
            difficulty="advanced",
            score=100,
            total_questions=10,
            correct_answers=10,
        )
        assert record_id > 0


class TestReadProgress:
    def test_empty_session_returns_empty(self, tmp_db):
        init_db(tmp_db)
        result = read_progress(db_path=tmp_db, session_id="unknown")
        data = json.loads(result)
        assert data["quiz_results"] == []
        assert data["technologies"] == {}

    def test_reads_written_results(self, tmp_db):
        init_db(tmp_db)
        write_quiz_result(tmp_db, "s1", "selenium", "beginner", 80, 5, 4)
        write_quiz_result(tmp_db, "s1", "selenium", "intermediate", 60, 5, 3)
        result = read_progress(db_path=tmp_db, session_id="s1")
        data = json.loads(result)
        assert len(data["quiz_results"]) == 2
        assert "selenium" in data["technologies"]
        assert data["technologies"]["selenium"]["avg_score"] == 70.0

    def test_isolates_sessions(self, tmp_db):
        init_db(tmp_db)
        write_quiz_result(tmp_db, "s1", "selenium", "beginner", 90, 5, 5)
        write_quiz_result(tmp_db, "s2", "tosca", "beginner", 50, 5, 2)
        result = read_progress(db_path=tmp_db, session_id="s1")
        data = json.loads(result)
        assert len(data["quiz_results"]) == 1
        assert "tosca" not in data["technologies"]
