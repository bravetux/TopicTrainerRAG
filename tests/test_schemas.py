# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from src.models.schemas import QuizQuestion, QuizResult, TrainingModule, LearningPath


class TestQuizQuestion:
    def test_valid_question(self):
        q = QuizQuestion(
            question="What is a locator in Selenium?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="A locator identifies a web element.",
            difficulty="beginner",
            topic="selenium",
        )
        assert q.question == "What is a locator in Selenium?"
        assert len(q.options) == 4

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Q",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                explanation="E",
                difficulty="expert",
                topic="selenium",
            )


class TestQuizResult:
    def test_valid_result(self):
        q = QuizQuestion(
            question="Q?", options=["A", "B", "C", "D"],
            correct_answer="A", explanation="E",
            difficulty="beginner", topic="selenium",
        )
        result = QuizResult(
            technology="selenium",
            difficulty="beginner",
            questions=[q],
            total_questions=1,
            passing_score=70,
        )
        assert result.total_questions == 1
        assert result.passing_score == 70

    def test_empty_questions_allowed(self):
        result = QuizResult(
            technology="tosca", difficulty="advanced",
            questions=[], total_questions=0, passing_score=70,
        )
        assert result.questions == []


class TestTrainingModule:
    def test_valid_module(self):
        m = TrainingModule(
            title="Intro to Selenium",
            technology="selenium",
            difficulty="beginner",
            duration_minutes=30,
            learning_objectives=["Understand locators"],
            content="# Selenium\nSelenium is...",
            exercises=["Write a login test"],
            references=["selenium_basics.txt"],
        )
        assert m.duration_minutes == 30

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            TrainingModule(
                title="T", technology="t", difficulty="master",
                duration_minutes=10, learning_objectives=[],
                content="c", exercises=[], references=[],
            )


class TestLearningPath:
    def test_valid_path(self):
        lp = LearningPath(
            session_id="abc123",
            current_level="beginner",
            recommended_topics=["selenium basics", "locators"],
            next_milestone="Complete Selenium beginner quiz",
            estimated_hours=4.5,
            weak_areas=["xpath"],
            strong_areas=["css selectors"],
        )
        assert lp.estimated_hours == 4.5

    def test_negative_hours_rejected(self):
        with pytest.raises(ValidationError):
            LearningPath(
                session_id="x", current_level="beginner",
                recommended_topics=[], next_milestone="m",
                estimated_hours=-1.0,
                weak_areas=[], strong_areas=[],
            )
