"""Pydantic models for structured agent outputs."""
from typing import Literal
from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    topic: str


class QuizResult(BaseModel):
    technology: str
    difficulty: str
    questions: list[QuizQuestion]
    total_questions: int
    passing_score: int = Field(default=70, ge=0, le=100)


class TrainingModule(BaseModel):
    title: str
    technology: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    duration_minutes: int = Field(ge=1)
    learning_objectives: list[str]
    content: str
    exercises: list[str]
    references: list[str]


class LearningPath(BaseModel):
    session_id: str
    current_level: str
    recommended_topics: list[str]
    next_milestone: str
    estimated_hours: float = Field(ge=0.0)
    weak_areas: list[str]
    strong_areas: list[str]
