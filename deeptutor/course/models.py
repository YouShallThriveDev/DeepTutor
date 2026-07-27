"""Persistent models for the interactive Courses workspace."""
from __future__ import annotations

from enum import Enum
import time
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _now() -> float:
    return time.time()


class CourseStatus(str, Enum):
    DRAFT = "draft"
    OUTLINE_READY = "outline_ready"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class ClassStatus(str, Enum):
    LOCKED = "locked"
    NEXT = "next"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETE = "complete"


class AssignmentStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    COMPLETE = "complete"


class ResourceLink(BaseModel):
    """A learner-supplied public URL used as untrusted research material."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("res"))
    url: str
    title: str = ""
    note: str = ""
    final_url: str = ""
    fetched_title: str = ""
    excerpt: str = ""
    status: str = "pending"  # pending | researched | unavailable
    error: str = ""
    truncated: bool = False
    created_at: float = Field(default_factory=_now)


class CourseInputs(BaseModel):
    """Immutable source snapshot captured before a course is proposed."""

    model_config = ConfigDict(extra="ignore")

    user_intent: str = ""
    chat_session_id: str = ""
    chat_selections: list[dict[str, Any]] = Field(default_factory=list)
    notebook_refs: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_bases: list[str] = Field(default_factory=list)
    question_categories: list[int] = Field(default_factory=list)
    question_entries: list[int] = Field(default_factory=list)
    resource_links: list[ResourceLink] = Field(default_factory=list)
    research_brief: str = ""
    language: str = "en"
    captured_at: float = Field(default_factory=_now)


class CourseProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    description: str = ""
    scope: str = ""
    target_level: str = "mixed"
    estimated_classes: int = 4
    duration_weeks: int = 4
    rationale: str = ""
    capstone_title: str = ""
    capstone_description: str = ""


class Tutorial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("tutorial"))
    title: str = ""
    content: str = ""
    estimated_minutes: int = 20
    status: str = "not_started"  # not_started | in_progress | complete


class Assignment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("assignment"))
    title: str = ""
    prompt: str = ""
    deliverable: str = ""
    rubric: list[str] = Field(default_factory=list)
    status: AssignmentStatus = AssignmentStatus.TODO
    learner_submission: str = ""
    feedback: str = ""


class ClassProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    brief: str = ""
    milestones: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    progress: int = 0


class CourseClassPlan(BaseModel):
    """Editable class plan generated before persistent class workspaces exist."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("class"))
    title: str = ""
    summary: str = ""
    learning_objectives: list[str] = Field(default_factory=list)
    knowledge_points: list[dict[str, str]] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    tutorial_titles: list[str] = Field(default_factory=list)
    assignment_titles: list[str] = Field(default_factory=list)
    project_title: str = ""
    resource_ids: list[str] = Field(default_factory=list)
    order: int = 0


class CourseOutline(BaseModel):
    model_config = ConfigDict(extra="ignore")

    course_id: str
    classes: list[CourseClassPlan] = Field(default_factory=list)
    course_objectives: list[str] = Field(default_factory=list)
    version: int = 1
    updated_at: float = Field(default_factory=_now)


class CourseClass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("class"))
    course_id: str
    title: str = ""
    summary: str = ""
    order: int = 0
    status: ClassStatus = ClassStatus.LOCKED
    learning_objectives: list[str] = Field(default_factory=list)
    knowledge_points: list[dict[str, str]] = Field(default_factory=list)
    prerequisite_ids: list[str] = Field(default_factory=list)
    resource_ids: list[str] = Field(default_factory=list)
    book_ids: list[str] = Field(default_factory=list)
    # Content is compiled from the approved plan on first availability, never filled with generic shells.
    content_status: str = "pending"  # pending | generating | ready | error
    content_error: str = ""
    tutorials: list[Tutorial] = Field(default_factory=list)
    assignments: list[Assignment] = Field(default_factory=list)
    notes: str = ""
    project: ClassProject = Field(default_factory=ClassProject)
    chat_session_id: str = ""
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)


class CourseProgress(BaseModel):
    model_config = ConfigDict(extra="ignore")

    course_id: str
    current_class_id: str = ""
    completed_class_ids: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=_now)


class Course(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: _id("course"))
    title: str = ""
    description: str = ""
    status: CourseStatus = CourseStatus.DRAFT
    proposal: CourseProposal | None = None
    knowledge_bases: list[str] = Field(default_factory=list)
    resource_links: list[ResourceLink] = Field(default_factory=list)
    language: str = "en"
    class_count: int = 0
    completed_class_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
