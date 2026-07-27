import asyncio
from pathlib import Path

from deeptutor.course.engine import CourseEngine
from deeptutor.course.models import Course, CourseClassPlan, CourseInputs, CourseOutline, CourseProposal
from deeptutor.course.storage import CourseStorage
from deeptutor.services.path_service import PathService


def test_confirm_outline_creates_classes_and_unlocks_prerequisite(tmp_path, monkeypatch):
    PathService.reset_instance()
    monkeypatch.setattr(PathService, "_instance", PathService(Path(tmp_path / "data")))
    storage = CourseStorage()
    engine = CourseEngine(storage)
    course = Course(title="Course", proposal=CourseProposal(title="Course"))
    storage.save_course(course)
    storage.save_inputs(course.id, CourseInputs(user_intent="Learn"))
    outline = CourseOutline(
        course_id=course.id,
        classes=[
            CourseClassPlan(title="Foundations", learning_objectives=["Explain a concept"]),
            CourseClassPlan(title="Build", prerequisites=["Foundations"], learning_objectives=["Apply it"]),
        ],
    )
    created = asyncio.run(engine.confirm_outline(course.id, outline))
    assert created[0].status.value == "next"
    assert created[1].status.value == "locked"
    engine.update_class(course.id, created[0].id, {"status": "complete"})
    assert storage.load_class(course.id, created[1].id).status.value == "next"


def test_resource_links_are_deduplicated():
    links = CourseEngine._normalize_links([
        {"url": "https://example.com", "title": "One"},
        {"url": "https://example.com", "title": "Duplicate"},
        {"url": "https://example.org"},
    ])
    assert [link.url for link in links] == ["https://example.com", "https://example.org"]
