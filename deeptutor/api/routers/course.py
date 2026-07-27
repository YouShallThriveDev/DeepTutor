"""REST API for interactive Courses."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deeptutor.course import CourseOutline, CourseProposal, get_course_engine

router = APIRouter()


class CreateCourseRequest(BaseModel):
    user_intent: str
    chat_session_id: str = ""
    chat_selections: list[dict[str, Any]] = Field(default_factory=list)
    notebook_refs: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_bases: list[str] = Field(default_factory=list)
    question_categories: list[int] = Field(default_factory=list)
    question_entries: list[int] = Field(default_factory=list)
    resource_links: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "en"


class ConfirmProposalRequest(BaseModel):
    proposal: dict[str, Any] | None = None


class ConfirmOutlineRequest(BaseModel):
    outline: dict[str, Any] | None = None


class PatchClassRequest(BaseModel):
    patch: dict[str, Any]


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "course"}


@router.get("/courses")
async def list_courses() -> dict[str, Any]:
    return {"courses": [item.model_dump(mode="json") for item in get_course_engine().list_courses()]}


@router.get("/courses/{course_id}")
async def get_course(course_id: str) -> dict[str, Any]:
    detail = get_course_engine().detail(course_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Course not found")
    course, outline, classes, progress = detail
    return {
        "course": course.model_dump(mode="json"),
        "outline": outline.model_dump(mode="json") if outline else None,
        "classes": [item.model_dump(mode="json") for item in classes],
        "progress": progress.model_dump(mode="json"),
    }


@router.get("/courses/{course_id}/classes/{class_id}")
async def get_class(course_id: str, class_id: str) -> dict[str, Any]:
    item = get_course_engine().storage.load_class(course_id, class_id)
    if not item:
        raise HTTPException(status_code=404, detail="Class not found")
    return {"class": item.model_dump(mode="json")}


@router.post("/courses")
async def create_course(body: CreateCourseRequest) -> dict[str, Any]:
    try:
        course, proposal, inputs = await get_course_engine().create_course(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Course research/proposal failed: {exc}") from exc
    return {
        "course": course.model_dump(mode="json"),
        "proposal": proposal.model_dump(mode="json"),
        "research_brief": inputs.research_brief,
        "resource_links": [item.model_dump(mode="json") for item in inputs.resource_links],
    }


@router.post("/courses/{course_id}/confirm-proposal")
async def confirm_proposal(course_id: str, body: ConfirmProposalRequest) -> dict[str, Any]:
    try:
        proposal = CourseProposal.model_validate(body.proposal) if body.proposal else None
        course, outline = await get_course_engine().confirm_proposal(course_id, proposal)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Course outline failed: {exc}") from exc
    return {"course": course.model_dump(mode="json"), "outline": outline.model_dump(mode="json")}


@router.post("/courses/{course_id}/confirm-outline")
async def confirm_outline(course_id: str, body: ConfirmOutlineRequest) -> dict[str, Any]:
    try:
        outline = CourseOutline.model_validate(body.outline) if body.outline else None
        classes = await get_course_engine().confirm_outline(course_id, outline)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Class creation failed: {exc}") from exc
    return {"classes": [item.model_dump(mode="json") for item in classes]}


@router.patch("/courses/{course_id}/classes/{class_id}")
async def patch_class(course_id: str, class_id: str, body: PatchClassRequest) -> dict[str, Any]:
    try:
        item = get_course_engine().update_class(course_id, class_id, body.patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"class": item.model_dump(mode="json")}


@router.delete("/courses/{course_id}")
async def delete_course(course_id: str) -> dict[str, Any]:
    if not get_course_engine().delete_course(course_id):
        raise HTTPException(status_code=404, detail="Course not found")
    return {"deleted": True, "course_id": course_id}
