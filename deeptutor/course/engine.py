"""Course lifecycle: research -> proposal -> editable outline -> class workspaces."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from deeptutor.book.inputs import build_book_inputs
from deeptutor.learning.models import KnowledgePoint, KnowledgeType, LearningModule
from deeptutor.learning.service import LearningService
from deeptutor.tools.web_fetch import fetch_url_as_markdown

from .agents import CourseIdeationAgent, CoursePlannerAgent, CourseResearchAgent
from .models import (
    Assignment,
    AssignmentStatus,
    ClassProject,
    ClassStatus,
    Course,
    CourseClass,
    CourseInputs,
    CourseOutline,
    CourseProgress,
    CourseProposal,
    CourseStatus,
    ResourceLink,
    Tutorial,
)
from .storage import CourseStorage, get_course_storage


class CourseEngine:
    def __init__(self, storage: CourseStorage | None = None) -> None:
        self.storage = storage or get_course_storage()
        self._lock = asyncio.Lock()

    def list_courses(self) -> list[Course]:
        values = [self.storage.load_course(course_id) for course_id in self.storage.list_ids()]
        return sorted((value for value in values if value), key=lambda value: value.updated_at, reverse=True)

    def detail(self, course_id: str) -> tuple[Course, CourseOutline | None, list[CourseClass], CourseProgress] | None:
        course = self.storage.load_course(course_id)
        if not course:
            return None
        progress = self.storage.load_progress(course_id) or CourseProgress(course_id=course_id)
        return course, self.storage.load_outline(course_id), self.storage.list_classes(course_id), progress

    async def create_course(self, **raw: Any) -> tuple[Course, CourseProposal, CourseInputs]:
        intent = str(raw.get("user_intent") or "").strip()
        if not intent:
            raise ValueError("user_intent is required")
        links = self._normalize_links(raw.get("resource_links") or [])
        source_snapshot, ideation_context = await build_book_inputs(
            user_intent=intent,
            chat_session_id=str(raw.get("chat_session_id") or ""),
            chat_selections=raw.get("chat_selections") or [],
            notebook_refs=raw.get("notebook_refs") or [],
            knowledge_bases=raw.get("knowledge_bases") or [],
            question_categories=raw.get("question_categories") or [],
            question_entries=raw.get("question_entries") or [],
            language=str(raw.get("language") or "en"),
        )
        researched = await self._research_links(links)
        evidence = self._resource_evidence(researched)
        research_brief = await CourseResearchAgent(language=source_snapshot.language).process(intent, evidence)
        proposal = await CourseIdeationAgent(language=source_snapshot.language).process(
            intent, ideation_context.render(), research_brief
        )
        course = Course(
            title=proposal.title,
            description=proposal.description,
            proposal=proposal,
            knowledge_bases=source_snapshot.knowledge_bases,
            resource_links=researched,
            language=source_snapshot.language,
        )
        inputs = CourseInputs(
            user_intent=intent,
            chat_session_id=source_snapshot.chat_session_id,
            chat_selections=[item.model_dump() for item in source_snapshot.chat_selections],
            notebook_refs=[item.model_dump() for item in source_snapshot.notebook_refs],
            knowledge_bases=source_snapshot.knowledge_bases,
            question_categories=source_snapshot.question_categories,
            question_entries=source_snapshot.question_entries,
            resource_links=researched,
            research_brief=research_brief,
            language=source_snapshot.language,
        )
        self.storage.save_course(course)
        self.storage.save_inputs(course.id, inputs)
        self.storage.save_progress(CourseProgress(course_id=course.id))
        self.storage.append_log(course.id, "course created; resources researched before proposal")
        return course, proposal, inputs

    async def confirm_proposal(self, course_id: str, proposal: CourseProposal | None = None) -> tuple[Course, CourseOutline]:
        course = self.storage.load_course(course_id)
        inputs = self.storage.load_inputs(course_id)
        if not course or not inputs:
            raise ValueError("Course not found")
        if proposal is not None:
            course.proposal = proposal
            course.title = proposal.title or course.title
            course.description = proposal.description or course.description
        approved = course.proposal or proposal
        if not approved:
            raise ValueError("Course proposal is missing")
        outline = await CoursePlannerAgent(language=course.language).process(
            course.id, approved, inputs.research_brief, [item.id for item in course.resource_links]
        )
        course.status = CourseStatus.OUTLINE_READY
        course.class_count = len(outline.classes)
        course.updated_at = time.time()
        self.storage.save_course(course)
        self.storage.save_outline(outline)
        self.storage.append_log(course.id, "proposal confirmed; class outline generated")
        return course, outline

    async def confirm_outline(self, course_id: str, outline: CourseOutline | None = None) -> list[CourseClass]:
        async with self._lock:
            course = self.storage.load_course(course_id)
            final = outline or self.storage.load_outline(course_id)
            if not course or not final:
                raise ValueError("Course outline not found")
            final.course_id = course_id
            id_by_title = {item.title.lower(): item.id for item in final.classes}
            classes: list[CourseClass] = []
            for index, plan in enumerate(final.classes):
                plan.order = index
                prereq_ids = [id_by_title.get(value.lower(), value) for value in plan.prerequisites]
                tutorials = [
                    Tutorial(
                        title=title,
                        content=(
                            f"## {title}\n\nThis interactive tutorial belongs to **{plan.title}**. "
                            "Open the class tutor to explore the material, ask for examples, and check your understanding."
                        ),
                    )
                    for title in (plan.tutorial_titles or ["Guided tutorial"])
                ]
                assignments = [
                    Assignment(
                        title=title,
                        prompt=f"Complete the practical work for: {title}.",
                        deliverable="A concise explanation, artifact, or implementation demonstrating the objective.",
                        rubric=["Addresses the learning objective", "Shows reasoning", "Reflects on what was learned"],
                    )
                    for title in (plan.assignment_titles or ["Practice assignment"])
                ]
                item = CourseClass(
                    id=plan.id,
                    course_id=course_id,
                    title=plan.title,
                    summary=plan.summary,
                    order=index,
                    status=ClassStatus.NEXT if index == 0 and not prereq_ids else ClassStatus.LOCKED,
                    learning_objectives=plan.learning_objectives,
                    knowledge_points=plan.knowledge_points,
                    prerequisite_ids=prereq_ids,
                    resource_ids=plan.resource_ids,
                    tutorials=tutorials,
                    assignments=assignments,
                    project=ClassProject(
                        title=plan.project_title or f"{plan.title} project",
                        brief=f"Build a small project that applies the objectives from {plan.title}.",
                        milestones=["Plan", "Build", "Reflect"],
                        success_criteria=["Uses the class objectives", "Explains design choices"],
                    ),
                )
                classes.append(item)
                self.storage.save_class(item)
            self.storage.save_outline(final)
            self._sync_learning_path(course_id, classes)
            course.status = CourseStatus.READY
            course.class_count = len(classes)
            course.updated_at = time.time()
            self.storage.save_course(course)
            self.storage.append_log(course_id, f"outline confirmed; {len(classes)} class workspaces created")
            return classes

    def update_class(self, course_id: str, class_id: str, patch: dict[str, Any]) -> CourseClass:
        item = self.storage.load_class(course_id, class_id)
        if not item:
            raise ValueError("Class not found")
        allowed = {"notes", "book_ids", "chat_session_id", "status", "tutorials", "assignments", "project"}
        values = {key: value for key, value in patch.items() if key in allowed}
        if "status" in values:
            values["status"] = ClassStatus(values["status"])
        if "tutorials" in values:
            values["tutorials"] = [Tutorial.model_validate(value) for value in values["tutorials"]]
        if "assignments" in values:
            values["assignments"] = [Assignment.model_validate(value) for value in values["assignments"]]
        if "project" in values:
            values["project"] = ClassProject.model_validate(values["project"])
        for key, value in values.items():
            setattr(item, key, value)
        item.updated_at = time.time()
        self.storage.save_class(item)
        if item.status == ClassStatus.COMPLETE:
            self._update_progress_after_completion(course_id, item.id)
        return self.storage.load_class(course_id, class_id) or item

    def delete_course(self, course_id: str) -> bool:
        return self.storage.delete(course_id)

    @staticmethod
    def _normalize_links(raw: list[Any]) -> list[ResourceLink]:
        links: list[ResourceLink] = []
        seen: set[str] = set()
        for value in raw[:12]:
            try:
                link = ResourceLink.model_validate(value)
            except Exception:
                continue
            normalized = link.url.strip()
            if normalized and normalized not in seen:
                link.url = normalized
                seen.add(normalized)
                links.append(link)
        return links

    async def _research_links(self, links: list[ResourceLink]) -> list[ResourceLink]:
        async def fetch(link: ResourceLink) -> ResourceLink:
            outcome = await fetch_url_as_markdown(link.url, max_chars=14000)
            if outcome.ok:
                link.status = "researched"
                link.final_url = outcome.url
                link.fetched_title = outcome.title
                link.excerpt = outcome.markdown
                link.truncated = outcome.truncated
            else:
                link.status = "unavailable"
                link.error = outcome.error
            return link
        return await asyncio.gather(*(fetch(link) for link in links)) if links else []

    @staticmethod
    def _resource_evidence(links: list[ResourceLink]) -> str:
        blocks = []
        for link in links:
            if link.status == "researched":
                blocks.append(f"[Resource {link.id}] {link.title or link.fetched_title or link.url}\nURL: {link.final_url or link.url}\n{link.excerpt}")
        return "\n\n---\n\n".join(blocks)

    def _sync_learning_path(self, course_id: str, classes: list[CourseClass]) -> None:
        modules: list[LearningModule] = []
        for item in classes:
            points = []
            for index, raw in enumerate(item.knowledge_points or [{"name": objective, "type": "concept"} for objective in item.learning_objectives]):
                try:
                    kind = KnowledgeType(str(raw.get("type") or "concept").lower())
                except ValueError:
                    kind = KnowledgeType.CONCEPT
                points.append(KnowledgePoint(
                    id=f"{course_id}_{item.id}_kp{index}",
                    name=str(raw.get("name") or f"Objective {index + 1}"),
                    type=kind,
                    module_id=item.id,
                ))
            if points:
                modules.append(LearningModule(id=item.id, name=item.title, order=item.order, knowledge_points=points))
        if modules:
            service = LearningService()
            progress = service.get_or_create(course_id)
            service.init_modules(progress, modules)
            progress.current_module_id = modules[0].id
            service.save(progress)

    def _update_progress_after_completion(self, course_id: str, class_id: str) -> None:
        progress = self.storage.load_progress(course_id) or CourseProgress(course_id=course_id)
        if class_id not in progress.completed_class_ids:
            progress.completed_class_ids.append(class_id)
        classes = self.storage.list_classes(course_id)
        for item in classes:
            if item.status == ClassStatus.LOCKED and all(pid in progress.completed_class_ids for pid in item.prerequisite_ids):
                item.status = ClassStatus.NEXT
                item.updated_at = time.time()
                self.storage.save_class(item)
        next_item = next((item for item in classes if item.status in {ClassStatus.NEXT, ClassStatus.IN_PROGRESS}), None)
        progress.current_class_id = next_item.id if next_item else ""
        progress.updated_at = time.time()
        self.storage.save_progress(progress)
        course = self.storage.load_course(course_id)
        if course:
            course.completed_class_count = len(progress.completed_class_ids)
            course.updated_at = time.time()
            self.storage.save_course(course)


_engine: CourseEngine | None = None


def get_course_engine() -> CourseEngine:
    global _engine
    if _engine is None:
        _engine = CourseEngine()
    return _engine
