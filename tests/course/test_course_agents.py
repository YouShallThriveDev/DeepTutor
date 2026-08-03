from unittest.mock import AsyncMock

import pytest

from deeptutor.course.agents import CourseClassCompilerAgent, CourseGenerationError, CoursePlannerAgent
from deeptutor.course.models import CourseClassPlan, CourseProposal


@pytest.mark.asyncio
async def test_planner_rejects_empty_model_response_instead_of_fabricating_classes():
    agent = CoursePlannerAgent()
    agent.json = AsyncMock(return_value={})

    with pytest.raises(CourseGenerationError, match="required"):
        await agent.process("course_1", CourseProposal(title="Test", estimated_classes=3), "", [])


@pytest.mark.asyncio
async def test_planner_requires_distinct_complete_class_plans():
    agent = CoursePlannerAgent()
    agent.json = AsyncMock(return_value={"course_objectives": ["Ship a small app", "Explain the tradeoffs"], "classes": [
        {"title": "Foundations", "summary": "Learn the concrete foundations needed to begin building the application.", "learning_objectives": ["Explain the request lifecycle", "Create a minimal route"], "knowledge_points": [], "prerequisites": [], "tutorial_titles": ["Trace a request", "Build the first route"], "assignment_titles": ["Route exercise"], "project_title": "Route prototype", "resource_ids": []},
        {"title": "Data flow", "summary": "Connect the route to a persistent data flow and test the important behavior.", "learning_objectives": ["Model application data", "Validate a create flow"], "knowledge_points": [], "prerequisites": ["Foundations"], "tutorial_titles": ["Model the data", "Test the flow"], "assignment_titles": ["Data exercise"], "project_title": "Data prototype", "resource_ids": []},
        {"title": "Integration", "summary": "Integrate the working pieces into a small coherent end-to-end feature.", "learning_objectives": ["Integrate client and server", "Review the feature"], "knowledge_points": [], "prerequisites": ["Data flow"], "tutorial_titles": ["Connect the client", "Review the feature"], "assignment_titles": ["Integration exercise"], "project_title": "Integrated feature", "resource_ids": []},
    ]})

    outline = await agent.process("course_1", CourseProposal(title="Test", estimated_classes=3), "", [])
    assert [item.title for item in outline.classes] == ["Foundations", "Data flow", "Integration"]
    assert outline.classes[1].prerequisites == [outline.classes[0].id]




def _planner_item(index: int, title: str | None = None) -> dict:
    label = title or f"Class {index}"
    previous = [] if index <= 1 else [f"Class {index - 1}"]
    return {
        "title": label,
        "summary": f"Build a concrete agent framework capability in step {index} with working implementation and review checkpoints.",
        "learning_objectives": [f"Explain capability {index}", f"Implement capability {index}"],
        "knowledge_points": [{"name": f"Capability {index}", "type": "concept"}],
        "prerequisites": previous,
        "tutorial_titles": [f"Explore capability {index}", f"Implement capability {index}"],
        "assignment_titles": [f"Practice capability {index}"],
        "project_title": f"Agent framework milestone {index}",
        "resource_ids": [],
    }


@pytest.mark.asyncio
async def test_planner_repairs_nearly_complete_outline_with_missing_class():
    agent = CoursePlannerAgent()
    first = {
        "course_objectives": ["Build a Flue agent", "Explain agent framework tradeoffs"],
        "classes": [_planner_item(index) for index in range(1, 12)],
    }
    repair = {"classes": [_planner_item(12, "Capstone Integration and Production Readiness")]}
    agent.json = AsyncMock(side_effect=[first, repair])

    outline = await agent.process("course_flue", CourseProposal(title="Flue Agent Framework", estimated_classes=12), "", [])

    assert len(outline.classes) == 12
    assert outline.classes[-1].title == "Capstone Integration and Production Readiness"
    assert agent.json.await_count == 2


@pytest.mark.asyncio
async def test_planner_repairs_mostly_complete_outline_with_multiple_missing_classes():
    agent = CoursePlannerAgent()
    first = {
        "course_objectives": ["Build a Flue agent", "Explain agent framework tradeoffs"],
        "classes": [_planner_item(index) for index in range(1, 9)],
    }
    repair = {"classes": [_planner_item(index) for index in range(9, 13)]}
    agent.json = AsyncMock(side_effect=[first, repair])

    outline = await agent.process("course_flue", CourseProposal(title="Flue Agent Framework", estimated_classes=12), "", [])

    assert len(outline.classes) == 12
    assert [item.order for item in outline.classes] == list(range(12))
    assert outline.classes[-1].title == "Class 12"


@pytest.mark.asyncio
async def test_planner_still_rejects_when_missing_class_repair_is_not_exact():
    agent = CoursePlannerAgent()
    agent.json = AsyncMock(side_effect=[
        {"course_objectives": ["Build a Flue agent", "Explain agent framework tradeoffs"], "classes": [_planner_item(index) for index in range(1, 12)]},
        {"classes": []},
    ])

    with pytest.raises(CourseGenerationError, match="required"):
        await agent.process("course_flue", CourseProposal(title="Flue Agent Framework", estimated_classes=12), "", [])

@pytest.mark.asyncio
async def test_compiler_materialises_distinct_workspaces_without_waiting_for_llm_json():
    agent = CourseClassCompilerAgent()
    agent._compiler_json = AsyncMock(return_value={})
    plan = CourseClassPlan(
        title="Project setup",
        summary="Set up a Laravel, Inertia, and Vue application with a verified local workflow.",
        learning_objectives=["Create the application skeleton", "Verify the local development workflow"],
        tutorial_titles=["Bootstrap Laravel with Inertia", "Configure a local database"],
        assignment_titles=["Create the project skeleton"],
        project_title="Project management foundation",
    )
    tutorials, assignments, project = await agent.process(
        CourseProposal(title="Laravel and Vue", capstone_title="Project manager"), plan, ""
    )

    assert len(tutorials) == 2
    assert "Hands-on workflow" in tutorials[0].content
    assert "Create the application skeleton" in tutorials[0].content
    assert assignments[0].rubric and len(assignments[0].rubric) >= 3
    assert len(project.milestones) >= 3
    assert "Project manager" in project.brief


@pytest.mark.asyncio
async def test_compiler_uses_valid_grok_enrichment_over_baseline():
    agent = CourseClassCompilerAgent()
    rich = "# Rich tutorial\n\n" + ("Follow these Laravel and Vue steps carefully, verify the behavior, and explain the trade-off. " * 8)
    agent._compiler_json = AsyncMock(side_effect=[
        {"tutorial": {"title": "Bootstrap Laravel with Inertia", "content": rich, "estimated_minutes": 45}},
        {"tutorial": {"title": "Configure a local database", "content": rich, "estimated_minutes": 35}},
        {"assignment": {"title": "Create the project skeleton", "prompt": "Build a Laravel, Inertia, and Vue skeleton with explicit setup constraints and document how you verified the local workflow.", "deliverable": "A runnable skeleton plus verification notes and a short explanation of decisions.", "rubric": ["Runs locally", "Uses Inertia and Vue", "Verification is documented"]}},
        {"project": {"title": "Project management foundation", "brief": "Create the foundational project-management application shell with a verified local development workflow and a clear path to the capstone.", "milestones": ["Install dependencies", "Render an Inertia page", "Verify SQLite"], "success_criteria": ["App runs", "Page renders", "Database connects"]}},
    ])
    plan = CourseClassPlan(
        title="Project setup",
        summary="Set up a Laravel, Inertia, and Vue application with a verified local workflow.",
        learning_objectives=["Create the application skeleton", "Verify the local development workflow"],
        tutorial_titles=["Bootstrap Laravel with Inertia", "Configure a local database"],
        assignment_titles=["Create the project skeleton"],
        project_title="Project management foundation",
    )

    tutorials, assignments, project = await agent.process(CourseProposal(title="Laravel and Vue"), plan, "")

    assert tutorials[0].estimated_minutes == 45
    assert tutorials[0].content.startswith("# Rich tutorial")
    assert assignments[0].prompt.startswith("Build a Laravel")
    assert project.brief.startswith("Create the foundational")
