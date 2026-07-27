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


@pytest.mark.asyncio
async def test_compiler_materialises_distinct_workspaces_without_waiting_for_llm_json():
    agent = CourseClassCompilerAgent()
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
