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
async def test_compiler_rejects_placeholder_content():
    agent = CourseClassCompilerAgent()
    agent.json = AsyncMock(return_value={"tutorials": [{"title": "One", "content": "too short"}], "assignments": [], "project": {}})
    plan = CourseClassPlan(title="Specific class", summary="A class", learning_objectives=["One", "Two"], tutorial_titles=["One", "Two"], assignment_titles=["Practice"], project_title="Project")

    with pytest.raises(CourseGenerationError, match="could not finish every artifact"):
        await agent.process(CourseProposal(title="Test"), plan, "")


@pytest.mark.asyncio
async def test_compiler_recovers_missing_artifacts_without_using_placeholders():
    agent = CourseClassCompilerAgent()
    content = "# Do the work\n\n" + ("Use the concrete setup steps and verify each result. " * 12)
    first = {"tutorials": [{"title": "First", "content": content, "estimated_minutes": 20}], "assignments": [], "project": {}}
    repaired = {
        "tutorials": [{"title": "Second", "content": content, "estimated_minutes": 25}],
        "assignments": [{"title": "Practice", "prompt": "Build and explain a realistic implementation with explicit technical constraints and verification steps.", "deliverable": "A runnable implementation plus a concise explanation of the design choices.", "rubric": ["Works end to end", "Explains choices", "Verifies the result"]}],
        "project": {"title": "Project", "brief": "Build a small, working increment that uses the class objectives and can be reviewed against concrete behavior.", "milestones": ["Plan the increment", "Implement the behavior", "Verify and reflect"], "success_criteria": ["Feature works", "Tests or verification are present", "Tradeoffs are explained"]},
    }
    agent.json = AsyncMock(side_effect=[first, repaired])
    plan = CourseClassPlan(title="Specific class", summary="A class", learning_objectives=["One", "Two"], tutorial_titles=["First", "Second"], assignment_titles=["Practice"], project_title="Project")

    tutorials, assignments, project = await agent.process(CourseProposal(title="Test"), plan, "")
    assert [item.title for item in tutorials] == ["First", "Second"]
    assert assignments[0].title == "Practice"
    assert project.title == "Project"
