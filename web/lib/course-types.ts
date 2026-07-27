export type CourseStatus = "draft" | "outline_ready" | "ready" | "error" | "archived";
export type ClassStatus = "locked" | "next" | "in_progress" | "review" | "complete";
export type AssignmentStatus = "todo" | "in_progress" | "submitted" | "complete";

export interface ResourceLink {
  id: string;
  url: string;
  title: string;
  note: string;
  final_url: string;
  fetched_title: string;
  excerpt: string;
  status: "pending" | "researched" | "unavailable" | string;
  error: string;
  truncated: boolean;
}

export interface CourseProposal {
  title: string;
  description: string;
  scope: string;
  target_level: string;
  estimated_classes: number;
  duration_weeks: number;
  rationale: string;
  capstone_title: string;
  capstone_description: string;
}

export interface Tutorial { id: string; title: string; content: string; estimated_minutes: number; status: string; }
export interface Assignment { id: string; title: string; prompt: string; deliverable: string; rubric: string[]; status: AssignmentStatus; learner_submission: string; feedback: string; }
export interface ClassProject { title: string; brief: string; milestones: string[]; success_criteria: string[]; progress: number; }

export interface CourseClassPlan {
  id: string; title: string; summary: string; learning_objectives: string[];
  knowledge_points: Array<{name: string; type: string}>; prerequisites: string[];
  tutorial_titles: string[]; assignment_titles: string[]; project_title: string;
  resource_ids: string[]; order: number;
}
export interface CourseOutline { course_id: string; classes: CourseClassPlan[]; course_objectives: string[]; version: number; updated_at: number; }
export interface CourseClass {
  id: string; course_id: string; title: string; summary: string; order: number;
  status: ClassStatus; learning_objectives: string[]; knowledge_points: Array<{name: string; type: string}>;
  prerequisite_ids: string[]; resource_ids: string[]; book_ids: string[];
  tutorials: Tutorial[]; assignments: Assignment[]; notes: string; project: ClassProject;
  chat_session_id: string; updated_at: number;
}
export interface Course { id: string; title: string; description: string; status: CourseStatus; proposal: CourseProposal | null; knowledge_bases: string[]; resource_links: ResourceLink[]; language: string; class_count: number; completed_class_count: number; created_at: number; updated_at: number; }
export interface CourseProgress { course_id: string; current_class_id: string; completed_class_ids: string[]; updated_at: number; }
export interface CourseDetail { course: Course; outline: CourseOutline | null; classes: CourseClass[]; progress: CourseProgress; research_brief?: string; }
