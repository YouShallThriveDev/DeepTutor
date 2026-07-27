import { apiFetch, apiUrl, wsUrl } from "@/lib/api";
import { runBookSocketOperation, type BookWsEvent } from "@/lib/book-ws-operation";
import type { Course, CourseClass, CourseDetail, CourseOutline, CourseProposal, ResourceLink } from "@/lib/course-types";

const BASE = "/api/v1/course";
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(apiUrl(`${BASE}${path}`), { headers: { "Content-Type": "application/json", ...(init?.headers || {}) }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try { const data = await res.json(); detail = data.detail || data.message || detail; } catch { /* ignore */ }
    throw new Error(`course api ${path} → ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export interface CreateCoursePayload {
  user_intent: string; language?: string; knowledge_bases?: string[]; notebook_refs?: Array<Record<string, unknown>>;
  chat_selections?: Array<{session_id: string; message_ids: number[]}>; question_categories?: number[]; question_entries?: number[];
  resource_links?: Array<Pick<ResourceLink, "url" | "title" | "note">>;
}
function socketOperation<T extends BookWsEvent>(message: BookWsEvent, resultType: string): Promise<T> {
  return runBookSocketOperation<T>(() => new WebSocket(wsUrl(`${BASE}/ws`)), { message, resultType });
}

export const courseApi = {
  list: () => request<{courses: Course[]}>("/courses"),
  get: (id: string) => request<CourseDetail>(`/courses/${encodeURIComponent(id)}`),
  create: (payload: CreateCoursePayload) => socketOperation<{type: "create_result"; course: Course; proposal: CourseProposal; research_brief: string; resource_links: ResourceLink[]}>({ type: "create", ...payload }, "create_result"),
  confirmProposal: (id: string, proposal?: CourseProposal) => socketOperation<{type: "confirm_proposal_result"; course: Course; outline: CourseOutline}>({ type: "confirm_proposal", course_id: id, proposal: proposal || null }, "confirm_proposal_result"),
  confirmOutline: (id: string, outline?: CourseOutline) => request<{classes: CourseClass[]}>(`/courses/${encodeURIComponent(id)}/confirm-outline`, { method: "POST", body: JSON.stringify({outline: outline || null}) }),
  patchClass: (courseId: string, classId: string, patch: Record<string, unknown>) => request<{class: CourseClass}>(`/courses/${encodeURIComponent(courseId)}/classes/${encodeURIComponent(classId)}`, { method: "PATCH", body: JSON.stringify({patch}) }),
  remove: (id: string) => request<{deleted: boolean}>(`/courses/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
