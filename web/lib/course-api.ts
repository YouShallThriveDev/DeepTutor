import { apiFetch, apiUrl } from "@/lib/api";
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
export const courseApi = {
  list: () => request<{courses: Course[]}>("/courses"),
  get: (id: string) => request<CourseDetail>(`/courses/${encodeURIComponent(id)}`),
  create: (payload: CreateCoursePayload) => request<{course: Course; proposal: CourseProposal; research_brief: string; resource_links: ResourceLink[]}>("/courses", { method: "POST", body: JSON.stringify(payload) }),
  confirmProposal: (id: string, proposal?: CourseProposal) => request<{course: Course; outline: CourseOutline}>(`/courses/${encodeURIComponent(id)}/confirm-proposal`, { method: "POST", body: JSON.stringify({proposal: proposal || null}) }),
  confirmOutline: (id: string, outline?: CourseOutline) => request<{classes: CourseClass[]}>(`/courses/${encodeURIComponent(id)}/confirm-outline`, { method: "POST", body: JSON.stringify({outline: outline || null}) }),
  patchClass: (courseId: string, classId: string, patch: Record<string, unknown>) => request<{class: CourseClass}>(`/courses/${encodeURIComponent(courseId)}/classes/${encodeURIComponent(classId)}`, { method: "PATCH", body: JSON.stringify({patch}) }),
  remove: (id: string) => request<{deleted: boolean}>(`/courses/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
