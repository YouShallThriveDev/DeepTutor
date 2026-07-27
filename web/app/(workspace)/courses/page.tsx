"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { BookOpen, CheckCircle2, ChevronLeft, ExternalLink, GraduationCap, LayoutGrid, Library, List, Loader2, Plus, Sparkles, Trash2, X } from "lucide-react";
import { useAppShell } from "@/context/AppShellContext";
import { courseApi } from "@/lib/course-api";
import type { Assignment, Course, CourseClass, CourseDetail, CourseOutline, CourseProposal, ResourceLink, Tutorial } from "@/lib/course-types";
import { bookApi } from "@/lib/book-api";
import type { Book } from "@/lib/book-types";
import { UnifiedWSClient, type StreamEvent } from "@/lib/unified-ws";
import AssistantResponse from "@/components/common/AssistantResponse";
import { AskUserOptions, extractAskUserPayload } from "@/components/chat/home/AskUserOptions";

const columns: Array<{key: CourseClass["status"]; label: string}> = [
  { key: "locked", label: "Locked" }, { key: "next", label: "Next up" }, { key: "in_progress", label: "In progress" }, { key: "review", label: "Review" }, { key: "complete", label: "Complete" },
];

export default function CoursesPage() {
  const { language } = useAppShell();
  const [courses, setCourses] = useState<Course[]>([]);
  const [detail, setDetail] = useState<CourseDetail | null>(null);
  const [view, setView] = useState<"library" | "creator" | "outline" | "board" | "class">("library");
  const [selectedClass, setSelectedClass] = useState<CourseClass | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [proposal, setProposal] = useState<CourseProposal | null>(null);
  const [outline, setOutline] = useState<CourseOutline | null>(null);
  const [researchBrief, setResearchBrief] = useState("");

  const refresh = useCallback(async () => {
    const result = await courseApi.list(); setCourses(result.courses);
  }, []);
  const openCourse = useCallback(async (id: string) => {
    setBusy(true); setError("");
    try { const next = await courseApi.get(id); setDetail(next); setOutline(next.outline); setView("board"); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not open course"); }
    finally { setBusy(false); }
  }, []);
  useEffect(() => { void refresh().catch(() => setError("Could not load courses")).finally(() => setLoading(false)); }, [refresh]);

  async function create(payload: {intent: string; resources: Array<Pick<ResourceLink, "url" | "title" | "note">>; knowledge_bases: string[]}) {
    setBusy(true); setError("");
    try {
      const result = await courseApi.create({ user_intent: payload.intent, resource_links: payload.resources, knowledge_bases: payload.knowledge_bases, language });
      setDetail({ course: result.course, outline: null, classes: [], progress: { course_id: result.course.id, current_class_id: "", completed_class_ids: [], updated_at: Date.now() / 1000 } });
      setProposal(result.proposal); setResearchBrief(result.research_brief); setView("creator"); await refresh();
    } catch (err) { setError(err instanceof Error ? err.message : "Course research failed"); }
    finally { setBusy(false); }
  }
  async function buildOutline() {
    if (!detail || !proposal) return; setBusy(true); setError("");
    try { const result = await courseApi.confirmProposal(detail.course.id, proposal); setDetail((old) => old ? {...old, course: result.course, outline: result.outline} : old); setOutline(result.outline); setView("outline"); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not create outline"); }
    finally { setBusy(false); }
  }
  async function createClasses() {
    if (!detail || !outline) return; setBusy(true); setError("");
    try { await courseApi.confirmOutline(detail.course.id, outline); await openCourse(detail.course.id); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not create classes"); }
    finally { setBusy(false); }
  }
  async function patchClass(item: CourseClass, patch: Record<string, unknown>) {
    if (!detail) return;
    try {
      const result = await courseApi.patchClass(detail.course.id, item.id, patch);
      const next = await courseApi.get(detail.course.id); setDetail(next); setSelectedClass(result.class);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not save class"); }
  }
  async function deleteCourse(id: string) {
    if (!window.confirm("Delete this course and its classes?")) return;
    await courseApi.remove(id); setDetail(null); setView("library"); await refresh();
  }

  return <div className="h-screen overflow-hidden bg-[var(--background)] text-[var(--foreground)]">
    {error && <div className="fixed right-4 top-4 z-50 max-w-md rounded-lg bg-rose-600 px-4 py-2 text-sm text-white shadow-lg">{error}<button className="ml-3" onClick={() => setError("")}><X size={14}/></button></div>}
    {view === "library" && <CourseLibrary courses={courses} loading={loading} onNew={() => { setDetail(null); setProposal(null); setOutline(null); setResearchBrief(""); setView("creator"); }} onOpen={openCourse} onDelete={deleteCourse} />}
    {view === "creator" && <CourseCreator pending={proposal} researchBrief={researchBrief} busy={busy} onBack={() => setView("library")} onCreate={create} onProposal={setProposal} onConfirm={buildOutline} />}
    {view === "outline" && detail && outline && <OutlineEditor outline={outline} busy={busy} onBack={() => setView("creator")} onChange={setOutline} onConfirm={createClasses} />}
    {view === "board" && detail && <CourseBoard detail={detail} busy={busy} onBack={() => setView("library")} onOpenClass={(item) => { setSelectedClass(item); setView("class"); }} onPatch={patchClass} />}
    {view === "class" && detail && selectedClass && <ClassWorkspace course={detail.course} item={selectedClass} resources={detail.course.resource_links} onBack={() => setView("board")} onPatch={patchClass} />}
  </div>;
}

function Header({ children, onBack }: {children: ReactNode; onBack?: () => void}) { return <header className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--card)]/50 px-6 py-4"> <div className="flex items-center gap-3">{onBack && <button className="rounded p-1 hover:bg-[var(--secondary)]" onClick={onBack}><ChevronLeft size={19}/></button>}<GraduationCap className="text-[var(--primary)]"/><div>{children}</div></div></header>; }

function CourseLibrary({courses, loading, onNew, onOpen, onDelete}: {courses: Course[]; loading: boolean; onNew: () => void; onOpen: (id: string) => void; onDelete: (id: string) => void}) {
  return <div className="flex h-full flex-col"><Header><h1 className="text-lg font-semibold">Courses</h1><p className="text-xs text-[var(--muted-foreground)]">Build interactive, project-driven learning paths.</p></Header><main className="flex-1 overflow-y-auto px-6 py-6"><div className="mb-6 flex items-center justify-between"><div className="grid grid-cols-3 gap-3"><Metric icon={<Library size={15}/>} label="Courses" value={courses.length}/><Metric icon={<GraduationCap size={15}/>} label="Classes" value={courses.reduce((n,c)=>n+c.class_count,0)}/><Metric icon={<CheckCircle2 size={15}/>} label="Completed" value={courses.reduce((n,c)=>n+c.completed_class_count,0)}/></div><button onClick={onNew} className="inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)]"><Plus size={16}/>New course</button></div>{loading ? <div className="flex justify-center py-20"><Loader2 className="animate-spin"/></div> : courses.length === 0 ? <Empty onNew={onNew}/> : <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{courses.map(course => <article key={course.id} onClick={() => onOpen(course.id)} className="group cursor-pointer rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-[var(--primary)]/40"><div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--primary)]/10 text-[var(--primary)]"><GraduationCap size={24}/></div><h2 className="line-clamp-2 text-base font-semibold">{course.title}</h2><p className="mt-2 line-clamp-3 text-sm text-[var(--muted-foreground)]">{course.description || "Interactive learning course"}</p><div className="mt-5 flex items-center justify-between text-xs text-[var(--muted-foreground)]"><span>{course.class_count} classes · {course.completed_class_count} complete</span><button onClick={(event)=>{event.stopPropagation(); onDelete(course.id);}} className="rounded p-1 opacity-0 hover:bg-rose-500/10 hover:text-rose-500 group-hover:opacity-100" title="Delete"><Trash2 size={15}/></button></div></article>)}</div>}</main></div>;
}
function Metric({icon,label,value}:{icon:ReactNode;label:string;value:number}) { return <div className="min-w-24 rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2"><div className="flex items-center gap-1 text-xs text-[var(--muted-foreground)]">{icon}{label}</div><div className="mt-1 text-lg font-semibold">{value}</div></div>; }
function Empty({onNew}:{onNew:()=>void}) { return <div className="mx-auto mt-20 max-w-md rounded-2xl border border-dashed border-[var(--border)] p-10 text-center"><GraduationCap className="mx-auto text-[var(--primary)]" size={32}/><h2 className="mt-4 font-semibold">Create your first course</h2><p className="mt-2 text-sm text-[var(--muted-foreground)]">Research your resources, plan classes, and learn with an adaptive tutor.</p><button onClick={onNew} className="mt-5 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm text-[var(--primary-foreground)]">Create course</button></div>; }

function CourseCreator({pending,researchBrief,busy,onBack,onCreate,onProposal,onConfirm}: {pending:CourseProposal|null;researchBrief:string;busy:boolean;onBack:()=>void;onCreate:(data:{intent:string;resources:Array<Pick<ResourceLink,"url"|"title"|"note">>;knowledge_bases:string[]})=>Promise<void>;onProposal:(value:CourseProposal)=>void;onConfirm:()=>Promise<void>}) {
  const [intent,setIntent]=useState(""); const [resources,setResources]=useState<Array<Pick<ResourceLink,"url"|"title"|"note">>>([]); const [url,setUrl]=useState(""); const [title,setTitle]=useState(""); const [note,setNote]=useState(""); const [kbs,setKbs]=useState<string[]>([]); const [availableKbs,setAvailableKbs]=useState<Array<{name:string}>>([]);
  useEffect(()=>{void import("@/lib/knowledge-api").then(({listKnowledgeBases})=>listKnowledgeBases().then(setAvailableKbs).catch(()=>undefined));},[]);
  const addResource=()=>{if(!url.trim())return;setResources([...resources,{url:url.trim(),title:title.trim(),note:note.trim()}]);setUrl("");setTitle("");setNote("");};
  return <div className="flex h-full flex-col"><Header onBack={onBack}><h1 className="text-lg font-semibold">{pending ? "Review course proposal" : "Create a course"}</h1><p className="text-xs text-[var(--muted-foreground)]">Sources are researched before the course is planned.</p></Header><main className="flex-1 overflow-y-auto"><div className="mx-auto max-w-3xl space-y-6 px-6 py-7">{!pending ? <><section className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5"><label className="text-sm font-medium">Learning intent</label><textarea value={intent} onChange={e=>setIntent(e.target.value)} rows={4} placeholder="What would you like to learn and build?" className="mt-2 w-full rounded-lg border border-[var(--border)] bg-[var(--background)] p-3 text-sm"/></section><section className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5"><div><h2 className="font-medium">Custom resources</h2><p className="mt-1 text-xs text-[var(--muted-foreground)]">Public URLs are fetched safely and researched before your course is proposed.</p></div><div className="mt-4 grid gap-2 sm:grid-cols-2"><input value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://resource.example" className="rounded border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm sm:col-span-2"/><input value={title} onChange={e=>setTitle(e.target.value)} placeholder="Optional title" className="rounded border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"/><input value={note} onChange={e=>setNote(e.target.value)} placeholder="Why is this useful?" className="rounded border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"/></div><button onClick={addResource} className="mt-3 rounded border border-[var(--border)] px-3 py-1.5 text-xs hover:bg-[var(--secondary)]">Add resource</button>{resources.length>0&&<div className="mt-3 space-y-2">{resources.map((resource,index)=><div key={`${resource.url}-${index}`} className="flex items-center justify-between rounded bg-[var(--secondary)]/50 px-3 py-2 text-xs"><span className="truncate">{resource.title || resource.url}</span><button onClick={()=>setResources(resources.filter((_,i)=>i!==index))} className="text-rose-500">Remove</button></div>)}</div>}</section><section className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5"><h2 className="font-medium">Knowledge bases</h2><div className="mt-3 flex flex-wrap gap-2">{availableKbs.map(kb=><label key={kb.name} className="inline-flex items-center gap-2 rounded border border-[var(--border)] px-2 py-1 text-xs"><input type="checkbox" checked={kbs.includes(kb.name)} onChange={()=>setKbs(kbs.includes(kb.name)?kbs.filter(n=>n!==kb.name):[...kbs,kb.name])}/>{kb.name}</label>)}{availableKbs.length===0&&<span className="text-xs text-[var(--muted-foreground)]">No knowledge bases selected.</span>}</div></section><button disabled={busy||!intent.trim()} onClick={()=>void onCreate({intent,resources,knowledge_bases:kbs})} className="inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] disabled:opacity-50">{busy?<Loader2 className="animate-spin" size={16}/>:<Sparkles size={16}/>}Research resources & generate proposal</button></> : <><section className="rounded-2xl border border-sky-500/30 bg-sky-500/5 p-5"><h2 className="font-medium">Research brief</h2><p className="mt-2 whitespace-pre-wrap text-sm text-[var(--muted-foreground)]">{researchBrief || "No custom resources were supplied."}</p></section><ProposalForm proposal={pending} onChange={onProposal}/><button disabled={busy} onClick={()=>void onConfirm()} className="inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] disabled:opacity-50">{busy?<Loader2 className="animate-spin" size={16}/>:<CheckCircle2 size={16}/>}Confirm proposal & generate class plan</button></>}</div></main></div>;
}
function ProposalForm({proposal,onChange}:{proposal:CourseProposal;onChange:(proposal:CourseProposal)=>void}) { const field=(key:keyof CourseProposal,label:string,area=false)=><label className="block text-sm"><span className="font-medium">{label}</span>{area?<textarea rows={3} value={String(proposal[key]||"")} onChange={e=>onChange({...proposal,[key]:e.target.value})} className="mt-1 w-full rounded border border-[var(--border)] bg-[var(--background)] p-2"/>:<input value={String(proposal[key]||"")} onChange={e=>onChange({...proposal,[key]:key==="estimated_classes"||key==="duration_weeks"?Number(e.target.value):e.target.value})} className="mt-1 w-full rounded border border-[var(--border)] bg-[var(--background)] p-2"/>}</label>;return <section className="space-y-4 rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5"><h2 className="font-medium">Course proposal</h2>{field("title","Title")}{field("description","Description",true)}<div className="grid grid-cols-2 gap-3">{field("estimated_classes","Classes")}{field("duration_weeks","Weeks")}</div>{field("capstone_title","Capstone title")}{field("capstone_description","Capstone description",true)}</section>; }

function OutlineEditor({outline,busy,onBack,onChange,onConfirm}:{outline:CourseOutline;busy:boolean;onBack:()=>void;onChange:(outline:CourseOutline)=>void;onConfirm:()=>Promise<void>}) { const update=(index:number,patch:Partial<CourseOutline["classes"][number]>)=>onChange({...outline,classes:outline.classes.map((item,i)=>i===index?{...item,...patch}:item)});return <div className="flex h-full flex-col"><Header onBack={onBack}><h1 className="text-lg font-semibold">Review class plan</h1><p className="text-xs text-[var(--muted-foreground)]">Edit class titles, summaries, and objectives before class workspaces are created.</p></Header><main className="flex-1 overflow-y-auto"><div className="mx-auto max-w-3xl space-y-3 px-6 py-6">{outline.classes.map((item,index)=><section key={item.id} className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-4"><div className="text-xs text-[var(--muted-foreground)]">Class {index+1}</div><input value={item.title} onChange={e=>update(index,{title:e.target.value})} className="mt-1 w-full bg-transparent text-base font-semibold outline-none"/><textarea rows={2} value={item.summary} onChange={e=>update(index,{summary:e.target.value})} className="mt-2 w-full rounded border border-[var(--border)] bg-[var(--background)] p-2 text-sm"/><textarea rows={3} value={item.learning_objectives.join("\n")} onChange={e=>update(index,{learning_objectives:e.target.value.split("\n").map(s=>s.trim()).filter(Boolean)})} placeholder="One learning objective per line" className="mt-2 w-full rounded border border-[var(--border)] bg-[var(--background)] p-2 text-sm"/></section>)}<button disabled={busy} onClick={()=>void onConfirm()} className="inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm text-[var(--primary-foreground)] disabled:opacity-50">{busy?<Loader2 className="animate-spin" size={16}/>:<CheckCircle2 size={16}/>}Create interactive classes</button></div></main></div>; }

function CourseBoard({detail,busy,onBack,onOpenClass,onPatch}:{detail:CourseDetail;busy:boolean;onBack:()=>void;onOpenClass:(item:CourseClass)=>void;onPatch:(item:CourseClass,patch:Record<string,unknown>)=>Promise<void>}) { const [mode,setMode]=useState<"board"|"list">("board");return <div className="flex h-full flex-col"><Header onBack={onBack}><h1 className="text-lg font-semibold">{detail.course.title}</h1><p className="text-xs text-[var(--muted-foreground)]">{detail.course.completed_class_count}/{detail.course.class_count} classes complete</p></Header><div className="flex items-center justify-between border-b border-[var(--border)] px-6 py-2"><div className="text-xs text-[var(--muted-foreground)]">{detail.course.proposal?.capstone_title && `Capstone: ${detail.course.proposal.capstone_title}`}</div><div className="flex rounded border border-[var(--border)]"><button onClick={()=>setMode("board")} className={`p-1.5 ${mode==="board"?"bg-[var(--secondary)]":""}`}><LayoutGrid size={15}/></button><button onClick={()=>setMode("list")} className={`p-1.5 ${mode==="list"?"bg-[var(--secondary)]":""}`}><List size={15}/></button></div></div>{busy?<div className="p-8"><Loader2 className="animate-spin"/></div>:mode==="board"?<main className="flex-1 overflow-x-auto p-5"><div className="grid min-w-[1080px] grid-cols-5 gap-3">{columns.map(column=><section key={column.key} className="rounded-xl bg-[var(--secondary)]/35 p-3"><h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">{column.label}</h2><div className="space-y-3">{detail.classes.filter(item=>item.status===column.key).map(item=><ClassCard key={item.id} item={item} onOpen={onOpenClass} onAdvance={onPatch}/>)}</div></section>)}</div></main>:<main className="flex-1 overflow-y-auto p-6"><div className="mx-auto max-w-4xl space-y-3">{detail.classes.map(item=><ClassCard key={item.id} item={item} onOpen={onOpenClass} onAdvance={onPatch} list/>)}</div></main>}</div>; }
function ClassCard({item,onOpen,onAdvance,list=false}:{item:CourseClass;onOpen:(item:CourseClass)=>void;onAdvance:(item:CourseClass,patch:Record<string,unknown>)=>Promise<void>;list?:boolean}) { const nextStatus:Record<CourseClass["status"],CourseClass["status"]>={locked:"locked",next:"in_progress",in_progress:"review",review:"complete",complete:"complete"};return <article className={`rounded-lg border border-[var(--border)] bg-[var(--card)] p-3 shadow-sm ${list?"flex items-center justify-between":""}`}><div><button disabled={item.status==="locked"} onClick={()=>onOpen(item)} className="text-left text-sm font-semibold hover:text-[var(--primary)] disabled:cursor-not-allowed disabled:opacity-55">{item.title}</button><p className="mt-1 text-xs text-[var(--muted-foreground)]">{item.learning_objectives.length} objectives · {item.assignments.length} assignments · {item.tutorials.length} tutorials</p>{item.book_ids.length>0&&<p className="mt-1 text-xs text-[var(--primary)]">{item.book_ids.length} linked book{item.book_ids.length===1?"":"s"}</p>}</div>{item.status!=="locked"&&item.status!=="complete"&&<button onClick={()=>void onAdvance(item,{status:nextStatus[item.status]})} className="mt-3 rounded border border-[var(--border)] px-2 py-1 text-xs hover:bg-[var(--secondary)]">{item.status==="next"?"Start":item.status==="in_progress"?"Review":"Complete"}</button>}</article>; }

function ClassWorkspace({course,item,resources,onBack,onPatch}:{course:Course;item:CourseClass;resources:ResourceLink[];onBack:()=>void;onPatch:(item:CourseClass,patch:Record<string,unknown>)=>Promise<void>}) { const [mode,setMode]=useState<"board"|"list">("board"); const [tutor,setTutor]=useState(false); const [books,setBooks]=useState<Book[]>([]); useEffect(()=>{void bookApi.list().then(result=>setBooks(result.books)).catch(()=>undefined);},[]); const linked=resources.filter(resource=>item.resource_ids.includes(resource.id)); const saveNotes=(notes:string)=>void onPatch(item,{notes}); const toggleTutorial=(tutorial:Tutorial)=>void onPatch(item,{tutorials:item.tutorials.map(value=>value.id===tutorial.id?{...value,status:value.status==="complete"?"not_started":"complete"}:value)}); const updateAssignment=(assignment:Assignment,status:Assignment["status"])=>void onPatch(item,{assignments:item.assignments.map(value=>value.id===assignment.id?{...value,status}:value)}); const addBook=(id:string)=>{if(id&&!item.book_ids.includes(id))void onPatch(item,{book_ids:[...item.book_ids,id]});}; return <div className="flex h-full flex-col"><Header onBack={onBack}><h1 className="text-lg font-semibold">{item.title}</h1><p className="text-xs text-[var(--muted-foreground)]">Interactive class workspace · {item.status.replace("_"," ")}</p></Header><div className="flex items-center justify-between border-b border-[var(--border)] px-6 py-2"><div className="flex gap-2"><button onClick={()=>setMode("board")} className={`rounded px-2 py-1 text-xs ${mode==="board"?"bg-[var(--secondary)]":""}`}>Board</button><button onClick={()=>setMode("list")} className={`rounded px-2 py-1 text-xs ${mode==="list"?"bg-[var(--secondary)]":""}`}>List</button></div><button onClick={()=>setTutor(true)} className="rounded-lg bg-[var(--primary)] px-3 py-1.5 text-xs font-medium text-[var(--primary-foreground)]">Open adaptive tutor</button></div><main className="flex-1 overflow-auto p-5">{mode==="board"?<div className="grid min-w-[980px] grid-cols-5 gap-3"><BoardSection title="Objectives"><ul className="space-y-2">{item.learning_objectives.map(objective=><li key={objective} className="rounded bg-[var(--card)] p-2 text-xs">{objective}</li>)}</ul></BoardSection><BoardSection title="Tutorials">{item.tutorials.map(tutorial=><article key={tutorial.id} className="rounded bg-[var(--card)] p-2 text-xs"><b>{tutorial.title}</b><p className="mt-1 text-[var(--muted-foreground)]">{tutorial.estimated_minutes} min · {tutorial.status.replace("_"," ")}</p><button onClick={()=>toggleTutorial(tutorial)} className="mt-2 text-[var(--primary)]">{tutorial.status==="complete"?"Reopen":"Mark complete"}</button></article>)}</BoardSection><BoardSection title="Assignments">{item.assignments.map(assignment=><article key={assignment.id} className="rounded bg-[var(--card)] p-2 text-xs"><b>{assignment.title}</b><p className="mt-1 text-[var(--muted-foreground)]">{assignment.status.replace("_"," ")}</p><button onClick={()=>updateAssignment(assignment,assignment.status==="complete"?"todo":"complete")} className="mt-2 text-[var(--primary)]">{assignment.status==="complete"?"Reopen":"Mark complete"}</button></article>)}</BoardSection><BoardSection title="Resources & notes">{linked.map(resource=><a key={resource.id} href={resource.final_url||resource.url} target="_blank" rel="noreferrer" className="mb-2 flex items-center gap-1 rounded bg-[var(--card)] p-2 text-xs text-[var(--primary)]"><ExternalLink size={12}/>{resource.title||resource.fetched_title||resource.url}</a>)}<textarea defaultValue={item.notes} onBlur={e=>saveNotes(e.target.value)} placeholder="Personal class notes…" rows={7} className="mt-2 w-full rounded bg-[var(--card)] p-2 text-xs"/><div className="mt-2"><select defaultValue="" onChange={e=>addBook(e.target.value)} className="w-full rounded bg-[var(--card)] p-2 text-xs"><option value="">Link a separately-created Book…</option>{books.map(book=><option key={book.id} value={book.id}>{book.title}</option>)}</select>{item.book_ids.map(id=>{const book=books.find(value=>value.id===id);return book?<a key={id} href={`/book?book=${encodeURIComponent(id)}`} className="mt-2 flex items-center gap-1 text-xs text-[var(--primary)]"><BookOpen size={12}/>{book.title}</a>:null;})}</div></BoardSection><BoardSection title="Class project"><article className="rounded bg-[var(--card)] p-2 text-xs"><b>{item.project.title}</b><p className="mt-2 whitespace-pre-wrap text-[var(--muted-foreground)]">{item.project.brief}</p><ul className="mt-2 list-disc pl-4">{item.project.milestones.map(milestone=><li key={milestone}>{milestone}</li>)}</ul></article></BoardSection></div>:<div className="mx-auto max-w-3xl space-y-5"><SectionList title="Objectives" items={item.learning_objectives}/><SectionList title="Tutorials" items={item.tutorials.map(t=>`${t.title} — ${t.status}`)}/><SectionList title="Assignments" items={item.assignments.map(a=>`${a.title} — ${a.status}`)}/><SectionList title="Project" items={[item.project.title,item.project.brief,...item.project.milestones]}/></div>}</main>{tutor&&<TutorPanel course={course} item={item} onClose={()=>setTutor(false)} onSession={(session_id)=>void onPatch(item,{chat_session_id:session_id})}/>}</div>; }
function BoardSection({title,children}:{title:string;children:ReactNode}) {return <section className="rounded-xl bg-[var(--secondary)]/35 p-3"><h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">{title}</h2><div className="space-y-2">{children}</div></section>;}
function SectionList({title,items}:{title:string;items:string[]}) {return <section><h2 className="font-semibold">{title}</h2><ul className="mt-2 space-y-2">{items.filter(Boolean).map((entry,index)=><li key={`${entry}-${index}`} className="rounded border border-[var(--border)] bg-[var(--card)] p-3 text-sm">{entry}</li>)}</ul></section>;}

function TutorPanel({ course, item, onClose, onSession }: { course: Course; item: CourseClass; onClose: () => void; onSession: (id: string) => void }) {
  const { language } = useAppShell();
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string; events?: StreamEvent[] }>>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const clientRef = useRef<UnifiedWSClient | null>(null);
  const turnRef = useRef("");

  useEffect(() => () => clientRef.current?.disconnect(), []);
  const ensureClient = () => {
    if (clientRef.current) return clientRef.current;
    const client = new UnifiedWSClient((event: StreamEvent) => {
      if (event.turn_id) turnRef.current = event.turn_id;
      if (event.type === "session" && event.session_id) onSession(event.session_id);
      setMessages((previous) => {
        const next = [...previous];
        const last = next[next.length - 1];
        if (event.type === "content" || event.type === "thinking") {
          if (last?.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + (event.content || ""), events: [...(last.events || []), event] };
          } else {
            next.push({ role: "assistant", content: event.content || "", events: [event] });
          }
        } else if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, events: [...(last.events || []), event] };
        } else if (event.type === "tool_result" || event.type === "progress") {
          next.push({ role: "assistant", content: "", events: [event] });
        }
        return next;
      });
      if (event.type === "done" || event.type === "error") setBusy(false);
    }, () => setBusy(false));
    clientRef.current = client;
    return client;
  };
  const send = () => {
    const content = input.trim();
    if (!content || busy) return;
    setMessages((previous) => [...previous, { role: "user", content }]);
    setInput(""); setBusy(true);
    const client = ensureClient(); client.connect();
    window.setTimeout(() => client.send({
      type: "start_turn", content, session_id: item.chat_session_id || undefined,
      capability: "mastery_path", knowledge_bases: course.knowledge_bases, language,
      config: { mastery_path_id: course.id, course_context: JSON.stringify({ course: course.title, class_title: item.title, objectives: item.learning_objectives, tutorials: item.tutorials.map((tutorial) => tutorial.title), assignments: item.assignments.map((assignment) => assignment.title) }) },
    }), 200);
  };
  const submitReply = (payload: { text?: string; answers?: Array<{ questionId: string; text: string }> }) => {
    if (!turnRef.current) return;
    ensureClient().send({ type: "submit_user_reply", turn_id: turnRef.current, ...payload });
  };

  return <aside className="absolute inset-y-0 right-0 z-30 flex w-[420px] max-w-full flex-col border-l border-[var(--border)] bg-[var(--card)] shadow-2xl">
    <header className="flex items-center justify-between border-b border-[var(--border)] p-4"><div><b className="text-sm">Adaptive class tutor</b><p className="text-xs text-[var(--muted-foreground)]">Mastery checks, interactive answers, and feedback</p></div><button onClick={onClose}><X size={18}/></button></header>
    <div className="flex-1 overflow-y-auto p-4">{messages.length === 0 ? <p className="rounded border border-dashed border-[var(--border)] p-3 text-sm text-[var(--muted-foreground)]">Start with a diagnostic, ask for a tutorial, or work through an assignment. This tutor has your current class context.</p> : messages.map((message, index) => {
      const ask = extractAskUserPayload(message.events);
      return <div key={index} className={`mb-3 rounded-lg p-3 text-sm ${message.role === "user" ? "ml-8 bg-[var(--primary)] text-[var(--primary-foreground)]" : "mr-4 bg-[var(--secondary)]"}`}>
        {message.role === "assistant" && message.content ? <AssistantResponse content={message.content} className="text-sm" /> : message.content}
        {message.role === "assistant" && ask ? <div className="mt-3"><AskUserOptions data={ask} onSubmit={submitReply} /></div> : null}
      </div>;
    })}</div>
    <form onSubmit={(event) => { event.preventDefault(); send(); }} className="border-t border-[var(--border)] p-3"><textarea value={input} onChange={(event) => setInput(event.target.value)} rows={3} placeholder="Ask the tutor or start a diagnostic…" className="w-full rounded border border-[var(--border)] bg-[var(--background)] p-2 text-sm"/><button disabled={busy} className="mt-2 rounded bg-[var(--primary)] px-3 py-1.5 text-xs text-[var(--primary-foreground)] disabled:opacity-50">{busy ? "Thinking…" : "Send"}</button></form>
  </aside>;
}
