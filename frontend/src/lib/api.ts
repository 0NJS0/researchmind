import type {
  ActivityEvent, AutopilotRole, Chunk, GapDetail, Gap, Health, Job, LibraryPaper, PaperDetail, PublicConfig, ReportEntry,
  ResearchPaper, SearchQuery, Session, SessionSummary, Synthesis, Topic, Source, AutopilotStatus, TopicPapers,
  Validation, ValidationSummary,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg}]
    return detail.map((d: { loc?: unknown[]; msg?: string }) =>
      `${(d.loc ?? []).filter((p) => p !== "body").join(".")}: ${d.msg}`).join("; ");
  }
  if (status === 0) return "Cannot reach the ResearchMind server";
  return `Request failed (${status})`;
}

export async function request<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init;
  let res: Response;
  try {
    res = await fetch(path, {
      ...rest,
      headers: { Accept: "application/json", ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
      body: json !== undefined ? JSON.stringify(json) : rest.body,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the ResearchMind server");
  }
  const text = await res.text();
  let body: unknown = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, body));
  return body as T;
}

const q = (params: Record<string, string | number | boolean | undefined | null>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const str = s.toString();
  return str ? `?${str}` : "";
};
const enc = encodeURIComponent;

/** Upload with progress (fetch has no upload progress events). */
function uploadWithProgress<T>(url: string, file: File, onProgress?: (pct: number) => void,
                               fields: Record<string, string> = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100)); };
    xhr.onerror = () => reject(new ApiError(0, "Cannot reach the ResearchMind server"));
    xhr.onload = () => {
      let body: unknown = null;
      try { body = JSON.parse(xhr.responseText); } catch { body = xhr.responseText; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, errorMessage(xhr.status, body)));
    };
    const form = new FormData();
    form.append("file", file);
    for (const [k, v] of Object.entries(fields)) if (v) form.append(k, v);
    xhr.send(form);
  });
}

export const api = {
  health: (refresh = false) => request<Health>(`/api/health${q({ refresh })}`),
  config: () => request<PublicConfig>("/api/config"),

  library: {
    topics: () => request<{ topics: Topic[] }>("/api/library/topics").then((r) => r.topics),
    createTopic: (name: string) => request<{ name: string }>("/api/library/topics", { method: "POST", json: { name } }),
    deleteTopic: (topic: string) => request(`/api/library/topics/${enc(topic)}`, { method: "DELETE" }),
    papers: (topic?: string) => request<{ papers: LibraryPaper[] }>(`/api/library/papers${q({ topic })}`).then((r) => r.papers),
    topicPapers: (topic: string, includeRejected = false) =>
      request<TopicPapers>(`/api/library/topics/${enc(topic)}/papers${q({ include_rejected: includeRejected })}`),
    deleteTopicPaper: (topic: string, paper: { id: string | null; library_paper_id: string }) =>
      request<{ status: string }>(`/api/library/topics/${enc(topic)}/papers${q(
        paper.id ? { paper_id: paper.id } : { library_id: paper.library_paper_id })}`, { method: "DELETE" }),
    pdfUrl: (topic: string, libraryId: string) => `/api/library/topics/${enc(topic)}/papers/${enc(libraryId)}/pdf`,
    exportUrl: (topic: string, includePdfs = true) => `/api/library/topics/${enc(topic)}/export${q({ include_pdfs: includePdfs })}`,
    deletePaper: (topic: string, paperId: string) =>
      request(`/api/library/papers/${enc(topic)}/${enc(paperId)}`, { method: "DELETE" }),
    analysis: (topic: string, paperId: string) =>
      request<{ content: string }>(`/api/library/papers/${enc(topic)}/${enc(paperId)}/analysis`).then((r) => r.content),
    chunk: (id: number) => request<Chunk>(`/api/library/chunks/${id}`),
    /** Upload a PDF, optionally with its link / DOI and title (identifies the paper; if the topic already has
     *  it with only an abstract, the PDF becomes that paper's full text: status "attached"). */
    upload: (topic: string, file: File, onProgress?: (pct: number) => void,
             meta: { url?: string; doi?: string; title?: string } = {}) =>
      uploadWithProgress<{ status: "indexed" | "attached"; paper_id: string; ref?: string; pages: number; chunks: number }>(
        `/api/library/upload${q({ topic })}`, file, onProgress, meta),
    ingest: (topic?: string, reset = false) => request<Job>(`/api/library/ingest${q({ topic, reset })}`, { method: "POST" }),
    query: (question: string, topic: string, sessionId: string) =>
      request<{ answer: string; sources: Source[]; errors: string[] }>("/api/library/query",
        { method: "POST", json: { question, topic, session_id: sessionId } }),
    createReport: (topic: string) => request<Job>("/api/library/reports", { method: "POST", json: { topic } }),
    jobs: (topic?: string) => request<{ jobs: Job[] }>(`/api/library/jobs${q({ topic })}`).then((r) => r.jobs),
    job: (id: string) => request<Job>(`/api/library/jobs/${enc(id)}`),
    reports: (topic: string) => request<{ reports: ReportEntry[] }>(`/api/library/reports${q({ topic })}`).then((r) => r.reports),
    report: (topic: string, filename: string) =>
      request<{ content: string }>(`/api/library/reports/${enc(topic)}/${enc(filename)}`).then((r) => r.content),
    deleteReport: (topic: string, filename: string) =>
      request(`/api/library/reports/${enc(topic)}/${enc(filename)}`, { method: "DELETE" }),
  },

  validator: {
    projectTypes: () => request<{ project_types: { key: string; label: string }[] }>("/api/validator/project-types")
      .then((r) => r.project_types),
    list: () => request<{ validations: ValidationSummary[] }>("/api/validator").then((r) => r.validations),
    create: (idea: string, projectType: string, topic: string) =>
      request<Validation>("/api/validator", { method: "POST", json: { idea, project_type: projectType, topic } }),
    get: (id: string) => request<Validation>(`/api/validator/${id}`),
    remove: (id: string) => request(`/api/validator/${id}`, { method: "DELETE" }),
  },

  autopilot: {
    status: () => request<AutopilotStatus>("/api/autopilot"),
    setTopic: (topic: string, focus?: string) =>
      request<AutopilotStatus & { status: "active" | "pending" }>("/api/autopilot/topic", { method: "POST", json: { topic, focus } }),
    setFocus: (topic: string, focus: string) =>
      request(`/api/autopilot/topics/${enc(topic)}/focus`, { method: "PUT", json: { focus } }),
    pause: () => request<AutopilotStatus>("/api/autopilot/pause", { method: "POST" }),
    resume: () => request<AutopilotStatus>("/api/autopilot/resume", { method: "POST" }),
    searchNow: () => request<AutopilotStatus>("/api/autopilot/search-now", { method: "POST" }),
  },

  research: {
    list: () => request<{ sessions: SessionSummary[] }>("/api/research").then((r) => r.sessions),
    create: (topic: string, description: string) =>
      request<Session>("/api/research", { method: "POST", json: { topic, description } }),
    get: (id: string) => request<Session>(`/api/research/${id}`),
    remove: (id: string) => request(`/api/research/${id}`, { method: "DELETE" }),
    /** Put this research's topic on the autopilot (and search now if it is already the active topic). */
    toAutopilot: (id: string) => request<Session>(`/api/research/${id}/search`, { method: "POST", json: {} }),
    cancel: (id: string) => request<{ status: string }>(`/api/research/${id}/cancel`, { method: "POST" }),
    queries: (id: string) => request<{ queries: SearchQuery[] }>(`/api/research/${id}/queries`).then((r) => r.queries),
    papers: (id: string, selection: "selected" | "rejected" | "candidate" | "all" = "selected") =>
      request<{ papers: ResearchPaper[] }>(`/api/research/${id}/papers${q({ selection })}`).then((r) => r.papers),
    addPaper: (id: string, body: { url?: string; doi?: string; title?: string }) =>
      request<{ status: string; paper: ResearchPaper; processing: boolean; queued?: boolean }>(`/api/research/${id}/papers`,
        { method: "POST", json: body }),
    paper: (paperId: string) => request<PaperDetail>(`/api/papers/${paperId}`),
    /** Attach a paper's PDF (e.g. downloaded in the browser); the autopilot re-analyses it from the full text. */
    attachPdf: (paperId: string, file: File, onProgress?: (pct: number) => void) =>
      uploadWithProgress<{ status: string; topic: string | null; autopilot_role: AutopilotRole }>(
        `/api/papers/${paperId}/pdf`, file, onProgress),
    analyzePaper: (paperId: string) => request(`/api/papers/${paperId}/analyze`, { method: "POST" }),
    gaps: (id: string, includeMerged = false) =>
      request<{ gaps: Gap[] }>(`/api/research/${id}/gaps${q({ include_merged: includeMerged })}`).then((r) => r.gaps),
    gap: (gapId: string) => request<GapDetail>(`/api/gaps/${gapId}`),
    rebuildGaps: (id: string) => request(`/api/research/${id}/analyze-gaps`, { method: "POST" }),
    synthesis: (id: string) =>
      request<{ synthesis: Synthesis | null; updated_at: string | null }>(`/api/research/${id}/synthesis`),
    events: (id: string, after = 0, latest = false) =>
      request<{ events: ActivityEvent[] }>(`/api/research/${id}/events${q({ after, latest: latest || undefined })}`).then((r) => r.events),
  },
};
