export interface ServiceStatus { ok: boolean; message: string }
export interface Health {
  status: "healthy" | "degraded" | "unhealthy";
  config: ServiceStatus;
  database: ServiceStatus;
  vectorstore: ServiceStatus;
  llm: ServiceStatus;
  searxng: ServiceStatus;
  openalex?: ServiceStatus;
  autopilot?: ServiceStatus;
}
export interface PublicConfig {
  llm_provider: string;
  llm_model: string;
  llm_fallback_models: string[];
  llm_fast_model: string;
  llm_context_tokens: number;
  llm_max_output_tokens: number;
  embedding_model: string;
  searxng_configured: boolean;
  database_configured: boolean;
  limits: Record<string, number>;
}

// ── library ──
export interface Topic { name: string; papers: number; chunks: number }
export interface LibraryPaper {
  topic: string;
  paper_id: string;
  source: string;
  chunks: number;
  pages: number;
  indexed_at: string | null;
  has_analysis: boolean;
  title: string;
  origin: "agent" | "upload";
  year: number | null;
  venue: string | null;
  citation_count: number | null;
  url: string | null;
  doi: string | null;
}
export interface Source {
  chunk_id: number;
  paper_id: string;
  page: number;
  score: number | null;
  source: string;
  snippet: string;
}
export interface Chunk { id: number; topic: string; paper_id: string; source: string; page: number; chunk_index: number; content: string }
export interface Job {
  id: string;
  kind: "report" | "ingest";
  topic: string;
  status: "running" | "completed" | "failed";
  stage: string;
  done: number;
  total: number;
  current: string;
  error: string;
  warnings: string[];
  file: string;
  started_at: string;
  finished_at: string | null;
}
export interface ReportEntry { type: "living" | "combined" | "paper"; filename: string; paper_id: string | null; created_at: string }
export type StageName = "retriever" | "classifier" | "analysis" | "comparison" | "gap" | "report";
export type StreamEvent =
  | { type: "stage"; stage: StageName; status: "running" | "done"; detail?: string; errors?: string[] }
  | { type: "answer"; answer: string; sources: Source[]; errors: string[]; citation_check?: CitationCheck };
export interface CitationCheck { total: number; verified: number; unverified: { paper_id: string; page: number; reason: string }[] }

// ── research agent ──
export type SessionStatus =
  | "created" | "searching" | "processing" | "summarizing" | "analyzing"
  | "completed" | "failed" | "cancelled" | "paused" | "autopilot";
export interface SessionSummary {
  id: string;
  topic: string;
  description: string;
  status: SessionStatus;
  status_message: string;
  created_at: string;
  updated_at: string;
  paper_count: number;
  gap_count: number;
  running: boolean;
  topic_name?: string | null;
  autopilot_role?: AutopilotRole;
}
/** How the autopilot relates to a topic: researching it now, switching to it next, or not on it. */
export type AutopilotRole = "active" | "next" | "idle";
export interface TopicAnalysis { field: string; summary: string; key_concepts: string[]; subtopics: string[]; synonyms: string[] }
export interface Session {
  id: string;
  topic: string;
  description: string;
  status: SessionStatus;
  status_message: string;
  topic_analysis: TopicAnalysis | null;
  iteration: number;
  created_at: string;
  updated_at: string;
  synthesis_updated_at: string | null;
  counts: { papers: number; candidates: number; rejected: number; summarized: number; gaps: number; queries: number };
  running: boolean;
  autopilot?: boolean;
  topic_name?: string | null;
  autopilot_role?: AutopilotRole;
  /** Only in the reply to starting/continuing research: "pending" = switches after the current paper. */
  autopilot_status?: "active" | "pending";
}
export interface ResearchPaper {
  id: string;
  session_id: string;
  ref: string;
  title: string;
  authors: string[];
  abstract: string;
  url: string;
  pdf_url: string;
  doi: string;
  arxiv_id: string;
  source: string;
  venue: string;
  publication_date: string;
  year: number | null;
  citation_count: number | null;
  content_status: "pending" | "full_text" | "abstract" | "metadata_only" | "unavailable";
  content_source: string;
  selection_status: "candidate" | "selected" | "rejected";
  relevance_score: number | null;
  relevance_reason: string;
  relevance_method: string;
  processing_status: string;
  error: string;
  added_by: string;
  has_summary: boolean;
  content_length: number | null;
  created_at: string;
}
export interface Statement { text: string; inferred: boolean }
export interface PaperSummary { analysis_level: string; sections: Record<string, Statement[]>; model: string; updated_at: string }
export interface Relation {
  related_ref: string;
  related_title: string;
  relationship: string;
  similarities: string;
  differences: string;
  contradiction: string;
  other_paper_id: string;
}
export interface PaperGapLink { id: string; ref: string; title: string; relationship: string; evidence: string; confidence: string; status: string }
export interface PaperDetail { paper: ResearchPaper; summary: PaperSummary | null; relations: Relation[]; gaps: PaperGapLink[] }
export interface Gap {
  id: string;
  ref: string;
  title: string;
  description: string;
  gap_type: string;
  category: string;
  confidence: "low" | "medium" | "high";
  status: "open" | "contested" | "partially_addressed" | "resolved" | "merged";
  merged_into: string | null;
  evidence_count: number;
  paper_count: number;
  related_papers: { id: string; ref: string; title: string }[];
  created_at: string;
  updated_at: string;
  verification_status?: "unverified" | "verified_open" | "partially_addressed" | "addressed";
  verification_note?: string;
  verification_refs?: { title: string; url: string; year: number | null; venue: string; how: string; citation_count: number | null }[] | null;
  verified_at?: string | null;
}
export interface Evidence { id: string; paper_id: string; paper_ref: string; paper_title: string; evidence: string; relationship: string; basis: string; created_at: string }
export interface GapHistory { id: string; action: string; detail: string; paper_ref: string | null; created_at: string }
export interface GapDetail { gap: Gap; evidence: Evidence[]; history: GapHistory[]; merged_into: { id: string; ref: string; title: string } | null }
export interface SynthesisItem { statement: string; paper_refs: string[]; paper_ids: string[]; strength?: string }
export interface Synthesis {
  overview: string;
  known_findings: SynthesisItem[];
  common_methods: SynthesisItem[];
  contradictory_findings: SynthesisItem[];
  recurring_limitations: SynthesisItem[];
  under_researched: SynthesisItem[];
  research_questions: SynthesisItem[];
}
export interface SearchQuery { id: string; query: string; source: string; iteration: number; status: string; result_count: number; error: string; created_at: string }
export interface ActivityEvent { id: number; type: string; message: string; data: Record<string, unknown> | null; created_at: string }

// ── autopilot ──
export type AutopilotStateName = "idle" | "searching" | "screening" | "processing" | "verifying" | "waiting" | "paused" | "stopped" | "error";
export interface QueuedPaper {
  id: string;
  ref: string;
  title: string;
  year: number | null;
  venue: string;
  citation_count: number | null;
  publication_type: string;
  peer_reviewed: boolean | null;
  url: string;
  relevance_score: number | null;
  priority_score: number | null;
  priority_detail: { relevance: number; recency: number; impact: number; venue: number } | null;
  added_by: string;
}
export interface ProcessedPaper {
  id: string;
  ref: string;
  title: string;
  year: number | null;
  venue: string;
  citation_count: number | null;
  content_status: string;
  processing_status: string;
  library_paper_id: string;
  processed_at: string;
  added_by: string;
}
export interface AutopilotStatus {
  enabled: boolean;
  state: AutopilotStateName;
  message: string;
  active_topic: string | null;
  requested_topic: string | null;
  next_search_at: string | null;
  paused_until: string | null;
  last_error: string;
  updated_at: string;
  free_requests_remaining: number | null;
  current_paper: { id: string; ref: string; title: string; year: number | null; venue: string } | null;
  queue: QueuedPaper[];
  recent: ProcessedPaper[];
  session_id: string | null;
  focus: string;
  counts: { queued: number; processed: number; selected: number };
  living_report: { filename: string; updated_at: string } | null;
}

// ── topic papers (Papers page) ──
export type TopicPaperStatus = "processed" | "partial" | "processing" | "queued" | "failed" | "rejected" | "candidate" | "library";
export interface TopicPaper {
  id: string | null;
  ref: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string;
  citation_count: number | null;
  peer_reviewed: boolean | null;
  publication_type: string;
  doi: string;
  url: string;
  pdf_url: string;
  status: TopicPaperStatus;
  content_status: string;
  relevance_score: number | null;
  priority_score: number | null;
  relevance_reason: string;
  origin: "upload" | "agent";
  library_paper_id: string;
  has_analysis: boolean;
  has_summary: boolean;
  has_pdf: boolean;
  gap_count: number;
  added_at: string | null;
  processed_at: string | null;
  error: string;
  /** Only the abstract (or title) could be read: attach the PDF, or wait for the weekly free-copy check. */
  needs_full_text: boolean;
  fulltext_checked_at: string | null;
}
export interface TopicPapers { topic: string; session_id: string | null; autopilot_managed: boolean; papers: TopicPaper[] }

// ── topic validator ──
export interface Momentum { per_year: number; trend: "rising" | "steady" | "cooling" | "new" | "quiet" }
export interface ValidatorNeighbour {
  title: string;
  year: number | null;
  venue: string;
  doi: string;
  url: string;
  pdf_url: string;
  publication_type: string;
  peer_reviewed: boolean | null;
  citation_count: number | null;
  momentum: Momentum | null;
  authors: string[];
  abstract: string;
  closeness: "same" | "very_close" | "related";
  overlap: string;
  difference: string;
  source: string;
  ref?: string;
}
export interface ValidationResult {
  verdict: "duplicated" | "crowded" | "clear_lane" | "too_broad";
  reason: string;
  project_type_label: string;
  specificity: { level: "field" | "neighbourhood" | "topic"; restated: string; missing: { knob: string; question: string }[]; queries: string[] };
  neighbours: ValidatorNeighbour[];
  pivot_questions: string[];
  search: { queries: string[]; records_found: number; candidates_considered: number; openalex_total: number; searxng_failed: boolean; openalex_failed?: boolean };
  saturation: { total_matching: number; per_year: { year: number; count: number }[]; growth_last3_vs_prev3: number | null; density: string };
  rival_momentum: { close_rivals: number; rising: number; newest_close_rival: number | null };
}
export interface Validation {
  id: string;
  idea: string;
  project_type: string;
  topic: string;
  status: "running" | "completed" | "failed";
  stage: string;
  result: ValidationResult | null;
  error: string;
  created_at: string;
  running: boolean;
}
export interface ValidationSummary { id: string; idea: string; project_type: string; topic: string; status: string; stage: string; verdict: string | null; created_at: string }
