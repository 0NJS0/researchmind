import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownUp, CheckCircle2, CircleDashed, Download, ExternalLink, FileDown, FileStack, FileText, Lightbulb, Loader2,
  FileUp, Search, Trash2, TriangleAlert, Upload, XCircle,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ExportTopicButton } from "@/components/ExportTopicButton";
import { UploadPaperDialog } from "@/components/UploadPaperDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, EmptyState, PageHeader, Skeleton } from "@/components/ui/misc";
import { Tooltip } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { useTopic } from "@/lib/topic";
import type { TopicPaper, TopicPaperStatus } from "@/lib/types";
import { cn, relativeTime } from "@/lib/utils";
import { AnalysisSheet } from "./LibraryPage";
import { PaperSheet } from "./research/PaperSheet";

const STATUS: Record<TopicPaperStatus, { label: string; tone: "success" | "primary" | "warning" | "danger" | "neutral"; icon: typeof CheckCircle2 }> = {
  processed: { label: "Processed", tone: "success", icon: CheckCircle2 },
  partial: { label: "Partly processed", tone: "warning", icon: TriangleAlert },
  processing: { label: "Processing", tone: "primary", icon: Loader2 },
  queued: { label: "In queue", tone: "neutral", icon: CircleDashed },
  failed: { label: "Failed", tone: "danger", icon: XCircle },
  rejected: { label: "Rejected", tone: "neutral", icon: XCircle },
  candidate: { label: "Candidate", tone: "neutral", icon: CircleDashed },
  library: { label: "In Library", tone: "neutral", icon: FileText },
};

const FILTERS = [
  { key: "all", label: "All" },
  { key: "processed", label: "Processed" },
  { key: "queued", label: "In queue" },
  { key: "upload", label: "Uploaded" },
  { key: "agent", label: "Found by agent" },
  { key: "fulltext", label: "Needs full text" },
  { key: "rejected", label: "Rejected" },
] as const;
type FilterKey = (typeof FILTERS)[number]["key"];

const SORTS = [
  { key: "priority", label: "Priority" },
  { key: "newest", label: "Newest" },
  { key: "cited", label: "Most cited" },
  { key: "added", label: "Recently added" },
  { key: "title", label: "Title A–Z" },
] as const;
type SortKey = (typeof SORTS)[number]["key"];

function matches(p: TopicPaper, filter: FilterKey, text: string) {
  if (filter === "processed" && !["processed", "partial"].includes(p.status)) return false;
  if (filter === "queued" && !["queued", "processing"].includes(p.status)) return false;
  if (filter === "upload" && p.origin !== "upload") return false;
  if (filter === "agent" && p.origin !== "agent") return false;
  if (filter === "fulltext" && !p.needs_full_text) return false;
  if (filter === "rejected" && p.status !== "rejected") return false;
  if (filter !== "rejected" && p.status === "rejected") return false;
  const q = text.trim().toLowerCase();
  if (!q) return true;
  return [p.title, p.venue, p.ref, p.doi, p.authors.join(" "), String(p.year ?? "")].some((v) => v?.toLowerCase().includes(q));
}

export function sortPapers(papers: TopicPaper[], sort: SortKey): TopicPaper[] {
  const by = {
    priority: (a: TopicPaper, b: TopicPaper) => (b.priority_score ?? -1) - (a.priority_score ?? -1),
    newest: (a: TopicPaper, b: TopicPaper) => (b.year ?? 0) - (a.year ?? 0),
    cited: (a: TopicPaper, b: TopicPaper) => (b.citation_count ?? -1) - (a.citation_count ?? -1),
    added: (a: TopicPaper, b: TopicPaper) => (b.added_at ?? "").localeCompare(a.added_at ?? ""),
    title: (a: TopicPaper, b: TopicPaper) => a.title.localeCompare(b.title),
  }[sort];
  return [...papers].sort(by);
}

function authorsLine(authors: string[]) {
  if (!authors.length) return "";
  return authors.slice(0, 3).join(", ") + (authors.length > 3 ? " et al." : "");
}

export function toCsv(papers: TopicPaper[]): string {
  const cols: [string, (p: TopicPaper) => unknown][] = [
    ["ref", (p) => p.ref], ["title", (p) => p.title], ["authors", (p) => p.authors.join("; ")], ["year", (p) => p.year],
    ["venue", (p) => p.venue], ["citations", (p) => p.citation_count],
    ["peer_reviewed", (p) => (p.peer_reviewed == null ? "" : p.peer_reviewed ? "yes" : "no")],
    ["doi", (p) => p.doi], ["url", (p) => p.url], ["pdf_url", (p) => p.pdf_url], ["status", (p) => p.status],
    ["source", (p) => p.origin], ["evidence", (p) => p.content_status], ["research_gaps", (p) => p.gap_count],
    ["relevance", (p) => p.relevance_score], ["priority", (p) => p.priority_score],
  ];
  const esc = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.map(([h]) => h).join(","), ...papers.map((p) => cols.map(([, f]) => esc(f(p))).join(","))].join("\n");
}

function downloadCsv(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function Links({ paper }: { paper: TopicPaper }) {
  return (
    <div className="flex flex-wrap gap-1.5" onClick={(e) => e.stopPropagation()}>
      {paper.url && (
        <a href={paper.url} target="_blank" rel="noreferrer noopener"
          className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs hover:border-primary/50 hover:text-primary">
          <ExternalLink className="size-3" />Open
        </a>
      )}
      {paper.pdf_url && (
        <a href={paper.pdf_url} target="_blank" rel="noreferrer noopener"
          className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs hover:border-primary/50 hover:text-primary">
          <FileDown className="size-3" />PDF
        </a>
      )}
      {paper.doi && (
        <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer noopener"
          className="inline-flex items-center rounded-md border px-2 py-0.5 text-xs hover:border-primary/50 hover:text-primary">DOI</a>
      )}
    </div>
  );
}

/** Attach the PDF of a paper the agent could only read the abstract of (e.g. downloaded from SSRN in the browser). */
function AttachPdfButton({ paper, onDone }: { paper: TopicPaper; onDone: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [pct, setPct] = useState<number | null>(null);
  const attach = async (file: File | undefined) => {
    if (!file || !paper.id) return;
    setPct(0);
    try {
      const r = await api.research.attachPdf(paper.id, file, setPct);
      toast.success(r.autopilot_role === "active"
        ? "PDF attached. The autopilot re-analyses this paper from the full text next."
        : "PDF attached. The paper is re-analysed when this topic is on the autopilot.");
      onDone();
    } catch (e) { toast.error((e as Error).message); }
    finally { setPct(null); if (input.current) input.current.value = ""; }
  };
  return (
    <>
      <input ref={input} type="file" accept="application/pdf,.pdf" className="hidden" aria-label={`PDF for ${paper.title}`}
        onChange={(e) => attach(e.target.files?.[0])} />
      <Tooltip content="Attach the PDF (download it in your browser first)">
        <Button variant="outline" size="sm" onClick={() => input.current?.click()} loading={pct !== null}
          disabled={paper.status === "processing"} aria-label={`Attach PDF for ${paper.title}`}>
          <FileUp />{pct !== null ? `${pct}%` : "Attach PDF"}
        </Button>
      </Tooltip>
    </>
  );
}

function RowActions({ topic, paper, onDeleted }: { topic: string; paper: TopicPaper; onDeleted: () => void }) {
  return (
    <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
      {paper.has_pdf && (
        <Tooltip content="Download PDF">
          <Button asChild variant="ghost" size="icon-sm" aria-label={`Download PDF of ${paper.title}`}>
            <a href={api.library.pdfUrl(topic, paper.library_paper_id)} download><Download /></a>
          </Button>
        </Tooltip>
      )}
      <ConfirmDialog
        title="Delete this paper?"
        description={<>
          <span className="font-medium text-foreground">“{paper.title}”</span> will be removed everywhere: its PDF and files,
          its summary and analysis, its search index entries, and its evidence in the research gaps and reports.
          The autopilot will not add it again. This cannot be undone.
        </>}
        confirmLabel="Delete paper"
        onConfirm={async () => {
          try {
            await api.library.deleteTopicPaper(topic, paper);
            toast.success("Paper deleted");
            onDeleted();
          } catch (e) { toast.error((e as Error).message); }
        }}
        trigger={(open) => (
          <Tooltip content="Delete paper">
            <Button variant="ghost" size="icon-sm" aria-label={`Delete ${paper.title}`} onClick={open}
              className="text-muted-foreground hover:text-danger" disabled={paper.status === "processing"}>
              <Trash2 />
            </Button>
          </Tooltip>
        )} />
    </div>
  );
}

function PaperRow({ paper, onOpen, actions, onChanged }: {
  paper: TopicPaper; onOpen: () => void; actions: React.ReactNode; onChanged: () => void;
}) {
  const s = STATUS[paper.status];
  const Icon = s.icon;
  return (
    <li>
      <div onClick={onOpen}
        className={cn("grid cursor-pointer gap-3 px-4 py-3.5 hover:bg-muted/40 md:grid-cols-[minmax(0,1fr)_64px_84px_140px_72px]",
          paper.status === "rejected" && "opacity-70")}>
        <div className="min-w-0">
          <div className="flex items-start gap-2">
            {paper.ref && <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] font-medium">{paper.ref}</span>}
            <button type="button" onClick={(e) => { e.stopPropagation(); onOpen(); }}
              className="line-clamp-2 text-left text-sm font-medium leading-snug hover:text-primary cursor-pointer">{paper.title}</button>
          </div>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {[authorsLine(paper.authors), paper.venue].filter(Boolean).join(" · ") || "—"}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Badge tone={paper.origin === "upload" ? "primary" : "neutral"}>{paper.origin === "upload" ? "Uploaded" : "Found by agent"}</Badge>
            {paper.peer_reviewed === true && <Badge tone="success">Peer reviewed</Badge>}
            {paper.peer_reviewed === false && <Badge tone="outline">Preprint</Badge>}
            {paper.content_status === "abstract" && <Badge tone="warning">Abstract only</Badge>}
            {paper.content_status === "metadata_only" && <Badge tone="warning">Title only</Badge>}
            {paper.needs_full_text && paper.id && (
              <span onClick={(e) => e.stopPropagation()}><AttachPdfButton paper={paper} onDone={onChanged} /></span>
            )}
            {paper.gap_count > 0 && <Badge tone="outline"><Lightbulb />{paper.gap_count} gap{paper.gap_count > 1 ? "s" : ""}</Badge>}
            <Links paper={paper} />
          </div>
          {paper.status === "rejected" && paper.relevance_reason && (
            <p className="mt-1.5 line-clamp-1 text-xs italic text-muted-foreground">{paper.relevance_reason}</p>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground md:block md:text-sm md:text-foreground">
          <span className="md:hidden">Year </span><span className="tabular-nums">{paper.year ?? "—"}</span>
        </div>
        <div className="-mt-2 flex items-center gap-4 text-xs text-muted-foreground md:mt-0 md:block md:text-sm md:text-foreground">
          <span className="md:hidden">Citations </span><span className="tabular-nums">{paper.citation_count ?? "—"}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2 md:flex-col md:items-start md:gap-1">
          <Badge tone={s.tone}><Icon className={paper.status === "processing" ? "animate-spin" : undefined} />{s.label}</Badge>
          {paper.priority_score != null && ["queued", "processing"].includes(paper.status) && (
            <Tooltip content="Priority: relevance, recency, citations and venue combined">
              <span className="text-xs text-muted-foreground">priority {Math.min(99, Math.round(paper.priority_score * 100))}</span>
            </Tooltip>
          )}
          {paper.processed_at && <span className="text-xs text-muted-foreground">{relativeTime(paper.processed_at)}</span>}
        </div>
        <div className="flex md:justify-end">{actions}</div>
      </div>
    </li>
  );
}

export default function PapersPage() {
  const { topic, setTopic, topics, loading } = useTopic();
  const [filter, setFilter] = useState<FilterKey>("all");
  const [sort, setSort] = useState<SortKey>("priority");
  const [text, setText] = useState("");
  const [researchPaper, setResearchPaper] = useState<string | null>(null);
  const [libraryPaper, setLibraryPaper] = useState<TopicPaper | null>(null);
  const [uploading, setUploading] = useState(false);
  const { data, isLoading, error } = useQuery({
    queryKey: ["topic-papers", topic],
    queryFn: () => api.library.topicPapers(topic, true),  // rejected ones are filtered client-side (for the count)
    enabled: !!topic,
    refetchInterval: 20_000,
  });

  const qc = useQueryClient();
  const refresh = () => {
    for (const key of ["topic-papers", "topics", "autopilot", "reports", "papers", "report", "paper"]) qc.invalidateQueries({ queryKey: [key] });
  };
  const all = data?.papers ?? [];
  const shown = useMemo(() => sortPapers(all.filter((p) => matches(p, filter, text)), sort), [all, filter, text, sort]);
  const counts = useMemo(() => ({
    all: all.filter((p) => p.status !== "rejected").length,
    processed: all.filter((p) => ["processed", "partial"].includes(p.status)).length,
    queued: all.filter((p) => ["queued", "processing"].includes(p.status)).length,
    upload: all.filter((p) => p.origin === "upload" && p.status !== "rejected").length,
    agent: all.filter((p) => p.origin === "agent" && p.status !== "rejected").length,
    fulltext: all.filter((p) => p.needs_full_text).length,
    rejected: all.filter((p) => p.status === "rejected").length,
  }), [all]);

  if (loading) return <Skeleton className="h-64" />;
  if (!topics.length) {
    return <EmptyState icon={FileStack} title="No topics yet" description="Create a topic in the Library first."
      action={<Button asChild><Link to="/library">Go to Library</Link></Button>} />;
  }

  const open = (p: TopicPaper) => {
    if (p.id) setResearchPaper(p.id);
    else setLibraryPaper(p);
  };

  return (
    <>
      <PageHeader title="Papers"
        description="Every paper in a topic — your uploads and the ones the autopilot found — with links and key details."
        actions={<>
          <Button variant="outline" onClick={() => setUploading(true)}><Upload />Upload paper</Button>
          <Button variant="outline" disabled={!shown.length}
            onClick={() => downloadCsv(`${topic}_papers.csv`, toCsv(shown))}><Download />CSV</Button>
          <ExportTopicButton topic={topic} variant="primary" />
        </>} />

      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center">
        <select aria-label="Topic" value={topic} onChange={(e) => setTopic(e.target.value)}
          className="h-9 rounded-md border border-input bg-card px-3 text-sm font-medium shadow-xs lg:w-72">
          {topics.map((t) => <option key={t.name} value={t.name}>{t.name.replace(/_/g, " ")}</option>)}
        </select>
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Search title, author, venue, DOI, year…"
            className="pl-9" aria-label="Search papers" />
        </div>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          <ArrowDownUp className="size-4" />
          <select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value as SortKey)}
            className="h-9 rounded-md border border-input bg-card px-2 text-sm text-foreground shadow-xs">
            {SORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </label>
      </div>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {FILTERS.map((f) => (
          <button key={f.key} onClick={() => setFilter(f.key)}
            className={cn("rounded-full border px-3 py-1 text-xs font-medium cursor-pointer",
              filter === f.key ? "border-primary bg-primary-soft text-primary" : "text-muted-foreground hover:bg-muted")}>
            {f.label} <span className="ml-1 tabular-nums opacity-70">{counts[f.key]}</span>
          </button>
        ))}
      </div>

      {data?.autopilot_managed && (
        <p className="mb-3 text-xs text-muted-foreground">The autopilot is working on this topic — the list refreshes automatically.</p>
      )}

      {isLoading ? <div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-20" />)}</div>
        : error ? <Alert title="Could not load papers">{(error as Error).message}</Alert>
        : !shown.length ? (
          <EmptyState icon={FileStack} title={all.length ? "No papers match" : "No papers in this topic yet"}
            description={all.length ? "Try another filter or search." : "Upload PDFs in the Library, or start the autopilot on this topic."} />
        ) : (
          <div className="overflow-hidden rounded-lg border bg-card">
            <div className="hidden grid-cols-[minmax(0,1fr)_64px_84px_140px_72px] gap-3 border-b bg-muted/60 px-4 py-2.5 text-xs font-medium text-muted-foreground md:grid">
              <span>Paper · {shown.length} shown</span><span>Year</span><span>Citations</span><span>Status</span><span />
            </div>
            <ul className="divide-y">{shown.map((p) => (
              <PaperRow key={p.id ?? p.library_paper_id} paper={p} onOpen={() => open(p)}
                actions={<RowActions topic={topic} paper={p} onDeleted={refresh} />} onChanged={refresh} />
            ))}</ul>
          </div>
        )}

      <PaperSheet paperId={researchPaper} onClose={() => setResearchPaper(null)} onOpenPaper={setResearchPaper}
        running={!!data?.autopilot_managed} />
      <AnalysisSheet onClose={() => setLibraryPaper(null)} paper={libraryPaper && {
        topic, paper_id: libraryPaper.library_paper_id, title: libraryPaper.title, source: libraryPaper.venue,
        pages: 0, has_analysis: libraryPaper.has_analysis,
      }} />
      {topic && <UploadPaperDialog topic={topic} open={uploading} onOpenChange={setUploading} />}
    </>
  );
}
