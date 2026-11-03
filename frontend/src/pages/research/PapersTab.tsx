import { useQuery } from "@tanstack/react-query";
import { FileStack, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { EmptyState, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import type { ResearchPaper } from "@/lib/types";
import { cn, humanize } from "@/lib/utils";
import { PaperRef } from "./common";
import { CONTENT_LABEL } from "./PaperSheet";

const FILTERS = [
  { value: "selected", label: "Selected" },
  { value: "rejected", label: "Rejected" },
  { value: "all", label: "All results" },
] as const;

const PROCESSING_TONE: Record<string, "success" | "primary" | "danger" | "warning" | "neutral"> = {
  done: "success", failed: "danger", analysis_failed: "warning", skipped: "neutral",
  fetching: "primary", summarizing: "primary", analyzing: "primary", pending: "neutral",
};

export function filterPapers(papers: ResearchPaper[], text: string) {
  const q = text.trim().toLowerCase();
  if (!q) return papers;
  return papers.filter((p) => [p.title, p.ref, p.venue, p.authors?.join(" "), String(p.year ?? "")]
    .some((v) => v?.toLowerCase().includes(q)));
}

export function RelevanceBar({ score }: { score: number | null }) {
  if (score == null) return <span className="text-xs text-muted-foreground">—</span>;
  const pct = Math.round(score * 100);
  return (
    <div className="flex items-center gap-2" title={`Relevance ${pct}%`}>
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", pct >= 70 ? "bg-success" : pct >= 45 ? "bg-primary" : "bg-muted-foreground/50")} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-muted-foreground">{pct}</span>
    </div>
  );
}

export function PapersTab({ sessionId, onOpen }: { sessionId: string; onOpen: (id: string) => void }) {
  const [selection, setSelection] = useState<(typeof FILTERS)[number]["value"]>("selected");
  const [text, setText] = useState("");
  const { data: papers, isLoading } = useQuery({
    queryKey: ["papers", sessionId, selection], queryFn: () => api.research.papers(sessionId, selection),
  });
  const shown = useMemo(() => filterPapers(papers ?? [], text), [papers, text]);

  return (
    <div>
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex rounded-lg border bg-card p-0.5" role="tablist" aria-label="Paper filter">
          {FILTERS.map((f) => (
            <button key={f.value} role="tab" aria-selected={selection === f.value} onClick={() => setSelection(f.value)}
              className={cn("rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground cursor-pointer",
                selection === f.value && "bg-muted text-foreground")}>{f.label}</button>
          ))}
        </div>
        <div className="relative sm:ml-auto sm:w-64">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Filter by title, author, venue…" className="pl-9" aria-label="Filter papers" />
        </div>
      </div>

      {isLoading ? <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16" />)}</div> :
        !shown.length ? <EmptyState icon={FileStack} title={text ? "No matching papers" : "No papers here yet"}
          description={selection === "selected" && !text ? "Selected papers appear as the agent screens search results." : undefined} /> : (
          <ul className="divide-y overflow-hidden rounded-lg border bg-card">
            {shown.map((p) => {
              const content = CONTENT_LABEL[p.content_status];
              return (
                <li key={p.id}>
                  <button onClick={() => onOpen(p.id)} className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-muted/40 cursor-pointer">
                    <div className="w-9 shrink-0 pt-0.5">{p.ref ? <PaperRef refId={p.ref} /> : null}</div>
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-2 text-sm font-medium leading-snug">{p.title}</p>
                      <p className="mt-0.5 truncate text-xs text-muted-foreground">
                        {[p.authors?.[0] && `${p.authors[0]}${p.authors.length > 1 ? " et al." : ""}`, p.venue || p.source, p.year].filter(Boolean).join(" · ")}
                      </p>
                      {p.selection_status === "rejected" && p.relevance_reason &&
                        <p className="mt-1 line-clamp-1 text-xs text-muted-foreground italic">{p.relevance_reason}</p>}
                    </div>
                    <div className="hidden shrink-0 flex-col items-end gap-1.5 sm:flex">
                      <RelevanceBar score={p.relevance_score} />
                      <div className="flex gap-1">
                        {p.selection_status === "selected" && content && <Badge tone={content.tone}>{content.label}</Badge>}
                        {p.selection_status === "selected" && p.processing_status !== "done" &&
                          <Badge tone={PROCESSING_TONE[p.processing_status] ?? "neutral"}>{humanize(p.processing_status)}</Badge>}
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
    </div>
  );
}
