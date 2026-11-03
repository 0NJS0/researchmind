import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, FileDown, GitCompareArrows, Lightbulb, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogTitle, SheetContent } from "@/components/ui/dialog";
import { Alert, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import { humanize } from "@/lib/utils";
import { PaperRef } from "./common";

const SECTION_ORDER = ["research_problem", "motivation", "research_questions", "methodology", "dataset", "key_findings",
  "contributions", "limitations", "future_work", "relevance"];

export const CONTENT_LABEL: Record<string, { label: string; tone: "success" | "warning" | "neutral" | "danger" }> = {
  full_text: { label: "Full text", tone: "success" },
  abstract: { label: "Abstract only", tone: "warning" },
  metadata_only: { label: "Metadata only", tone: "neutral" },
  unavailable: { label: "No content", tone: "danger" },
  pending: { label: "Not fetched", tone: "neutral" },
};

const RELATION_TONE: Record<string, "danger" | "primary" | "neutral" | "success"> = {
  contradicts: "danger", extends: "primary", complements: "success",
};

export function PaperSheet({ paperId, onClose, onOpenPaper, running }: {
  paperId: string | null; onClose: () => void; onOpenPaper: (id: string) => void; running: boolean;
}) {
  const qc = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["paper", paperId], queryFn: () => api.research.paper(paperId!), enabled: !!paperId,
  });
  const paper = data?.paper;
  const content = paper ? CONTENT_LABEL[paper.content_status] : undefined;

  return (
    <Dialog open={!!paperId} onOpenChange={(o) => !o && onClose()}>
      <SheetContent aria-describedby={undefined}>
        {isLoading || !paper ? (
          <div className="space-y-3 p-6">{error ? <Alert>{(error as Error).message}</Alert> : <><Skeleton className="h-6 w-3/4" /><Skeleton className="h-40" /></>}</div>
        ) : (
          <>
            <div className="border-b px-6 py-5 pr-12">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <PaperRef refId={paper.ref || "—"} />
                {content && <Badge tone={content.tone}>{content.label}</Badge>}
                {paper.relevance_score != null && <Badge tone="outline">Relevance {(paper.relevance_score * 100).toFixed(0)}%</Badge>}
                {paper.added_by === "user" && <Badge tone="primary">Added by you</Badge>}
              </div>
              <DialogTitle className="text-lg leading-snug">{paper.title}</DialogTitle>
              <p className="mt-1 text-sm text-muted-foreground">
                {[paper.authors?.slice(0, 4).join(", ") + (paper.authors?.length > 4 ? " et al." : ""), paper.venue, paper.year]
                  .filter(Boolean).join(" · ")}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {paper.url && <Button asChild variant="outline" size="sm"><a href={paper.url} target="_blank" rel="noreferrer noopener"><ExternalLink />Source</a></Button>}
                {paper.pdf_url && <Button asChild variant="outline" size="sm"><a href={paper.pdf_url} target="_blank" rel="noreferrer noopener"><FileDown />PDF</a></Button>}
                {paper.doi && <Button asChild variant="outline" size="sm"><a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer noopener">DOI</a></Button>}
                <Button variant="ghost" size="sm" disabled={running} onClick={async () => {
                  try { await api.research.analyzePaper(paper.id); toast.success("Re-analysing this paper"); qc.invalidateQueries({ queryKey: ["session"] }); }
                  catch (e) { toast.error((e as Error).message); }
                }}><RefreshCw />Re-analyse</Button>
              </div>
            </div>

            <div className="flex-1 space-y-7 overflow-y-auto px-6 py-5 scrollbar-thin">
              {paper.relevance_reason && <p className="text-sm text-muted-foreground"><span className="font-medium text-foreground">Why selected: </span>{paper.relevance_reason}</p>}
              {paper.error && paper.processing_status !== "done" && <Alert tone="warning">{paper.error}</Alert>}

              {data.summary ? (
                <section className="space-y-5">
                  {paper.content_status !== "full_text" && (
                    <Alert tone="info">This summary is based on the {paper.content_status === "abstract" ? "abstract only" : "title and metadata only"} — details may be missing.</Alert>
                  )}
                  {SECTION_ORDER.filter((k) => data.summary!.sections[k]?.length).map((key) => (
                    <div key={key}>
                      <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{humanize(key)}</h4>
                      <ul className="space-y-1.5">
                        {data.summary!.sections[key].map((s, i) => (
                          <li key={i} className="flex gap-2 text-sm leading-relaxed">
                            <span className="mt-2 size-1.5 shrink-0 rounded-full bg-muted-foreground/40" />
                            <span>{s.text}{s.inferred && <Badge tone="outline" className="ml-2 align-middle" title="The model's interpretation, not stated by the authors">inferred</Badge>}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </section>
              ) : paper.abstract ? (
                <section><h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Abstract</h4>
                  <p className="text-sm leading-relaxed">{paper.abstract}</p></section>
              ) : <p className="text-sm text-muted-foreground">Not summarised yet.</p>}

              {data.relations.length > 0 && (
                <section>
                  <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold"><GitCompareArrows className="size-4" />Related papers</h4>
                  <ul className="space-y-2">
                    {data.relations.map((r, i) => (
                      <li key={i} className="rounded-lg border p-3 text-sm">
                        <div className="flex flex-wrap items-center gap-2">
                          <PaperRef refId={r.related_ref} onClick={() => onOpenPaper(r.other_paper_id)} />
                          <Badge tone={RELATION_TONE[r.relationship] ?? "neutral"}>{humanize(r.relationship)}</Badge>
                          <span className="truncate text-xs text-muted-foreground">{r.related_title}</span>
                        </div>
                        {r.similarities && <p className="mt-2 text-muted-foreground"><span className="font-medium text-foreground">Similar: </span>{r.similarities}</p>}
                        {r.differences && <p className="mt-1 text-muted-foreground"><span className="font-medium text-foreground">Different: </span>{r.differences}</p>}
                        {r.contradiction && <p className="mt-1 text-danger"><span className="font-medium">Conflict: </span>{r.contradiction}</p>}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {data.gaps.length > 0 && (
                <section>
                  <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold"><Lightbulb className="size-4" />Evidence for research gaps</h4>
                  <ul className="space-y-2">
                    {data.gaps.map((g, i) => (
                      <li key={i} className="rounded-lg border p-3 text-sm">
                        <div className="flex items-center gap-2"><PaperRef refId={g.ref} /><span className="font-medium">{g.title}</span></div>
                        <p className="mt-1 text-muted-foreground"><Badge tone={g.relationship === "contradicts" ? "danger" : "neutral"} className="mr-1.5">{humanize(g.relationship)}</Badge>{g.evidence}</p>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Dialog>
  );
}
