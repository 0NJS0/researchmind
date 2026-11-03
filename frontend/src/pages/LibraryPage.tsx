import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, FileSearch, FileText, FolderPlus, MoreHorizontal, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { NewTopicDialog } from "@/components/AppLayout";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Markdown } from "@/components/Markdown";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogTitle, SheetContent } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown";
import { Alert, EmptyState, PageHeader, Progress, Skeleton } from "@/components/ui/misc";
import { useJob, useRunningJob } from "@/hooks/useJob";
import { api } from "@/lib/api";
import { useTopic } from "@/lib/topic";
import type { LibraryPaper } from "@/lib/types";
import { pluralize, relativeTime } from "@/lib/utils";

export function AnalysisSheet({ paper, onClose }: {
  paper: Pick<LibraryPaper, "topic" | "paper_id" | "title" | "source" | "pages" | "has_analysis"> | null; onClose: () => void;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["analysis", paper?.topic, paper?.paper_id],
    queryFn: () => api.library.analysis(paper!.topic, paper!.paper_id),
    enabled: !!paper?.has_analysis,
  });
  return (
    <Dialog open={!!paper} onOpenChange={(o) => !o && onClose()}>
      <SheetContent aria-describedby={undefined}>
        <div className="border-b px-6 py-4 pr-12">
          <DialogTitle className="truncate">{paper?.title}</DialogTitle>
          <p className="text-xs text-muted-foreground">{[paper?.source, paper?.pages ? pluralize(paper.pages, "page") : ""].filter(Boolean).join(" · ")}</p>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-5 scrollbar-thin">
          {!paper?.has_analysis ? (
            <EmptyState icon={FileSearch} title="No analysis yet"
              description="Generate a topic report on the Reports page — each paper gets its own structured analysis." />
          ) : isLoading ? <Skeleton className="h-64" /> : error ? <Alert>{(error as Error).message}</Alert> :
            <Markdown>{data ?? ""}</Markdown>}
        </div>
      </SheetContent>
    </Dialog>
  );
}

function PapersTable({ topic }: { topic: string }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState<LibraryPaper | null>(null);
  const { data: papers, isLoading, error } = useQuery({ queryKey: ["papers", topic], queryFn: () => api.library.papers(topic) });

  if (isLoading) return <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-14" />)}</div>;
  if (error) return <Alert title="Could not load papers">{(error as Error).message}</Alert>;
  if (!papers?.length) return <EmptyState icon={BookOpen} title="No papers in this topic" description="Upload PDFs above to start building your library." />;

  return (
    <>
      <div className="overflow-hidden rounded-lg border bg-card">
        <table className="w-full table-fixed text-sm">
          <thead className="bg-muted/60 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-2.5 font-medium">Paper</th>
              <th className="hidden w-20 px-4 py-2.5 font-medium sm:table-cell">Year</th>
              <th className="hidden w-24 px-4 py-2.5 font-medium md:table-cell">Citations</th>
              <th className="hidden w-28 px-4 py-2.5 font-medium md:table-cell">Indexed</th>
              <th className="w-28 px-4 py-2.5 font-medium">Analysis</th>
              <th className="w-12" />
            </tr>
          </thead>
          <tbody className="divide-y">
            {papers.map((p) => (
              <tr key={p.paper_id} className="group hover:bg-muted/40">
                <td className="max-w-0 px-4 py-3">
                  <button className="block w-full truncate text-left font-medium hover:text-primary cursor-pointer" onClick={() => setOpen(p)} title={p.title}>
                    {p.title}
                  </button>
                  <p className="flex items-center gap-1.5 truncate text-xs text-muted-foreground">
                    <Badge tone={p.origin === "agent" ? "primary" : "neutral"}>{p.origin === "agent" ? "Found by agent" : "Uploaded"}</Badge>
                    <span className="truncate">{p.venue || p.source}</span>
                  </p>
                </td>
                <td className="hidden px-4 py-3 tabular-nums text-muted-foreground sm:table-cell">{p.year ?? "—"}</td>
                <td className="hidden px-4 py-3 tabular-nums text-muted-foreground md:table-cell">{p.citation_count ?? "—"}</td>
                <td className="hidden px-4 py-3 text-muted-foreground md:table-cell">{relativeTime(p.indexed_at)}</td>
                <td className="px-4 py-3">
                  {p.has_analysis
                    ? <button onClick={() => setOpen(p)} className="cursor-pointer"><Badge tone="success"><FileText />Ready</Badge></button>
                    : <Badge tone="outline">None</Badge>}
                </td>
                <td className="px-2 py-3 text-right">
                  <ConfirmDialog
                    title="Delete paper?"
                    description={<>“{p.title}” and its analysis will be removed from the Library and from the topic's research and reports. This cannot be undone.</>}
                    onConfirm={async () => {
                      try {
                        await api.library.deletePaper(topic, p.paper_id);
                        toast.success("Paper deleted");
                        qc.invalidateQueries({ queryKey: ["papers", topic] });
                        qc.invalidateQueries({ queryKey: ["topics"] });
                      } catch (e) { toast.error((e as Error).message); }
                    }}
                    trigger={(openConfirm) => (
                      <Button variant="ghost" size="icon-sm" aria-label={`Delete ${p.paper_id}`} onClick={openConfirm}
                        className="text-muted-foreground opacity-60 group-hover:opacity-100 hover:text-danger">
                        <Trash2 />
                      </Button>
                    )} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <AnalysisSheet paper={open} onClose={() => setOpen(null)} />
    </>
  );
}

export default function LibraryPage() {
  const { topic, topics, loading } = useTopic();
  const qc = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const running = useRunningJob(topic, "ingest");
  const runningAll = useRunningJob("*", "ingest");   // a whole-library re-index
  const activeJobId = jobId ?? running.data?.id ?? runningAll.data?.id ?? null;
  const { data: job } = useJob(activeJobId, (j) => {
    if (j.status === "completed") toast.success(`Re-indexing finished · ${j.current}`);
    else toast.error(`Re-indexing failed: ${j.error}`);
    qc.invalidateQueries({ queryKey: ["papers"] });
    qc.invalidateQueries({ queryKey: ["topics"] });
    setJobId(null);
  });
  const current = topics.find((t) => t.name === topic);

  const startIngest = async (reset: boolean) => {
    try {
      const j = await api.library.ingest(reset ? undefined : topic, reset);
      setJobId(j.id);
      toast.info(reset ? "Re-indexing the whole library…" : "Re-indexing this topic…");
    } catch (e) { toast.error((e as Error).message); }
  };

  if (loading) return <Skeleton className="h-64" />;
  if (!topics.length) {
    return (
      <>
        <PageHeader title="Library" description="Organise papers into topics, then ask questions or generate reports." />
        <EmptyState icon={FolderPlus} title="Create your first topic"
          description="A topic is a collection of related papers — for example “LLM toxicity detection”."
          action={<Button onClick={() => setCreating(true)}><FolderPlus />New topic</Button>} />
        <NewTopicDialog open={creating} onOpenChange={setCreating} />
      </>
    );
  }

  return (
    <>
      <PageHeader
        title={topic.replace(/_/g, " ")}
        description={current ? `${pluralize(current.papers, "paper")} · ${pluralize(current.chunks, "indexed passage")}` : undefined}
        actions={
          <>
            <Button variant="outline" onClick={() => startIngest(false)} disabled={!!job && job.status === "running"}>
              <RefreshCw />Re-index topic
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild><Button variant="outline" size="icon" aria-label="More actions"><MoreHorizontal /></Button></DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onSelect={() => setCreating(true)}><FolderPlus />New topic</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => startIngest(true)}><RefreshCw />Re-index entire library</DropdownMenuItem>
                <DropdownMenuSeparator />
                <ConfirmDialog
                  title="Delete topic?"
                  description={<>All papers, analyses and reports in “{topic}” will be permanently deleted.</>}
                  confirmLabel="Delete topic"
                  onConfirm={async () => {
                    try {
                      await api.library.deleteTopic(topic);
                      toast.success("Topic deleted");
                      await qc.invalidateQueries({ queryKey: ["topics"] });
                    } catch (e) { toast.error((e as Error).message); }
                  }}
                  trigger={(openConfirm) => (
                    <DropdownMenuItem danger onSelect={(e) => { e.preventDefault(); openConfirm(); }}><Trash2 />Delete topic</DropdownMenuItem>
                  )} />
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        } />

      {job?.status === "running" && (
        <Alert tone="info" className="mb-6" title={`Re-indexing · ${job.stage}`}>
          {job.total > 0 && <><Progress value={(job.done / job.total) * 100} className="mt-2" />
            <p className="mt-1 text-xs">{job.done} / {job.total} passages embedded</p></>}
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="order-2 lg:order-1">
          <h2 className="mb-3 text-sm font-semibold">Papers</h2>
          <PapersTable topic={topic} />
        </section>
        <aside className="order-1 lg:order-2">
          <Card>
            <CardHeader>
              <CardTitle>Add papers</CardTitle>
              <CardDescription>PDFs are indexed into this topic for search and reports.</CardDescription>
            </CardHeader>
            <CardContent><UploadDropzone key={topic} topic={topic} /></CardContent>
          </Card>
        </aside>
      </div>
      <NewTopicDialog open={creating} onOpenChange={setCreating} />
    </>
  );
}
