import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, Download, FileText, Files, Loader2, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ExportTopicButton } from "@/components/ExportTopicButton";
import { Markdown } from "@/components/Markdown";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, EmptyState, PageHeader, Progress, Skeleton } from "@/components/ui/misc";
import { useJob, useRunningJob } from "@/hooks/useJob";
import { api } from "@/lib/api";
import { useTopic } from "@/lib/topic";
import type { ReportEntry } from "@/lib/types";
import { cn, downloadText, formatDateTime, relativeTime, copyText } from "@/lib/utils";

const keyOf = (r: ReportEntry) => `${r.type}:${r.paper_id ?? r.filename}`;

function headings(md: string) {
  return md.split("\n").filter((l) => /^#{1,2}\s/.test(l)).map((l) => ({
    level: l.startsWith("## ") ? 2 : 1, text: l.replace(/^#+\s*/, "").replace(/[*`]/g, ""),
  }));
}

function slug(text: string) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function ReportReader({ topic, entry, onDeleted }: { topic: string; entry: ReportEntry; onDeleted: () => void }) {
  const { data: content, isLoading, error } = useQuery({
    queryKey: ["report", topic, keyOf(entry)],
    queryFn: () => entry.type === "paper" ? api.library.analysis(topic, entry.paper_id!) : api.library.report(topic, entry.filename),
    refetchInterval: entry.type === "living" ? 30_000 : false,  // the autopilot rewrites it after every paper
  });
  const toc = useMemo(() => headings(content ?? ""), [content]);
  const filename = entry.type === "paper" ? entry.filename : `${topic}_${entry.filename}`;

  if (isLoading) return <Skeleton className="h-96" />;
  if (error) return <Alert>{(error as Error).message}</Alert>;
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_200px]">
      <Card className="min-w-0 p-6 sm:p-8">
        <div className="mb-6 flex flex-wrap items-center gap-2 border-b pb-4">
          <Badge tone={entry.type === "paper" ? "neutral" : entry.type === "living" ? "success" : "primary"}>
            {entry.type === "living" ? "Living report" : entry.type === "combined" ? "Topic report" : "Paper analysis"}</Badge>
          <span className="text-xs text-muted-foreground">{formatDateTime(entry.created_at)}</span>
          <div className="ml-auto flex gap-1">
            <Button variant="ghost" size="sm" onClick={async () => { if (await copyText(content ?? "")) toast.success("Copied as Markdown"); else toast.error("Could not copy"); }}><Copy />Copy</Button>
            <Button variant="ghost" size="sm" onClick={() => downloadText(filename, content ?? "")}><Download />Download</Button>
            {entry.type === "combined" && (
              <ConfirmDialog title="Delete report?" description="This report file will be permanently deleted."
                onConfirm={async () => {
                  try { await api.library.deleteReport(topic, entry.filename); toast.success("Report deleted"); onDeleted(); }
                  catch (e) { toast.error((e as Error).message); }
                }}
                trigger={(open) => <Button variant="ghost" size="sm" className="hover:text-danger" onClick={open}><Trash2 />Delete</Button>} />
            )}
          </div>
        </div>
        <Markdown className="[&_h1]:scroll-mt-20 [&_h2]:scroll-mt-20">{content ?? ""}</Markdown>
      </Card>
      {toc.length > 2 && (
        <nav className="hidden xl:block" aria-label="Table of contents">
          <div className="sticky top-20">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Contents</p>
            <ul className="space-y-1.5 border-l text-sm">
              {toc.map((h, i) => (
                <li key={i}>
                  <button onClick={() => {
                    const el = Array.from(document.querySelectorAll(".prose-rm h1, .prose-rm h2"))
                      .find((n) => slug(n.textContent ?? "") === slug(h.text));
                    el?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                    className={cn("-ml-px block border-l border-transparent pl-3 text-left text-muted-foreground hover:border-primary hover:text-foreground cursor-pointer",
                      h.level === 2 && "pl-5 text-xs")}>{h.text}</button>
                </li>
              ))}
            </ul>
          </div>
        </nav>
      )}
    </div>
  );
}

export default function ReportsPage() {
  const { topic, topics, setTopic } = useTopic();
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const wanted = params.get("topic");
  // Links like the autopilot's "Living report" name their topic, which may differ from the sidebar's:
  // switch to it once, then drop the parameter so the sidebar can change the topic again.
  useEffect(() => {
    if (!wanted || !topics.length) return;
    if (topics.some((t) => t.name === wanted)) setTopic(wanted);
    setParams({}, { replace: true });
  }, [wanted, topics, setTopic, setParams]);
  const [selected, setSelected] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const running = useRunningJob(topic, "report");
  const { data: reports, isLoading } = useQuery({
    queryKey: ["reports", topic], queryFn: () => api.library.reports(topic), enabled: !!topic,
  });
  const current = topics.find((t) => t.name === topic);
  const { data: job } = useJob(jobId ?? running.data?.id ?? null, (j) => {
    qc.invalidateQueries({ queryKey: ["reports", topic] });
    qc.invalidateQueries({ queryKey: ["papers", topic] });
    if (j.status === "completed") {
      toast.success("Report ready");
      setSelected(`combined:${j.file}`);
    } else toast.error(`Report failed: ${j.error}`);
    setJobId(null);
  });
  const [starting, setStarting] = useState(false);
  const generating = starting || job?.status === "running";

  const generate = async () => {
    if (generating) return;
    setStarting(true);
    try {
      const j = await api.library.createReport(topic);
      setJobId(j.id);
    } catch (e) { toast.error((e as Error).message); } finally { setStarting(false); }
  };

  if (!topics.length) {
    return <EmptyState icon={FileText} title="No topics yet" description="Create a topic and upload papers first."
      action={<Button asChild><Link to="/library">Go to Library</Link></Button>} />;
  }

  const living = reports?.filter((r) => r.type === "living") ?? [];
  const combined = reports?.filter((r) => r.type === "combined") ?? [];
  const perPaper = reports?.filter((r) => r.type === "paper") ?? [];
  const active = reports?.find((r) => keyOf(r) === selected) ?? living[0] ?? combined[0] ?? perPaper[0];

  return (
    <>
      <PageHeader title="Reports"
        description="The living report is updated by the autopilot after every paper. A full re-analysis report re-reads every paper from scratch."
        actions={<><ExportTopicButton topic={topic} /><Button onClick={generate} loading={generating} disabled={!current?.papers}>
          {!generating && <Sparkles />}{generating ? "Generating…" : "Full re-analysis"}</Button></>} />

      {generating && job && (
        <Card className="mb-6 p-5">
          <div className="flex items-center gap-3">
            <Loader2 className="size-5 animate-spin text-primary" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium capitalize">{job.stage}{job.current ? ` · ${job.current}` : ""}</p>
              <p className="text-xs text-muted-foreground">
                {job.total ? `${job.done} of ${job.total} papers analysed` : "Preparing"} · runs in the background, you can leave this page
              </p>
            </div>
          </div>
          {job.total > 0 && <Progress value={(job.done / job.total) * 100} className="mt-4" />}
        </Card>
      )}
      {current?.papers === 0 && <Alert tone="info" className="mb-6">Upload papers to this topic before generating a report.</Alert>}

      {isLoading ? <Skeleton className="h-64" /> : !reports?.length ? (
        <EmptyState icon={Files} title="No reports yet" description="Generate a report to get per-paper analyses and a cross-paper synthesis." />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[240px_minmax(0,1fr)]">
          <aside className="space-y-5">
            {[["Living report", living], ["Full re-analysis reports", combined], ["Paper analyses", perPaper]].map(([label, list]) => (list as ReportEntry[]).length > 0 && (
              <div key={label as string}>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label as string}</p>
                <ul className="space-y-1">
                  {(list as ReportEntry[]).map((r) => (
                    <li key={keyOf(r)}>
                      <button onClick={() => setSelected(keyOf(r))}
                        className={cn("w-full rounded-md px-3 py-2 text-left text-sm hover:bg-muted cursor-pointer",
                          active && keyOf(active) === keyOf(r) && "bg-primary-soft text-primary hover:bg-primary-soft")}>
                        <p className="truncate font-medium">{r.type === "living" ? "Always up to date" : r.type === "combined" ? formatDateTime(r.created_at) : r.paper_id!.replace(/_/g, " ")}</p>
                        {r.type === "living" && <p className="text-xs text-muted-foreground">Updated {relativeTime(r.created_at)}</p>}
                        {r.type === "combined" && <p className="text-xs text-muted-foreground">{r.filename}</p>}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </aside>
          {active && <ReportReader key={keyOf(active)} topic={topic} entry={active}
            onDeleted={() => { setSelected(null); qc.invalidateQueries({ queryKey: ["reports", topic] }); }} />}
        </div>
      )}
    </>
  );
}
