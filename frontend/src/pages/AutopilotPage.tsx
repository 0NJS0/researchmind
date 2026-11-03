import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, ArrowRightLeft, BookMarked, Bot, CheckCircle2, Clock, FileText, FolderOpen, FolderPlus, Lightbulb, ListOrdered,
  Loader2, Pause, Play, Plus, Search, Sparkles, TriangleAlert,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { NewTopicResearchDialog } from "@/components/NewTopicResearchDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label, Textarea } from "@/components/ui/input";
import { Alert, EmptyState, PageHeader, Progress, Skeleton } from "@/components/ui/misc";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { useTopic } from "@/lib/topic";
import type { AutopilotStatus, QueuedPaper } from "@/lib/types";
import { cn, formatDateTime, relativeTime } from "@/lib/utils";
import { ActivityFeed } from "./research/ActivityFeed";
import { AutopilotRoleBadge } from "./research/common";
import { GapsTab } from "./research/GapsTab";
import { PaperSheet } from "./research/PaperSheet";
import { SynthesisTab } from "./research/SynthesisTab";
import { useSessionEvents } from "./research/useSessionEvents";
import { usePageVisible } from "@/hooks/usePageVisible";

const BUSY_STATES = ["processing", "searching", "screening", "verifying"];

export function useAutopilot(interval = 4000) {
  return useQuery({ queryKey: ["autopilot"], queryFn: api.autopilot.status, refetchInterval: interval });
}

const STATE_STYLE: Record<string, { label: string; tone: "primary" | "success" | "warning" | "neutral" | "danger"; spin?: boolean }> = {
  processing: { label: "Processing", tone: "primary", spin: true },
  searching: { label: "Searching", tone: "primary", spin: true },
  screening: { label: "Screening", tone: "primary", spin: true },
  verifying: { label: "Checking a gap", tone: "primary", spin: true },
  waiting: { label: "Waiting for new papers", tone: "success" },
  idle: { label: "Idle", tone: "neutral" },
  paused: { label: "Paused · quota", tone: "warning" },
  stopped: { label: "Paused", tone: "neutral" },
  error: { label: "Retrying", tone: "danger" },
};

export function AutopilotBadge({ status }: { status: AutopilotStatus }) {
  const s = STATE_STYLE[status.state] ?? { label: status.state, tone: "neutral" as const };
  return <Badge tone={s.tone}>{s.spin ? <Loader2 className="animate-spin" /> : <Bot />}{s.label}</Badge>;
}

function ScoreChip({ paper }: { paper: QueuedPaper }) {
  const d = paper.priority_detail;
  const content = d ? (
    <div className="space-y-0.5">
      <p className="font-medium">Why this position</p>
      <p>Relevance {Math.round(d.relevance * 100)}% · Recency {Math.round(d.recency * 100)}%</p>
      <p>Impact {Math.round(d.impact * 100)}% · Venue {Math.round(d.venue * 100)}%</p>
      {paper.added_by === "user" && <p>Your upload — analysed first</p>}
    </div>
  ) : "Not ranked yet";
  return (
    <Tooltip content={content}>
      <span className="cursor-help rounded-md bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground">
        {paper.priority_score != null ? Math.min(99, Math.round(paper.priority_score * 100)) : "–"}
      </span>
    </Tooltip>
  );
}

function PaperMeta({ year, venue, citations, peerReviewed }: { year: number | null; venue: string; citations: number | null; peerReviewed?: boolean | null }) {
  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
      {year && <span>{year}</span>}
      {venue && <span className="max-w-64 truncate">{venue}</span>}
      {citations != null && <span>{citations} citations</span>}
      {peerReviewed === true && <Badge tone="success">Peer reviewed</Badge>}
      {peerReviewed === false && <Badge tone="outline">Preprint</Badge>}
    </div>
  );
}

function TopicControl({ status }: { status: AutopilotStatus }) {
  const { topics } = useTopic();
  const qc = useQueryClient();
  const [topic, setTopic] = useState(status.requested_topic || status.active_topic || topics[0]?.name || "");
  const [focus, setFocus] = useState(status.focus);
  const [edited, setEdited] = useState(false);
  const [busy, setBusy] = useState(false);
  const { data: sessions } = useQuery({ queryKey: ["research"], queryFn: api.research.list });
  // Show the selected topic's own focus (not the active topic's), unless the user is editing it.
  const topicFocus = topic === status.active_topic ? status.focus
    : sessions?.find((s) => s.topic_name === topic)?.description ?? "";
  useEffect(() => { if (!edited) setFocus(topicFocus); }, [topicFocus, edited]);
  useEffect(() => { if (!topic && topics[0]) setTopic(topics[0].name); }, [topics, topic]);

  const isActive = topic === status.active_topic;
  const apply = async () => {
    setBusy(true);
    try {
      // Only send a focus the user wrote for this topic; otherwise the topic keeps its own.
      const res = await api.autopilot.setTopic(topic, edited || isActive ? focus : undefined);
      setEdited(false);
      qc.invalidateQueries({ queryKey: ["research"] });
      qc.setQueryData(["autopilot"], res);
      toast.success(res.status === "pending"
        ? `Will switch to “${topic.replace(/_/g, " ")}” after the current task`
        : isActive ? "Focus saved" : `Now researching “${topic.replace(/_/g, " ")}”`);
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Research topic</CardTitle>
        <CardDescription>The agent searches, reads and reports on this topic only, until you change it.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor="ap-topic">Topic</Label>
          <select id="ap-topic" value={topic} onChange={(e) => { setTopic(e.target.value); setEdited(false); }}
            className="h-9 w-full rounded-md border border-input bg-card px-3 text-sm shadow-xs">
            {topics.map((t) => <option key={t.name} value={t.name}>{t.name.replace(/_/g, " ")} ({t.papers} papers)</option>)}
          </select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="ap-focus">Focus <span className="font-normal text-muted-foreground">(optional)</span></Label>
          <Textarea id="ap-focus" rows={3} value={focus} maxLength={2000} onChange={(e) => { setFocus(e.target.value); setEdited(true); }}
            placeholder="What to emphasise or exclude — e.g. multilingual toxicity, empirical studies, papers after 2022…" />
          <p className="text-xs text-muted-foreground">Changing the focus makes the agent re-plan its searches.</p>
        </div>
        <Button className="w-full" onClick={apply} loading={busy} disabled={!topic}>
          {!busy && (isActive ? <CheckCircle2 /> : <ArrowRightLeft />)}
          {isActive ? "Save focus" : status.active_topic ? "Switch to this topic" : "Start autopilot"}
        </Button>
      </CardContent>
    </Card>
  );
}

function Stat({ icon: Icon, label, value }: { icon: typeof FileText; label: string; value: number | string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card p-3">
      <div className="rounded-md bg-primary-soft p-1.5 text-primary"><Icon className="size-4" /></div>
      <div><p className="text-lg font-semibold leading-none tabular-nums">{value}</p><p className="mt-1 text-xs text-muted-foreground">{label}</p></div>
    </div>
  );
}

function QueueList({ status }: { status: AutopilotStatus }) {
  return (
    <>
            {!status.queue.length ? (
              <p className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                {status.active_topic ? "The queue is empty — the agent searches for more papers when it's due." : "No topic selected."}
              </p>
            ) : (
              <ol className="divide-y overflow-hidden rounded-lg border bg-card">
                {status.queue.map((p, i) => (
                  <li key={p.id} className="flex items-start gap-3 px-4 py-3">
                    <span className="w-5 pt-0.5 text-right text-xs font-semibold tabular-nums text-muted-foreground">{i + 1}</span>
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-2 text-sm font-medium leading-snug">
                        {p.url ? <a href={p.url} target="_blank" rel="noreferrer noopener" className="hover:text-primary">{p.title}</a> : p.title}
                        {p.added_by === "user" && <Badge tone="primary" className="ml-2 align-middle">Your upload</Badge>}
                      </p>
                      <PaperMeta year={p.year} venue={p.venue} citations={p.citation_count} peerReviewed={p.peer_reviewed} />
                    </div>
                    <ScoreChip paper={p} />
                  </li>
                ))}
              </ol>
            )}
    </>
  );
}

function RecentList({ status }: { status: AutopilotStatus }) {
  return (
    <>
            {!status.recent.length ? (
              <p className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">Nothing processed yet.</p>
            ) : (
              <ul className="divide-y overflow-hidden rounded-lg border bg-card">
                {status.recent.map((p) => (
                  <li key={p.id} className="flex items-start gap-3 px-4 py-3">
                    <CheckCircle2 className={cn("mt-0.5 size-4 shrink-0", p.processing_status === "done" ? "text-success" : "text-warning")} />
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-2 text-sm font-medium leading-snug"><span className="mr-1.5 font-mono text-xs text-muted-foreground">{p.ref}</span>{p.title}</p>
                      <PaperMeta year={p.year} venue={p.venue} citations={p.citation_count} />
                    </div>
                    <span className="shrink-0 text-xs text-muted-foreground">{relativeTime(p.processed_at)}</span>
                  </li>
                ))}
              </ul>
            )}
    </>
  );
}

/** While work runs, refresh what the live stream would have refreshed, without holding a connection open. */
function usePollWhileRunning(sessionId: string, running: boolean, qc: ReturnType<typeof useQueryClient>) {
  const visible = usePageVisible();
  useEffect(() => {
    if (!sessionId || !running || !visible) return;
    const t = setInterval(() => {
      for (const key of ["gaps", "synthesis", "papers", "session"]) qc.invalidateQueries({ queryKey: [key, sessionId] });
      qc.invalidateQueries({ queryKey: ["paper"] });   // an open paper panel
    }, 20_000);
    return () => clearInterval(t);
  }, [sessionId, running, visible, qc]);
  return visible;
}

/** The active topic's queue, latest papers, research gaps, synthesis and activity, in one place. */
function ActiveTopicTabs({ status }: { status: AutopilotStatus }) {
  const [tab, setTab] = useState("queue");
  const [paperId, setPaperId] = useState<string | null>(null);
  const sid = status.session_id ?? "";
  const running = BUSY_STATES.includes(status.state);
  const qc = useQueryClient();
  const visible = usePollWhileRunning(sid, running, qc);
  // Live stream only while the Activity tab is on screen: each open stream holds one of the browser's ~6
  // connections to this server (plain http), and the other tabs are refreshed by polling instead.
  const events = useSessionEvents(sid, !!sid && running && visible && tab === "activity");
  useEffect(() => { if (!sid && tab !== "queue" && tab !== "recent") setTab("queue"); }, [sid, tab]);
  return (
    <>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="queue"><ListOrdered />Up next<span className="text-xs text-muted-foreground">{status.counts.queued}</span></TabsTrigger>
          <TabsTrigger value="recent"><Sparkles />Recently added</TabsTrigger>
          {sid && <TabsTrigger value="gaps"><Lightbulb />Gaps</TabsTrigger>}
          {sid && <TabsTrigger value="synthesis"><BookMarked />Synthesis</TabsTrigger>}
          {sid && <TabsTrigger value="activity"><Activity />Activity</TabsTrigger>}
        </TabsList>
        <TabsContent value="queue"><QueueList status={status} /></TabsContent>
        <TabsContent value="recent"><RecentList status={status} /></TabsContent>
        {sid && <TabsContent value="gaps"><GapsTab sessionId={sid} onOpenPaper={setPaperId} /></TabsContent>}
        {sid && <TabsContent value="synthesis"><SynthesisTab sessionId={sid} onOpenPaper={setPaperId} /></TabsContent>}
        {sid && <TabsContent value="activity"><ActivityFeed events={events} running={running} /></TabsContent>}
      </Tabs>
      {/* The autopilot owns this topic's research, so manual re-analysis is unavailable here. */}
      <PaperSheet paperId={paperId} onClose={() => setPaperId(null)} onOpenPaper={setPaperId} running />
    </>
  );
}

/** Every topic, with what the autopilot is doing on it; open one for its papers, gaps and synthesis. */
function AllTopics({ status }: { status: AutopilotStatus }) {
  const { topics } = useTopic();
  const qc = useQueryClient();
  const { data: sessions = [] } = useQuery({ queryKey: ["research"], queryFn: api.research.list, refetchInterval: 15_000 });
  const byTopic = new Map(sessions.filter((s) => s.topic_name).map((s) => [s.topic_name as string, s]));
  const legacy = sessions.filter((s) => !s.topic_name);   // one-off sessions from before topics and the autopilot merged
  const role = (name: string) => (status.requested_topic === name ? "next" : status.active_topic === name && status.enabled ? "active" : "idle");
  const [busy, setBusy] = useState<string | null>(null);
  const start = async (key: string, fn: () => Promise<{ status?: string; autopilot_status?: string }>) => {
    if (busy) return;
    setBusy(key);
    try {
      const r = await fn();
      qc.invalidateQueries({ queryKey: ["autopilot"] });
      qc.invalidateQueries({ queryKey: ["research"] });
      qc.invalidateQueries({ queryKey: ["topics"] });
      toast.success((r.status ?? r.autopilot_status) === "pending"
        ? "The autopilot switches to it after its current paper" : "Put on the autopilot");
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(null); }
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle>All topics</CardTitle>
        <CardDescription>Open a topic for its papers, research gaps and synthesis.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 p-2 pt-0">
        {topics.map((t) => {
          const s = byTopic.get(t.name);
          const r = role(t.name);
          return (
            <div key={t.name} className="flex items-center gap-2 rounded-md px-2 py-2 hover:bg-muted/50">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{t.name.replace(/_/g, " ")}</p>
                <p className="text-xs text-muted-foreground">
                  {s ? `${s.paper_count} papers · ${s.gap_count} gaps` : `${t.papers} papers`}
                </p>
              </div>
              {r !== "idle" ? <AutopilotRoleBadge role={r} /> : (
                <Tooltip content="Put this topic on the autopilot">
                  <Button size="icon-sm" variant="ghost" aria-label={`Put ${t.name} on the autopilot`}
                    disabled={!!busy} onClick={() => start(t.name, () => api.autopilot.setTopic(t.name))}><Play /></Button>
                </Tooltip>
              )}
              {s && (
                <Tooltip content="Open papers, gaps and synthesis">
                  <Button asChild size="icon-sm" variant="ghost" aria-label={`Open ${t.name}`}>
                    <Link to={`/research/${s.id}`}><FolderOpen /></Link>
                  </Button>
                </Tooltip>
              )}
            </div>
          );
        })}
        {legacy.map((s) => (
          <div key={s.id} className="flex items-center gap-2 rounded-md px-2 py-2 hover:bg-muted/50">
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{s.topic}</p>
              <p className="text-xs text-muted-foreground">Earlier one-off research · {s.paper_count} papers</p>
            </div>
            <Tooltip content="Turn it into a topic on the autopilot (keeps everything found)">
              <Button size="icon-sm" variant="ghost" aria-label={`Put ${s.topic} on the autopilot`}
                disabled={!!busy} onClick={() => start(s.id, () => api.research.toAutopilot(s.id))}><Play /></Button>
            </Tooltip>
            <Button asChild size="icon-sm" variant="ghost" aria-label={`Open ${s.topic}`}><Link to={`/research/${s.id}`}><FolderOpen /></Link></Button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default function AutopilotPage() {
  const { topics, loading: topicsLoading } = useTopic();
  const qc = useQueryClient();
  const { data: status, isLoading, error } = useAutopilot();
  const [creating, setCreating] = useState(false);

  const act = async (fn: () => Promise<AutopilotStatus>, msg: string) => {
    try { qc.setQueryData(["autopilot"], await fn()); toast.success(msg); } catch (e) { toast.error((e as Error).message); }
  };

  if (isLoading || topicsLoading) return <div className="space-y-4"><Skeleton className="h-10 w-64" /><Skeleton className="h-64" /></div>;
  if (!status) return <Alert title="Could not load the autopilot">{(error as Error)?.message ?? "No response from the server"}</Alert>;
  if (!topics.length) {
    return <>
      <EmptyState icon={FolderPlus} title="Start your first topic"
        description="Describe a research topic. The autopilot finds papers on it, analyses them one at a time and maps the research gaps."
        action={<Button onClick={() => setCreating(true)}><Plus />New topic</Button>} />
      <NewTopicResearchDialog open={creating} onOpenChange={setCreating} />
    </>;
  }

  const topicLabel = status.active_topic?.replace(/_/g, " ");
  const quotaPct = status.free_requests_remaining != null ? (status.free_requests_remaining / 1000) * 100 : null;

  return (
    <>
      <PageHeader title="Autopilot"
        description="Runs continuously: finds new papers for the active topic, processes them one at a time — newest and best-published first — and adds each to the Library and the topic's living report."
        actions={<>
          <Button variant="outline" onClick={() => setCreating(true)}><Plus />New topic</Button>
          {status.active_topic && (status.enabled
            ? <Button variant="outline" onClick={() => act(api.autopilot.pause, "Autopilot paused")}><Pause />Pause</Button>
            : <Button onClick={() => act(api.autopilot.resume, "Autopilot resumed")}><Play />Resume</Button>)}
        </>} />

      {status.requested_topic && status.requested_topic !== status.active_topic && (
        <Alert tone="info" className="mb-5" title={`Switching to “${status.requested_topic.replace(/_/g, " ")}” after the current task`}>
          The paper or search in progress on “{topicLabel}” is finished first.
        </Alert>
      )}
      {status.state === "error" && status.last_error && <Alert tone="warning" className="mb-5" title="The last step failed — retrying">{status.last_error}</Alert>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0 space-y-6">
          <Card className="p-5">
            {!status.active_topic ? (
              <div className="flex items-center gap-3 text-sm text-muted-foreground"><Bot className="size-5" />Choose a topic to start the autopilot.</div>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <AutopilotBadge status={status} />
                  <span className="text-sm font-semibold">{topicLabel}</span>
                  {status.focus && <span className="truncate text-xs text-muted-foreground">· {status.focus}</span>}
                </div>
                {!status.current_paper && <p className="mt-3 text-sm">{status.message || "—"}</p>}
                {status.current_paper && (
                  <div className="mt-3 flex items-start gap-3 rounded-lg bg-primary-soft/50 p-3">
                    <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-primary" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium">{status.current_paper.ref}: {status.current_paper.title}</p>
                      <p className="text-xs text-muted-foreground">Summary → comparison → gaps → synthesis → Library & report</p>
                    </div>
                  </div>
                )}
                {status.state === "waiting" && status.next_search_at && (
                  <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground"><Clock className="size-3.5" />Next search {formatDateTime(status.next_search_at)}</p>
                )}
                {status.state === "paused" && status.paused_until && (
                  <p className="mt-2 flex items-center gap-1.5 text-xs text-warning"><TriangleAlert className="size-3.5" />Resumes {formatDateTime(status.paused_until)}</p>
                )}
                <div className="mt-4 flex flex-wrap gap-2">
                  {status.living_report && <Button asChild size="sm"><Link to={`/reports?topic=${encodeURIComponent(status.active_topic ?? "")}`}><FileText />Living report</Link></Button>}
                  {status.session_id && <Button asChild size="sm" variant="outline"><Link to={`/research/${status.session_id}`}><FolderOpen />All papers, gaps & searches</Link></Button>}
                  <Button size="sm" variant="ghost" onClick={() => act(api.autopilot.searchNow, "Searching for new papers after the current task")}
                    disabled={!status.enabled}><Search />Search now</Button>
                </div>
              </>
            )}
          </Card>

          {status.active_topic && (
            <div className="grid grid-cols-3 gap-3">
              <Stat icon={CheckCircle2} label="Processed" value={status.counts.processed} />
              <Stat icon={ListOrdered} label="In queue" value={status.counts.queued} />
              <Stat icon={BookMarked} label="Selected" value={status.counts.selected} />
            </div>
          )}

          <ActiveTopicTabs status={status} />
        </div>

        <aside className="space-y-6">
          <TopicControl key={status.active_topic ?? "none"} status={status} />
          {quotaPct != null && (
            <Card className="p-5">
              <p className="text-sm font-semibold">Free requests today</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{status.free_requests_remaining}<span className="text-sm font-normal text-muted-foreground"> / 1000</span></p>
              <Progress value={quotaPct} className="mt-3" />
              <p className="mt-2 text-xs text-muted-foreground">The autopilot stops for the day before the last requests, keeping them for your questions.</p>
            </Card>
          )}
          <AllTopics status={status} />
        </aside>
      </div>
      <NewTopicResearchDialog open={creating} onOpenChange={setCreating} />
    </>
  );
}
