import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, ArrowLeft, BookMarked, Bot, Clock, FilePlus2, FileStack, Lightbulb, ListTree, MoreHorizontal, RefreshCw, Search, Square, Trash2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown";
import { Input, Label } from "@/components/ui/input";
import { Alert, Skeleton } from "@/components/ui/misc";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, ApiError } from "@/lib/api";
import { relativeTime } from "@/lib/utils";
import { ActivityFeed } from "./ActivityFeed";
import { ACTIVE, AutopilotRoleBadge, SessionStatusBadge } from "./common";
import { GapsTab } from "./GapsTab";
import { PapersTab } from "./PapersTab";
import { PaperSheet } from "./PaperSheet";
import { QueriesTab } from "./QueriesTab";
import { SynthesisTab } from "./SynthesisTab";
import { useSessionEvents } from "./useSessionEvents";
import { usePageVisible } from "@/hooks/usePageVisible";

function AddPaperDialog({ sessionId, open, onOpenChange }: { sessionId: string; open: boolean; onOpenChange: (o: boolean) => void }) {
  const [value, setValue] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const qc = useQueryClient();
  const submit = async () => {
    setBusy(true);
    const v = value.trim();
    const isDoi = /^10\.\d{4,9}\//.test(v) || v.toLowerCase().startsWith("doi:");
    try {
      const res = await api.research.addPaper(sessionId, isDoi ? { doi: v, title } : { url: v, title });
      toast[res.status === "duplicate" ? "info" : "success"](res.status === "duplicate" ? `Already in this review as ${res.paper.ref}`
        : res.queued ? `Added as ${res.paper.ref} — the autopilot analyses it with priority` : `Added as ${res.paper.ref}`);
      qc.invalidateQueries({ queryKey: ["session", sessionId] });
      qc.invalidateQueries({ queryKey: ["papers", sessionId] });
      onOpenChange(false);
      setValue(""); setTitle("");
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add a paper</DialogTitle>
          <DialogDescription>Paste a paper URL (arXiv, publisher page or direct PDF) or a DOI. It is fetched, summarised and added to the gap analysis.</DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); if (value.trim()) submit(); }}>
          <div className="space-y-1.5">
            <Label htmlFor="pu">URL or DOI</Label>
            <Input id="pu" autoFocus value={value} onChange={(e) => setValue(e.target.value)} placeholder="https://arxiv.org/abs/2401.00001 or 10.1145/…" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="pt">Title <span className="font-normal text-muted-foreground">(optional)</span></Label>
            <Input id="pt" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={!value.trim()}><FilePlus2 />Add paper</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function Stat({ icon: Icon, label, value, sub }: { icon: typeof FileStack; label: string; value: number; sub?: string }) {
  return (
    <Card className="flex items-center gap-3 p-4">
      <div className="rounded-lg bg-primary-soft p-2 text-primary"><Icon className="size-4" /></div>
      <div>
        <p className="text-xl font-semibold tabular-nums leading-none">{value}</p>
        <p className="mt-1 text-xs text-muted-foreground">{label}{sub ? ` · ${sub}` : ""}</p>
      </div>
    </Card>
  );
}

export default function ResearchSessionPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [adding, setAdding] = useState(false);
  const [paperId, setPaperId] = useState<string | null>(null);
  const [tab, setTab] = useState("activity");
  // The same page instance is reused when moving to another topic's research: start it fresh.
  useEffect(() => { setTab("activity"); setPaperId(null); setAdding(false); }, [id]);

  const { data: session, isLoading, error } = useQuery({
    queryKey: ["session", id], queryFn: () => api.research.get(id),
    // Poll while a job runs or the topic is (or is about to be) on the autopilot.
    refetchInterval: (q) => (q.state.data?.running || (q.state.data?.autopilot_role ?? "idle") !== "idle" ? 5000 : false),
  });
  const { data: ap } = useQuery({ queryKey: ["autopilot"], queryFn: api.autopilot.status, refetchInterval: 4000 });
  // An autopilot topic is "working" only while the autopilot is busy on it, not while it waits or is paused.
  const autopilotBusy = !!ap && ap.session_id === id && ["processing", "searching", "screening", "verifying"].includes(ap.state);
  const running = session?.autopilot ? autopilotBusy : !!session?.running;
  const visible = usePageVisible();
  const events = useSessionEvents(id, running && visible && tab === "activity");  // stream only while watched
  useEffect(() => {  // other tabs refresh by polling while work runs
    if (!running || !visible) return;
    const t = setInterval(() => {
      for (const key of ["gaps", "synthesis", "papers"]) qc.invalidateQueries({ queryKey: [key, id] });
      qc.invalidateQueries({ queryKey: ["paper"] });   // an open paper panel
    }, 20_000);
    return () => clearInterval(t);
  }, [id, running, visible, qc]);

  if (isLoading) return <div className="space-y-4"><Skeleton className="h-10 w-2/3" /><Skeleton className="h-24" /><Skeleton className="h-64" /></div>;
  if (!session) {
    return <Alert title={error instanceof ApiError && error.status === 404 ? "Research session not found" : "Could not load this session"}>
      <Link to="/autopilot" className="text-primary hover:underline">Back to the autopilot</Link></Alert>;
  }

  const refreshSession = () => {
    for (const key of [["session", id], ["autopilot"], ["research"], ["topics"], ["topic-papers"]]) qc.invalidateQueries({ queryKey: key });
  };
  const act = async (fn: () => Promise<unknown>, success: string) => {
    try { await fn(); toast.success(success); refreshSession(); } catch (e) { toast.error((e as Error).message); }
  };
  const stale = !running && ACTIVE.includes(session.status);
  const role = session.autopilot_role ?? (session.autopilot ? "active" : "idle");

  return (
    <>
      <Link to="/autopilot" className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" />Autopilot
      </Link>

      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {session.topic_name ? <AutopilotRoleBadge role={role} /> : <SessionStatusBadge status={session.status} running={running} />}
            {session.topic_analysis?.field && <Badge tone="outline">{session.topic_analysis.field}</Badge>}
            <span className="text-xs text-muted-foreground">Started {relativeTime(session.created_at)}</span>
          </div>
          <h1 className="text-xl font-semibold tracking-tight">{session.topic}</h1>
          {session.description && <p className="mt-1 text-sm text-muted-foreground">{session.description}</p>}
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {running && !session.autopilot ? (
            <Button variant="outline" onClick={() => act(() => api.research.cancel(id), "Stopping after the current step")}><Square />Stop</Button>
          ) : role === "active" ? (
            <>
              <Button variant="outline" onClick={() => act(() => api.autopilot.searchNow(), "Searching for new papers now")}><Search />Search now</Button>
              <Button asChild variant="outline"><Link to="/autopilot"><Bot />Autopilot</Link></Button>
            </>
          ) : role === "next" ? (
            <Button asChild variant="outline"><Link to="/autopilot"><Clock />Starts after the current paper</Link></Button>
          ) : (
            <Button onClick={async () => {
              try {
                const r = await api.research.toAutopilot(id);
                toast.success(r.autopilot_status === "pending"
                  ? "The autopilot switches to this topic after its current paper" : "Put on the autopilot");
                refreshSession();
              } catch (e) { toast.error((e as Error).message); }
            }}><Bot />Put on autopilot</Button>
          )}
          <Button variant="outline" onClick={() => setAdding(true)}><FilePlus2 />Add paper</Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild><Button variant="outline" size="icon" aria-label="More actions"><MoreHorizontal /></Button></DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem disabled={running || !!session.autopilot} onSelect={() => act(() => api.research.rebuildGaps(id), "Rebuilding gaps from existing summaries")}>
                <RefreshCw />Rebuild gaps & synthesis
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <ConfirmDialog title={session.topic_name ? "Delete this topic?" : "Delete this research?"}
                description={session.topic_name
                  ? "The topic is deleted everywhere: its Library papers and PDFs, reports, research gaps, synthesis and activity."
                  : "All papers, summaries, gaps and activity for this session will be deleted."}
                onConfirm={async () => {
                  try {
                    if (session.topic_name) await api.library.deleteTopic(session.topic_name);
                    else await api.research.remove(id);
                    qc.invalidateQueries({ queryKey: ["research"] });
                    qc.invalidateQueries({ queryKey: ["topics"] });
                    toast.success(session.topic_name ? "Topic deleted" : "Research deleted");
                    navigate("/autopilot");
                  } catch (e) { toast.error((e as Error).message); }
                }}
                trigger={(open) => <DropdownMenuItem danger onSelect={(e) => { e.preventDefault(); open(); }}><Trash2 />{session.topic_name ? "Delete topic" : "Delete research"}</DropdownMenuItem>} />
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {role === "active" && (
        <Alert tone="info" className="mb-5" title="The autopilot is researching this topic continuously">
          New papers are found, processed one at a time and added here, to the Library and to the living report.
        </Alert>
      )}
      {role === "idle" && session.topic_name && (
        <Alert tone="info" className="mb-5" title="Not on the autopilot right now">
          Everything found so far is kept. Use “Put on autopilot” to continue researching this topic.
        </Alert>
      )}
      {running && !session.autopilot && session.status_message && (
        <Alert tone="info" className="mb-5"><span className="font-medium">Now: </span>{session.status_message}</Alert>
      )}
      {session.status === "paused" && (
        <Alert tone="warning" className="mb-5" title="Paused — the LLM's daily free quota is used up">
          {session.status_message} Continue once the quota resets; finished work is kept and unfinished papers resume.
        </Alert>
      )}
      {session.status === "failed" && <Alert className="mb-5" title="The last run failed">{session.status_message}</Alert>}
      {stale && <Alert tone="warning" className="mb-5">This session shows as active but no job is running (the server may have restarted). Use “Put on autopilot” to resume.</Alert>}

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat icon={FileStack} label="Papers selected" value={session.counts.papers} sub={`${session.counts.candidates} screened`} />
        <Stat icon={BookMarked} label="Summarised" value={session.counts.summarized} />
        <Stat icon={Lightbulb} label="Research gaps" value={session.counts.gaps} />
        <Stat icon={Search} label="Searches" value={session.counts.queries} sub={`${session.iteration} rounds`} />
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="activity"><Activity />Activity</TabsTrigger>
          <TabsTrigger value="papers"><FileStack />Papers<span className="text-xs text-muted-foreground">{session.counts.papers}</span></TabsTrigger>
          <TabsTrigger value="gaps"><Lightbulb />Gaps<span className="text-xs text-muted-foreground">{session.counts.gaps}</span></TabsTrigger>
          <TabsTrigger value="synthesis"><BookMarked />Synthesis</TabsTrigger>
          <TabsTrigger value="queries"><ListTree />Searches</TabsTrigger>
        </TabsList>
        <TabsContent value="activity"><ActivityFeed events={events} running={running} /></TabsContent>
        <TabsContent value="papers"><PapersTab sessionId={id} onOpen={setPaperId} /></TabsContent>
        <TabsContent value="gaps"><GapsTab sessionId={id} onOpenPaper={setPaperId} /></TabsContent>
        <TabsContent value="synthesis"><SynthesisTab sessionId={id} onOpenPaper={setPaperId} /></TabsContent>
        <TabsContent value="queries"><QueriesTab sessionId={id} /></TabsContent>
      </Tabs>

      <PaperSheet paperId={paperId} onClose={() => setPaperId(null)} onOpenPaper={setPaperId} running={running} />
      <AddPaperDialog sessionId={id} open={adding} onOpenChange={setAdding} />
    </>
  );
}
