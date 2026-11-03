import { ArrowUp, BookOpen, ChevronDown, Copy, MessageSquareText, RotateCcw, ShieldCheck, Square, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Markdown } from "@/components/Markdown";
import { STAGES, StageStepper, type StageState } from "@/components/StageStepper";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Alert, EmptyState, PageHeader } from "@/components/ui/misc";
import { postSSE } from "@/lib/sse";
import { storage } from "@/lib/storage";
import { useTopic } from "@/lib/topic";
import type { CitationCheck, Source, StageName, StreamEvent } from "@/lib/types";
import { cn, uid, copyText } from "@/lib/utils";

interface Turn {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  errors: string[];
  citationCheck?: CitationCheck;
  stages: Record<StageName, StageState>;
  status: "streaming" | "done" | "error" | "stopped";
}

const SUGGESTIONS = [
  "What are the main methods used across these papers?",
  "Which datasets are used, and how large are they?",
  "Where do the papers disagree, and why?",
  "What limitations do the authors report?",
];

const freshStages = (): Record<StageName, StageState> =>
  Object.fromEntries(STAGES.map((s) => [s, { status: "pending" }])) as Record<StageName, StageState>;

function sessionId(topic: string) {
  const key = `rm-chat-${topic}`;
  let id = storage.get(key);
  if (!id) { id = uid(); storage.set(key, id); }
  return id;
}

function SourcesList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  const papers = new Set(sources.map((s) => s.paper_id)).size;
  return (
    <div className="mt-4 rounded-lg border bg-muted/30">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-xs font-medium text-muted-foreground cursor-pointer">
        <BookOpen className="size-3.5" />{sources.length} passages from {papers} {papers === 1 ? "paper" : "papers"}
        <ChevronDown className={cn("ml-auto size-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ul className="divide-y border-t">
          {sources.map((s) => (
            <li key={s.chunk_id} className="px-4 py-3">
              <p className="text-xs font-medium">{s.paper_id} · p.{s.page}
                {s.score != null && <span className="ml-2 font-normal text-muted-foreground">relevance {(s.score * 100).toFixed(0)}%</span>}</p>
              <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-muted-foreground">{s.snippet}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TurnView({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  return (
    <div className="animate-fade-in space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground shadow-sm">{turn.question}</div>
      </div>
      <div className="rounded-xl border bg-card p-5 shadow-xs">
        {turn.status === "streaming" && <StageStepper stages={turn.stages} />}
        {turn.answer ? <Markdown sources={turn.sources} className={cn(turn.status === "streaming" && "mt-4")}>{turn.answer}</Markdown>
          : turn.status === "streaming" ? <p className="mt-3 text-sm text-muted-foreground">Working through your papers — free models can take a minute.</p>
          : turn.status === "stopped" ? <p className="text-sm text-muted-foreground">Stopped.</p> : null}
        {turn.errors.length > 0 && (
          <Alert tone="warning" className="mt-4" title="Some steps had problems">
            <ul className="list-disc pl-4">{turn.errors.map((e) => <li key={e}>{e}</li>)}</ul>
          </Alert>
        )}
        {turn.citationCheck && turn.citationCheck.total > 0 && (
          <p className={cn("mt-4 flex items-center gap-1.5 text-xs",
            turn.citationCheck.unverified.length ? "text-warning" : "text-muted-foreground")}>
            {turn.citationCheck.unverified.length ? <TriangleAlert className="size-3.5" /> : <ShieldCheck className="size-3.5" />}
            {turn.citationCheck.verified} of {turn.citationCheck.total} citations match a retrieved passage
            {turn.citationCheck.unverified.length > 0 && " — amber ones could not be verified"}
          </p>
        )}
        <SourcesList sources={turn.sources} />
        {turn.status !== "streaming" && (
          <div className="mt-3 flex gap-1">
            {turn.answer && <Button variant="ghost" size="sm" onClick={async () => { if (await copyText(turn.answer)) toast.success("Copied"); else toast.error("Could not copy"); }}><Copy />Copy</Button>}
            <Button variant="ghost" size="sm" onClick={onRetry}><RotateCcw />Ask again</Button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function AskPage() {
  const { topic, topics } = useTopic();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const streaming = turns.some((t) => t.status === "streaming");
  const current = topics.find((t) => t.name === topic);

  useEffect(() => { setTurns([]); abortRef.current?.abort(); }, [topic]);
  // Leaving the page stops the answer being generated (it would spend free requests for nothing).
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns.length]);

  const patch = (id: string, fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? fn(t) : t)));

  const ask = async (question: string) => {
    question = question.trim();
    if (question.length < 3 || streaming) return;
    const id = uid();
    setTurns((ts) => [...ts, { id, question, answer: "", sources: [], errors: [], stages: freshStages(), status: "streaming" }]);
    setInput("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      for await (const msg of postSSE("/api/library/query/stream", { question, topic, session_id: sessionId(topic) }, controller.signal)) {
        if (msg.data === "[DONE]") break;
        const ev = JSON.parse(msg.data) as StreamEvent;
        if (ev.type === "stage") {
          patch(id, (t) => ({ ...t, stages: { ...t.stages, [ev.stage]: {
            status: ev.status === "done" && ev.errors?.length ? "error" : ev.status, detail: ev.detail } } }));
        } else if (ev.type === "answer") {
          patch(id, (t) => ({ ...t, answer: ev.answer, sources: ev.sources, errors: ev.errors,
            citationCheck: ev.citation_check, status: "done" }));
        }
      }
      patch(id, (t) => (t.status === "streaming" ? { ...t, status: "done" } : t));
    } catch (e) {
      if (controller.signal.aborted) patch(id, (t) => ({ ...t, status: "stopped" }));
      else patch(id, (t) => ({ ...t, status: "error", errors: [(e as Error).message] }));
    }
  };

  if (!topics.length) {
    return <EmptyState icon={MessageSquareText} title="No papers to ask about yet"
      description="Create a topic and upload papers in the Library first." action={<Button asChild><Link to="/library">Go to Library</Link></Button>} />;
  }

  return (
    <div className="flex min-h-[calc(100dvh-8rem)] flex-col">
      <PageHeader title="Ask your papers"
        description={<>Answers come only from papers in <span className="font-medium text-foreground">{topic.replace(/_/g, " ")}</span>
          {current ? ` (${current.papers} papers)` : ""}, with page-level citations.</>}
        actions={turns.length > 0 && <Button variant="outline" size="sm" onClick={() => { abortRef.current?.abort(); setTurns([]); storage.set(`rm-chat-${topic}`, null); }}>New conversation</Button>} />

      <div className="flex-1 space-y-8 pb-6">
        {turns.length === 0 && (
          <div className="mx-auto max-w-2xl pt-6 text-center">
            <div className="mx-auto mb-4 w-fit rounded-2xl bg-primary-soft p-3 text-primary"><MessageSquareText className="size-6" /></div>
            <h2 className="text-lg font-semibold">What would you like to know?</h2>
            <p className="mt-1 text-sm text-muted-foreground">Try one of these, or ask your own question.</p>
            <div className="mt-6 grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => ask(s)}
                  className="rounded-lg border bg-card px-4 py-3 text-left text-sm shadow-xs transition-colors hover:border-primary/40 hover:bg-primary-soft/40 cursor-pointer">{s}</button>
              ))}
            </div>
            {current?.papers === 0 && (
              <p className="mt-6 flex items-center justify-center gap-2 text-sm text-warning"><TriangleAlert className="size-4" />This topic has no papers yet.</p>
            )}
          </div>
        )}
        {turns.map((t) => <TurnView key={t.id} turn={t} onRetry={() => ask(t.question)} />)}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={(e) => { e.preventDefault(); ask(input); }}
        className="sticky bottom-4 rounded-2xl border bg-card p-2 shadow-lg">
        <div className="flex items-end gap-2">
          <Textarea value={input} onChange={(e) => setInput(e.target.value)} rows={1} maxLength={2000}
            placeholder="Ask about methods, results, disagreements, gaps…" aria-label="Your question"
            className="max-h-40 min-h-10 border-0 shadow-none focus-visible:outline-none"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); ask(input); } }} />
          {streaming
            ? <Button type="button" size="icon" variant="secondary" onClick={() => abortRef.current?.abort()} aria-label="Stop"><Square /></Button>
            : <Button type="submit" size="icon" disabled={input.trim().length < 3} aria-label="Send"><ArrowUp /></Button>}
        </div>
      </form>
    </div>
  );
}
