import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownRight, ArrowRight, ArrowUpRight, CheckCircle2, ExternalLink, FlaskConical, HelpCircle, History, Loader2,
  Minus, OctagonX, Search, ShieldAlert, Telescope, Trash2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { YearBars } from "@/components/YearBars";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label, Textarea } from "@/components/ui/input";
import { Alert, EmptyState, PageHeader, Skeleton } from "@/components/ui/misc";
import { ApiError, api } from "@/lib/api";
import { storage } from "@/lib/storage";
import { useTopic } from "@/lib/topic";
import type { Momentum, Validation, ValidationResult, ValidatorNeighbour } from "@/lib/types";
import { cn, relativeTime } from "@/lib/utils";

export const VERDICT: Record<ValidationResult["verdict"], { title: string; tone: "danger" | "warning" | "success" | "info"; icon: typeof CheckCircle2; badge: "danger" | "warning" | "success" | "primary" }> = {
  duplicated: { title: "Already done — this would not pass the duplication test", tone: "danger", icon: OctagonX, badge: "danger" },
  crowded: { title: "Crowded — you need a sharper, clearly different angle", tone: "warning", icon: ShieldAlert, badge: "warning" },
  clear_lane: { title: "Clear lane so far — nothing close was found", tone: "success", icon: CheckCircle2, badge: "success" },
  too_broad: { title: "A field, not yet a topic — narrow it down first", tone: "info", icon: Telescope, badge: "primary" },
};
const VERDICT_SHORT: Record<string, string> = { duplicated: "Already done", crowded: "Crowded", clear_lane: "Clear lane", too_broad: "Too broad" };
const CLOSENESS: Record<ValidatorNeighbour["closeness"], { label: string; tone: "danger" | "warning" | "neutral" }> = {
  same: { label: "Same study", tone: "danger" },
  very_close: { label: "Very close", tone: "warning" },
  related: { label: "Related", tone: "neutral" },
};
const LEVEL: Record<string, { label: string; tone: "warning" | "primary" | "success" }> = {
  field: { label: "Field", tone: "warning" }, neighbourhood: { label: "Neighbourhood", tone: "primary" }, topic: { label: "Testable topic", tone: "success" },
};

function MomentumBadge({ m }: { m: Momentum | null }) {
  if (!m || m.trend === "quiet" || m.trend === "new") return null;
  const Icon = m.trend === "rising" ? ArrowUpRight : m.trend === "cooling" ? ArrowDownRight : Minus;
  return (
    <Badge tone={m.trend === "rising" ? "success" : "outline"} title="Citations per year over the last two full years">
      <Icon />{m.trend === "rising" ? "Rising" : m.trend === "cooling" ? "Cooling" : "Steady"} · {m.per_year}/yr
    </Badge>
  );
}

function NeighbourRow({ n }: { n: ValidatorNeighbour }) {
  const c = CLOSENESS[n.closeness] ?? CLOSENESS.related;
  return (
    <li className="px-4 py-3.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={c.tone}>{c.label}</Badge>
        {n.source === "library" && <Badge tone="primary">In your library{n.ref ? ` (${n.ref})` : ""}</Badge>}
        {n.publication_type === "review" && <Badge tone="outline">Review</Badge>}
        <MomentumBadge m={n.momentum} />
      </div>
      <p className="mt-1.5 text-sm font-medium leading-snug">
        {n.url ? <a href={n.url} target="_blank" rel="noreferrer noopener" className="hover:text-primary">{n.title}<ExternalLink className="ml-1 inline size-3 opacity-60" /></a> : n.title}
      </p>
      <p className="mt-0.5 text-xs text-muted-foreground">
        {[n.authors?.slice(0, 3).join(", ") + (n.authors?.length > 3 ? " et al." : ""), n.venue, n.year,
          n.citation_count != null ? `${n.citation_count} citations` : ""].filter(Boolean).join(" · ")}
      </p>
      {(n.overlap || n.difference) && (
        <dl className="mt-2 grid gap-1 text-sm sm:grid-cols-2">
          {n.overlap && <div><dt className="text-xs font-medium text-muted-foreground">Overlap</dt><dd>{n.overlap}</dd></div>}
          {n.difference && <div><dt className="text-xs font-medium text-muted-foreground">Difference</dt><dd>{n.difference}</dd></div>}
        </dl>
      )}
    </li>
  );
}

function ResultView({ v }: { v: Validation }) {
  const r = v.result!;
  const verdict = VERDICT[r.verdict] ?? VERDICT.crowded;
  const Icon = verdict.icon;
  const level = LEVEL[r.specificity?.level] ?? LEVEL.neighbourhood;
  const growth = r.saturation.growth_last3_vs_prev3;
  return (
    <div className="space-y-6">
      <Alert tone={verdict.tone === "success" ? "info" : verdict.tone === "info" ? "info" : verdict.tone}
        className={cn(verdict.tone === "success" && "border-success/40 bg-success-soft")}>
        <div className="flex gap-3">
          <Icon className={cn("mt-0.5 size-5 shrink-0", verdict.tone === "danger" ? "text-danger" : verdict.tone === "warning" ? "text-warning" :
            verdict.tone === "success" ? "text-success" : "text-primary")} />
          <div>
            <p className="font-semibold text-foreground">{verdict.title}</p>
            <p className="mt-1">{r.reason}</p>
          </div>
        </div>
      </Alert>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2"><CardTitle>How specific is the idea?</CardTitle><Badge tone={level.tone}>{level.label}</Badge></div>
              <CardDescription>{r.project_type_label}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {r.specificity.restated && <p><span className="text-muted-foreground">As stated: </span>{r.specificity.restated}</p>}
              {r.specificity.missing.length > 0 && (
                <div>
                  <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Still undefined</p>
                  <ul className="space-y-1.5">
                    {r.specificity.missing.map((m, i) => (
                      <li key={i} className="flex gap-2"><HelpCircle className="mt-0.5 size-4 shrink-0 text-primary" />
                        <span>{m.knob && <span className="font-medium capitalize">{m.knob}: </span>}{m.question}</span></li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>

          <section>
            <h2 className="mb-3 text-sm font-semibold">Closest rivals <span className="font-normal text-muted-foreground">· {r.neighbours.length}</span></h2>
            {r.neighbours.length ? (
              <ul className="divide-y overflow-hidden rounded-lg border bg-card">{r.neighbours.map((n, i) => <NeighbourRow key={i} n={n} />)}</ul>
            ) : <p className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">No close studies were found in this search.</p>}
          </section>

          {r.pivot_questions.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Questions to find your distinct contribution</CardTitle>
                <CardDescription>The validator asks; the answers — and the gap — are yours.</CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="list-decimal space-y-1.5 pl-5 text-sm">{r.pivot_questions.map((q, i) => <li key={i}>{q}</li>)}</ol>
              </CardContent>
            </Card>
          )}
        </div>

        <aside className="space-y-6">
          <Card className="p-5">
            <p className="text-sm font-semibold">Field density</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">{r.saturation.total_matching ? r.saturation.total_matching.toLocaleString() : "—"}</p>
            <p className="text-xs text-muted-foreground">matching works in OpenAlex · <span className="capitalize">{r.saturation.density}</span></p>
            {r.saturation.per_year.some((d) => d.count) && (
              <div className="mt-4">
                <p className="mb-2 text-xs text-muted-foreground">Publications per year</p>
                <YearBars data={r.saturation.per_year} label="Publications" height={96} />
              </div>
            )}
            {growth != null && (
              <p className="mt-3 flex items-center gap-1 text-xs text-muted-foreground">
                {growth >= 0 ? <ArrowUpRight className="size-3.5 text-success" /> : <ArrowDownRight className="size-3.5" />}
                {growth >= 5 ? "More than 5× the output of the previous 3 years — a new, fast-growing area"
                  : `${growth >= 0 ? "+" : ""}${Math.round(growth * 100)}% in the last 3 years vs the 3 before`}
              </p>
            )}
          </Card>
          <Card className="p-5">
            <p className="text-sm font-semibold">Rival momentum</p>
            <dl className="mt-2 space-y-1.5 text-sm">
              <div className="flex justify-between"><dt className="text-muted-foreground">Close rivals</dt><dd className="tabular-nums">{r.rival_momentum.close_rivals}</dd></div>
              <div className="flex justify-between"><dt className="text-muted-foreground">Gaining citations fast</dt><dd className="tabular-nums">{r.rival_momentum.rising}</dd></div>
              <div className="flex justify-between"><dt className="text-muted-foreground">Newest close rival</dt><dd className="tabular-nums">{r.rival_momentum.newest_close_rival ?? "—"}</dd></div>
            </dl>
          </Card>
          <p className="text-xs leading-relaxed text-muted-foreground">
            <Search className="mr-1 inline size-3" />Searched {r.search.queries.length} queries across SearXNG and OpenAlex
            ({r.search.records_found} records, {r.search.candidates_considered} compared closely). A search this size can miss studies —
            confirm big decisions with Scopus / Web of Science and your supervisor.
            {r.search.searxng_failed && " SearXNG was unavailable for part of the search."}
            {r.search.openalex_failed && " OpenAlex was unavailable (or its daily credits were used up), so citation data and some rivals may be missing."}
          </p>
        </aside>
      </div>
    </div>
  );
}

export default function ValidatorPage() {
  const qc = useQueryClient();
  const { topics } = useTopic();
  const [idea, setIdea] = useState("");
  const [projectType, setProjectType] = useState("empirical_quantitative");
  const [topic, setTopic] = useState("");
  const [selected, setSelected] = useState<string | null>(() => storage.get("rm-validation"));
  const [busy, setBusy] = useState(false);
  const { data: types = [] } = useQuery({ queryKey: ["validator-types"], queryFn: api.validator.projectTypes, staleTime: Infinity });
  const { data: history = [] } = useQuery({ queryKey: ["validations"], queryFn: api.validator.list });
  const { data: current, isLoading, error: currentError } = useQuery({
    queryKey: ["validation", selected], queryFn: () => api.validator.get(selected!), enabled: !!selected,
    refetchInterval: (q) => (q.state.data?.running ? 2000 : false), retry: false,
  });
  useEffect(() => { storage.set("rm-validation", selected); }, [selected]);
  // The saved test no longer exists (deleted elsewhere, database reset): forget it.
  useEffect(() => { if (currentError instanceof ApiError && currentError.status === 404) setSelected(null); }, [currentError]);
  // Open the most recent test when nothing (or a deleted test) is selected.
  useEffect(() => {
    if (history.length && (!selected || !history.some((h) => h.id === selected))) setSelected(history[0].id);
  }, [history, selected]);
  useEffect(() => { if (current && !current.running) qc.invalidateQueries({ queryKey: ["validations"] }); }, [current?.running, current, qc]);

  const run = async () => {
    setBusy(true);
    try {
      const v = await api.validator.create(idea.trim(), projectType, topic);
      setSelected(v.id);
      qc.invalidateQueries({ queryKey: ["validations"] });
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <>
      <PageHeader title="Topic validator"
        description="Stress-test a research idea before you invest in it: is it specific enough, has it already been done, and how crowded is the field? The AI is the critic — the gap stays yours." />

      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="space-y-6">
          <Card>
            <CardHeader><CardTitle>Your idea</CardTitle></CardHeader>
            <CardContent>
              <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); if (idea.trim().length >= 10) run(); }}>
                <div className="space-y-1.5">
                  <Label htmlFor="v-idea">Research idea</Label>
                  <Textarea id="v-idea" rows={5} value={idea} maxLength={2000} onChange={(e) => setIdea(e.target.value)}
                    placeholder="e.g. The long-term effect of youth unemployment on mental health, using UK cohort data" />
                  <p className="text-xs text-muted-foreground">Be as concrete as you can: who, what, which outcome, which data.</p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="v-type">Project type</Label>
                  <select id="v-type" value={projectType} onChange={(e) => setProjectType(e.target.value)}
                    className="h-9 w-full rounded-md border border-input bg-card px-3 text-sm shadow-xs">
                    {types.map((t) => <option key={t.key} value={t.key}>{t.label}</option>)}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="v-topic">Also compare with my papers in</Label>
                  <select id="v-topic" value={topic} onChange={(e) => setTopic(e.target.value)}
                    className="h-9 w-full rounded-md border border-input bg-card px-3 text-sm shadow-xs">
                    <option value="">— none —</option>
                    {topics.map((t) => <option key={t.name} value={t.name}>{t.name.replace(/_/g, " ")}</option>)}
                  </select>
                </div>
                <Button type="submit" className="w-full" loading={busy} disabled={idea.trim().length < 10}>
                  {!busy && <FlaskConical />}Run the tests
                </Button>
                <p className="text-xs text-muted-foreground">Uses about 2 free LLM requests and a few free OpenAlex lookups.</p>
              </form>
            </CardContent>
          </Card>

          {history.length > 0 && (
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground"><History className="size-3.5" />Earlier tests</p>
              <ul className="space-y-1">
                {history.map((h) => (
                  <li key={h.id} className="group flex items-start gap-1">
                    <button onClick={() => setSelected(h.id)}
                      className={cn("min-w-0 flex-1 rounded-md px-3 py-2 text-left hover:bg-muted cursor-pointer", selected === h.id && "bg-primary-soft hover:bg-primary-soft")}>
                      <p className="line-clamp-2 text-sm">{h.idea}</p>
                      <p className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                        {h.verdict ? <Badge tone={VERDICT[h.verdict as ValidationResult["verdict"]]?.badge ?? "neutral"}>{VERDICT_SHORT[h.verdict] ?? h.verdict}</Badge>
                          : h.status === "running" ? <Badge tone="primary"><Loader2 className="animate-spin" />Running</Badge> : <Badge tone="danger">Failed</Badge>}
                        {relativeTime(h.created_at)}
                      </p>
                    </button>
                    <ConfirmDialog title="Delete this test?" description="The idea and its results are removed."
                      confirmLabel="Delete test"
                      onConfirm={async () => {
                        try {
                          await api.validator.remove(h.id);
                          if (selected === h.id) setSelected(null);
                          qc.invalidateQueries({ queryKey: ["validations"] });
                        } catch (e) { toast.error((e as Error).message); }
                      }}
                      trigger={(open) => (
                        <Button variant="ghost" size="icon-sm" aria-label="Delete test" onClick={open}
                          className="mt-1 md:opacity-0 md:group-hover:opacity-100 md:focus:opacity-100"><Trash2 /></Button>
                      )} />
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>

        <section className="min-w-0">
          {!selected ? (
            <EmptyState icon={FlaskConical} title="Test an idea before you commit to it"
              description={<>The validator checks whether your idea is a testable topic, finds its closest rivals in the
                literature, runs a duplication test and shows how crowded and how active the field is.</>} />
          ) : currentError && !current ? (
            <Alert title="Could not load this test">{(currentError as Error).message}</Alert>
          ) : isLoading || !current ? <Skeleton className="h-96" /> : current.running ? (
            <Card className="p-8">
              <div className="flex items-center gap-3"><Loader2 className="size-5 animate-spin text-primary" /><p className="font-medium">{current.stage || "Working…"}</p></div>
              <p className="mt-3 text-sm text-muted-foreground">“{current.idea}”</p>
              <ol className="mt-5 space-y-1.5 text-sm text-muted-foreground">
                {["Checking how specific the idea is", "Searching the literature for the closest studies", "Running the duplication test on the closest studies"].map((s) => (
                  <li key={s} className={cn("flex items-center gap-2", current.stage === s && "font-medium text-foreground")}>
                    <ArrowRight className="size-3.5" />{s}
                  </li>
                ))}
              </ol>
            </Card>
          ) : current.status === "failed" ? (
            <Alert title="The test could not finish">{current.error}</Alert>
          ) : current.result ? (
            <>
              <p className="mb-4 text-sm"><span className="text-muted-foreground">Idea: </span>“{current.idea}”</p>
              <ResultView v={current} />
            </>
          ) : null}
        </section>
      </div>
    </>
  );
}
