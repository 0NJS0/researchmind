import { useQuery } from "@tanstack/react-query";
import { ChevronDown, History, ShieldAlert, ShieldCheck, ShieldX } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import type { Gap } from "@/lib/types";
import { cn, humanize, relativeTime } from "@/lib/utils";
import { PaperRef } from "./common";

const STATUS_TONE: Record<Gap["status"], "primary" | "danger" | "warning" | "success" | "neutral"> = {
  open: "primary", contested: "danger", partially_addressed: "warning", resolved: "success", merged: "neutral",
};
const LEVELS = ["low", "medium", "high"] as const;

export function ConfidenceMeter({ level }: { level: Gap["confidence"] }) {
  const n = LEVELS.indexOf(level) + 1;
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground" aria-label={`Confidence ${level}`}>
      <span className="flex gap-0.5">
        {LEVELS.map((l, i) => <span key={l} className={cn("h-3 w-1.5 rounded-sm", i < n ? "bg-primary" : "bg-muted")} />)}
      </span>
      {humanize(level)} confidence
    </span>
  );
}

const CHECK: Record<string, { label: string; tone: "success" | "warning" | "danger" | "outline"; icon: typeof ShieldCheck }> = {
  verified_open: { label: "Checked: still open", tone: "success", icon: ShieldCheck },
  partially_addressed: { label: "Checked: partly addressed elsewhere", tone: "warning", icon: ShieldAlert },
  addressed: { label: "Checked: addressed elsewhere", tone: "danger", icon: ShieldX },
};

export function CheckBadge({ gap }: { gap: Gap }) {
  const c = CHECK[gap.verification_status ?? ""];
  if (!c) return <Badge tone="outline" title="Not yet tested against the wider literature">Not yet checked</Badge>;
  const Icon = c.icon;
  return <Badge tone={c.tone} title="Duplication test against papers outside this review"><Icon />{c.label}</Badge>;
}

export function GapCard({ gap, onOpenPaper }: { gap: Gap; onOpenPaper?: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const { data, isLoading } = useQuery({ queryKey: ["gap", gap.id, gap.updated_at], queryFn: () => api.research.gap(gap.id), enabled: open });

  return (
    <article className={cn("rounded-lg border bg-card shadow-xs transition-shadow", open && "shadow-md", gap.status === "merged" && "opacity-70")}>
      <button className="flex w-full items-start gap-3 p-4 text-left cursor-pointer" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="mt-0.5 font-mono text-xs font-semibold text-muted-foreground">{gap.ref}</span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold leading-snug">{gap.title}</h3>
          {gap.description && <p className={cn("mt-1 text-sm text-muted-foreground", !open && "line-clamp-2")}>{gap.description}</p>}
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <Badge tone={STATUS_TONE[gap.status]}>{humanize(gap.status)}</Badge>
            <Badge tone="outline">{humanize(gap.gap_type)}</Badge>
            <ConfidenceMeter level={gap.confidence} />
            <CheckBadge gap={gap} />
            <span className="text-xs text-muted-foreground">· {gap.paper_count} {gap.paper_count === 1 ? "paper" : "papers"}</span>
          </div>
        </div>
        <ChevronDown className={cn("mt-1 size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="border-t px-4 pb-4 pt-3">
          {isLoading || !data ? <Skeleton className="h-24" /> : (
            <div className="grid gap-5 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Evidence</h4>
                <ul className="space-y-2">
                  {data.evidence.map((e) => (
                    <li key={e.id} className="rounded-md bg-muted/50 p-2.5 text-sm">
                      <div className="mb-1 flex flex-wrap items-center gap-1.5">
                        <PaperRef refId={e.paper_ref} title={e.paper_title} onClick={onOpenPaper ? () => onOpenPaper(e.paper_id) : undefined} />
                        <Badge tone={e.relationship === "contradicts" ? "danger" : e.relationship === "supports" ? "success" : "neutral"}>
                          {humanize(e.relationship)}</Badge>
                        <span className="text-xs text-muted-foreground">{humanize(e.basis)}</span>
                      </div>
                      <p className="text-foreground/85">{e.evidence}</p>
                    </li>
                  ))}
                  {!data.evidence.length && <p className="text-sm text-muted-foreground">No evidence recorded.</p>}
                </ul>
              </div>
              <div>
                {gap.verification_status && gap.verification_status !== "unverified" && (
                  <div className="mb-4 rounded-md border p-2.5 text-sm">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Duplication test</p>
                    {gap.verification_note && <p className="text-foreground/85">{gap.verification_note}</p>}
                    {(gap.verification_refs ?? []).map((r, i) => (
                      <p key={i} className="mt-1.5 text-xs">
                        {r.url ? <a href={r.url} target="_blank" rel="noreferrer noopener" className="font-medium text-primary hover:underline">{r.title}</a>
                          : <span className="font-medium">{r.title}</span>} ({r.year ?? "n.d."}) — {r.how}
                      </p>
                    ))}
                    {gap.verified_at && <p className="mt-1.5 text-xs text-muted-foreground">Checked {relativeTime(gap.verified_at)}</p>}
                  </div>
                )}
                <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground"><History className="size-3.5" />How it evolved</h4>
                <ol className="space-y-2 border-l pl-3">
                  {data.history.map((h) => (
                    <li key={h.id} className="relative text-xs">
                      <span className="absolute -left-[17px] top-1 size-2 rounded-full bg-border" />
                      <p className="text-foreground/85">{h.detail || humanize(h.action)}</p>
                      <p className="text-muted-foreground">{relativeTime(h.created_at)}</p>
                    </li>
                  ))}
                </ol>
                {data.merged_into && <p className="mt-3 text-xs text-muted-foreground">Merged into {data.merged_into.ref}: {data.merged_into.title}</p>}
              </div>
            </div>
          )}
        </div>
      )}
    </article>
  );
}
