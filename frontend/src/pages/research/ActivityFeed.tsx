import { AlertTriangle, CheckCircle2, CircleDot, FileText, Info, Lightbulb, Sparkles, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/misc";
import type { ActivityEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

const ICONS: Record<string, { icon: typeof Info; className: string }> = {
  step: { icon: CircleDot, className: "text-primary" },
  success: { icon: CheckCircle2, className: "text-success" },
  info: { icon: Info, className: "text-muted-foreground" },
  warning: { icon: AlertTriangle, className: "text-warning" },
  error: { icon: XCircle, className: "text-danger" },
  paper: { icon: FileText, className: "text-primary" },
  gap: { icon: Lightbulb, className: "text-warning" },
  done: { icon: Sparkles, className: "text-success" },
};

function time(iso: string) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ActivityFeed({ events, running }: { events: ActivityEvent[]; running: boolean }) {
  const [follow, setFollow] = useState(true);
  const [onlyImportant, setOnlyImportant] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (follow && running) endRef.current?.scrollIntoView({ block: "nearest" }); }, [events.length, follow, running]);

  const shown = onlyImportant ? events.filter((e) => !["info"].includes(e.type)) : events;
  if (!events.length) return <EmptyState icon={Sparkles} title="No activity yet" description="Steps appear here as the agent works." />;

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <Button size="sm" variant={onlyImportant ? "secondary" : "ghost"} onClick={() => setOnlyImportant(!onlyImportant)}>
          {onlyImportant ? "Showing key steps" : "Showing everything"}
        </Button>
        {running && <Button size="sm" variant="ghost" onClick={() => setFollow(!follow)}>{follow ? "Auto-scroll on" : "Auto-scroll off"}</Button>}
      </div>
      <ol className="relative space-y-0.5 border-l pl-5">
        {shown.map((e) => {
          const { icon: Icon, className } = ICONS[e.type] ?? ICONS.info;
          return (
            <li key={e.id} className={cn("relative py-1.5 animate-fade-in", e.type === "step" && "pt-3")}>
              <span className="absolute -left-[27px] top-2 flex size-4 items-center justify-center rounded-full bg-background">
                <Icon className={cn("size-4", className)} />
              </span>
              <div className="flex gap-3">
                <p className={cn("min-w-0 flex-1 text-sm", e.type === "step" ? "font-medium" : "text-foreground/85",
                  e.type === "error" && "text-danger")}>{e.message}</p>
                <time className="shrink-0 pt-0.5 text-[11px] tabular-nums text-muted-foreground">{time(e.created_at)}</time>
              </div>
              {Array.isArray(e.data?.queries) && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {(e.data!.queries as string[]).map((q) => <span key={q} className="rounded-md bg-muted px-2 py-0.5 text-xs">{q}</span>)}
                </div>
              )}
              {typeof e.data?.reason === "string" && <p className="mt-0.5 text-xs text-muted-foreground">{e.data.reason as string}</p>}
            </li>
          );
        })}
        {running && (
          <li className="relative py-2">
            <span className="absolute -left-[25px] top-3 size-3 animate-pulse rounded-full bg-primary/60" />
            <p className="text-sm text-muted-foreground">Working…</p>
          </li>
        )}
      </ol>
      <div ref={endRef} />
    </div>
  );
}
