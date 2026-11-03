import { Bot, CheckCircle2, CircleDashed, Clock, Loader2, PauseCircle, Square, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { AutopilotRole, SessionStatus } from "@/lib/types";

export const ACTIVE: SessionStatus[] = ["created", "searching", "processing", "summarizing", "analyzing"];

const STATUS: Record<SessionStatus, { label: string; tone: "primary" | "success" | "danger" | "warning" | "neutral"; icon: typeof Loader2 }> = {
  created: { label: "Starting", tone: "primary", icon: Loader2 },
  searching: { label: "Searching", tone: "primary", icon: Loader2 },
  processing: { label: "Processing", tone: "primary", icon: Loader2 },
  summarizing: { label: "Summarising", tone: "primary", icon: Loader2 },
  analyzing: { label: "Analysing", tone: "primary", icon: Loader2 },
  completed: { label: "Completed", tone: "success", icon: CheckCircle2 },
  failed: { label: "Failed", tone: "danger", icon: XCircle },
  cancelled: { label: "Stopped", tone: "neutral", icon: Square },
  paused: { label: "Paused · quota", tone: "warning", icon: PauseCircle },
  autopilot: { label: "Autopilot", tone: "primary", icon: Bot },
};

export function SessionStatusBadge({ status, running }: { status: SessionStatus; running?: boolean }) {
  const s = STATUS[status] ?? { label: status, tone: "neutral" as const, icon: CircleDashed };
  const Icon = s.icon;
  const spinning = ACTIVE.includes(status) && running !== false;
  return <Badge tone={s.tone}><Icon className={spinning ? "animate-spin" : undefined} />{s.label}</Badge>;
}

/** Whether the autopilot is researching this topic now, switches to it next, or isn't on it. */
export function AutopilotRoleBadge({ role }: { role?: AutopilotRole }) {
  if (role === "active") return <Badge tone="primary"><Bot />On autopilot</Badge>;
  if (role === "next") return <Badge tone="warning"><Clock />Up next</Badge>;
  return <Badge tone="neutral"><PauseCircle />Not on autopilot</Badge>;
}

export function PaperRef({ refId, title, onClick }: { refId: string; title?: string; onClick?: () => void }) {
  if (!onClick) {
    // A plain label (it often sits inside a clickable row: a button inside a button is invalid and eats clicks).
    return <span title={title} className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 font-mono text-[11px] font-medium text-foreground/80">{refId}</span>;
  }
  return (
    <button type="button" onClick={onClick} title={title} disabled={!onClick}
      className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 font-mono text-[11px] font-medium text-foreground/80 hover:bg-primary-soft hover:text-primary disabled:hover:bg-muted disabled:hover:text-foreground/80 cursor-pointer disabled:cursor-default">
      {refId}
    </button>
  );
}
