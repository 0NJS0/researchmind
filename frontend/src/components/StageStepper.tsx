import { Check, Circle, Loader2, TriangleAlert } from "lucide-react";
import type { StageName } from "@/lib/types";
import { cn } from "@/lib/utils";

export const STAGE_LABELS: Record<StageName, string> = {
  retriever: "Finding passages",
  classifier: "Identifying field",
  analysis: "Reading papers",
  comparison: "Comparing",
  gap: "Spotting gaps",
  report: "Writing answer",
};
export const STAGES = Object.keys(STAGE_LABELS) as StageName[];

export interface StageState { status: "pending" | "running" | "done" | "error"; detail?: string }

export function StageStepper({ stages }: { stages: Record<StageName, StageState> }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-1 gap-y-2" aria-label="Progress">
      {STAGES.map((name, i) => {
        const s = stages[name];
        return (
          <li key={name} className="flex items-center gap-1">
            <span className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
              s.status === "done" && "bg-success-soft text-success",
              s.status === "running" && "bg-primary-soft text-primary",
              s.status === "error" && "bg-warning-soft text-warning",
              s.status === "pending" && "text-muted-foreground")}
              title={s.detail || undefined}>
              {s.status === "done" ? <Check className="size-3.5" /> : s.status === "running" ? <Loader2 className="size-3.5 animate-spin" /> :
                s.status === "error" ? <TriangleAlert className="size-3.5" /> : <Circle className="size-3" />}
              {STAGE_LABELS[name]}
            </span>
            {i < STAGES.length - 1 && <span className="h-px w-3 bg-border" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}
