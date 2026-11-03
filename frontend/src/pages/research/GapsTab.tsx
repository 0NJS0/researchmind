import { useQuery } from "@tanstack/react-query";
import { Lightbulb } from "lucide-react";
import { useMemo, useState } from "react";
import { EmptyState, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import type { Gap } from "@/lib/types";
import { cn } from "@/lib/utils";
import { GapCard } from "./GapCard";

const GROUPS: { key: string; label: string; statuses: Gap["status"][] }[] = [
  { key: "open", label: "Open", statuses: ["open"] },
  { key: "contested", label: "Contested", statuses: ["contested"] },
  { key: "addressed", label: "Addressed", statuses: ["partially_addressed", "resolved"] },
  { key: "merged", label: "Merged", statuses: ["merged"] },
];
const CONF_ORDER = { high: 0, medium: 1, low: 2 };

export function GapsTab({ sessionId, onOpenPaper }: { sessionId: string; onOpenPaper: (id: string) => void }) {
  const [group, setGroup] = useState("all");
  const { data: gaps, isLoading } = useQuery({
    queryKey: ["gaps", sessionId], queryFn: () => api.research.gaps(sessionId, true),
  });
  const counts = useMemo(() => Object.fromEntries(GROUPS.map((g) => [g.key, (gaps ?? []).filter((x) => g.statuses.includes(x.status)).length])), [gaps]);
  const shown = useMemo(() => {
    const list = (gaps ?? []).filter((g) => group === "all" ? g.status !== "merged" : GROUPS.find((x) => x.key === group)!.statuses.includes(g.status));
    return [...list].sort((a, b) => CONF_ORDER[a.confidence] - CONF_ORDER[b.confidence] || b.paper_count - a.paper_count);
  }, [gaps, group]);

  if (isLoading) return <div className="space-y-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-24" />)}</div>;
  if (!gaps?.length) return <EmptyState icon={Lightbulb} title="No research gaps yet"
    description="Gaps are proposed as papers are analysed, and only kept when a paper provides evidence for them." />;

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {[{ key: "all", label: "All active", n: gaps.filter((g) => g.status !== "merged").length }, ...GROUPS.map((g) => ({ key: g.key, label: g.label, n: counts[g.key] }))]
          .filter((g) => g.key === "all" || g.n > 0)
          .map((g) => (
            <button key={g.key} onClick={() => setGroup(g.key)}
              className={cn("rounded-full border px-3 py-1 text-xs font-medium cursor-pointer",
                group === g.key ? "border-primary bg-primary-soft text-primary" : "text-muted-foreground hover:bg-muted")}>
              {g.label} <span className="ml-1 tabular-nums opacity-70">{g.n}</span>
            </button>
          ))}
      </div>
      <p className="mb-3 text-xs text-muted-foreground">Confidence comes from the evidence: independent supporting papers raise it, contradicting papers lower it, and future-work statements alone keep it low.</p>
      <div className="space-y-3">{shown.map((g) => <GapCard key={g.id} gap={g} onOpenPaper={onOpenPaper} />)}</div>
    </div>
  );
}
