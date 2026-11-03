import { useQuery } from "@tanstack/react-query";
import { BookMarked } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { EmptyState, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import type { Synthesis, SynthesisItem } from "@/lib/types";
import { relativeTime } from "@/lib/utils";
import { PaperRef } from "./common";

const SECTIONS: { key: keyof Omit<Synthesis, "overview">; title: string; hint: string }[] = [
  { key: "known_findings", title: "What is known", hint: "Findings reported by the papers" },
  { key: "contradictory_findings", title: "Contradictions", hint: "Where papers disagree" },
  { key: "common_methods", title: "Common methods", hint: "How the field approaches the problem" },
  { key: "recurring_limitations", title: "Recurring limitations", hint: "Weaknesses seen across papers" },
  { key: "under_researched", title: "Under-researched", hint: "Areas with little coverage" },
  { key: "research_questions", title: "Open questions", hint: "Questions emerging from the literature" },
];

function Item({ item, onOpenPaper }: { item: SynthesisItem; onOpenPaper: (id: string) => void }) {
  return (
    <li className="flex gap-2 text-sm leading-relaxed">
      <span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary/50" />
      <div>
        {item.statement}
        <span className="ml-1.5 inline-flex flex-wrap gap-1 align-middle">
          {item.paper_refs.map((ref, i) => <PaperRef key={ref} refId={ref} onClick={() => onOpenPaper(item.paper_ids[i])} />)}
          {item.strength && <Badge tone={item.strength === "established" ? "success" : item.strength === "unverified" ? "warning" : "neutral"}>{item.strength}</Badge>}
        </span>
      </div>
    </li>
  );
}

export function SynthesisTab({ sessionId, onOpenPaper }: { sessionId: string; onOpenPaper: (id: string) => void }) {
  const { data, isLoading } = useQuery({ queryKey: ["synthesis", sessionId], queryFn: () => api.research.synthesis(sessionId) });
  if (isLoading) return <Skeleton className="h-64" />;
  const s = data?.synthesis;
  if (!s) return <EmptyState icon={BookMarked} title="No synthesis yet" description="The synthesis is updated after each paper is analysed." />;

  return (
    <div className="space-y-5">
      <Card className="border-primary/20 bg-primary-soft/30 p-5">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold">Overview</h3>
          {data?.updated_at && <span className="text-xs text-muted-foreground">Updated {relativeTime(data.updated_at)}</span>}
        </div>
        <p className="text-sm leading-relaxed">{s.overview || "—"}</p>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        {SECTIONS.filter((sec) => s[sec.key]?.length).map((sec) => (
          <Card key={sec.key} className="p-5">
            <h3 className="text-sm font-semibold">{sec.title}</h3>
            <p className="mb-3 text-xs text-muted-foreground">{sec.hint}</p>
            <ul className="space-y-2">{s[sec.key].map((item, i) => <Item key={i} item={item} onOpenPaper={onOpenPaper} />)}</ul>
          </Card>
        ))}
      </div>
    </div>
  );
}
