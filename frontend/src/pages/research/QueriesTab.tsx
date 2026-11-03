import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { EmptyState, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import { humanize } from "@/lib/utils";

export function QueriesTab({ sessionId }: { sessionId: string }) {
  const { data: queries, isLoading } = useQuery({ queryKey: ["queries", sessionId], queryFn: () => api.research.queries(sessionId) });
  if (isLoading) return <Skeleton className="h-48" />;
  if (!queries?.length) return <EmptyState icon={Search} title="No searches yet" />;
  const rounds = [...new Set(queries.map((q) => q.iteration))];
  return (
    <div className="space-y-6">
      {rounds.map((round) => (
        <section key={round}>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Round {round}</h3>
          <ul className="divide-y overflow-hidden rounded-lg border bg-card">
            {queries.filter((q) => q.iteration === round).map((q) => (
              <li key={q.id} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                <Search className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate">{q.query}</span>
                <Badge tone="outline">{humanize(q.source)}</Badge>
                {q.status === "failed" ? <Badge tone="danger" title={q.error}>Failed</Badge> :
                  <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">{q.result_count} results</span>}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
