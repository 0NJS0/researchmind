import { useQuery } from "@tanstack/react-query";
import { Activity, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { Health, ServiceStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "./ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";

export const SERVICES: { key: keyof Omit<Health, "status">; label: string; hint: string }[] = [
  { key: "database", label: "Database", hint: "PostgreSQL + pgvector (DATABASE_URL)" },
  { key: "llm", label: "Language model", hint: "OpenRouter (LLM_API_KEY, LLM_MODEL)" },
  { key: "vectorstore", label: "Paper index", hint: "Embeddings stored in pgvector (EMBEDDING_MODEL)" },
  { key: "searxng", label: "Web search", hint: "SearXNG for the research agent (SEARXNG_URL)" },
  { key: "openalex", label: "Paper metadata", hint: "OpenAlex: citations, peer review, open-access PDFs (OPENALEX_API_KEY)" },
  { key: "autopilot", label: "Autopilot worker", hint: "The background worker that researches the active topic" },
  { key: "config", label: "Configuration", hint: "Required environment variables" },
];

export function StatusDot({ ok, className }: { ok?: boolean; className?: string }) {
  return (
    <span className={cn("relative inline-flex size-2 rounded-full", ok === undefined ? "bg-muted-foreground/40" : ok ? "bg-success" : "bg-danger", className)}>
      {ok === false && <span className="absolute inset-0 animate-ping rounded-full bg-danger opacity-60" />}
    </span>
  );
}

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: () => api.health(), refetchInterval: 60_000, retry: false });
}

export function ServiceRow({ label, hint, status }: { label: string; hint: string; status?: ServiceStatus }) {
  return (
    <div className="flex items-start gap-3 py-2">
      <StatusDot ok={status?.ok} className="mt-1.5" />
      <div className="min-w-0">
        <p className="text-sm font-medium">{label}</p>
        <p className="break-words text-xs text-muted-foreground">{status?.message || hint}</p>
      </div>
    </div>
  );
}

export function HealthIndicator() {
  const { data, isLoading, isError, refetch, isFetching } = useHealth();
  const label = isError ? "Server unreachable" : isLoading ? "Checking…" :
    data?.status === "healthy" ? "All systems normal" : data?.status === "degraded" ? "Partially available" : "Needs attention";
  const ok = isError ? false : isLoading ? undefined : data?.status === "healthy" ? true : data?.status === "degraded" ? undefined : false;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="flex items-center gap-2 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted cursor-pointer">
          <StatusDot ok={ok} className={data?.status === "degraded" ? "bg-warning" : undefined} />
          <span className="hidden sm:inline">{label}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80">
        <div className="mb-1 flex items-center justify-between">
          <p className="flex items-center gap-2 text-sm font-semibold"><Activity className="size-4" />Service status</p>
          <Button variant="ghost" size="icon-sm" onClick={() => api.health(true).then(() => refetch()).catch(() => refetch())} aria-label="Refresh status">
            <RefreshCw className={cn(isFetching && "animate-spin")} />
          </Button>
        </div>
        {isError ? <p className="py-2 text-sm text-danger">The ResearchMind API is not responding.</p> :
          <div className="divide-y">{SERVICES.map((s) => <ServiceRow key={s.key} label={s.label} hint={s.hint} status={data?.[s.key]} />)}</div>}
        <Link to="/status" className="mt-2 block text-xs text-primary hover:underline">Open status page →</Link>
      </PopoverContent>
    </Popover>
  );
}
