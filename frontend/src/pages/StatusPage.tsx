import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { useState } from "react";
import { SERVICES, ServiceRow, useHealth } from "@/components/HealthIndicator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, PageHeader, Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import { humanize } from "@/lib/utils";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-2.5 sm:flex-row sm:items-center sm:justify-between">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="text-sm font-medium break-all sm:text-right">{children}</dd>
    </div>
  );
}

export default function StatusPage() {
  const { data: health, isLoading, isError, refetch } = useHealth();
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: api.config });
  const [refreshing, setRefreshing] = useState(false);

  return (
    <>
      <PageHeader title="Status" description="Connected services and active settings. Secrets such as API keys are never shown."
        actions={<Button variant="outline" loading={refreshing} onClick={async () => {
          setRefreshing(true);
          try { await api.health(true); } catch { /* shown by the refetch below */ } finally { await refetch(); setRefreshing(false); }
        }}>{!refreshing && <RefreshCw />}Re-check</Button>} />

      {isError && <Alert className="mb-6" title="The ResearchMind API is not responding">Check that the backend container is running.</Alert>}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Services</CardTitle>
            <CardDescription>Checked every minute (cached for 30 s on the server).</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-48" /> :
              <div className="divide-y">{SERVICES.map((s) => <ServiceRow key={s.key} label={s.label} hint={s.hint} status={health?.[s.key]} />)}</div>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Models</CardTitle>
            <CardDescription>Configured through environment variables on the server.</CardDescription>
          </CardHeader>
          <CardContent>
            {!config ? <Skeleton className="h-48" /> : (
              <dl className="divide-y">
                <Row label="Provider"><Badge tone="primary">{config.llm_provider}</Badge></Row>
                <Row label="Main model">{config.llm_model || "—"}</Row>
                <Row label="Fallback models">{config.llm_fallback_models.length ? config.llm_fallback_models.join(", ") : "none"}</Row>
                <Row label="Screening model">{config.llm_fast_model || "—"}</Row>
                <Row label="Embedding model">{config.embedding_model}</Row>
                <Row label="Context budget">{config.llm_context_tokens.toLocaleString()} tokens · {config.llm_max_output_tokens.toLocaleString()} output</Row>
              </dl>
            )}
          </CardContent>
        </Card>

        {config && (
          <Card className="lg:col-span-2">
            <CardHeader><CardTitle>Research agent limits</CardTitle></CardHeader>
            <CardContent>
              <dl className="grid gap-x-8 sm:grid-cols-2">
                {Object.entries(config.limits).map(([k, v]) => <Row key={k} label={humanize(k)}>{v}</Row>)}
              </dl>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
