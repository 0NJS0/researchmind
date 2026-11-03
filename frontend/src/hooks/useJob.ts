import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { ApiError, api } from "@/lib/api";
import type { Job } from "@/lib/types";

/** Poll a background job while it runs; call onFinish once when it completes or fails. */
export function useJob(jobId: string | null, onFinish?: (job: Job) => void) {
  const finished = useRef<string | null>(null);
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: async (): Promise<Job> => {
      try {
        return await api.library.job(jobId!);
      } catch (e) {
        // Jobs live in server memory: after a server restart the job is gone and will never finish.
        if (e instanceof ApiError && e.status === 404) {
          return {
            id: jobId!, kind: "report", topic: "", status: "failed", stage: "", done: 0, total: 0, current: "",
            error: "The server restarted while this job was running. Please start it again.",
            warnings: [], file: "", started_at: "", finished_at: null,
          };
        }
        throw e;
      }
    },
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === "running" || !q.state.data ? 1500 : false),
  });
  useEffect(() => {
    const job = query.data;
    if (job && job.status !== "running" && finished.current !== job.id) {
      finished.current = job.id;
      qc.invalidateQueries({ queryKey: ["jobs"] });
      onFinish?.(job);
    }
  }, [query.data, onFinish, qc]);
  return query;
}

/** The running job for a topic (if any), so progress survives navigation and reloads. */
export function useRunningJob(topic: string, kind: Job["kind"]) {
  return useQuery({
    queryKey: ["jobs", topic],
    queryFn: () => api.library.jobs(topic),
    enabled: !!topic,
    select: (jobs) => jobs.find((j) => j.kind === kind && j.status === "running") ?? null,
    refetchInterval: 10_000,
  });
}
