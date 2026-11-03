import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ActivityEvent } from "@/lib/types";

const HISTORY = 300; // most recent events shown (a topic researched for weeks has thousands)

/** Activity events for a session: loads the latest history, then follows the live SSE stream while work runs.
 *  Everything an effect run fetches is tagged with its own session, so a topic switch never mixes feeds. */
export function useSessionEvents(sessionId: string, running: boolean) {
  const [state, setState] = useState<{ sessionId: string; events: ActivityEvent[] }>({ sessionId: "", events: [] });
  const qc = useQueryClient();

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    let source: EventSource | null = null;
    let refreshTimer: ReturnType<typeof setTimeout> | undefined;
    let lastId = 0;  // newest event this run has seen (history + stream)

    // Merge by id: history and live events can arrive in either order.
    const merge = (evs: ActivityEvent[]) => {
      if (cancelled || !evs.length) return;
      lastId = Math.max(lastId, ...evs.map((e) => e.id));
      setState((cur) => {
        const base = cur.sessionId === sessionId ? cur.events : [];
        const seen = new Set(base.map((e) => e.id));
        const fresh = evs.filter((e) => !seen.has(e.id));
        if (!fresh.length && cur.sessionId === sessionId) return cur;
        return { sessionId, events: [...base, ...fresh].sort((a, b) => a.id - b.id).slice(-HISTORY * 2) };
      });
    };
    const refresh = () => {
      // Coalesce bursts of events into one refetch of what the events describe.
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => {
        if (cancelled) return;
        for (const key of ["session", "papers", "gaps", "synthesis", "queries"]) qc.invalidateQueries({ queryKey: [key, sessionId] });
        qc.invalidateQueries({ queryKey: ["paper"] });   // an open paper panel
      }, 800);
    };

    setState((cur) => (cur.sessionId === sessionId ? cur : { sessionId, events: [] }));

    api.research.events(sessionId, 0, true).then((evs) => {
      merge(evs);
      if (cancelled || !running) return;
      // Open the live stream only after the history, from the last event we have.
      source = new EventSource(`/api/research/${sessionId}/events/stream?after=${lastId}`);
      source.addEventListener("activity", (e) => {
        merge([JSON.parse((e as MessageEvent).data) as ActivityEvent]);
        refresh();
      });
      source.addEventListener("status", () => {
        source?.close();
        // Catch anything written between the last streamed event and the end of the stream.
        api.research.events(sessionId, lastId).then(merge).catch(() => {});
        refresh();
        qc.invalidateQueries({ queryKey: ["research"] });
      });
      source.onerror = () => {
        // The browser reconnects by itself; if it gave up, fall back to one catch-up fetch.
        if (source?.readyState === EventSource.CLOSED) api.research.events(sessionId, lastId).then(merge).catch(() => {});
      };
    }).catch(() => {});

    return () => {
      cancelled = true;
      clearTimeout(refreshTimer);
      source?.close();
    };
  }, [sessionId, running, qc]);

  return state.sessionId === sessionId ? state.events : [];
}
