import { useQuery } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";
import { storage } from "./storage";
import type { Topic } from "./types";

interface TopicState {
  topic: string;
  setTopic: (t: string) => void;
  topics: Topic[];
  loading: boolean;
}
const EMPTY: Topic[] = [];
const TopicContext = createContext<TopicState>({ topic: "", setTopic: () => {}, topics: [], loading: true });

export function TopicProvider({ children }: { children: ReactNode }) {
  const [topic, setTopicState] = useState(() => storage.get("rm-topic") || "");
  // The autopilot adds papers in the background, so paper counts (Ask, sidebar, Autopilot) must refresh on their own.
  const { data, isLoading, isError } = useQuery({
    queryKey: ["topics"], queryFn: api.library.topics, refetchInterval: 30_000, refetchOnWindowFocus: true,
  });
  const topics = data ?? EMPTY;
  const setTopic = useCallback((t: string) => {
    storage.set("rm-topic", t || null);
    setTopicState(t);
  }, []);
  // Keep the selection valid when topics change (deleted topic, first load).
  useEffect(() => {
    if (isLoading || isError || !data) return;  // e.g. the server is restarting: keep the saved choice
    if (topic && !topics.some((t) => t.name === topic)) setTopic(topics[0]?.name ?? "");
    else if (!topic && topics.length) setTopic(topics[0].name);
  }, [topics, topic, isLoading, isError, data, setTopic]);
  return <TopicContext.Provider value={{ topic, setTopic, topics, loading: isLoading }}>{children}</TopicContext.Provider>;
}

export const useTopic = () => useContext(TopicContext);
