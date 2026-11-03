import { errorMessage } from "./api";
/** Incremental parser for text/event-stream bodies (used for POST streams, which EventSource can't do). */
export interface SSEMessage { event: string; data: string; id?: string }

export class SSEParser {
  private buffer = "";

  /** Feed a decoded text chunk; returns the complete messages it finished. */
  push(chunk: string): SSEMessage[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const out: SSEMessage[] = [];
    let idx: number;
    while ((idx = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 2);
      const msg = parseBlock(block);
      if (msg) out.push(msg);
    }
    return out;
  }
}

function parseBlock(block: string): SSEMessage | null {
  let event = "message";
  let id: string | undefined;
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") data.push(value);
    else if (field === "event") event = value;
    else if (field === "id") id = value;
  }
  if (!data.length) return null;
  return { event, data: data.join("\n"), id };
}

/** POST JSON and yield parsed SSE messages until the stream ends or `signal` aborts. */
export async function* postSSE(url: string, body: unknown, signal?: AbortSignal): AsyncGenerator<SSEMessage> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let body: unknown = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    throw new Error(errorMessage(res.status, body));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const msg of parser.push(decoder.decode(value, { stream: true }))) yield msg;
  }
  for (const msg of parser.push(decoder.decode() + "\n\n")) yield msg;
}
