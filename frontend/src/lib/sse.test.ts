import { describe, expect, it } from "vitest";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses messages split across chunks", () => {
    const p = new SSEParser();
    expect(p.push('data: {"a"')).toEqual([]);
    expect(p.push(': 1}\n\ndata: [DO')).toEqual([{ event: "message", data: '{"a": 1}', id: undefined }]);
    expect(p.push("NE]\n\n")).toEqual([{ event: "message", data: "[DONE]", id: undefined }]);
  });

  it("handles named events, ids, CRLF, multi-line data and keep-alive comments", () => {
    const p = new SSEParser();
    const msgs = p.push(": keep-alive\r\n\r\nid: 7\r\nevent: activity\r\ndata: line1\r\ndata: line2\r\n\r\n");
    expect(msgs).toEqual([{ event: "activity", data: "line1\nline2", id: "7" }]);
  });
});
