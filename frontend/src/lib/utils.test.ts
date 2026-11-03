import { afterEach, describe, expect, it, vi } from "vitest";
import { copyText, uid } from "./utils";

describe("works over plain http (no secure context)", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uid falls back when crypto.randomUUID is missing", () => {
    vi.stubGlobal("crypto", { getRandomValues: (a: Uint8Array) => a.map((_, i) => (i * 37) % 256) });
    const id = uid();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });

  it("copyText falls back when navigator.clipboard is missing", async () => {
    vi.stubGlobal("navigator", {});
    const exec = vi.fn(() => true);
    Object.defineProperty(document, "execCommand", { value: exec, configurable: true });
    expect(await copyText("hello")).toBe(true);
    expect(exec).toHaveBeenCalledWith("copy");
  });
});
