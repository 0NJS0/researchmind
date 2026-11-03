import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./api";

afterEach(() => vi.restoreAllMocks());

function mockFetch(status: number, body: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status }));
}

describe("request", () => {
  it("returns parsed JSON", async () => {
    mockFetch(200, { ok: 1 });
    await expect(request("/x")).resolves.toEqual({ ok: 1 });
  });

  it("surfaces FastAPI detail messages", async () => {
    mockFetch(409, { detail: "A job is already running for this topic" });
    await expect(request("/x")).rejects.toMatchObject({ status: 409, message: "A job is already running for this topic" });
  });

  it("formats validation errors", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "topic"], msg: "String should have at least 3 characters" }] });
    await expect(request("/x")).rejects.toThrow("topic: String should have at least 3 characters");
  });

  it("reports network failures as status 0", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const err = (await request("/x").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
  });
});
