import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Gap, ResearchPaper } from "@/lib/types";
import { GapCard } from "./GapCard";
import { filterPapers } from "./PapersTab";

vi.mock("@/lib/api", () => ({
  api: {
    research: {
      gap: vi.fn().mockResolvedValue({
        gap: {}, merged_into: null,
        evidence: [{ id: "e1", paper_id: "p1", paper_ref: "P1", paper_title: "T", evidence: "Only Java benchmarks were used.",
          relationship: "supports", basis: "stated_limitation", created_at: "2026-01-01T00:00:00Z" }],
        history: [{ id: "h1", action: "created", detail: "Identified while analysing P1", paper_ref: "P1", created_at: "2026-01-01T00:00:00Z" }],
      }),
    },
  },
}));

const gap: Gap = {
  id: "g1", ref: "G1", title: "No evaluation on industrial codebases", description: "Only open-source benchmarks.",
  gap_type: "evaluation", category: "potential_gap", confidence: "medium", status: "contested", merged_into: null,
  evidence_count: 2, paper_count: 2, related_papers: [], created_at: "", updated_at: "u1",
};

describe("GapCard", () => {
  it("shows status and confidence, and loads evidence and history when expanded", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><GapCard gap={gap} /></QueryClientProvider>);
    expect(screen.getByText("No evaluation on industrial codebases")).toBeInTheDocument();
    expect(screen.getByText("Contested")).toBeInTheDocument();
    expect(screen.getByLabelText("Confidence medium")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { expanded: false }));
    expect(await screen.findByText("Only Java benchmarks were used.")).toBeInTheDocument();
    expect(screen.getByText("Identified while analysing P1")).toBeInTheDocument();
    expect(screen.getByText("Stated limitation")).toBeInTheDocument();
  });
});

describe("filterPapers", () => {
  const paper = (over: Partial<ResearchPaper>) => ({ title: "", ref: "", venue: "", authors: [], year: null, ...over }) as ResearchPaper;
  const papers = [
    paper({ title: "LLM test generation", ref: "P1", authors: ["Alice Smith"], year: 2023 }),
    paper({ title: "Search-based testing", ref: "P2", venue: "ICSE" }),
  ];
  it("matches title, author, venue, ref and year case-insensitively", () => {
    expect(filterPapers(papers, "llm").map((p) => p.ref)).toEqual(["P1"]);
    expect(filterPapers(papers, "smith").map((p) => p.ref)).toEqual(["P1"]);
    expect(filterPapers(papers, "icse").map((p) => p.ref)).toEqual(["P2"]);
    expect(filterPapers(papers, "2023").map((p) => p.ref)).toEqual(["P1"]);
    expect(filterPapers(papers, "  ")).toHaveLength(2);
  });
});
