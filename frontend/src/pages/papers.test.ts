import { describe, expect, it } from "vitest";
import type { TopicPaper } from "@/lib/types";
import { sortPapers, toCsv } from "./PapersPage";

const paper = (over: Partial<TopicPaper>): TopicPaper => ({ has_pdf: false,
  id: "x", ref: "P1", title: "T", authors: [], year: null, venue: "", citation_count: null, peer_reviewed: null,
  publication_type: "", doi: "", url: "", pdf_url: "", status: "processed", content_status: "full_text",
  relevance_score: null, priority_score: null, relevance_reason: "", origin: "agent", library_paper_id: "",
  has_analysis: false, has_summary: true, gap_count: 0, added_at: null, processed_at: null, error: "",
  needs_full_text: false, fulltext_checked_at: null, ...over,
});

describe("Papers page helpers", () => {
  it("sorts by newest, citations and priority with missing values last", () => {
    const ps = [paper({ ref: "A", year: 2020, citation_count: 50, priority_score: 0.5 }),
      paper({ ref: "B", year: 2025, citation_count: null, priority_score: 0.9 }),
      paper({ ref: "C", year: null, citation_count: 300, priority_score: null })];
    expect(sortPapers(ps, "newest").map((p) => p.ref)).toEqual(["B", "A", "C"]);
    expect(sortPapers(ps, "cited").map((p) => p.ref)).toEqual(["C", "A", "B"]);
    expect(sortPapers(ps, "priority").map((p) => p.ref)).toEqual(["B", "A", "C"]);
  });

  it("exports CSV with quoting", () => {
    const csv = toCsv([paper({ title: 'Toxicity, "bias" and LLMs', authors: ["A. One", "B. Two"], doi: "10.1/x", year: 2024 })]);
    const [header, row] = csv.split("\n");
    expect(header.startsWith("ref,title,authors,year")).toBe(true);
    expect(row).toContain('"Toxicity, ""bias"" and LLMs"');
    expect(row).toContain("A. One; B. Two");
  });
});
