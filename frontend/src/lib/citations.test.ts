import { describe, expect, it } from "vitest";
import { linkCitations, parseCiteHref } from "./citations";

describe("citations", () => {
  it("links single and multiple citations", () => {
    expect(linkCitations("RoBERTa wins [toxicity, p.4].")).toBe("RoBERTa wins [toxicity · p.4](cite:toxicity/4).");
    expect(linkCitations("Both agree [a_b, p.2; c-d, p.7]"))
      .toBe("Both agree [a_b · p.2](cite:a_b/2) [c-d · p.7](cite:c-d/7)");
  });

  it("leaves normal links, plain brackets and code blocks alone", () => {
    expect(linkCitations("[docs](https://x.org) and [note]")).toBe("[docs](https://x.org) and [note]");
    const code = "```\n[paper, p.1]\n```";
    expect(linkCitations(code)).toBe(code);
  });

  it("parses cite hrefs", () => {
    expect(parseCiteHref("cite:my%20paper/12")).toEqual({ paperId: "my paper", page: 12 });
    expect(parseCiteHref("https://example.org")).toBeNull();
  });
});
