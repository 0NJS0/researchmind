/**
 * Turn citations like "[paper_id, p.4]" or "[A, p.2; B, p.7]" into markdown links with a cite: URL,
 * which the Markdown renderer shows as clickable chips. Code blocks are left untouched.
 */
const CITATION = /\[((?:[A-Za-z0-9._-]+,\s*(?:p\.|pp?\.\s?|page\s)\s*\d+(?:\s*[-–]\s*\d+)?)(?:\s*;\s*[A-Za-z0-9._-]+,\s*(?:p\.|pp?\.\s?|page\s)\s*\d+(?:\s*[-–]\s*\d+)?)*)\](?!\()/g;
const PART = /([A-Za-z0-9._-]+),\s*(?:p\.|pp?\.\s?|page\s)\s*(\d+)/;

export interface Citation { paperId: string; page: number }

export function linkCitations(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```)/g)
    .map((segment) => segment.startsWith("```") ? segment : segment.replace(CITATION, (_m, inner: string) =>
      inner.split(";").map((part) => {
        const m = part.trim().match(PART);
        return m ? `[${m[1]} · p.${m[2]}](cite:${encodeURIComponent(m[1])}/${m[2]})` : part;
      }).join(" ")))
    .join("");
}

export function parseCiteHref(href: string | undefined): Citation | null {
  const m = href?.match(/^cite:([^/]+)\/(\d+)$/);
  return m ? { paperId: decodeURIComponent(m[1]), page: Number(m[2]) } : null;
}
