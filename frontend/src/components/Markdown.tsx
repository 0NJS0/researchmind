import { FileText, TriangleAlert } from "lucide-react";
import { useMemo } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import { linkCitations, parseCiteHref } from "@/lib/citations";
import type { Source } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";

function CitationChip({ paperId, page, sources }: { paperId: string; page: number; sources?: Source[] }) {
  const matches = (sources ?? []).filter((s) => s.paper_id === paperId && s.page === page);
  const unverified = sources !== undefined && matches.length === 0;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" title={unverified ? "Not found in the retrieved passages — check before relying on it" : undefined}
          className={cn("mx-0.5 inline-flex translate-y-[-1px] items-center gap-1 rounded-md px-1.5 py-0.5 align-baseline text-[0.72em] font-medium cursor-pointer",
            unverified ? "bg-warning-soft text-warning ring-1 ring-warning/40" : "bg-primary-soft text-primary hover:bg-primary/15")}>
          {unverified ? <TriangleAlert className="size-3" /> : <FileText className="size-3" />}{paperId} · p.{page}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-96">
        <p className="text-xs font-semibold">{paperId} · page {page}</p>
        {matches.length ? (
          <div className="mt-2 max-h-64 space-y-2 overflow-y-auto scrollbar-thin">
            {matches.map((s) => (
              <blockquote key={s.chunk_id} className="border-l-2 border-primary/40 pl-3 text-xs leading-relaxed text-muted-foreground">
                {s.snippet}{s.snippet.length >= 400 ? "…" : ""}
              </blockquote>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">The cited passage is not among the retrieved excerpts shown for this answer.</p>
        )}
      </PopoverContent>
    </Popover>
  );
}

export function Markdown({ children, sources, className }: { children: string; sources?: Source[]; className?: string }) {
  const text = useMemo(() => linkCitations(children || ""), [children]);
  return (
    <div className={cn("prose-rm", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => (url.startsWith("cite:") ? url : defaultUrlTransform(url))}
        components={{
          a: ({ href, children: label, ...props }) => {
            const cite = parseCiteHref(href);
            if (cite) return <CitationChip paperId={cite.paperId} page={cite.page} sources={sources} />;
            return <a href={href} target="_blank" rel="noreferrer noopener" {...props}>{label}</a>;
          },
        }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
