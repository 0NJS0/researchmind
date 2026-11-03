import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileUp, Loader2, UploadCloud, XCircle } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import { cn, uid } from "@/lib/utils";
import { Progress } from "./ui/misc";

interface Item { id: string; name: string; status: "uploading" | "indexing" | "done" | "error"; progress: number; message?: string }

const MAX_MB = 100;

export function UploadDropzone({ topic }: { topic: string }) {
  const [items, setItems] = useState<Item[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();

  const update = (id: string, patch: Partial<Item>) => setItems((xs) => xs.map((x) => (x.id === id ? { ...x, ...patch } : x)));

  const uploadAll = useCallback(async (files: File[]) => {
    const queued = files.map((f) => ({ file: f, item: { id: uid(), name: f.name, status: "uploading" as const, progress: 0 } }));
    // Keep every row of this batch visible (older rows are dropped first).
    setItems((xs) => [...queued.map((q) => q.item), ...xs].slice(0, Math.max(30, queued.length)));
    // Sequential: indexing calls the embedding API, which is rate limited on the free tier.
    for (const { file, item } of queued) {
      if (!file.name.toLowerCase().endsWith(".pdf")) { update(item.id, { status: "error", message: "Not a PDF" }); continue; }
      if (file.size > MAX_MB * 1024 * 1024) { update(item.id, { status: "error", message: `Larger than ${MAX_MB} MB` }); continue; }
      try {
        const res = await api.library.upload(topic, file, (pct) =>
          update(item.id, pct >= 100 ? { progress: 100, status: "indexing" } : { progress: pct }));
        update(item.id, { status: "done", progress: 100, message: res.status === "attached"
          ? `Already in the topic as ${res.ref} (abstract only): added as its full text`
          : `${res.pages} pages · ${res.chunks} passages indexed` });
      } catch (e) {
        update(item.id, { status: "error", message: (e as Error).message });
      }
      for (const key of [["papers", topic], ["topics"], ["topic-papers"], ["autopilot"]]) qc.invalidateQueries({ queryKey: key });
    }
  }, [topic, qc]);

  return (
    <div>
      <div
        role="button" tabIndex={0} aria-label="Upload PDF files"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); uploadAll(Array.from(e.dataTransfer.files)); }}
        className={cn("flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors",
          dragging ? "border-primary bg-primary-soft" : "border-border hover:border-primary/50 hover:bg-muted/50")}>
        <div className="rounded-full bg-primary-soft p-2.5 text-primary"><UploadCloud className="size-5" /></div>
        <p className="text-sm font-medium">Drop PDFs here or <span className="text-primary">browse</span></p>
        <p className="text-xs text-muted-foreground">Up to {MAX_MB} MB each · text is extracted, split into passages and embedded</p>
        <input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple hidden
          onChange={(e) => { uploadAll(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      </div>
      {items.length > 0 && (
        <ul className="mt-3 space-y-2">
          {items.map((it) => (
            <li key={it.id} className="flex items-center gap-3 rounded-md border bg-card px-3 py-2 text-sm">
              {it.status === "done" ? <CheckCircle2 className="size-4 shrink-0 text-success" /> :
                it.status === "error" ? <XCircle className="size-4 shrink-0 text-danger" /> :
                it.status === "indexing" ? <Loader2 className="size-4 shrink-0 animate-spin text-primary" /> :
                <FileUp className="size-4 shrink-0 text-muted-foreground" />}
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{it.name}</p>
                {it.status === "uploading" ? <Progress value={it.progress} className="mt-1.5" /> :
                  <p className={cn("truncate text-xs", it.status === "error" ? "text-danger" : "text-muted-foreground")}>
                    {it.status === "indexing" ? "Extracting text and embedding…" : it.message}</p>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
