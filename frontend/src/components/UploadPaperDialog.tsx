import { useQueryClient } from "@tanstack/react-query";
import { FileUp, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input, Label } from "@/components/ui/input";
import { Progress } from "@/components/ui/misc";
import { api } from "@/lib/api";

const MAX_MB = 100;

/** Upload one paper's PDF together with its link / DOI and title. */
export function UploadPaperDialog({ topic, open, onOpenChange }: { topic: string; open: boolean; onOpenChange: (o: boolean) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [link, setLink] = useState("");
  const [title, setTitle] = useState("");
  const [pct, setPct] = useState<number | null>(null);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();

  const reset = () => { setFile(null); setLink(""); setTitle(""); setPct(null); setError(""); if (input.current) input.current.value = ""; };
  // Closing while the upload runs would hide its result; wait for it instead.
  const close = (o: boolean) => { if (o || pct !== null) return; reset(); onOpenChange(false); };
  const pick = (f: File | undefined) => {
    setError("");
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pdf")) return setError("Choose a PDF file.");
    if (f.size > MAX_MB * 1024 * 1024) return setError(`The file is larger than ${MAX_MB} MB.`);
    setFile(f);
  };
  const submit = async () => {
    if (!file) return;
    const v = link.trim();
    const isDoi = /^(doi:)?10\.\d{4,9}\//i.test(v);
    setPct(0); setError("");
    try {
      const r = await api.library.upload(topic, file, setPct, isDoi ? { doi: v.replace(/^doi:/i, ""), title } : { url: v, title });
      toast.success(r.status === "attached"
        ? `The topic already had this paper as ${r.ref} (abstract only): the PDF is now its full text and it is re-analysed.`
        : `Uploaded (${r.pages} pages). The autopilot analyses it with priority when this topic is active.`);
      for (const key of ["topic-papers", "topics", "papers", "autopilot"]) qc.invalidateQueries({ queryKey: [key] });
      setPct(null);
      reset();
      onOpenChange(false);
    } catch (e) {
      setError((e as Error).message);   // shown in the dialog, so a failure is never silent
      setPct(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => close(o)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload a paper</DialogTitle>
          <DialogDescription>
            Add a PDF to “{topic.replace(/_/g, " ")}”. The link or DOI is optional but helps identify the paper
            (authors, venue, citations) and avoids duplicates.
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <div className="space-y-1.5">
            <Label htmlFor="up-file">PDF file</Label>
            <input ref={input} id="up-file" type="file" accept="application/pdf,.pdf" className="hidden"
              onChange={(e) => pick(e.target.files?.[0])} />
            <Button type="button" variant="outline" className="w-full justify-start" onClick={() => input.current?.click()}>
              <FileUp />{file ? <span className="truncate">{file.name}</span> : "Choose a PDF…"}
            </Button>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="up-link">Link or DOI <span className="font-normal text-muted-foreground">(optional)</span></Label>
            <Input id="up-link" value={link} onChange={(e) => setLink(e.target.value)}
              placeholder="https://openaccess.thecvf.com/… · https://arxiv.org/abs/… · 10.1109/…" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="up-title">Title <span className="font-normal text-muted-foreground">(optional)</span></Label>
            <Input id="up-title" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={500} />
          </div>
          {pct !== null && <Progress value={pct} />}
          {pct === 100 && <p className="text-xs text-muted-foreground">Extracting text and indexing…</p>}
          {error && <p role="alert" className="rounded-md bg-danger-soft px-3 py-2 text-sm text-danger">{error}</p>}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => close(false)} disabled={pct !== null}>Cancel</Button>
            <Button type="submit" disabled={!file} loading={pct !== null}><Upload />Upload</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
