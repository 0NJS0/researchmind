import { useQueryClient } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input, Label, Textarea } from "@/components/ui/input";
import { api } from "@/lib/api";

/** Start research on a topic: creates the Library topic (or reuses it) and puts it on the autopilot. */
export function NewTopicResearchDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [topic, setTopic] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const qc = useQueryClient();
  const submit = async () => {
    setBusy(true);
    try {
      const session = await api.research.create(topic.trim(), description.trim());
      toast.success(session.autopilot_status === "pending"
        ? "Topic created. The autopilot switches to it after its current paper."
        : "Topic created and put on the autopilot.");
      onOpenChange(false);
      setTopic(""); setDescription("");
      qc.invalidateQueries({ queryKey: ["topics"] });
      qc.invalidateQueries({ queryKey: ["autopilot"] });
      qc.invalidateQueries({ queryKey: ["research"] });
    } catch (e) {
      toast.error((e as Error).message);
    } finally { setBusy(false); }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>New topic</DialogTitle>
          <DialogDescription>
            The autopilot keeps finding relevant papers on it, analyses them one at a time (newest and best-published
            first) and maps the research gaps. If it is busy with another topic, it switches after the current paper.
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); if (topic.trim().length >= 3) submit(); }}>
          <div className="space-y-1.5">
            <Label htmlFor="nt-topic">Research topic</Label>
            <Input id="nt-topic" autoFocus value={topic} maxLength={300} onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Large language models for automated unit test generation" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="nt-focus">Focus <span className="font-normal text-muted-foreground">(optional)</span></Label>
            <Textarea id="nt-focus" rows={3} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)}
              placeholder="Anything to emphasise or exclude — e.g. only empirical studies, industrial settings, papers after 2020…" />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={topic.trim().length < 3}><Sparkles />Start on autopilot</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
