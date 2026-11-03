import { ChevronDown, Download, FileArchive, FileText } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "./ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "./ui/dropdown";

/** One-click download of everything about a topic (ZIP built by the server). */
export function ExportTopicButton({ topic, variant = "outline" }: { topic: string; variant?: "outline" | "primary" }) {
  const start = (includePdfs: boolean) => {
    // A plain link download: the browser streams the ZIP straight to disk and shows its own progress.
    const a = document.createElement("a");
    a.href = api.library.exportUrl(topic, includePdfs);
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast.info("Preparing the download — large topics with many PDFs can take a moment.");
  };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant={variant} disabled={!topic}><Download />Download everything<ChevronDown className="opacity-60" /></Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-72">
        <DropdownMenuItem onSelect={() => start(true)}>
          <FileArchive />
          <div><p>ZIP with PDFs</p><p className="text-xs text-muted-foreground">Reports, gaps, synthesis, analyses, paper list and all stored PDFs</p></div>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => start(false)}>
          <FileText />
          <div><p>ZIP without PDFs</p><p className="text-xs text-muted-foreground">Smaller: everything except the PDF files</p></div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
