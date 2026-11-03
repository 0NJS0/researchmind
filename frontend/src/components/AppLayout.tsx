import * as DialogPrimitive from "@radix-ui/react-dialog";
import { useQueryClient } from "@tanstack/react-query";
import {
  Activity, Bot, BookOpen, FileStack, FlaskConical, Check, ChevronsUpDown, FileText, FolderPlus, Menu, MessageSquareText, Monitor, Moon, Sun, X,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useTheme } from "@/lib/theme";
import { useTopic } from "@/lib/topic";
import { cn } from "@/lib/utils";
import { HealthIndicator } from "./HealthIndicator";
import { useQuery } from "@tanstack/react-query";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "./ui/dropdown";
import { Input, Label } from "./ui/input";

const NAV = [
  { to: "/autopilot", label: "Autopilot", icon: Bot, description: "Topics, research gaps & synthesis" },
  { to: "/papers", label: "Papers", icon: FileStack, description: "All papers in a topic" },
  { to: "/validator", label: "Topic validator", icon: FlaskConical, description: "Stress-test an idea" },
  { to: "/library", label: "Library", icon: BookOpen, description: "Topics & papers" },
  { to: "/ask", label: "Ask", icon: MessageSquareText, description: "Question your papers" },
  { to: "/reports", label: "Reports", icon: FileText, description: "Literature reports" },
  { to: "/status", label: "Status", icon: Activity, description: "Services & settings" },
];

export function NewTopicDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const { setTopic } = useTopic();
  const qc = useQueryClient();
  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const { name: created } = await api.library.createTopic(name.trim());
      await qc.invalidateQueries({ queryKey: ["topics"] });
      setTopic(created);
      toast.success(`Topic “${created}” created`);
      setName("");
      onOpenChange(false);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New topic</DialogTitle>
          <DialogDescription>Topics group related papers, e.g. “LLM toxicity detection”.</DialogDescription>
        </DialogHeader>
        <form onSubmit={(e) => { e.preventDefault(); submit(); }} className="space-y-2">
          <Label htmlFor="topic-name">Name</Label>
          <Input id="topic-name" autoFocus value={name} maxLength={120} placeholder="e.g. Graph neural networks"
            onChange={(e) => setName(e.target.value)} />
          <p className="text-xs text-muted-foreground">Spaces become underscores.</p>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={!name.trim()}>Create topic</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function TopicSwitcher() {
  const { topic, setTopic, topics } = useTopic();
  const [creating, setCreating] = useState(false);
  const current = topics.find((t) => t.name === topic);
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex w-full items-center gap-2 rounded-lg border bg-card px-3 py-2 text-left shadow-xs hover:bg-muted cursor-pointer">
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Topic</p>
              <p className="truncate text-sm font-medium">{topic ? topic.replace(/_/g, " ") : "No topic yet"}</p>
              {current && <p className="text-xs text-muted-foreground">{current.papers} papers</p>}
            </div>
            <ChevronsUpDown className="size-4 text-muted-foreground" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-60">
          <div className="max-h-72 overflow-y-auto scrollbar-thin">
            {topics.map((t) => (
              <DropdownMenuItem key={t.name} onSelect={() => setTopic(t.name)}>
                <Check className={cn(t.name === topic ? "opacity-100" : "opacity-0")} />
                <span className="flex-1 truncate">{t.name.replace(/_/g, " ")}</span>
                <span className="text-xs text-muted-foreground">{t.papers}</span>
              </DropdownMenuItem>
            ))}
            {!topics.length && <p className="px-2.5 py-2 text-sm text-muted-foreground">No topics yet</p>}
          </div>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => setCreating(true)}><FolderPlus />New topic…</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <NewTopicDialog open={creating} onOpenChange={setCreating} />
    </>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const options = [
    { value: "light" as const, icon: Sun, label: "Light" },
    { value: "dark" as const, icon: Moon, label: "Dark" },
    { value: "system" as const, icon: Monitor, label: "System" },
  ];
  return (
    <div className="flex rounded-lg border bg-card p-0.5" role="radiogroup" aria-label="Theme">
      {options.map(({ value, icon: Icon, label }) => (
        <button key={value} role="radio" aria-checked={theme === value} aria-label={label} title={label}
          onClick={() => setTheme(value)}
          className={cn("flex flex-1 items-center justify-center rounded-md py-1.5 text-muted-foreground cursor-pointer",
            theme === value && "bg-muted text-foreground")}>
          <Icon className="size-4" />
        </button>
      ))}
    </div>
  );
}

function AutopilotPill({ onNavigate }: { onNavigate?: () => void }) {
  const { data } = useQuery({ queryKey: ["autopilot"], queryFn: api.autopilot.status, refetchInterval: 10_000 });
  if (!data?.active_topic) return null;
  const working = ["processing", "searching", "screening", "verifying"].includes(data.state);
  return (
    <NavLink to="/autopilot" onClick={onNavigate}
      className="block rounded-lg border bg-card px-3 py-2.5 text-xs shadow-xs hover:border-primary/40">
      <p className="flex items-center gap-1.5 font-medium">
        <span className={cn("size-2 rounded-full", !data.enabled ? "bg-muted-foreground/50" : working ? "animate-pulse bg-primary"
          : data.state === "paused" || data.state === "error" ? "bg-warning" : "bg-success")} />
        Autopilot · {data.active_topic.replace(/_/g, " ")}
      </p>
      <p className="mt-1 line-clamp-2 text-muted-foreground">{data.enabled ? data.message || data.state : "Paused"}</p>
    </NavLink>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-5 p-4">
      <div className="flex items-center gap-2.5 px-1">
        <img src="/favicon.svg" alt="" className="size-8" />
        <div>
          <p className="text-sm font-semibold leading-tight">ResearchMind</p>
          <p className="text-xs text-muted-foreground">Literature intelligence</p>
        </div>
      </div>
      <TopicSwitcher />
      <nav className="flex flex-col gap-0.5" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} onClick={onNavigate}
            className={({ isActive }) => cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              isActive && "bg-primary-soft text-primary hover:bg-primary-soft hover:text-primary")}>
            <Icon className="size-4" />{label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto space-y-3"><AutopilotPill onNavigate={onNavigate} /><ThemeToggle /></div>
    </div>
  );
}

export function AppLayout({ children }: { children?: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setMobileOpen(false), [location.pathname]);
  const current = NAV.find((n) => location.pathname.startsWith(n.to));

  return (
    <div className="flex min-h-dvh">
      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 border-r bg-sidebar lg:block">
        <Sidebar />
      </aside>

      {/* Mobile menu: a real modal (focus trap, Escape, scroll lock, screen-reader dialog) */}
      <DialogPrimitive.Root open={mobileOpen} onOpenChange={setMobileOpen}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40 lg:hidden" />
          <DialogPrimitive.Content aria-describedby={undefined}
            className="fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] border-r bg-sidebar shadow-xl animate-fade-in focus:outline-none lg:hidden">
            <DialogPrimitive.Title className="sr-only">Menu</DialogPrimitive.Title>
            <DialogPrimitive.Close className="absolute right-3 top-4 text-muted-foreground cursor-pointer" aria-label="Close menu">
              <X className="size-5" />
            </DialogPrimitive.Close>
            <Sidebar onNavigate={() => setMobileOpen(false)} />
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur sm:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open menu">
            <Menu />
          </Button>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{current?.label ?? "ResearchMind"}</p>
          </div>
          <HealthIndicator />
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 lg:py-8">
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  );
}
