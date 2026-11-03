import { Component, type ReactNode } from "react";
import { Button } from "./ui/button";

/** Catches a crash in one page so the app shows a message instead of going blank. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    // Navigating to another page clears the error.
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    const outdated = /dynamically imported module|Failed to fetch|Importing a module script failed|MIME type/i
      .test(this.state.error.message);
    return (
      <div role="alert" className="mx-auto mt-16 max-w-md rounded-lg border bg-card p-6 text-center">
        <p className="font-semibold">{outdated ? "ResearchMind was updated" : "Something went wrong on this page"}</p>
        <p className="mt-2 text-sm text-muted-foreground">
          {outdated ? "Reload to get the new version." : this.state.error.message}
        </p>
        <Button className="mt-4" onClick={() => window.location.reload()}>Reload</Button>
      </div>
    );
  }
}

/** After a redeploy, an open tab asks for code files that no longer exist: reload once to get the new version. */
export function reloadOnStaleBuild() {
  window.addEventListener("vite:preloadError", (event) => {
    const key = "rm-reloaded-for-update";
    try {
      if (sessionStorage.getItem(key)) return;   // reloaded once already: let the error boundary explain
      sessionStorage.setItem(key, "1");
    } catch { /* storage unavailable: still try once */ }
    event.preventDefault();
    window.location.reload();
  });
}
