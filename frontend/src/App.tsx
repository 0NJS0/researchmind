import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Skeleton } from "./components/ui/misc";
import { TopicProvider } from "./lib/topic";

const LibraryPage = lazy(() => import("./pages/LibraryPage"));
const AskPage = lazy(() => import("./pages/AskPage"));
const ReportsPage = lazy(() => import("./pages/ReportsPage"));
const ResearchSessionPage = lazy(() => import("./pages/research/ResearchSessionPage"));
const StatusPage = lazy(() => import("./pages/StatusPage"));
const AutopilotPage = lazy(() => import("./pages/AutopilotPage"));
const PapersPage = lazy(() => import("./pages/PapersPage"));
const ValidatorPage = lazy(() => import("./pages/ValidatorPage"));

function PageFallback() {
  return <div className="space-y-4"><Skeleton className="h-8 w-48" /><Skeleton className="h-40 w-full" /></div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <TopicProvider>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Navigate to="/autopilot" replace />} />
            {/* The research list became part of the Autopilot page (old links and bookmarks still work). */}
            <Route path="/research" element={<Navigate to="/autopilot" replace />} />
            {[
              ["/library", LibraryPage], ["/ask", AskPage], ["/reports", ReportsPage],
              ["/research/:id", ResearchSessionPage], ["/status", StatusPage], ["/autopilot", AutopilotPage], ["/papers", PapersPage], ["/validator", ValidatorPage],
            ].map(([path, Page]) => {
              const Component = Page as React.ComponentType;
              return <Route key={path as string} path={path as string}
                element={<ErrorBoundary resetKey={path as string}><Suspense fallback={<PageFallback />}><Component /></Suspense></ErrorBoundary>} />;
            })}
            <Route path="*" element={<Navigate to="/library" replace />} />
          </Route>
        </Routes>
      </TopicProvider>
    </BrowserRouter>
  );
}
