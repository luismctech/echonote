import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastProvider } from "./components/Toaster";
import { MeetingsProvider } from "./state/useMeetingsStore";
import "./i18n"; // ← initialise i18next before any component renders

// Self-hosted fonts — DESIGN.md §4.1 (no external CDN)
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";

import "./index.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root missing from index.html");

// Provider order matters:
//   ErrorBoundary  →  catches render-time crashes from any child
//   ToastProvider  →  toast API must exist before any consumer mounts
//   MeetingsProvider →  reads useToast() via useIpcAction internally
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <MeetingsProvider>
          <App />
        </MeetingsProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
