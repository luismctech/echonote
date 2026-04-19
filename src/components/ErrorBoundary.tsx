/**
 * Top-level React error boundary.
 *
 * Sprint 1 day 1: protects the whole app from a render-time crash blanking
 * the webview. Renders a fallback that:
 *   - shows what went wrong with a copyable stack,
 *   - links to the bug tracker pre-filled with the error,
 *   - offers a hard reload button.
 *
 * Class component because React still gates `componentDidCatch` /
 * `getDerivedStateFromError` to class components.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  /** Optional custom fallback. Defaults to {@link DefaultFallback}. */
  fallback?: (
    error: Error,
    info: ErrorInfo | null,
    reset: () => void,
  ) => ReactNode;
  children: ReactNode;
};

type State = {
  error: Error | null;
  info: ErrorInfo | null;
};

const INITIAL_STATE: State = { error: null, info: null };

export class ErrorBoundary extends Component<Props, State> {
  override state: State = INITIAL_STATE;

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ info });
    // Log to the webview console — Tauri DevTools (debug build) will pick it up.
    console.error("[ErrorBoundary]", error, info);
  }

  reset = (): void => {
    this.setState(INITIAL_STATE);
  };

  override render(): ReactNode {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) {
      return this.props.fallback(error, info, this.reset);
    }
    return <DefaultFallback error={error} info={info} onReset={this.reset} />;
  }
}

// ---------------------------------------------------------------------------
// Default fallback UI
// ---------------------------------------------------------------------------

const ISSUE_URL = "https://github.com/AlbertoMZCruz/echonote/issues/new";

function DefaultFallback({
  error,
  info,
  onReset,
}: {
  error: Error;
  info: ErrorInfo | null;
  onReset: () => void;
}) {
  const fullDetail = formatErrorDetail(error, info);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(fullDetail);
    } catch {
      // Clipboard may not be available; fall through silently.
    }
  };

  const onReload = () => {
    window.location.reload();
  };

  const onReport = () => {
    const body = encodeURIComponent(
      `**What happened?**\n<describe what you were doing>\n\n**Error**\n\`\`\`\n${fullDetail}\n\`\`\`\n`,
    );
    const title = encodeURIComponent(
      `[crash] ${error.message.slice(0, 80)}`,
    );
    window.open(
      `${ISSUE_URL}?title=${title}&body=${body}`,
      "_blank",
      "noopener,noreferrer",
    );
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col items-start gap-6 px-6 py-12">
      <header className="flex flex-col gap-1">
        <p className="font-mono text-xs uppercase tracking-wider text-rose-600 dark:text-rose-400">
          fatal — render error
        </p>
        <h1 className="text-2xl font-semibold tracking-tight">
          Something broke in the EchoNote UI.
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          The current session is safe — your meetings on disk are untouched.
          You can copy the error and try again, or open a bug report.
        </p>
      </header>

      <section className="w-full rounded-md border border-rose-200 bg-rose-50 p-3 dark:border-rose-900 dark:bg-rose-950/40">
        <p className="font-mono text-xs font-semibold text-rose-900 dark:text-rose-200">
          {error.name}: {error.message}
        </p>
      </section>

      <details className="w-full rounded-md border border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
        <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">
          Stack trace + component tree
        </summary>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-all border-t border-zinc-200 p-3 font-mono text-[10px] leading-tight text-zinc-700 dark:border-zinc-800 dark:text-zinc-300">
          {fullDetail}
        </pre>
      </details>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onReset}
          className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500"
        >
          Try again
        </button>
        <button
          type="button"
          onClick={onReload}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          Reload window
        </button>
        <button
          type="button"
          onClick={onCopy}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          Copy error
        </button>
        <button
          type="button"
          onClick={onReport}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          Report bug
        </button>
      </div>
    </main>
  );
}

function formatErrorDetail(error: Error, info: ErrorInfo | null): string {
  const stack = error.stack ?? `${error.name}: ${error.message}`;
  if (!info) return stack;
  return `${stack}\n\nComponent stack:${info.componentStack ?? "<unavailable>"}`;
}
