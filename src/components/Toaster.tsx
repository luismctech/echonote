/**
 * Toast notification layer.
 *
 * Sprint 1 day 1: a tiny, dependency-free toast system. The provider holds
 * a reducer, the hook (`useToast`) gives any component access to `push` /
 * `dismiss`, and the `<Toaster />` component renders the stack.
 *
 * Why hand-rolled instead of `sonner` / `react-hot-toast`:
 * - Bundle stays slim (this is an offline desktop app).
 * - Tauri webview is exactly one user; we don't need queueing across tabs.
 * - We want strict control over copy-to-clipboard for error toasts so users
 *   can paste into a bug report without screenshots.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ToastKind = "info" | "success" | "warning" | "error";

export type Toast = {
  id: string;
  kind: ToastKind;
  message: string;
  /** Optional secondary line — usually a stack/error detail. */
  detail?: string | undefined;
  /** Auto-dismiss after this many ms. `0` keeps it sticky. Defaults: error=0, others=4000. */
  durationMs: number;
  createdAt: number;
};

export type ToastInput = Omit<Toast, "id" | "createdAt" | "durationMs"> & {
  durationMs?: number | undefined;
};

type ToastAction =
  | { type: "PUSH"; toast: Toast }
  | { type: "DISMISS"; id: string }
  | { type: "CLEAR" };

function reducer(state: Toast[], action: ToastAction): Toast[] {
  switch (action.type) {
    case "PUSH":
      // Cap the stack at 5; drop the oldest non-error first.
      if (state.length < 5) return [...state, action.toast];
      const idx = state.findIndex((t) => t.kind !== "error");
      const next = idx >= 0 ? state.filter((_, i) => i !== idx) : state.slice(1);
      return [...next, action.toast];
    case "DISMISS":
      return state.filter((t) => t.id !== action.id);
    case "CLEAR":
      return [];
  }
}

// ---------------------------------------------------------------------------
// Context + hook
// ---------------------------------------------------------------------------

type ToastContextValue = {
  toasts: Toast[];
  push: (toast: ToastInput) => string;
  dismiss: (id: string) => void;
  clear: () => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return ctx;
}

let _toastCounter = 0;
function nextToastId(): string {
  _toastCounter += 1;
  return `t${Date.now().toString(36)}-${_toastCounter}`;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, dispatch] = useReducer(reducer, []);

  const push = useCallback((input: ToastInput): string => {
    const id = nextToastId();
    const toast: Toast = {
      id,
      kind: input.kind,
      message: input.message,
      detail: input.detail,
      durationMs:
        input.durationMs ?? (input.kind === "error" ? 0 : 4_000),
      createdAt: Date.now(),
    };
    dispatch({ type: "PUSH", toast });
    return id;
  }, []);

  const dismiss = useCallback((id: string) => {
    dispatch({ type: "DISMISS", id });
  }, []);

  const clear = useCallback(() => dispatch({ type: "CLEAR" }), []);

  const value = useMemo(
    () => ({ toasts, push, dismiss, clear }),
    [toasts, push, dismiss, clear],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <Toaster />
    </ToastContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

function Toaster() {
  const { toasts, dismiss } = useToast();
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2"
    >
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
      ))}
    </div>
  );
}

function ToastCard({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: () => void;
}) {
  // Auto-dismiss timer (only for non-sticky toasts).
  useEffect(() => {
    if (toast.durationMs <= 0) return;
    const t = setTimeout(onDismiss, toast.durationMs);
    return () => clearTimeout(t);
  }, [toast.durationMs, onDismiss]);

  const copyRef = useRef<HTMLButtonElement>(null);
  const onCopy = async () => {
    const payload = toast.detail
      ? `${toast.message}\n\n${toast.detail}`
      : toast.message;
    try {
      await navigator.clipboard.writeText(payload);
      if (copyRef.current) {
        copyRef.current.textContent = "copied";
        setTimeout(() => {
          if (copyRef.current) copyRef.current.textContent = "copy";
        }, 1_500);
      }
    } catch {
      // Clipboard may not be available in some Tauri configurations.
    }
  };

  return (
    <div
      role={toast.kind === "error" ? "alert" : "status"}
      className={`pointer-events-auto rounded-md border px-3 py-2 text-xs shadow-md backdrop-blur ${kindClasses(toast.kind)}`}
    >
      <div className="flex items-start gap-2">
        <span aria-hidden className="mt-0.5 font-mono">
          {kindGlyph(toast.kind)}
        </span>
        <div className="flex flex-1 flex-col gap-1">
          <p className="font-medium leading-snug">{toast.message}</p>
          {toast.detail && (
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-black/5 p-1.5 font-mono text-[10px] leading-tight dark:bg-white/5">
              {toast.detail}
            </pre>
          )}
          {toast.kind === "error" && (
            <div className="flex gap-2 pt-0.5">
              <button
                ref={copyRef}
                type="button"
                onClick={onCopy}
                className="rounded border border-current/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wide opacity-80 hover:opacity-100"
              >
                copy
              </button>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss notification"
          className="-m-1 rounded p-1 opacity-60 hover:opacity-100"
        >
          ×
        </button>
      </div>
    </div>
  );
}

function kindClasses(kind: ToastKind): string {
  switch (kind) {
    case "info":
      return "border-zinc-300 bg-white/95 text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900/95 dark:text-zinc-100";
    case "success":
      return "border-emerald-300 bg-emerald-50/95 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-100";
    case "warning":
      return "border-amber-300 bg-amber-50/95 text-amber-900 dark:border-amber-800 dark:bg-amber-950/80 dark:text-amber-100";
    case "error":
      return "border-rose-300 bg-rose-50/95 text-rose-900 dark:border-rose-800 dark:bg-rose-950/80 dark:text-rose-100";
  }
}

function kindGlyph(kind: ToastKind): string {
  switch (kind) {
    case "info":
      return "i";
    case "success":
      return "✓";
    case "warning":
      return "!";
    case "error":
      return "✗";
  }
}
