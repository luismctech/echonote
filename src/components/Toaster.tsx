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
import { useTranslation } from "react-i18next";

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
  /** Optional action button rendered inline. */
  action?: { label: string; onClick: () => void } | undefined;
  /** Auto-dismiss after this many ms. `0` keeps it sticky. Defaults: error=0, others=4000. */
  durationMs: number;
  createdAt: number;
};

export type ToastInput = Omit<Toast, "id" | "createdAt" | "durationMs"> & {
  durationMs?: number | undefined;
};

type ToastAction =
  | { type: "PUSH"; toast: Toast }
  | { type: "UPDATE"; id: string; fields: Partial<Omit<Toast, "id" | "createdAt">> }
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
    case "UPDATE":
      return state.map((t) =>
        t.id === action.id ? { ...t, ...action.fields } : t,
      );
    case "DISMISS":
      return state.filter((t) => t.id !== action.id);
    case "CLEAR":
      return [];
  }
}

// ---------------------------------------------------------------------------
// Context + hook
// ---------------------------------------------------------------------------
//
// Two separate contexts on purpose:
//   - `ToastApiContext` is the *write* surface (`push` / `dismiss` / `clear`).
//     Its value reference is stable across renders because all callbacks come
//     from `useReducer`'s dispatch, which React guarantees is identity-stable.
//   - `ToastListContext` is the *read* surface (the `Toast[]`). This changes
//     every time a toast is pushed or dismissed.
//
// We split them because consumers should be able to depend on the *api*
// (e.g. `useEffect(..., [toast])` to fire a one-shot notification) without
// re-running every time the toast list mutates. A combined context bit us
// in Sprint 1 day 8: pushing one toast caused effects keyed on `toast` to
// re-fire, which pushed *another* toast — infinite loop until the reducer's
// 5-toast cap clamped it, plus the auto-dismiss timer kept resetting because
// `<ToastCard>` was re-rendered on every state change.

type ToastApi = {
  push: (toast: ToastInput) => string;
  update: (id: string, fields: Partial<Omit<Toast, "id" | "createdAt">>) => void;
  dismiss: (id: string) => void;
  clear: () => void;
};

const ToastApiContext = createContext<ToastApi | null>(null);
const ToastListContext = createContext<Toast[]>([]);

/**
 * Stable handle to push, dismiss, and clear toasts. The returned object is
 * referentially stable across renders, so it is safe to put in `useEffect`
 * dependency arrays without causing re-runs.
 */
export function useToast(): ToastApi {
  const ctx = useContext(ToastApiContext);
  if (!ctx) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return ctx;
}

/**
 * Read-only view of the current toast stack. Only `<Toaster />` should need
 * this; expose it for tests too.
 */
export function useToastList(): Toast[] {
  return useContext(ToastListContext);
}

let _toastCounter = 0;
function nextToastId(): string {
  _toastCounter += 1;
  return `t${Date.now().toString(36)}-${_toastCounter}`;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, dispatch] = useReducer(reducer, []);

  // All three callbacks are wrapped in `useCallback` with no dependencies on
  // changing values, so `api` itself never changes identity. That's the whole
  // point of the API/list split — see the long comment above.
  const push = useCallback((input: ToastInput): string => {
    const id = nextToastId();
    const toast: Toast = {
      id,
      kind: input.kind,
      message: input.message,
      detail: input.detail,
      action: input.action,
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

  const update = useCallback(
    (id: string, fields: Partial<Omit<Toast, "id" | "createdAt">>) => {
      dispatch({ type: "UPDATE", id, fields });
    },
    [],
  );

  const clear = useCallback(() => dispatch({ type: "CLEAR" }), []);

  const api = useMemo<ToastApi>(
    () => ({ push, update, dismiss, clear }),
    [push, update, dismiss, clear],
  );

  return (
    <ToastApiContext.Provider value={api}>
      <ToastListContext.Provider value={toasts}>
        {children}
        <Toaster />
      </ToastListContext.Provider>
    </ToastApiContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

function Toaster() {
  const toasts = useToastList();
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2"
    >
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} />
      ))}
    </div>
  );
}

function ToastCard({ toast }: { toast: Toast }) {
  const { t } = useTranslation();
  // Pull `dismiss` from context instead of receiving it as a prop. That way
  // the auto-dismiss `useEffect` below depends only on values that are
  // intrinsic to this toast (id, durationMs) plus the stable `dismiss`
  // reference — so the timer is set up exactly once per toast and won't be
  // torn down + reinstalled when a sibling toast is added or removed.
  const { dismiss } = useToast();

  useEffect(() => {
    if (toast.durationMs <= 0) return;
    const t = setTimeout(() => dismiss(toast.id), toast.durationMs);
    return () => clearTimeout(t);
  }, [toast.id, toast.durationMs, dismiss]);

  const copyRef = useRef<HTMLButtonElement>(null);
  const onCopy = async () => {
    const payload = toast.detail
      ? `${toast.message}\n\n${toast.detail}`
      : toast.message;
    try {
      await navigator.clipboard.writeText(payload);
      if (copyRef.current) {
        copyRef.current.textContent = t("toast.copied");
        setTimeout(() => {
          if (copyRef.current) copyRef.current.textContent = t("toast.copy");
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
                {t("toast.copy")}
              </button>
            </div>
          )}
          {toast.action && (
            <div className="flex gap-2 pt-0.5">
              <button
                type="button"
                onClick={() => {
                  toast.action!.onClick();
                  dismiss(toast.id);
                }}
                className="rounded border border-current/30 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide opacity-80 hover:opacity-100"
              >
                {toast.action.label}
              </button>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={() => dismiss(toast.id)}
          aria-label={t("toast.dismiss")}
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
