import { useEffect, useRef, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";

const isMac =
  typeof navigator !== "undefined" &&
  /mac|iphone|ipad/i.test(navigator.userAgent);

/** Format milliseconds as "MM:SS". */
function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function NoteInput({
  elapsedMs,
  onSubmit,
  disabled,
}: Readonly<{
  /** Current recording elapsed time in ms (for the timestamp badge). */
  elapsedMs: number;
  /** Called with the note text when the user submits. */
  onSubmit: (text: string) => void;
  /** Whether the input is disabled (not recording). */
  disabled: boolean;
}>) {
  const { t } = useTranslation();
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Cmd+N (macOS) / Ctrl+N (other) focuses the note input.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((isMac ? e.metaKey : e.ctrlKey) && e.key === "n") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
    setText("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center gap-2 rounded-md border bg-surface-elevated px-3 py-2"
    >
      <span className="shrink-0 rounded bg-surface-inset px-1.5 py-0.5 font-mono text-micro text-content-tertiary">
        {formatTimestamp(elapsedMs)}
      </span>
      <input
        ref={inputRef}
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        placeholder={`${t("live.notePlaceholder")} (${isMac ? "⌘N" : "Ctrl+N"})`}
        className="min-w-0 flex-1 bg-transparent text-ui-sm text-content-primary placeholder:text-content-placeholder focus:outline-none"
      />
      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="shrink-0 rounded bg-emerald-600 px-2 py-1 text-micro font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {t("live.addNote")}
      </button>
    </form>
  );
}
