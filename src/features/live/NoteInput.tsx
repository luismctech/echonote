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
      className="flex items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-800"
    >
      <span className="shrink-0 rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400">
        {formatTimestamp(elapsedMs)}
      </span>
      <input
        ref={inputRef}
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        placeholder={`${t("live.notePlaceholder")} (${isMac ? "⌘N" : "Ctrl+N"})`}
        className="min-w-0 flex-1 bg-transparent text-xs text-zinc-800 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-200 dark:placeholder:text-zinc-500"
      />
      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="shrink-0 rounded bg-emerald-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {t("live.addNote")}
      </button>
    </form>
  );
}
