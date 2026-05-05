/**
 * `ChatPanel` — interactive Q&A over the meeting transcript.
 *
 * Layout:
 *   ┌─────────────────────────────────────────────┐
 *   │  Chat   ·   model provenance (if known)     │
 *   ├─────────────────────────────────────────────┤
 *   │  scrollable message list                    │
 *   │  (user bubbles right, assistant left)       │
 *   │  streaming text with blinking cursor        │
 *   ├─────────────────────────────────────────────┤
 *   │  ┌──────────────────────────────┐  [Send]   │
 *   │  │  input field                 │           │
 *   │  └──────────────────────────────┘           │
 *   └─────────────────────────────────────────────┘
 *
 * Citation markers (`[seg:UUID]`) in assistant replies are rendered
 * as clickable links that scroll the transcript pane to the cited
 * segment. The parent `MeetingDetail` passes `onScrollToSegment` so
 * the ChatPanel stays decoupled from DOM details.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";
import i18next from "i18next";
import Markdown from "react-markdown";
import { Clock } from "lucide-react";

import type { UseChat, DisplayMessage } from "../../hooks/useChat";
import type { SegmentId } from "../../types/chat";

// Regex to split assistant text into plain text + citation markers.
const CITATION_RE = /\[seg:([0-9a-f-]{36})\]/gi;

// eslint-disable-next-line @typescript-eslint/no-empty-function
const noop = () => {};

/** Format milliseconds as MM:SS for citation chips. */
function formatCitationTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function ChatPanel({
  chat,
  onScrollToSegment,
  segmentTimestamps,
}: Readonly<{
  chat: UseChat;
  onScrollToSegment?: (segmentId: SegmentId) => void;
  segmentTimestamps?: Record<string, number>;
}>) {
  const { t } = useTranslation();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const rafRef = useRef<number | null>(null);

  const scrollToBottom = useCallback(() => {
    // During streaming the text changes on every token (20-50/s).
    // Debounce via rAF so we don't queue competing smooth-scroll
    // animations.
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      messagesEndRef.current?.scrollIntoView({ behavior: "instant" });
    });
  }, []);

  // Clean up any pending rAF on unmount.
  useEffect(() => () => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
  }, []);

  // Auto-scroll when new messages arrive or streaming text changes.
  useEffect(scrollToBottom, [
    chat.messages.length,
    chat.streamingText,
    scrollToBottom,
  ]);

  // Auto-focus the input when streaming finishes.
  useEffect(() => {
    if (!chat.streaming) {
      inputRef.current?.focus();
    }
  }, [chat.streaming]);

  // Destructure stable refs to avoid `chat` object as dependency (it
  // recreates on every render since it's a plain object return).
  const { send, streaming } = chat;

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!input.trim() || streaming) return;
      send(input);
      setInput("");
      scrollToBottom();
    },
    [input, send, streaming, scrollToBottom],
  );

  // Submit on Enter (without Shift). Shift+Enter inserts a newline.
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit],
  );

  const isEmpty = chat.messages.length === 0 && !chat.streaming;

  return (
    <section
      aria-label={t("chat.label")}
      className="flex min-h-0 flex-1 flex-col"
    >
      {/* ── Header bar (never scrolls) ── */}
      <div className="flex flex-shrink-0 items-center justify-between px-1 py-1">
        <span className="type-section-header text-content-placeholder">
          {t("chat.label")}
        </span>
        {chat.model && (
          <span className="text-micro text-content-placeholder">{chat.model}</span>
        )}
      </div>

      {/* ── Scrollable messages area ── */}
      <div className="flex min-h-0 flex-1 flex-col rounded-md border border-subtle bg-surface-sunken">
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3">
          {isEmpty && (
            <p className="py-4 text-center text-ui-sm text-content-placeholder">
              {t("chat.description")}
            </p>
          )}
          {chat.messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              onCitationClick={onScrollToSegment ?? noop}
              {...(segmentTimestamps != null && { segmentTimestamps })}
              showRole={
                i === 0 || chat.messages[i - 1]!.role !== msg.role
              }
            />
          ))}
          {chat.streaming && chat.streamingText && (
            <div className="flex flex-col items-start">
              <div className="max-w-[85%] rounded-lg border border-subtle bg-surface-elevated px-3 py-2 text-ui-md text-content-primary shadow-sm">
                {chat.streamingText}
                <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-content-tertiary" />
              </div>
            </div>
          )}
          {chat.streaming && !chat.streamingText && (
            <div className="flex flex-col items-start">
              <div className="rounded-lg border border-subtle bg-surface-elevated px-3 py-2 text-ui-md text-content-tertiary shadow-sm">
                <span className="inline-flex gap-1">
                  <span className="animate-bounce">·</span>
                  <span className="animate-bounce [animation-delay:150ms]">·</span>
                  <span className="animate-bounce [animation-delay:300ms]">·</span>
                </span>
              </div>
            </div>
          )}
          {chat.error && (
            <p className="text-ui-sm text-amber-700 dark:text-amber-400">
              {chat.error}
            </p>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input area (pinned to bottom) ── */}
        <form onSubmit={handleSubmit} className="flex flex-shrink-0 gap-2 border-t border-subtle p-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t("chat.placeholder")}
            rows={1}
            disabled={chat.streaming}
            className="min-h-[30px] flex-1 resize-none rounded-md border bg-surface-elevated px-2 py-1 text-ui-sm placeholder:text-content-placeholder focus:border-strong focus:outline-none disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={chat.streaming || !input.trim()}
            className="rounded-md border bg-surface-elevated px-3 py-1.5 text-ui-sm font-medium hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-60"
          >
            {t("chat.send")}
          </button>
        </form>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

function MessageBubble({
  message,
  onCitationClick,
  segmentTimestamps,
  showRole,
}: Readonly<{
  message: DisplayMessage;
  onCitationClick?: (segmentId: SegmentId) => void;
  segmentTimestamps?: Record<string, number>;
  showRole?: boolean;
}>) {
  const { t } = useTranslation();
  const isUser = message.role === "user";
  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      {showRole && (
        <span className="mb-0.5 px-1 text-micro font-medium text-content-placeholder">
          {isUser ? t("chat.roleUser") : t("chat.roleAssistant")}
        </span>
      )}
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-ui-md ${
          isUser
            ? "bg-accent-600 text-white"
            : "border border-subtle bg-surface-elevated text-content-primary shadow-sm"
        }`}
      >
        {isUser ? (
          message.content
        ) : (
          <AssistantContent
            text={message.content}
            citations={message.citations ?? []}
            hadCitations={message.hadCitations ?? false}
            onCitationClick={onCitationClick ?? noop}
            {...(segmentTimestamps != null && { segmentTimestamps })}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Citation-aware assistant content
// ---------------------------------------------------------------------------

function AssistantContent({
  text,
  citations,
  hadCitations,
  onCitationClick,
  segmentTimestamps,
}: Readonly<{
  text: string;
  citations?: SegmentId[];
  hadCitations?: boolean;
  onCitationClick?: (segmentId: SegmentId) => void;
  segmentTimestamps?: Record<string, number>;
}>) {
  const validCitations = useMemo(() => new Set(citations ?? []), [citations]);

  // Replace [seg:UUID] markers with placeholder tokens that won't be
  // mangled by the markdown parser, then swap them back post-render.
  const { cleaned, chips } = useMemo(() => {
    const chipMap: Map<string, string> = new Map();
    let idx = 0;
    const result = text.replaceAll(CITATION_RE, (_, segId: string) => {
      const placeholder = `%%CIT${idx}%%`;
      chipMap.set(placeholder, segId);
      idx++;
      return placeholder;
    });
    return { cleaned: result, chips: chipMap };
  }, [text]);

  // Custom component to intercept text nodes and replace placeholders
  // with citation chips.
  const renderChildren = useCallback(
    (children: ReactNode): ReactNode => {
      if (typeof children === "string") {
        return replacePlaceholders(children, chips, validCitations, onCitationClick, segmentTimestamps);
      }
      if (Array.isArray(children)) {
        return children.map((child, idx) => (
          // eslint-disable-next-line react/no-array-index-key
          <span key={idx}>{renderChildren(child)}</span>
        ));
      }
      return children;
    },
    [chips, validCitations, onCitationClick, segmentTimestamps],
  );

  const mdComponents = useMemo(
    () => ({
      p: ({ children }: { children?: ReactNode }) => <p>{renderChildren(children)}</p>,
      li: ({ children }: { children?: ReactNode }) => <li>{renderChildren(children)}</li>,
      strong: ({ children }: { children?: ReactNode }) => <strong>{renderChildren(children)}</strong>,
      em: ({ children }: { children?: ReactNode }) => <em>{renderChildren(children)}</em>,
    }),
    [renderChildren],
  );

  return (
    <div className="prose prose-sm max-w-none [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0 [&_h1]:text-ui-md [&_h2]:text-ui-md [&_h3]:text-ui-sm [&_h4]:text-ui-sm [&_pre]:text-ui-sm">
      <Markdown components={mdComponents}>
        {cleaned}
      </Markdown>
      {hadCitations === false && (
          <span className="mt-1 block text-micro italic text-content-placeholder">
          {i18next.t("chat.noCitations")}
        </span>
      )}
    </div>
  );
}

/** Replace `%%CIT0%%` placeholders with clickable citation chips. */
function replacePlaceholders(
  text: string,
  chips: Map<string, string>,
  validIds: Set<string>,
  onClick?: (id: SegmentId) => void,
  segmentTimestamps?: Record<string, number>,
): ReactNode[] {
  const parts: ReactNode[] = [];
  const placeholderRe = /%%CIT(\d+)%%/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = placeholderRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const placeholder = match[0];
    const segId = chips.get(placeholder);
    if (segId && validIds.has(segId)) {
      const ms = segmentTimestamps?.[segId];
      const label = ms == null ? segId.slice(0, 8) : formatCitationTime(ms);
      parts.push(
        <button
          key={`cit-${match.index}`}
          type="button"
          onClick={() => onClick?.(segId)}
          title={i18next.t("chat.scrollToSegment", { id: label })}
          className="mx-0.5 inline-flex items-center gap-0.5 rounded bg-blue-100 px-1.5 py-0.5 font-mono text-micro text-blue-700 hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
        >
          <Clock className="h-2.5 w-2.5" />
          {label}
        </button>,
      );
    } else if (segId) {
      // Invalid citation — render as plain text
      parts.push(`[seg:${segId}]`);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}
