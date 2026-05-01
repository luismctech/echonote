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

import type { UseChat, DisplayMessage } from "../../hooks/useChat";
import type { SegmentId } from "../../types/chat";

// Regex to split assistant text into plain text + citation markers.
const CITATION_RE = /\[seg:([0-9a-f-]{36})\]/gi;

// eslint-disable-next-line @typescript-eslint/no-empty-function
const noop = () => {};

export function ChatPanel({
  chat,
  onScrollToSegment,
}: Readonly<{
  chat: UseChat;
  onScrollToSegment?: (segmentId: SegmentId) => void;
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
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
          {t("chat.label")}
        </span>
        {chat.model && (
          <span className="text-[10px] text-zinc-400">{chat.model}</span>
        )}
      </div>

      {/* ── Scrollable messages area ── */}
      <div className="flex min-h-0 flex-1 flex-col rounded-md border border-zinc-100 bg-zinc-50 dark:border-zinc-900 dark:bg-zinc-900/40">
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3">
          {isEmpty && (
            <p className="py-4 text-center text-xs text-zinc-400">
              {t("chat.description")}
            </p>
          )}
          {chat.messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              onCitationClick={onScrollToSegment ?? noop}
            />
          ))}
          {chat.streaming && chat.streamingText && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-lg bg-zinc-100 px-3 py-2 text-sm text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200">
                {chat.streamingText}
                <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-zinc-500" />
              </div>
            </div>
          )}
          {chat.streaming && !chat.streamingText && (
            <div className="flex justify-start">
              <div className="rounded-lg bg-zinc-100 px-3 py-2 text-sm text-zinc-500 dark:bg-zinc-800">
                <span className="inline-flex gap-1">
                  <span className="animate-bounce">·</span>
                  <span className="animate-bounce [animation-delay:150ms]">·</span>
                  <span className="animate-bounce [animation-delay:300ms]">·</span>
                </span>
              </div>
            </div>
          )}
          {chat.error && (
            <p className="text-xs text-amber-700 dark:text-amber-400">
              {chat.error}
            </p>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input area (pinned to bottom) ── */}
        <form onSubmit={handleSubmit} className="flex flex-shrink-0 gap-2 border-t border-zinc-100 p-2 dark:border-zinc-800">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t("chat.placeholder")}
            rows={1}
            disabled={chat.streaming}
            className="min-h-[36px] flex-1 resize-none rounded-md border border-zinc-200 bg-white px-2 py-1.5 text-sm placeholder:text-zinc-400 focus:border-zinc-400 focus:outline-none disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-900 dark:placeholder:text-zinc-600 dark:focus:border-zinc-500"
          />
          <button
            type="submit"
            disabled={chat.streaming || !input.trim()}
            className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800"
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
}: Readonly<{
  message: DisplayMessage;
  onCitationClick?: (segmentId: SegmentId) => void;
}>) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
          isUser
            ? "bg-zinc-800 text-zinc-100 dark:bg-zinc-200 dark:text-zinc-900"
            : "bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
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
}: Readonly<{
  text: string;
  citations?: SegmentId[];
  hadCitations?: boolean;
  onCitationClick?: (segmentId: SegmentId) => void;
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
        return replacePlaceholders(children, chips, validCitations, onCitationClick);
      }
      if (Array.isArray(children)) {
        return children.map((child, idx) => (
          // eslint-disable-next-line react/no-array-index-key
          <span key={idx}>{renderChildren(child)}</span>
        ));
      }
      return children;
    },
    [chips, validCitations, onCitationClick],
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
    <div className="prose prose-sm prose-zinc dark:prose-invert max-w-none [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0 [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-xs [&_h4]:text-xs [&_pre]:text-xs">
      <Markdown components={mdComponents}>
        {cleaned}
      </Markdown>
      {hadCitations === false && (
        <span className="mt-1 block text-[10px] italic text-zinc-400">
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
      parts.push(
        <button
          key={`cit-${match.index}`}
          type="button"
          onClick={() => onClick?.(segId)}
          title={i18next.t("chat.scrollToSegment", { id: segId.slice(0, 8) })}
          className="mx-0.5 inline-flex items-center rounded bg-blue-100 px-1 py-0.5 font-mono text-[10px] text-blue-700 hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
        >
          {segId.slice(0, 8)}
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
