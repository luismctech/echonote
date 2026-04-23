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
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";

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
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!input.trim() || chat.streaming) return;
      chat.send(input);
      setInput("");
    },
    [input, chat],
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
      aria-label="Chat"
      className="flex flex-col gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-900 dark:bg-zinc-900"
    >
      {/* Header */}
      <header className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">Chat</h3>
        {chat.model && (
          <span className="text-[10px] text-zinc-400">{chat.model}</span>
        )}
      </header>

      {/* Messages area */}
      <div className="flex min-h-[120px] max-h-[320px] flex-col gap-2 overflow-y-auto rounded-md border border-zinc-100 bg-white p-2 dark:border-zinc-800 dark:bg-zinc-950">
        {isEmpty && (
          <p className="py-4 text-center text-xs text-zinc-400">
            Ask a question about this meeting's transcript.
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
                <span className="animate-bounce [animation-delay:150ms]">
                  ·
                </span>
                <span className="animate-bounce [animation-delay:300ms]">
                  ·
                </span>
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

      {/* Input area */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about this meeting…"
          rows={1}
          disabled={chat.streaming}
          className="min-h-[36px] flex-1 resize-none rounded-md border border-zinc-200 bg-white px-2 py-1.5 text-sm placeholder:text-zinc-400 focus:border-zinc-400 focus:outline-none disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-900 dark:placeholder:text-zinc-600 dark:focus:border-zinc-500"
        />
        <button
          type="submit"
          disabled={chat.streaming || !input.trim()}
          className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800"
        >
          Send
        </button>
      </form>
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
  const validCitations = new Set(citations ?? []);
  const parts = splitWithCitations(text, validCitations, onCitationClick);

  return (
    <span>
      {parts}
      {hadCitations === false && (
        <span className="mt-1 block text-[10px] italic text-zinc-400">
          respuesta sin citas verificables
        </span>
      )}
    </span>
  );
}

/**
 * Split text into plain runs + clickable citation chips.
 *
 * Only citations that appear in `validIds` (the set the backend
 * validated against real segments) become interactive links; unknown
 * markers are rendered as plain text.
 */
function splitWithCitations(
  text: string,
  validIds: Set<string>,
  onClick?: (id: SegmentId) => void,
): ReactNode[] {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  // Reset the regex (global flag) before iterating.
  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const segId: string | undefined = match[1];
    if (segId && validIds.has(segId)) {
      parts.push(
        <button
          key={`cit-${match.index}`}
          type="button"
          onClick={() => onClick?.(segId)}
          title={`Scroll to segment ${segId.slice(0, 8)}…`}
          className="mx-0.5 inline-flex items-center rounded bg-blue-100 px-1 py-0.5 font-mono text-[10px] text-blue-700 hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
        >
          {segId.slice(0, 8)}
        </button>,
      );
    } else {
      parts.push(match[0]);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}
