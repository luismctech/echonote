/**
 * `useChat` — manages the chat-with-transcript conversation for one
 * meeting.
 *
 * Owns the view-state the `ChatPanel` needs:
 *
 *   - `messages`    the full conversation history (user + assistant
 *                   turns), newest last
 *   - `streaming`   true while the backend is emitting tokens
 *   - `error`       the latest error (toast for IPC failures,
 *                   inline for mid-stream `Failed` events)
 *   - `model`       the model id reported by the `started` event
 *   - `send(text)`  submit a new user question; returns once the
 *                   stream terminates
 *
 * Reset semantics: passing a different `meetingId` wipes the entire
 * conversation — same "re-mount on meeting change" contract as
 * `useMeetingSummary`.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { askAboutMeeting } from "../ipc/client";
import type { ChatEvent, ChatMessage, SegmentId } from "../types/chat";
import type { MeetingId } from "../types/meeting";

/** A single message in the local conversation history. */
export type DisplayMessage = {
  role: "user" | "assistant";
  content: string;
  /** Validated segment citations (assistant turns only). */
  citations?: SegmentId[];
  /** Whether the model included any citations at all. */
  hadCitations?: boolean;
};

export type UseChat = {
  messages: DisplayMessage[];
  streaming: boolean;
  streamingText: string;
  error: string | null;
  model: string | null;
  send: (text: string) => void;
};

export function useChat(meetingId: MeetingId | null): UseChat {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);

  // Track active meeting so stale streams from a previous meeting
  // can't pollute the current one.
  const activeMeetingRef = useRef<MeetingId | null>(null);

  // Accumulated text inside the current streaming turn. Ref so the
  // event callback (closed over once per `send`) always reads the
  // latest value without re-closing.
  const accRef = useRef("");

  // Keep a ref to the latest messages so `send` can read the current
  // history without depending on the `messages` array (which changes
  // on every assistant turn and would destabilize the callback).
  const messagesRef = useRef<DisplayMessage[]>([]);

  useEffect(() => {
    activeMeetingRef.current = meetingId;
    setMessages([]);
    messagesRef.current = [];
    setStreaming(false);
    setStreamingText("");
    setError(null);
    setModel(null);
  }, [meetingId]);

  const send = useCallback(
    (text: string) => {
      if (!meetingId || streaming) return;
      const trimmed = text.trim();
      if (!trimmed) return;

      setError(null);
      setStreaming(true);
      setStreamingText("");
      accRef.current = "";

      const userMsg: DisplayMessage = { role: "user", content: trimmed };
      setMessages((prev) => {
        const next = [...prev, userMsg];
        messagesRef.current = next;
        return next;
      });

      // Build the history the backend expects: only user/assistant
      // turns (no system messages — the backend assembles those from
      // the transcript). We include the NEW user message as well,
      // because the backend expects `question` as a separate param.
      const history: ChatMessage[] = messagesRef.current.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const onEvent = (event: ChatEvent) => {
        if (activeMeetingRef.current !== meetingId) return;
        switch (event.kind) {
          case "started":
            setModel(event.model);
            break;
          case "token":
            accRef.current += event.delta;
            setStreamingText(accRef.current);
            break;
          case "finished": {
            const assistantMsg: DisplayMessage = {
              role: "assistant",
              content: event.text,
              citations: event.citations,
              hadCitations: event.hadCitations,
            };
            setMessages((prev) => {
              const next = [...prev, assistantMsg];
              messagesRef.current = next;
              return next;
            });
            setStreaming(false);
            setStreamingText("");
            break;
          }
          case "failed":
            setError(event.error);
            setStreaming(false);
            setStreamingText("");
            break;
        }
      };

      askAboutMeeting(meetingId, history, trimmed, onEvent).catch((err) => {
        if (activeMeetingRef.current !== meetingId) return;
        const detail = err instanceof Error ? err.message : String(err);
        setError(detail);
        setStreaming(false);
        setStreamingText("");
      });
    },
    [meetingId, streaming],
  );

  return { messages, streaming, streamingText, error, model, send };
}
