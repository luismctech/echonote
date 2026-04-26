/**
 * Chat-with-transcript domain types (CU-05).
 *
 * Mirrors `crates/echo-domain/src/ports/chat.rs` (request shapes) and
 * `crates/echo-app/src/use_cases/chat_with_transcript.rs` (event
 * stream). The `ask_about_meeting` IPC command takes a request and
 * pushes events through a `tauri::Channel<ChatEvent>` until a
 * terminal `finished` / `failed` event arrives.
 *
 * The Rust side derives `serde` with `#[serde(tag = "kind",
 * rename_all = "camelCase")]`, so every event on the wire carries a
 * literal `kind` discriminator and camelCase fields. Keep this in
 * sync if the Rust enum gains a new variant — TypeScript will narrow
 * exhaustiveness in `useChat` switch statements.
 */

/** UUIDv7 string identifying a transcript segment cited by the model. */
export type SegmentId = string;

/**
 * Author of a single message in the chat history.
 *
 * The three roles match the OpenAI / llama.cpp chat-template
 * vocabulary; they are folded into the model-specific framing
 * (`<|im_start|>system\n…<|im_end|>` for Qwen) by the backend
 * adapter.
 */
export type ChatRole = "system" | "user" | "assistant";

/** A single message inside the conversation history. */
export type ChatMessage = {
  role: ChatRole;
  /** Plain UTF-8 text. Empty strings are tolerated. */
  content: string;
};

/**
 * Discriminated union of every event the backend pushes through the
 * `Channel<ChatEvent>`.
 *
 * Stream lifecycle:
 *
 * 1. Exactly one `started` (front of the stream).
 * 2. Zero or more `token` events (incremental reply pieces).
 * 3. Exactly one terminal event: `finished` on success, `failed` on
 *    a mid-decode error. After this the channel closes and the
 *    `askAboutMeeting` promise resolves.
 */
export type ChatEvent =
  | {
      kind: "started";
      /**
       * Stable model identifier reported by the chat adapter
       * (currently the GGUF filename without extension). Lets the UI
       * show "powered by …" and stamp provenance on persisted
       * history.
       */
      model: string;
    }
  | {
      kind: "token";
      /** Text added to the assistant reply by this token. */
      delta: string;
    }
  | {
      kind: "finished";
      /**
       * The full reply, exactly as the model emitted it (including
       * the `[seg:UUID]` citation markers — the UI is in charge of
       * formatting them as clickable links).
       */
      text: string;
      /**
       * Segment ids the model cited, in first-mention order, after
       * validation against the meeting's real segments. May be
       * empty even when the reply text contains markers, if every
       * marker pointed at an unknown segment id.
       */
      citations: SegmentId[];
      /**
       * `false` when {@link citations} is empty after validation.
       * Lets the UI surface "respuesta sin citas verificables"
       * without re-checking `citations.length`. An empty list is
       * NOT an error; we never
       * re-prompt the model to add citations.
       */
      hadCitations: boolean;
    }
  | {
      kind: "failed";
      /**
       * Human-readable error message. Already wrapped in
       * `DomainError` before being stringified, so the prefix
       * (`llm:`, `model not loaded`, …) tells the UI whether the
       * issue was OOM, context-overflow, or generic.
       */
      error: string;
    };
