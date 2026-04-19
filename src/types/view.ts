/**
 * Top-level UI-state types shared across the shell, hooks, and views.
 *
 * These describe transient view concerns (probe status, live
 * transcript log lines, which pane is on screen) rather than persisted
 * domain entities. They are kept here so any presentation component
 * can consume them without reaching into `App.tsx`.
 */

import type { HealthStatus } from "./health";
import type { Meeting, MeetingId } from "./meeting";

/** Backend health probe state shown in the header pill. */
export type Probe =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; status: HealthStatus }
  | { kind: "error"; message: string };

/**
 * One row in the live transcript scroller.
 *
 * `chunk` rows carry the transcribed text + diagnostics; `skipped`
 * rows mark a silence window that the pipeline discarded. We keep
 * both so users can see *why* there's a gap without inspecting logs.
 */
export type StreamLine =
  | {
      kind: "chunk";
      key: string;
      chunkIndex: number;
      offsetMs: number;
      text: string;
      language: string | null;
      rtf: number;
      /** 0-based slot of the diarized speaker, or undefined if diarization is off. */
      speakerSlot?: number | undefined;
    }
  | {
      kind: "skipped";
      key: string;
      chunkIndex: number;
      offsetMs: number;
      durationMs: number;
      rms: number;
    };

/** Right-pane mode: live transcription or replay of a stored meeting. */
export type MainView =
  | { kind: "live" }
  | {
      kind: "meeting";
      id: MeetingId;
      meeting: Meeting | null;
      loading: boolean;
      error?: string;
    };
