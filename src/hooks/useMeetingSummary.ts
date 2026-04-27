/**
 * `useMeetingSummary` — load + generate the LLM summary for one meeting.
 *
 * Owns the small piece of view-state the `SummaryPanel` needs:
 *
 *   - `summary`     the most recent `Summary` for the meeting, or
 *                    `null` when none has been generated yet
 *   - `loading`     true while the initial `getSummary` is in flight
 *   - `generating`  true while `summarize_meeting` is running on the
 *                    backend (typically several seconds — the UI must
 *                    render a spinner instead of a frozen panel)
 *   - `error`       the latest *load* error, surfaced inline in the
 *                    panel; generation errors go straight to a toast
 *                    via `useIpcAction` (no need to mirror them here)
 *   - `generate()`  trigger a (re)generate; returns the new summary
 *                    on success, `undefined` on failure
 *   - `reset()`     drop local state, used when the active meeting
 *                    changes (the parent owns "which meeting is open")
 *
 * The hook treats meeting changes as full re-mounts: pass a different
 * `meetingId` and the loading/error/summary fields all re-initialise.
 * That matches how `MeetingDetail` is keyed, and keeps the state
 * machine inside the hook trivial.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getSummary, summarizeMeeting, summarizeWithCustomTemplate } from "../ipc/client";
import { useIpcAction } from "../ipc/useIpcAction";
import type { MeetingId } from "../types/meeting";
import type { Summary, TemplateId } from "../types/summary";
import type { CustomTemplateId } from "../types/custom-template";

/** The template selector can pick a built-in or a custom template. */
export type SelectedTemplate =
  | { kind: "builtin"; id: TemplateId }
  | { kind: "custom"; id: CustomTemplateId; name: string };

export type UseMeetingSummary = {
  summary: Summary | null;
  loading: boolean;
  generating: boolean;
  error: string | null;
  selectedTemplate: SelectedTemplate;
  setSelectedTemplate: (t: SelectedTemplate) => void;
  generate: () => Promise<Summary | undefined>;
};

export function useMeetingSummary(meetingId: MeetingId | null): UseMeetingSummary {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<SelectedTemplate>({
    kind: "builtin",
    id: "general",
  });

  const requestedRef = useRef<MeetingId | null>(null);

  useEffect(() => {
    requestedRef.current = meetingId;
    if (!meetingId) {
      setSummary(null);
      setLoading(false);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    setSummary(null);
    let cancelled = false;
    (async () => {
      try {
        const fetched = await getSummary(meetingId);
        if (cancelled || requestedRef.current !== meetingId) return;
        setSummary(fetched);
        // Sync the selector to the loaded template so "Regenerate"
        // targets the right one.
        if (fetched) {
          if (fetched.template === "custom") {
            setSelectedTemplate({
              kind: "custom",
              id: "", // id unknown from stored summary
              name: (fetched as { templateName?: string }).templateName ?? "Custom",
            });
          } else {
            setSelectedTemplate({ kind: "builtin", id: fetched.template as TemplateId });
          }
        }
      } catch (err) {
        if (cancelled || requestedRef.current !== meetingId) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        if (!cancelled && requestedRef.current === meetingId) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  const generateBuiltinCall = useIpcAction(
    "Couldn't generate summary.",
    summarizeMeeting,
  );

  const generateCustomCall = useIpcAction(
    "Couldn't generate summary.",
    summarizeWithCustomTemplate,
  );

  const generate = useCallback(async (): Promise<Summary | undefined> => {
    if (!meetingId) return undefined;
    setGenerating(true);
    try {
      let fresh: Summary | undefined;
      if (selectedTemplate.kind === "custom") {
        fresh = await generateCustomCall(meetingId, selectedTemplate.id);
      } else {
        fresh = await generateBuiltinCall(meetingId, selectedTemplate.id);
      }
      if (fresh && requestedRef.current === meetingId) {
        setSummary(fresh);
      }
      return fresh;
    } finally {
      setGenerating(false);
    }
  }, [meetingId, selectedTemplate, generateBuiltinCall, generateCustomCall]);

  return {
    summary,
    loading,
    generating,
    error,
    selectedTemplate,
    setSelectedTemplate,
    generate,
  };
}
