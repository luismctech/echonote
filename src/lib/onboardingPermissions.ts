import type { SystemAudioPermission } from "../types/streaming";

/** Result of the onboarding microphone probe. */
export type MicStatus = "idle" | "checking" | "granted" | "denied" | "deferred";

/** Progress of the system-audio permission request. */
export type SystemAudioStatus = "pending" | "checking" | SystemAudioPermission;

/**
 * What the permissions step does next:
 * - `wait`: a check is still running.
 * - `advance`: the microphone works (or its check was deferred) and
 *   system audio needs nothing more, so the step moves on by itself.
 * - `review`: the user should read why something is off (microphone
 *   failed, or system audio declined and ScreenCaptureKit will be used).
 */
export function permissionsOutcome(
  mic: MicStatus,
  systemAudio: SystemAudioStatus,
): "wait" | "advance" | "review" {
  if (mic === "idle" || mic === "checking") return "wait";
  if (systemAudio === "pending" || systemAudio === "checking") return "wait";
  if (mic === "denied" || systemAudio === "denied") return "review";
  return "advance";
}
