import type { Meeting } from "../types/meeting";
import type { MainView } from "../types/view";

/**
 * Put a meeting reloaded after an edit into the view, when that meeting
 * is still the one open. Keeping the store's copy current matters
 * because later actions (e.g. a title rename) build on it.
 */
export function withFreshMeeting(view: MainView, fresh: Meeting): MainView {
  if (view.kind !== "meeting" || view.id !== fresh.id) return view;
  return { ...view, meeting: fresh };
}
