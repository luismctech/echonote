import type { Speaker, SpeakerId } from "../../types/speaker";
import { SpeakerEditor } from "./SpeakerEditor";

/**
 * List of diarized speakers with an inline rename input each. Saves
 * on Enter or blur; clearing the input reverts to anonymous.
 */
export function SpeakersPanel({
  speakers,
  onRename,
}: {
  speakers: Speaker[];
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  return (
    <section
      aria-label="Speakers"
      className="flex flex-wrap gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-2 dark:border-zinc-900 dark:bg-zinc-900/40"
    >
      {speakers.map((sp) => (
        <SpeakerEditor key={sp.id} speaker={sp} onRename={onRename} />
      ))}
    </section>
  );
}
