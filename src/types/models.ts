/** Status of a single downloadable model. */
export type ModelInfo = {
  id: string;
  label: string;
  kind: "asr" | "llm" | "vad" | "embedder";
  present: boolean;
  sizeBytes: number;
};

/** Events streamed by the backend during a model download. */
export type DownloadEvent =
  | { kind: "progress"; downloaded: number; total: number }
  | { kind: "finished" }
  | { kind: "failed"; error: string }
  | { kind: "cancelled" };
