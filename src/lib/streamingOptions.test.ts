import { describe, expect, it } from "vitest";

import { streamingOptionsFor } from "./streamingOptions";

describe("streamingOptionsFor", () => {
  it("asks the backend to save the audio only when the user opted in", () => {
    expect(streamingOptionsFor({ language: "", diarize: false, saveAudio: true }).saveAudio).toBe(
      true,
    );
    expect(streamingOptionsFor({ language: "", diarize: false, saveAudio: false }).saveAudio).toBe(
      false,
    );
  });

  it("keeps the existing diarize, language and source options", () => {
    expect(
      streamingOptionsFor({ language: " es ", diarize: true, saveAudio: false, source: "mixed" }),
    ).toEqual({
      chunkMs: 5_000,
      silenceRmsThreshold: 0.005,
      diarize: true,
      saveAudio: false,
      language: "es",
      source: "mixed",
    });
    expect(
      streamingOptionsFor({
        language: "",
        diarize: false,
        saveAudio: false,
        source: "microphone",
      }),
    ).toEqual({ chunkMs: 5_000, silenceRmsThreshold: 0.005, diarize: false, saveAudio: false });
  });
});
