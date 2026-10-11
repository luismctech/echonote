import { describe, expect, it } from "vitest";

import { LABEL, resolveProvider } from "../ModelProviderLogo";

describe("resolveProvider", () => {
  it("maps Gemma LLMs to the Gemma provider", () => {
    expect(resolveProvider("llm-gemma4-e4b")).toBe("gemma");
    expect(resolveProvider("llm-gemma4-12b")).toBe("gemma");
    expect(LABEL.gemma).toContain("Gemma");
  });

  it("keeps every other LLM on Qwen and the remaining families unchanged", () => {
    expect(resolveProvider("llm-qwen3-8b")).toBe("qwen");
    expect(resolveProvider("asr-whisper-small")).toBe("openai");
    expect(resolveProvider("vad-silero")).toBe("silero");
    expect(resolveProvider("embedder-3dspeaker")).toBe("3dspeaker");
    expect(resolveProvider("segmenter-pyannote")).toBe("pyannote");
  });
});
