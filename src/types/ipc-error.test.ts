import { describe, expect, it } from "vitest";

import { errorMessage, userErrorMessage } from "./ipc-error";

describe("errorMessage", () => {
  it("reads the message of a structured IPC error instead of [object Object]", () => {
    expect(
      errorMessage({ code: "modelNotReady", message: "No language model installed.", retriable: false }),
    ).toBe("No language model installed.");
  });

  it("keeps Error messages and plain strings", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("plain text")).toBe("plain text");
  });

  it("never renders [object Object] for other values", () => {
    expect(errorMessage({ message: "partial shape" })).toBe("partial shape");
    expect(errorMessage({ foo: 1 })).toBe('{"foo":1}');
    expect(errorMessage(undefined)).toBe("undefined");
  });
});

describe("userErrorMessage", () => {
  const t = (key: string) => `t:${key}`;

  it("explains a missing language model in the UI language", () => {
    expect(
      userErrorMessage({ code: "modelNotReady", message: "No language model installed.", retriable: false }, t),
    ).toBe("t:errors.noLanguageModel");
  });

  it("falls back to the error's own message otherwise", () => {
    expect(userErrorMessage({ code: "llm", message: "context overflow", retriable: true }, t)).toBe(
      "context overflow",
    );
    expect(userErrorMessage(new Error("boom"), t)).toBe("boom");
  });
});
