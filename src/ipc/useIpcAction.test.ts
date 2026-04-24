/**
 * Unit tests for the pure wrapping logic behind `useIpcAction`.
 *
 * We test `runIpcAction` directly (no React, no jsdom) so the suite
 * stays as cheap as the existing `recording.test.ts` and `speakers.test.ts`.
 * The hook itself is a 4-line `useCallback` shim; once you trust the
 * wrapper invariants below, the hook trivially upholds them.
 */

import { describe, expect, it, vi } from "vitest";

import { runIpcAction } from "./useIpcAction";
import type { IpcError } from "../types/ipc-error";

describe("runIpcAction", () => {
  it("returns the resolved value when the IPC call succeeds", async () => {
    const push = vi.fn();
    const fn = vi.fn(async (a: number, b: number) => a + b);

    const result = await runIpcAction("never used", fn, push, [2, 3]);

    expect(result).toBe(5);
    expect(fn).toHaveBeenCalledWith(2, 3);
    expect(push).not.toHaveBeenCalled();
  });

  it("pushes an error toast and resolves to undefined when the call throws", async () => {
    const push = vi.fn();
    const boom = new Error("kaput");
    const fn = vi.fn(async () => {
      throw boom;
    });

    const result = await runIpcAction("Couldn't do X.", fn, push, []);

    expect(result).toBeUndefined();
    expect(push).toHaveBeenCalledTimes(1);
    expect(push).toHaveBeenCalledWith({
      kind: "error",
      message: "Couldn't do X.",
      detail: "kaput",
    });
  });

  it("stringifies non-Error rejections into the toast detail", async () => {
    const push = vi.fn();
    // Backends sometimes throw plain strings or numbers across the IPC
    // boundary; the wrapper must still produce a usable detail.
    const fn = vi.fn(async () => {
      throw "raw-string-failure"; // eslint-disable-line @typescript-eslint/no-throw-literal
    });

    const result = await runIpcAction("label", fn, push, []);

    expect(result).toBeUndefined();
    expect(push).toHaveBeenCalledWith({
      kind: "error",
      message: "label",
      detail: "raw-string-failure",
    });
  });

  it("does not swallow synchronous throws inside the IPC function", async () => {
    const push = vi.fn();
    const fn = vi.fn((() => {
      throw new Error("sync");
    }) as unknown as () => Promise<number>);

    const result = await runIpcAction("label", fn, push, []);

    expect(result).toBeUndefined();
    expect(push).toHaveBeenCalledWith({
      kind: "error",
      message: "label",
      detail: "sync",
    });
  });

  it("extracts message from a structured IpcError thrown by the backend", async () => {
    const push = vi.fn();
    const ipcErr: IpcError = {
      code: "notFound",
      message: "meeting abc123 not found",
      retriable: false,
    };
    const fn = vi.fn(async () => {
      throw ipcErr;
    });

    const result = await runIpcAction("Couldn't load meeting.", fn, push, []);

    expect(result).toBeUndefined();
    expect(push).toHaveBeenCalledWith({
      kind: "error",
      message: "Couldn't load meeting.",
      detail: "meeting abc123 not found",
    });
  });
});
