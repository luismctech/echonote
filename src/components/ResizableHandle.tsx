import { useCallback, useRef } from "react";

/**
 * A horizontal drag handle that sits between two vertically-stacked
 * panels and lets the user resize them by dragging up/down.
 *
 * `ratio` is the fraction of the container allocated to the **top** panel
 * (0 – 1). The caller clamps it to its own min/max.
 */
export function ResizableHandle({
  containerRef,
  ratio,
  onRatioChange,
}: Readonly<{
  containerRef: React.RefObject<HTMLDivElement | null>;
  ratio: number;
  onRatioChange: (r: number) => void;
}>) {
  const dragging = useRef(false);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const y = e.clientY - rect.top;
      onRatioChange(y / rect.height);
    },
    [containerRef, onRatioChange],
  );

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-valuenow={Math.round(ratio * 100)}
      className="group flex h-2 flex-shrink-0 cursor-row-resize items-center justify-center"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <div className="h-0.5 w-12 rounded-full bg-zinc-300 transition-colors group-hover:bg-zinc-400 group-active:bg-emerald-500 dark:bg-zinc-700 dark:group-hover:bg-zinc-500 dark:group-active:bg-emerald-500" />
    </div>
  );
}
