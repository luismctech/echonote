import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export interface MenuItem {
  label: string;
  icon?: ReactNode;
  danger?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

interface Position { x: number; y: number }

/**
 * `<ContextMenu />` — right-click context menu rendered as a portal.
 *
 * Wrap any element; right-click triggers the menu.
 * Closes on click-outside, Escape, or item selection.
 */
export function ContextMenu({
  items,
  children,
}: Readonly<{
  items: MenuItem[];
  children: ReactNode;
}>) {
  const [pos, setPos] = useState<Position | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setPos({ x: e.clientX, y: e.clientY });
    },
    [],
  );

  const close = useCallback(() => setPos(null), []);

  // Close on click outside or Escape
  useEffect(() => {
    if (!pos) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        close();
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [pos, close]);

  // Adjust position so menu doesn't overflow the viewport
  useEffect(() => {
    if (!pos || !menuRef.current) return;
    const rect = menuRef.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let { x, y } = pos;
    if (x + rect.width > vw - 8) x = vw - rect.width - 8;
    if (y + rect.height > vh - 8) y = vh - rect.height - 8;
    if (x !== pos.x || y !== pos.y) setPos({ x, y });
  }, [pos]);

  return (
    <>
      <div onContextMenu={handleContextMenu}>{children}</div>
      {pos &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            className="fixed z-[100] min-w-[160px] animate-ctx-in rounded-lg border bg-surface-elevated p-1 shadow-lg"
            style={{ left: pos.x, top: pos.y }}
          >
            {items.map((item) => (
              <button
                key={item.label}
                type="button"
                role="menuitem"
                disabled={item.disabled}
                onClick={() => {
                  item.onClick();
                  close();
                }}
                className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-ui-sm transition-colors disabled:opacity-40 ${
                  item.danger
                    ? "text-rose-600 hover:bg-rose-50 dark:text-rose-400 dark:hover:bg-rose-950/30"
                    : "text-content-primary hover:bg-surface-sunken"
                }`}
              >
                {item.icon && (
                  <span className="flex h-4 w-4 items-center justify-center text-content-tertiary">
                    {item.icon}
                  </span>
                )}
                {item.label}
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}
