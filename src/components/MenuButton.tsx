import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";

import type { MenuItem } from "./ContextMenu";

/**
 * `<MenuButton />` — a button that opens a small dropdown menu.
 *
 * The left-click sibling of `<ContextMenu />` (same `MenuItem` shape).
 * Opening focuses the first enabled item; ArrowUp/ArrowDown move
 * between items; Escape, an outside click or picking an item closes it
 * and returns focus to the trigger. `heading` adds a non-interactive
 * caption above the items (e.g. "Merge into…").
 */
export function MenuButton({
  label,
  ariaLabel,
  title,
  heading,
  items,
  disabled,
  align = "left",
  triggerClassName,
}: Readonly<{
  label: ReactNode;
  ariaLabel?: string;
  title?: string;
  heading?: string;
  items: MenuItem[];
  disabled?: boolean;
  align?: "left" | "right";
  triggerClassName?: string;
}>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  const close = useCallback((restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;
    menuRef.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
    const handleClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) close(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open, close]);

  const handleMenuKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      close(true);
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const buttons = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [],
    );
    if (buttons.length === 0) return;
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const step = e.key === "ArrowDown" ? 1 : -1;
    buttons[(current + step + buttons.length) % buttons.length]?.focus();
  };

  return (
    <div ref={rootRef} className="relative inline-block">
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={ariaLabel}
        title={title}
        onClick={() => setOpen((prev) => !prev)}
        className={triggerClassName}
      >
        {label}
      </button>
      {open && (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          tabIndex={-1}
          onKeyDown={handleMenuKey}
          className={`absolute z-30 mt-1 max-h-72 min-w-[180px] overflow-y-auto rounded-lg border bg-surface-elevated p-1 text-content-primary shadow-lg ${
            align === "right" ? "right-0" : "left-0"
          }`}
        >
          {heading && (
            <p
              role="presentation"
              className="px-2.5 pb-1 pt-1.5 text-micro font-medium uppercase tracking-wide text-content-placeholder"
            >
              {heading}
            </p>
          )}
          {items.map((item, i) => (
            <button
              // biome-ignore lint/suspicious/noArrayIndexKey: labels may repeat; the list is static while open.
              key={i}
              type="button"
              role="menuitem"
              disabled={item.disabled}
              onClick={() => {
                close(true);
                item.onClick();
              }}
              className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-ui-sm transition-colors focus-visible:outline-hidden disabled:opacity-40 ${
                item.danger
                  ? "text-rose-600 hover:bg-rose-50 focus-visible:bg-rose-50 dark:text-rose-400 dark:hover:bg-rose-950/30 dark:focus-visible:bg-rose-950/30"
                  : "hover:bg-surface-sunken focus-visible:bg-surface-sunken"
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
        </div>
      )}
    </div>
  );
}
