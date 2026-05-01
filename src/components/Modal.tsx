import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

/**
 * `<Modal />` — animated overlay container.
 *
 * Renders a backdrop + centered panel with enter/exit CSS animations.
 * Call `onClose` to trigger the exit animation; the component
 * unmounts itself after the animation finishes.
 */
export function Modal({
  open,
  onClose,
  children,
  className = "",
}: Readonly<{
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}>) {
  const [mounted, setMounted] = useState(open);
  const [closing, setClosing] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Mount when open turns true.
  useEffect(() => {
    if (open) {
      setMounted(true);
      setClosing(false);
    }
  }, [open]);

  // Trigger close animation.
  const handleClose = useCallback(() => {
    setClosing(true);
  }, []);

  // After exit animation ends, truly unmount.
  const handleAnimationEnd = useCallback(() => {
    if (closing) {
      setMounted(false);
      setClosing(false);
      onClose();
    }
  }, [closing, onClose]);

  // Expose handleClose when parent says !open.
  useEffect(() => {
    if (!open && mounted && !closing) {
      setClosing(true);
    }
  }, [open, mounted, closing]);

  // Escape key
  useEffect(() => {
    if (!mounted) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [mounted, handleClose]);

  if (!mounted) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        ref={overlayRef}
        className={`absolute inset-0 bg-black/40 ${closing ? "animate-overlay-out" : "animate-overlay-in"}`}
        onClick={handleClose}
        onAnimationEnd={handleAnimationEnd}
      />
      {/* Panel */}
      <div
        className={`relative ${closing ? "animate-modal-out" : "animate-modal-in"} ${className}`}
      >
        {children}
      </div>
    </div>
  );
}
