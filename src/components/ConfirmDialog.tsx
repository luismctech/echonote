import { useCallback, useId, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "./Button";
import { Modal } from "./Modal";

/** What a confirmation dialog asks and what runs when the user accepts. */
export type ConfirmRequest = {
  title: string;
  message: string;
  confirmLabel: string;
  /** Render the confirm button as destructive. */
  danger?: boolean;
  onConfirm: () => void;
};

/**
 * Small accessible confirmation dialog on top of `<Modal />`. The
 * cancel button gets the initial focus so Enter never confirms a
 * destructive action by accident; Escape and the backdrop cancel.
 */
export function ConfirmDialog({
  open,
  request,
  onCancel,
  onConfirm,
  onClosed,
}: Readonly<{
  open: boolean;
  request: ConfirmRequest;
  onCancel: () => void;
  onConfirm: () => void;
  /** Called once the exit animation finished (any close path). */
  onClosed: () => void;
}>) {
  const { t } = useTranslation();
  const titleId = useId();
  const messageId = useId();
  return (
    <Modal open={open} onClose={onClosed} className="w-full max-w-sm">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={messageId}
        className="flex flex-col gap-3 rounded-xl border bg-surface-elevated p-5 shadow-xl"
      >
        <h2 id={titleId} className="text-ui-lg font-semibold">
          {request.title}
        </h2>
        <p id={messageId} className="text-ui-sm text-content-secondary">
          {request.message}
        </p>
        <div className="flex justify-end gap-2">
          <Button size="sm" onClick={onCancel} autoFocus>
            {t("confirm.cancel")}
          </Button>
          <Button size="sm" variant={request.danger ? "danger" : "primary"} onClick={onConfirm}>
            {request.confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

/**
 * Imperative wrapper around {@link ConfirmDialog}: `ask(request)` opens
 * it and `dialog` is the element to render. The request stays mounted
 * during the exit animation so the text does not vanish mid-fade.
 */
export function useConfirm(): { ask: (request: ConfirmRequest) => void; dialog: ReactNode } {
  const [request, setRequest] = useState<ConfirmRequest | null>(null);
  const [open, setOpen] = useState(false);
  const ask = useCallback((next: ConfirmRequest) => {
    setRequest(next);
    setOpen(true);
  }, []);
  const dialog = request ? (
    <ConfirmDialog
      open={open}
      request={request}
      onCancel={() => setOpen(false)}
      onConfirm={() => {
        setOpen(false);
        request.onConfirm();
      }}
      onClosed={() => {
        setOpen(false);
        setRequest(null);
      }}
    />
  ) : null;
  return { ask, dialog };
}
