/**
 * `TemplateManager` — modal panel for CRUD on custom summary templates.
 *
 * Users can create, edit, and delete their own prompt templates that
 * appear in the SummaryPanel's template selector alongside the built-in ones.
 */

import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { CustomTemplate, CustomTemplateId } from "../../types/custom-template";
import {
  listCustomTemplates,
  createCustomTemplate,
  updateCustomTemplate,
  deleteCustomTemplate,
} from "../../ipc/client";

type FormState = {
  name: string;
  prompt: string;
};

const EMPTY_FORM: FormState = { name: "", prompt: "" };

export function TemplateManager({
  onClose,
  onChanged,
}: Readonly<{
  onClose: () => void;
  onChanged?: () => void;
}>) {
  const { t } = useTranslation();
  const [templates, setTemplates] = useState<CustomTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Editing state: null = creating new, string = editing existing id
  const [editingId, setEditingId] = useState<CustomTemplateId | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(() => {
    setLoading(true);
    listCustomTemplates()
      .then(setTemplates)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleNew = useCallback(() => {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setShowForm(true);
    setError(null);
  }, []);

  const handleEdit = useCallback((tpl: CustomTemplate) => {
    setEditingId(tpl.id);
    setForm({ name: tpl.name, prompt: tpl.prompt });
    setShowForm(true);
    setError(null);
  }, []);

  const handleCancel = useCallback(() => {
    setShowForm(false);
    setForm(EMPTY_FORM);
    setEditingId(null);
  }, []);

  const handleSave = useCallback(async () => {
    const trimmedName = form.name.trim();
    const trimmedPrompt = form.prompt.trim();
    if (!trimmedName || !trimmedPrompt) return;

    setSaving(true);
    setError(null);
    try {
      if (editingId) {
        await updateCustomTemplate(editingId, trimmedName, trimmedPrompt);
      } else {
        await createCustomTemplate(trimmedName, trimmedPrompt);
      }
      setShowForm(false);
      setForm(EMPTY_FORM);
      setEditingId(null);
      refresh();
      onChanged?.();
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [form, editingId, refresh, onChanged]);

  const handleDelete = useCallback(
    async (id: CustomTemplateId) => {
      setError(null);
      try {
        await deleteCustomTemplate(id);
        refresh();
        onChanged?.();
      } catch (err) {
        setError(String(err));
      }
    },
    [refresh, onChanged],
  );

  const saveKey = editingId ? "templates.save" : "templates.create";
  const saveLabel = saving ? t("templates.saving") : t(saveKey);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col gap-3 overflow-hidden rounded-xl border bg-surface-elevated p-5 shadow-xl">
        <header className="flex items-center justify-between">
          <h2 className="text-ui-lg font-semibold">{t("templates.title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-ui-sm text-content-tertiary hover:bg-surface-sunken"
          >
            {t("templates.close")}
          </button>
        </header>

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-ui-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-ui-md text-content-tertiary">{t("templates.loading")}</p>
        ) : showForm ? (
          /* ── Form view (replaces list) ── */
          <div className="flex flex-col gap-3">
            <p className="text-ui-sm font-medium text-content-secondary">
              {editingId ? t("templates.edit") : t("templates.new")}
            </p>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder={t("templates.namePlaceholder")}
              className="w-full rounded-md border bg-surface-elevated px-2.5 py-1.5 text-ui-md text-content-primary placeholder:text-content-placeholder focus:border-accent-400 focus:outline-none focus:ring-1 focus:ring-accent-400"
              maxLength={100}
              autoFocus
            />
            <textarea
              value={form.prompt}
              onChange={(e) => setForm((f) => ({ ...f, prompt: e.target.value }))}
              placeholder={t("templates.promptPlaceholder")}
              rows={6}
              className="w-full resize-y rounded-md border bg-surface-elevated px-2.5 py-1.5 text-ui-md text-content-primary placeholder:text-content-placeholder focus:border-accent-400 focus:outline-none focus:ring-1 focus:ring-accent-400"
              maxLength={4000}
            />
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={handleCancel}
                className="rounded-md px-3 py-1.5 text-ui-sm font-medium text-content-secondary hover:bg-surface-sunken"
              >
                {t("templates.cancel")}
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={saving || !form.name.trim() || !form.prompt.trim()}
                className="rounded-md bg-accent-600 px-3 py-1.5 text-ui-sm font-medium text-white hover:bg-accent-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {saveLabel}
              </button>
            </div>
          </div>
        ) : (
          /* ── List view ── */
          <>
            <div className="flex min-h-0 flex-col gap-3 overflow-y-auto">
              {templates.length === 0 && (
                <p className="py-4 text-center text-ui-md text-content-tertiary">{t("templates.empty")}</p>
              )}

              {templates.map((tpl) => (
                <div
                  key={tpl.id}
                  className="flex items-start gap-3 rounded-lg border border-subtle bg-surface-sunken/50 px-3 py-2.5"
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="truncate text-ui-md font-medium text-content-primary">
                      {tpl.name}
                    </span>
                    <span className="line-clamp-2 text-ui-sm text-content-tertiary">
                      {tpl.prompt}
                    </span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      type="button"
                      onClick={() => handleEdit(tpl)}
                      className="rounded-md p-1 text-content-tertiary hover:bg-surface-inset hover:text-content-primary"
                      title={t("templates.edit")}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                        <path d="M13.488 2.513a1.75 1.75 0 0 0-2.475 0L3.84 9.686a2.25 2.25 0 0 0-.602 1.07l-.56 2.243a.75.75 0 0 0 .912.912l2.243-.56a2.25 2.25 0 0 0 1.07-.602l7.174-7.174a1.75 1.75 0 0 0 0-2.474l-.588-.588ZM11.72 3.22a.25.25 0 0 1 .354 0l.588.588a.25.25 0 0 1 0 .354L11.95 4.874 10.807 3.93l.913-.71ZM10.1 4.636l1.143.944-5.498 5.498a.75.75 0 0 1-.357.2l-1.396.35.349-1.397a.75.75 0 0 1 .2-.357l5.56-5.238Z" />
                      </svg>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(tpl.id)}
                      className="rounded-md p-1 text-content-placeholder hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                      title={t("templates.delete")}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                        <path fillRule="evenodd" d="M5 3.25V4H2.75a.75.75 0 0 0 0 1.5h.3l.815 8.15A1.5 1.5 0 0 0 5.357 15h5.285a1.5 1.5 0 0 0 1.493-1.35l.815-8.15h.3a.75.75 0 0 0 0-1.5H11v-.75A2.25 2.25 0 0 0 8.75 1h-1.5A2.25 2.25 0 0 0 5 3.25Zm2.25-.75a.75.75 0 0 0-.75.75V4h3v-.75a.75.75 0 0 0-.75-.75h-1.5ZM6.05 6a.75.75 0 0 1 .787.713l.275 5.5a.75.75 0 0 1-1.498.075l-.275-5.5A.75.75 0 0 1 6.05 6Zm3.9 0a.75.75 0 0 1 .712.787l-.275 5.5a.75.75 0 0 1-1.498-.075l.275-5.5A.75.75 0 0 1 9.95 6Z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={handleNew}
              className="mt-1 self-start rounded-md bg-accent-600 px-3 py-1.5 text-ui-sm font-medium text-white hover:bg-accent-700"
            >
              {t("templates.new")}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
