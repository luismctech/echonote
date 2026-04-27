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
      <div className="flex max-h-[80vh] w-full max-w-lg flex-col gap-3 overflow-hidden rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-950">
        <header className="flex items-center justify-between">
          <h2 className="text-base font-semibold">{t("templates.title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-900"
          >
            {t("templates.close")}
          </button>
        </header>

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-sm text-zinc-500">{t("templates.loading")}</p>
        ) : (
          <div className="flex min-h-0 flex-col gap-3 overflow-y-auto">
            {templates.length === 0 && !showForm && (
              <p className="py-4 text-center text-sm text-zinc-500">{t("templates.empty")}</p>
            )}

            {templates.map((tpl) => (
              <div
                key={tpl.id}
                className="flex items-start gap-3 rounded-lg border border-zinc-100 bg-zinc-50/50 px-3 py-2.5 dark:border-zinc-800/60 dark:bg-zinc-900/30"
              >
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="truncate text-sm font-medium text-zinc-800 dark:text-zinc-200">
                    {tpl.name}
                  </span>
                  <span className="line-clamp-2 text-xs text-zinc-500 dark:text-zinc-400">
                    {tpl.prompt}
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <button
                    type="button"
                    onClick={() => handleEdit(tpl)}
                    className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
                    title={t("templates.edit")}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                      <path d="M13.488 2.513a1.75 1.75 0 0 0-2.475 0L3.84 9.686a2.25 2.25 0 0 0-.602 1.07l-.56 2.243a.75.75 0 0 0 .912.912l2.243-.56a2.25 2.25 0 0 0 1.07-.602l7.174-7.174a1.75 1.75 0 0 0 0-2.474l-.588-.588ZM11.72 3.22a.25.25 0 0 1 .354 0l.588.588a.25.25 0 0 1 0 .354L11.95 4.874 10.807 3.93l.913-.71ZM10.1 4.636l1.143.944-5.498 5.498a.75.75 0 0 1-.357.2l-1.396.35.349-1.397a.75.75 0 0 1 .2-.357l5.56-5.238Z" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(tpl.id)}
                    className="rounded-md p-1 text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                    title={t("templates.delete")}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                      <path fillRule="evenodd" d="M5 3.25V4H2.75a.75.75 0 0 0 0 1.5h.3l.815 8.15A1.5 1.5 0 0 0 5.357 15h5.285a1.5 1.5 0 0 0 1.493-1.35l.815-8.15h.3a.75.75 0 0 0 0-1.5H11v-.75A2.25 2.25 0 0 0 8.75 1h-1.5A2.25 2.25 0 0 0 5 3.25Zm2.25-.75a.75.75 0 0 0-.75.75V4h3v-.75a.75.75 0 0 0-.75-.75h-1.5ZM6.05 6a.75.75 0 0 1 .787.713l.275 5.5a.75.75 0 0 1-1.498.075l-.275-5.5A.75.75 0 0 1 6.05 6Zm3.9 0a.75.75 0 0 1 .712.787l-.275 5.5a.75.75 0 0 1-1.498-.075l.275-5.5A.75.75 0 0 1 9.95 6Z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}

            {showForm && (
              <div className="flex flex-col gap-2 rounded-lg border border-blue-200 bg-blue-50/30 p-3 dark:border-blue-800/60 dark:bg-blue-950/20">
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder={t("templates.namePlaceholder")}
                  className="w-full rounded-md border border-zinc-200 bg-white px-2.5 py-1.5 text-sm text-zinc-800 placeholder:text-zinc-400 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:placeholder:text-zinc-500"
                  maxLength={100}
                />
                <textarea
                  value={form.prompt}
                  onChange={(e) => setForm((f) => ({ ...f, prompt: e.target.value }))}
                  placeholder={t("templates.promptPlaceholder")}
                  rows={5}
                  className="w-full resize-y rounded-md border border-zinc-200 bg-white px-2.5 py-1.5 text-sm text-zinc-800 placeholder:text-zinc-400 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:placeholder:text-zinc-500"
                  maxLength={4000}
                />
                <div className="flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="rounded-md px-3 py-1.5 text-xs font-medium text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
                  >
                    {t("templates.cancel")}
                  </button>
                  <button
                    type="button"
                    onClick={handleSave}
                    disabled={saving || !form.name.trim() || !form.prompt.trim()}
                    className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
                  >
                    {saveLabel}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {!showForm && !loading && (
          <button
            type="button"
            onClick={handleNew}
            className="mt-1 self-start rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60"
          >
            {t("templates.new")}
          </button>
        )}
      </div>
    </div>
  );
}
