/**
 * `McpInstaller` — modal for one-click MCP server installation.
 *
 * Detects installed MCP-compatible clients (Claude Desktop, VS Code,
 * Cursor, Windsurf, Claude Code) and lets users install/uninstall
 * the EchoNote MCP server config with a single click.
 */

import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { LayoutGrid, ClipboardList, FileText, Search, NotebookPen, StickyNote, Users, FileDown, LayoutTemplate, Activity, Pencil, Trash2, UserPen, Plus, Minus } from "lucide-react";

import { Modal } from "../../components/Modal";
import { McpClientLogo } from "../../components/McpClientLogo";
import { CopyButton } from "../../components/CopyButton";
import {
  detectMcpClients,
  installMcpClient,
  uninstallMcpClient,
  getMcpConfigSnippet,
} from "../../ipc/client";

interface McpClient {
  id: string;
  label: string;
  detected: boolean;
  installed: boolean;
  configPath: string | null;
}

const CLIENT_ICONS: Record<string, JSX.Element> = {
  "claude-desktop": <McpClientLogo clientId="claude-desktop" />,
  "claude-code": <McpClientLogo clientId="claude-code" />,
  "vscode": <McpClientLogo clientId="vscode" />,
  "cursor": <McpClientLogo clientId="cursor" />,
  "windsurf": <McpClientLogo clientId="windsurf" />,
};

export function McpInstaller({
  onClose,
}: Readonly<{
  onClose: () => void;
}>) {
  const { t } = useTranslation();
  const [clients, setClients] = useState<McpClient[]>([]);
  const [loading, setLoading] = useState(true);
  const [snippet, setSnippet] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!successMsg) return;
    const id = setTimeout(() => setSuccessMsg(null), 3000);
    return () => clearTimeout(id);
  }, [successMsg]);

  const refresh = useCallback(() => {
    setLoading(true);
    detectMcpClients()
      .then(setClients)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    getMcpConfigSnippet().then(setSnippet).catch(() => {});
  }, [refresh]);

  const handleInstall = useCallback(
    async (clientId: string) => {
      setBusy(clientId);
      setError(null);
      setSuccessMsg(null);
      try {
        const result = await installMcpClient(clientId);
        if (result.success) {
          setSuccessMsg(result.message);
          refresh();
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const handleUninstall = useCallback(
    async (clientId: string) => {
      setBusy(clientId);
      setError(null);
      setSuccessMsg(null);
      try {
        const result = await uninstallMcpClient(clientId);
        if (result.success) {
          setSuccessMsg(result.message);
          refresh();
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const detected = clients.filter((c) => c.detected);
  const notDetected = clients.filter((c) => !c.detected);

  return (
    <Modal open onClose={onClose} className="w-full max-w-lg">
      <div className="flex max-h-[80vh] w-full flex-col gap-3 overflow-hidden rounded-xl border bg-surface-elevated p-5 shadow-xl">
        {/* Header */}
        <header className="flex shrink-0 items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-950/40 dark:text-violet-300">
              <LayoutGrid className="h-4.5 w-4.5" />
            </div>
            <h2 className="text-ui-lg font-semibold">{t("mcp.title")}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-ui-sm text-content-tertiary hover:bg-surface-sunken"
          >
            {t("mcp.close")}
          </button>
        </header>

        <p className="text-ui-sm text-content-secondary">{t("mcp.description")}</p>

        {/* Feedback */}
        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-ui-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}
        {successMsg && (
          <p className="rounded-md bg-emerald-50 px-3 py-2 text-ui-sm text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
            {successMsg}
          </p>
        )}

        {loading ? (
          <p className="py-6 text-center text-ui-md text-content-tertiary">{t("mcp.detecting")}</p>
        ) : (
          <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
            {/* Detected clients */}
            {detected.length > 0 && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <h3 className="text-ui-sm font-semibold text-content-primary">{t("mcp.detected")}</h3>
                  <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-micro font-medium text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
                    {detected.length}
                  </span>
                </div>
                {detected.map((client) => (
                  <ClientRow
                    key={client.id}
                    client={client}
                    busy={busy === client.id}
                    anyBusy={busy !== null}
                    onInstall={handleInstall}
                    onUninstall={handleUninstall}
                  />
                ))}
              </div>
            )}

            {/* Not detected */}
            {notDetected.length > 0 && (
              <div className="flex flex-col gap-2">
                <h3 className="text-ui-sm font-semibold text-content-tertiary">{t("mcp.notDetected")}</h3>
                {notDetected.map((client) => (
                  <ClientRow
                    key={client.id}
                    client={client}
                    busy={busy === client.id}
                    anyBusy={busy !== null}
                    onInstall={handleInstall}
                    onUninstall={handleUninstall}
                  />
                ))}
              </div>
            )}

            {/* Manual config snippet */}
            <div className="flex flex-col gap-2 border-t border-subtle pt-3">
              <div className="flex items-center justify-between">
                <h3 className="text-ui-sm font-semibold text-content-primary">{t("mcp.manualConfig")}</h3>
                <CopyButton getText={() => snippet} />
              </div>
              <p className="text-ui-xs text-content-tertiary">{t("mcp.manualHint")}</p>
              <pre className="max-h-28 overflow-auto rounded-md border bg-surface-sunken px-3 py-2 font-mono text-micro text-content-secondary">
                {snippet}
              </pre>
            </div>
          </div>
        )}

        {/* Tools info */}
        <div className="flex shrink-0 flex-col gap-1.5 border-t border-subtle pt-3">
          <h3 className="text-ui-xs font-semibold uppercase tracking-wide text-content-tertiary">{t("mcp.toolsAvailable")}</h3>

          {/* Read tools */}
          <p className="text-micro font-medium text-content-placeholder">{t("mcp.toolsRead")}</p>
          <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
            {[
              { key: "listMeetings", icon: <ClipboardList className="h-3 w-3" /> },
              { key: "getMeeting", icon: <FileText className="h-3 w-3" /> },
              { key: "searchMeetings", icon: <Search className="h-3 w-3" /> },
              { key: "getSummary", icon: <NotebookPen className="h-3 w-3" /> },
              { key: "listNotes", icon: <StickyNote className="h-3 w-3" /> },
              { key: "getSpeakers", icon: <Users className="h-3 w-3" /> },
              { key: "exportMeeting", icon: <FileDown className="h-3 w-3" /> },
              { key: "listTemplates", icon: <LayoutTemplate className="h-3 w-3" /> },
              { key: "getStatus", icon: <Activity className="h-3 w-3" /> },
            ].map((tool) => (
              <div key={tool.key} className="flex items-center gap-1.5 rounded-md px-2 py-1 text-ui-xs text-content-secondary">
                {tool.icon}
                {t(`mcp.tools.${tool.key}`)}
              </div>
            ))}
          </div>

          {/* Write tools */}
          <p className="mt-1 text-micro font-medium text-content-placeholder">{t("mcp.toolsWrite")}</p>
          <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
            {[
              { key: "renameMeeting", icon: <Pencil className="h-3 w-3" /> },
              { key: "deleteMeeting", icon: <Trash2 className="h-3 w-3" /> },
              { key: "renameSpeaker", icon: <UserPen className="h-3 w-3" /> },
              { key: "addNote", icon: <Plus className="h-3 w-3" /> },
              { key: "deleteNote", icon: <Minus className="h-3 w-3" /> },
            ].map((tool) => (
              <div key={tool.key} className="flex items-center gap-1.5 rounded-md px-2 py-1 text-ui-xs text-content-secondary">
                {tool.icon}
                {t(`mcp.tools.${tool.key}`)}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  );
}

function ClientRow({
  client,
  busy,
  anyBusy,
  onInstall,
  onUninstall,
}: Readonly<{
  client: McpClient;
  busy: boolean;
  anyBusy: boolean;
  onInstall: (id: string) => void;
  onUninstall: (id: string) => void;
}>) {
  const { t } = useTranslation();
  const icon = CLIENT_ICONS[client.id];

  let borderClass = "border-subtle bg-surface-sunken opacity-60";
  if (client.installed) {
    borderClass = "border-violet-200 bg-violet-50/50 dark:border-violet-900 dark:bg-violet-950/20";
  } else if (client.detected) {
    borderClass = "border-subtle bg-surface-sunken";
  }

  return (
    <div
      className={`flex items-center gap-3 rounded-lg border p-3 transition-colors ${borderClass}`}
    >
      {/* Icon */}
      <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${
        client.installed
          ? "bg-violet-100 text-violet-600 dark:bg-violet-900/40 dark:text-violet-300"
          : "bg-surface-inset text-content-tertiary"
      }`}>
        {icon}
      </div>

      {/* Info */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-ui-sm font-medium text-content-primary">{client.label}</span>
          {client.installed && (
            <span className="shrink-0 rounded bg-violet-100 px-1.5 py-0.5 text-micro font-medium text-violet-700 dark:bg-violet-950/40 dark:text-violet-300">
              {t("mcp.connected")}
            </span>
          )}
          {!client.detected && (
            <span className="shrink-0 rounded bg-neutral-100 px-1.5 py-0.5 text-micro font-medium text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
              {t("mcp.notInstalled")}
            </span>
          )}
        </div>
        {client.configPath && (
          <p className="truncate text-micro text-content-placeholder" title={client.configPath}>
            {client.configPath}
          </p>
        )}
      </div>

      {/* Action */}
      {client.installed ? (
        <button
          type="button"
          disabled={anyBusy}
          onClick={() => onUninstall(client.id)}
          className="shrink-0 rounded-full border border-red-200 px-3 py-1 text-ui-xs font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/30"
        >
          {busy ? t("mcp.removing") : t("mcp.remove")}
        </button>
      ) : (
        <button
          type="button"
          disabled={anyBusy}
          onClick={() => onInstall(client.id)}
          className="shrink-0 rounded-full bg-violet-600 px-3 py-1 text-ui-xs font-medium text-white transition-colors hover:bg-violet-700 disabled:opacity-50"
        >
          {busy ? t("mcp.installing") : t("mcp.install")}
        </button>
      )}
    </div>
  );
}
