import { useCallback, useEffect, useRef, useState } from "react";
import { cancelDownload, deleteModel, downloadModel, getModelStatus, setActiveLlm, getActiveLlm, setActiveAsr, getActiveAsr } from "../ipc/client";
import { isIpcError } from "../types/ipc-error";
import type { DownloadEvent, ModelInfo } from "../types/models";

export type DownloadProgress = {
  modelId: string;
  downloaded: number;
  total: number;
};

export type UseModelManager = {
  models: ModelInfo[];
  loading: boolean;
  downloading: DownloadProgress | null;
  error: string | null;
  activeLlm: string | null;
  activeAsr: string | null;
  refresh: () => void;
  download: (modelId: string) => void;
  cancelDl: (modelId: string) => void;
  remove: (modelId: string) => void;
  selectLlm: (modelId: string) => void;
  selectAsr: (modelId: string) => void;
};

export function useModelManager(): UseModelManager {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeLlm, setActiveLlmState] = useState<string | null>(null);
  const [activeAsr, setActiveAsrState] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchStatus = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([getModelStatus(), getActiveLlm(), getActiveAsr()])
      .then(([result, activeLlmId, activeAsrId]) => {
        if (mountedRef.current) {
          setModels(result);
          setActiveLlmState(activeLlmId);
          setActiveAsrState(activeAsrId);
        }
      })
      .catch((err) => {
        if (mountedRef.current) {
          let msg: string;
          if (isIpcError(err)) msg = err.message;
          else if (err instanceof Error) msg = err.message;
          else msg = String(err);
          setError(msg);
        }
      })
      .finally(() => {
        if (mountedRef.current) setLoading(false);
      });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchStatus();
    return () => {
      mountedRef.current = false;
    };
  }, [fetchStatus]);

  const download = useCallback(
    (modelId: string) => {
      if (downloading) return;
      setDownloading({ modelId, downloaded: 0, total: 0 });
      setError(null);

      const onEvent = (event: DownloadEvent) => {
        if (!mountedRef.current) return;
        switch (event.kind) {
          case "progress":
            setDownloading({ modelId, downloaded: event.downloaded, total: event.total });
            break;
          case "finished":
            setDownloading(null);
            fetchStatus();
            break;
          case "failed":
            setDownloading(null);
            setError(event.error);
            break;
          case "cancelled":
            setDownloading(null);
            break;
        }
      };

      downloadModel(modelId, onEvent).catch((err) => {
        if (mountedRef.current) {
          setDownloading(null);
          let msg: string;
          if (isIpcError(err)) msg = err.message;
          else if (err instanceof Error) msg = err.message;
          else msg = String(err);
          setError(msg);
        }
      });
    },
    [downloading, fetchStatus],
  );

  const cancelDl = useCallback(
    (modelId: string) => {
      cancelDownload(modelId).catch(() => {});
    },
    [],
  );

  const remove = useCallback(
    (modelId: string) => {
      setError(null);
      deleteModel(modelId)
        .then(() => {
          if (mountedRef.current) fetchStatus();
        })
        .catch((err) => {
          if (mountedRef.current) {
            let msg: string;
            if (isIpcError(err)) msg = err.message;
            else if (err instanceof Error) msg = err.message;
            else msg = String(err);
            setError(msg);
          }
        });
    },
    [fetchStatus],
  );

  const selectLlm = useCallback(
    (modelId: string) => {
      setError(null);
      setActiveLlm(modelId)
        .then(() => {
          if (mountedRef.current) setActiveLlmState(modelId);
        })
        .catch((err) => {
          if (mountedRef.current) {
            let msg: string;
            if (isIpcError(err)) msg = err.message;
            else if (err instanceof Error) msg = err.message;
            else msg = String(err);
            setError(msg);
          }
        });
    },
    [],
  );

  const selectAsr = useCallback(
    (modelId: string) => {
      setError(null);
      setActiveAsr(modelId)
        .then(() => {
          if (mountedRef.current) setActiveAsrState(modelId);
        })
        .catch((err) => {
          if (mountedRef.current) {
            let msg: string;
            if (isIpcError(err)) msg = err.message;
            else if (err instanceof Error) msg = err.message;
            else msg = String(err);
            setError(msg);
          }
        });
    },
    [],
  );

  return { models, loading, downloading, error, activeLlm, activeAsr, refresh: fetchStatus, download, cancelDl, remove, selectLlm, selectAsr };
}
