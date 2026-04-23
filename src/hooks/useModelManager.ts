import { useCallback, useEffect, useRef, useState } from "react";
import { downloadModel, getModelStatus } from "../ipc/client";
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
  refresh: () => void;
  download: (modelId: string) => void;
};

export function useModelManager(): UseModelManager {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchStatus = useCallback(() => {
    setLoading(true);
    setError(null);
    getModelStatus()
      .then((result) => {
        if (mountedRef.current) setModels(result);
      })
      .catch((err) => {
        if (mountedRef.current) {
          setError(err instanceof Error ? err.message : String(err));
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
        }
      };

      downloadModel(modelId, onEvent).catch((err) => {
        if (mountedRef.current) {
          setDownloading(null);
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    },
    [downloading, fetchStatus],
  );

  return { models, loading, downloading, error, refresh: fetchStatus, download };
}
