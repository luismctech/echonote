import { useEffect, useState } from "react";

import type { ModelRecommendation } from "../types/hardware";
import { getModelRecommendation } from "../ipc/client";

/**
 * Fetches hardware profile and model recommendations from the backend.
 * Caches the result for the lifetime of the component.
 */
export function useHardwareRecommendation() {
  const [data, setData] = useState<ModelRecommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelRecommendation()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Unknown error";
          setError(message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}
