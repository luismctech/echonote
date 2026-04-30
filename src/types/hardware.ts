/** Hardware profile detected at runtime by the backend. */
export type HardwareProfile = {
  totalRamBytes: number;
  availableRamBytes: number;
  cpuCores: number;
  cpuBrand: string;
  gpuName: string;
  gpuMaxWorkingSet: number;
  unifiedMemory: boolean;
};

/** A single model recommendation pick. */
export type ModelPick = {
  modelId: string;
  reason: string;
  confidence: "high" | "medium" | "low";
};

/** Full recommendation result from the backend. */
export type ModelRecommendation = {
  asr: ModelPick;
  llm: ModelPick | null;
  warning: string | null;
  hardware: HardwareProfile;
};
