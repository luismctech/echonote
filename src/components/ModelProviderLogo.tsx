/**
 * `ModelProviderLogo` — renders a brand icon for a model's provider.
 *
 * Derives the provider from the model ID prefix and renders a small
 * SVG mark inside a tinted rounded container.
 */

type Provider = "openai" | "qwen" | "silero" | "3dspeaker" | "pyannote";

function resolveProvider(modelId: string): Provider {
  if (modelId.startsWith("asr-")) return "openai";
  if (modelId.startsWith("llm-")) return "qwen";
  if (modelId.startsWith("vad-")) return "silero";
  if (modelId.startsWith("embedder-")) return "3dspeaker";
  if (modelId.startsWith("segmenter-")) return "pyannote";
  return "openai";
}

const BG: Record<Provider, string> = {
  openai: "bg-neutral-900 dark:bg-neutral-200",
  qwen: "bg-[#615EFF]",
  silero: "bg-amber-500",
  "3dspeaker": "bg-sky-600",
  pyannote: "bg-violet-600",
};

const LABEL: Record<Provider, string> = {
  openai: "OpenAI Whisper",
  qwen: "Qwen · Alibaba",
  silero: "Silero",
  "3dspeaker": "3D-Speaker · DAMO",
  pyannote: "pyannote",
};

export function ModelProviderLogo({
  modelId,
  size = 28,
}: Readonly<{
  modelId: string;
  size?: number;
}>) {
  const provider = resolveProvider(modelId);
  const iconSize = Math.round(size * 0.57);

  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-lg ${BG[provider]}`}
      style={{ width: size, height: size }}
      title={LABEL[provider]}
    >
      <ProviderSvg provider={provider} size={iconSize} />
    </div>
  );
}

function ProviderSvg({ provider, size }: Readonly<{ provider: Provider; size: number }>) {
  switch (provider) {
    case "openai":
      return <OpenAISvg size={size} />;
    case "qwen":
      return <QwenSvg size={size} />;
    case "silero":
      return <SileroSvg size={size} />;
    case "3dspeaker":
      return <SpeakerSvg size={size} />;
    case "pyannote":
      return <PyannoteSvg size={size} />;
  }
}

/* ------------------------------------------------------------------ */
/*  SVG marks                                                          */
/* ------------------------------------------------------------------ */

/** OpenAI hexagonal-knot logo (Simple Icons, CC0). */
function OpenAISvg({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        className="fill-white dark:fill-neutral-900"
        d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zM13.26 22.43a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"
      />
    </svg>
  );
}

/** Qwen sparkle mark — 4-pointed star. */
function QwenSvg({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        className="fill-white"
        d="M12 2L14 10L22 12L14 14L12 22L10 14L2 12L10 10Z"
      />
    </svg>
  );
}

/** Silero — audio waveform bars. */
function SileroSvg({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect className="fill-white" x="2" y="10" width="2.5" height="4" rx="1.25" />
      <rect className="fill-white" x="6.75" y="6" width="2.5" height="12" rx="1.25" />
      <rect className="fill-white" x="11" y="3" width="2.5" height="18" rx="1.25" />
      <rect className="fill-white" x="15.25" y="6" width="2.5" height="12" rx="1.25" />
      <rect className="fill-white" x="19.5" y="10" width="2.5" height="4" rx="1.25" />
    </svg>
  );
}

/** 3D-Speaker — person silhouette with sound arc. */
function SpeakerSvg({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="fill-white" cx="10" cy="7.5" r="3.5" />
      <path className="fill-white" d="M3.5 20a6.5 6.5 0 0 1 13 0z" />
      <path
        className="fill-none stroke-white"
        strokeWidth="1.8"
        strokeLinecap="round"
        d="M18 7c1 .9 1.6 2.2 1.6 3.5S19 12.6 18 13.5"
      />
      <path
        className="fill-none stroke-white"
        strokeWidth="1.8"
        strokeLinecap="round"
        d="M20.5 4.5c1.6 1.3 2.5 3.3 2.5 5.5s-.9 4.2-2.5 5.5"
      />
    </svg>
  );
}

/** pyannote — stacked timeline segments. */
function PyannoteSvg({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect className="fill-white" x="2" y="4" width="8" height="3" rx="1.5" />
      <rect className="fill-white/60" x="12" y="4" width="10" height="3" rx="1.5" />
      <rect className="fill-white/60" x="2" y="10.5" width="12" height="3" rx="1.5" />
      <rect className="fill-white" x="16" y="10.5" width="6" height="3" rx="1.5" />
      <rect className="fill-white" x="2" y="17" width="5" height="3" rx="1.5" />
      <rect className="fill-white/60" x="9" y="17" width="13" height="3" rx="1.5" />
    </svg>
  );
}
