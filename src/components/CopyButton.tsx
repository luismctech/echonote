import { useCallback, useState } from "react";
import { Check, Copy } from "lucide-react";

/** Small icon button that copies text to the clipboard. */
export function CopyButton({
  getText,
  title = "Copy",
}: Readonly<{
  getText: () => string;
  title?: string;
}>) {
  const [copied, setCopied] = useState(false);

  const handleClick = useCallback(() => {
    navigator.clipboard.writeText(getText()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [getText]);

  return (
    <button
      type="button"
      onClick={handleClick}
      className="rounded-md p-1 text-content-tertiary hover:bg-surface-inset hover:text-content-primary"
      title={title}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-emerald-500" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}
