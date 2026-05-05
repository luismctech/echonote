/**
 * `McpClientLogo` — official brand logos for MCP client applications.
 */

export function McpClientLogo({
  clientId,
  size = 20,
}: Readonly<{
  clientId: string;
  size?: number;
}>) {
  switch (clientId) {
    case "claude-desktop":
    case "claude-code":
      return <ClaudeLogo size={size} />;
    case "vscode":
      return <VSCodeLogo size={size} />;
    case "cursor":
      return <CursorLogo size={size} />;
    case "windsurf":
      return <WindsurfLogo size={size} />;
    default:
      return <DefaultLogo size={size} />;
  }
}

/** Anthropic Claude mark — warm terracotta roundmark. */
function ClaudeLogo({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M14.914 3.743L17.09 12.2l-4.607 4.372c-.278.263-.42.395-.582.444a.75.75 0 0 1-.464 0c-.162-.049-.303-.181-.582-.444L6.25 12.2l2.177-8.457c.063-.245.094-.368.156-.464a.75.75 0 0 1 .247-.247c.096-.062.219-.093.464-.156l5.156-1.327c.245-.063.368-.094.471-.079a.75.75 0 0 1 .335.134c.082.061.146.161.272.363l-.614 1.776z"
        className="fill-[#D97757]"
      />
      <path
        d="M16.5 4.5L14.914 3.743l.614-1.776c.126-.202.19-.302.272-.363a.75.75 0 0 1 .335-.134c.103-.015.226.016.471.079L21 2.877c.245.063.368.094.464.156a.75.75 0 0 1 .247.247c.062.096.093.219.156.464L24 12.2l-4.48 4.25L16.5 4.5z"
        className="fill-[#D97757]"
      />
      <path
        d="M11.437 17.016l.418.397c.278.263.42.395.42.587s-.142.324-.42.587l-4.607 4.372c-.278.264-.42.396-.582.445a.75.75 0 0 1-.464 0c-.162-.049-.303-.181-.582-.445L.843 18.587c-.278-.263-.42-.395-.42-.587s.142-.324.42-.587l.418-.397 5.088 4.897c.278.264.42.396.582.445a.75.75 0 0 0 .464 0c.162-.049.303-.181.582-.445l3.46-3.897z"
        className="fill-[#D97757]"
      />
      <path
        d="M6.25 12.2L1.66 16.45l-.818-.777c-.278-.264-.42-.396-.42-.588 0-.191.142-.323.42-.586L6.25 9.5l5.187 4.999c.278.263.42.395.42.587 0 .192-.142.324-.42.587l-.818.777L6.25 12.2z"
        className="fill-[#D97757]/70"
      />
    </svg>
  );
}

/** VS Code — blue cross/editor mark. */
function VSCodeLogo({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
      <mask id="vsc-m">
        <rect width="100" height="100" fill="#fff" />
      </mask>
      <path
        d="M71.29 2.97L34.22 36.4 15.86 22.18 9.3 25.02v49.96l6.56 2.84 18.36-14.22 37.07 33.43L90.7 91.14V8.86L71.29 2.97zM34.3 62.54l-16.2-12.54 16.2-12.54V62.54zm36.95 16.74L34.3 50 71.25 20.72v58.56z"
        className="fill-[#007ACC]"
      />
    </svg>
  );
}

/** Cursor — stylized cursor / arrow mark. */
function CursorLogo({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M5.727 2.3l14.11 9.115a.5.5 0 0 1-.014.848L5.748 21.687a.5.5 0 0 1-.748-.434V2.725a.5.5 0 0 1 .727-.425z"
        className="fill-current"
      />
      <path
        d="M13 13l7.5 7.5M13 13l-1-7"
        className="stroke-current fill-none"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Windsurf (Codeium) — wave/surf mark. */
function WindsurfLogo({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M3 8c2-3 5-5 9-5s7 2 9 5"
        className="stroke-[#09B6A2] fill-none"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M5 13c1.5-2.5 4-4 7-4s5.5 1.5 7 4"
        className="stroke-[#09B6A2] fill-none"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M8 18c1-1.5 2.5-2.5 4-2.5s3 1 4 2.5"
        className="stroke-[#09B6A2] fill-none"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Fallback generic terminal icon. */
function DefaultLogo({ size }: Readonly<{ size: number }>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2" y="3" width="20" height="18" rx="3" className="fill-none stroke-current" strokeWidth="1.5" />
      <path d="M6 9l4 3-4 3" className="fill-none stroke-current" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 17h5" className="fill-none stroke-current" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
