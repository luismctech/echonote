/**
 * EchoNote logo components — three variants rendered as inline SVG.
 *
 * - `<LogoMark />`     — static mark (no background)
 * - `<LogoGlow />`     — static mark with soft glow filter
 * - `<LogoAnimated />` — bars pulse with staggered timing
 *
 * All variants accept `size` (px, default 32) and an optional `className`.
 * The viewBox is centred on the mark so it can be used at any size.
 */

import type { SVGAttributes } from "react";

type LogoProps = {
  /** Width & height in pixels. @default 32 */
  size?: number;
  className?: string;
} & Omit<SVGAttributes<SVGSVGElement>, "width" | "height" | "viewBox">;

const BRAND = "#3BE8A5";

// The original design sits in a 512×512 canvas. The mark itself spans
// x=124..388  y=140..376, so we crop to that with a small margin.
const VB = "110 130 292 260";

/** Static logo mark — dots + bars, no background. */
export function LogoMark({ size = 32, className, ...rest }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox={VB}
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      {...rest}
    >
      <g fill={BRAND}>
        <circle cx={140} cy={256} r={16} />
        <rect x={180} y={196} width={32} height={120} rx={16} />
        <rect x={240} y={156} width={32} height={200} rx={16} />
        <rect x={300} y={196} width={32} height={120} rx={16} />
        <circle cx={372} cy={256} r={16} />
      </g>
    </svg>
  );
}

/** Logo mark with a soft green glow effect. */
export function LogoGlow({ size = 32, className, ...rest }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox={VB}
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      {...rest}
    >
      <defs>
        <filter id="echo-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={6} result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g fill={BRAND} filter="url(#echo-glow)">
        <circle cx={140} cy={256} r={16} />
        <rect x={180} y={196} width={32} height={120} rx={16} />
        <rect x={240} y={156} width={32} height={200} rx={16} />
        <rect x={300} y={196} width={32} height={120} rx={16} />
        <circle cx={372} cy={256} r={16} />
      </g>
    </svg>
  );
}

/** Animated logo — bars pulse with staggered timing. */
export function LogoAnimated({ size = 32, className, ...rest }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox={VB}
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      {...rest}
    >
      <g fill={BRAND}>
        <circle cx={140} cy={256} r={16} />

        {/* Bar 1 */}
        <rect x={180} y={196} width={32} height={120} rx={16}>
          <animate
            attributeName="height"
            values="120;160;120"
            dur="1.2s"
            begin="0s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
          <animate
            attributeName="y"
            values="196;176;196"
            dur="1.2s"
            begin="0s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
        </rect>

        {/* Bar 2 */}
        <rect x={240} y={156} width={32} height={200} rx={16}>
          <animate
            attributeName="height"
            values="200;240;200"
            dur="1.2s"
            begin="0.2s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
          <animate
            attributeName="y"
            values="156;136;156"
            dur="1.2s"
            begin="0.2s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
        </rect>

        {/* Bar 3 */}
        <rect x={300} y={196} width={32} height={120} rx={16}>
          <animate
            attributeName="height"
            values="120;160;120"
            dur="1.2s"
            begin="0.4s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
          <animate
            attributeName="y"
            values="196;176;196"
            dur="1.2s"
            begin="0.4s"
            repeatCount="indefinite"
            calcMode="spline"
            keyTimes="0;0.5;1"
            keySplines="0.4 0 0.2 1;0.4 0 0.2 1"
          />
        </rect>

        <circle cx={372} cy={256} r={16} />
      </g>
    </svg>
  );
}
