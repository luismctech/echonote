import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

// Design system tokens from docs/DESIGN.md §4.
// Colours are defined as CSS custom properties in index.css (light/dark)
// and consumed here via rgb() with <alpha-value> for opacity support.
const rgb = (v: string) => `rgb(var(${v}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', '"SF Mono"', "Consolas", "monospace"],
      },
      fontSize: {
        // Display — hero moments, onboarding, empty states
        "display-xl": ["48px", { lineHeight: "1.05" }],
        "display-lg": ["36px", { lineHeight: "1.1" }],
        "display-md": ["28px", { lineHeight: "1.15" }],
        // UI — primary interface scale (aligned to macOS HIG)
        "ui-lg": ["15px", { lineHeight: "1.45" }],
        "ui-md": ["13px", { lineHeight: "1.4" }],
        "ui-sm": ["12px", { lineHeight: "1.35" }],
        "ui-xs": ["11px", { lineHeight: "1.3" }],
        // Reading — transcripts and long-form notes
        "reading-lg": ["18px", { lineHeight: "1.65" }],
        "reading-md": ["16px", { lineHeight: "1.7" }],
        // Mono — timestamps, code
        "mono-md": ["12px", { lineHeight: "1.5" }],
        // Micro — badges, status indicators
        "micro": ["10px", { lineHeight: "1.3" }],
      },
      colors: {
        surface: {
          base: rgb("--bg-base"),
          elevated: rgb("--bg-elevated"),
          sunken: rgb("--bg-sunken"),
          inset: rgb("--bg-inset"),
        },
        content: {
          primary: rgb("--text-primary"),
          secondary: rgb("--text-secondary"),
          tertiary: rgb("--text-tertiary"),
          placeholder: rgb("--text-placeholder"),
        },
        accent: {
          50: rgb("--accent-50"),
          100: rgb("--accent-100"),
          400: rgb("--accent-400"),
          DEFAULT: rgb("--accent-600"),
          600: rgb("--accent-600"),
          700: rgb("--accent-700"),
          900: rgb("--accent-900"),
        },
        semantic: {
          success: rgb("--color-success"),
          warning: rgb("--color-warning"),
          danger: rgb("--color-danger"),
          info: rgb("--color-info"),
        },
      },
      borderColor: {
        subtle: "var(--border-subtle)",
        DEFAULT: "var(--border-default)",
        strong: "var(--border-strong)",
      },
      boxShadow: {
        sm: "var(--shadow-sm)",
        DEFAULT: "var(--shadow-md)",
        md: "var(--shadow-md)",
        lg: "var(--shadow-lg)",
        xl: "var(--shadow-xl)",
      },
      keyframes: {
        "rec-breathe": {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%": { transform: "scale(1.10)", opacity: "0.85" },
        },
        "rec-ring": {
          "0%": { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(1.8)", opacity: "0" },
        },
        "text-appear": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "progress-glow": {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "overlay-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "overlay-out": {
          "0%": { opacity: "1" },
          "100%": { opacity: "0" },
        },
        "modal-in": {
          "0%": { opacity: "0", transform: "scale(0.95) translateY(8px)" },
          "100%": { opacity: "1", transform: "scale(1) translateY(0)" },
        },
        "modal-out": {
          "0%": { opacity: "1", transform: "scale(1) translateY(0)" },
          "100%": { opacity: "0", transform: "scale(0.95) translateY(8px)" },
        },
        "slide-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "ctx-in": {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
      },
      animation: {
        "rec-breathe": "rec-breathe 3s cubic-bezier(0.32,0.72,0,1) infinite",
        "rec-ring": "rec-ring 1.5s cubic-bezier(0.32,0.72,0,1) infinite",
        "text-appear": "text-appear 150ms ease-out forwards",
        "progress-glow": "progress-glow 2s linear infinite",
        "overlay-in": "overlay-in 200ms ease-out forwards",
        "overlay-out": "overlay-out 150ms ease-in forwards",
        "modal-in": "modal-in 250ms cubic-bezier(0.16,1,0.3,1) forwards",
        "modal-out": "modal-out 150ms ease-in forwards",
        "slide-in": "slide-in 200ms ease-out forwards",
        "ctx-in": "ctx-in 120ms ease-out forwards",
      },
      typography: {
        DEFAULT: {
          css: {
            "--tw-prose-body": rgb("--text-primary"),
            "--tw-prose-headings": rgb("--text-primary"),
            "--tw-prose-links": rgb("--accent-600"),
            "--tw-prose-bold": rgb("--text-primary"),
            "--tw-prose-quotes": rgb("--text-secondary"),
            "--tw-prose-code": rgb("--text-primary"),
            "--tw-prose-hr": "var(--border-default)",
          },
        },
      },
    },
  },
  plugins: [typography],
} satisfies Config;
