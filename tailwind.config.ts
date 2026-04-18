import type { Config } from "tailwindcss";

// Minimal Tailwind config for Sprint 0. The full design system (tokens,
// typography scale, color palette) lands in Sprint 1 alongside the
// onboarding screens — see docs/DESIGN.md §4.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;
