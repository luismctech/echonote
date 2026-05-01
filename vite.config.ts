import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri exposes the target host + port via env. Binding to host makes the
// dev server reachable from the Tauri webview on non-loopback targets.
// See: https://tauri.app/v2/start/frontend/vite/
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],

  // Prevent Vite from obscuring Rust errors in the terminal.
  clearScreen: false,

  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // Skip the Rust side — tauri dev owns that.
      ignored: ["**/src-tauri/**", "**/target/**", "**/crates/**"],
    },
  },

  // Tauri uses Chromium on Windows/Linux and WebKit on macOS.
  build: {
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    minify: !process.env.TAURI_ENV_DEBUG,
    sourcemap: Boolean(process.env.TAURI_ENV_DEBUG),
  },
});
