/**
 * i18n configuration — initialises `i18next` with `react-i18next`.
 *
 * Languages supported:
 *   - `en` — English (fallback)
 *   - `es` — Spanish
 *
 * The initial language is resolved from `navigator.language` at boot.
 * All translations are bundled statically (no lazy loading / backend),
 * which is appropriate for a desktop app with a small string surface.
 *
 * Import this module once (in `main.tsx`) before rendering `<App />`.
 */

import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en.json";
import es from "./locales/es.json";

/** Detect the best initial language from the OS / browser locale. */
function detectLanguage(): string {
  const nav = navigator.language; // e.g. "es-MX", "en-US"
  if (nav.startsWith("es")) return "es";
  return "en";
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    es: { translation: es },
  },
  lng: detectLanguage(),
  fallbackLng: "en",
  interpolation: {
    escapeValue: false, // React already escapes
  },
});

export default i18n;
