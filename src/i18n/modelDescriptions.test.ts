import { describe, expect, it } from "vitest";

import catalogSource from "../../src-tauri/src/commands/models.rs?raw";
import en from "./locales/en.json";
import es from "./locales/es.json";

/**
 * Model ids shipped by the Rust catalog. Read from source so a model
 * added to the catalog without a translated description fails here
 * instead of rendering the untranslated English fallback in the UI.
 */
function catalogIds(): string[] {
  const ids = [...catalogSource.matchAll(/^\s*id: "([a-z0-9-]+)"\.into\(\),$/gm)].map(
    (m) => m[1] ?? "",
  );
  return [...new Set(ids)].sort();
}

const locales: Record<string, Record<string, string>> = {
  en: en.models.desc,
  es: es.models.desc,
};

describe("model descriptions", () => {
  it("reads a non-empty catalog", () => {
    expect(catalogIds()).toContain("embedder-campplus-multi");
  });

  it.each(Object.keys(locales))("%s describes every catalog model", (locale) => {
    const desc = locales[locale] ?? {};
    const missing = catalogIds().filter((id) => !desc[id]);
    expect(missing).toEqual([]);
  });

  it("does not recommend the English-only CAM++ for Spanish", () => {
    expect(en.models.desc["embedder-camplusplus"]).not.toMatch(/spanish/i);
    expect(es.models.desc["embedder-camplusplus"]).not.toMatch(/español/i);
  });
});
