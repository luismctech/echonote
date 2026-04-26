# EchoNote — Plan de Optimización

> Generado: 24 de abril de 2026
> Auditoría: 4 agentes (architect-review, performance-engineer, security-auditor, code-reviewer)
> 55 hallazgos consolidados en 3 fases

---

## Estado General

| Fase | Estado | Commit |
|------|--------|--------|
| **Fase 1 — Seguridad** | ✅ Completada | `79127ab` |
| **Fase 2 — Rendimiento** | ✅ Completada | `0ea7196` |
| **Fase 3 — Arquitectura** | ✅ Completada | `cd392b3` |

---

## Fase 1 — Seguridad ✅

Todas las correcciones de seguridad fueron implementadas y comiteadas.

| ID | Descripción | Archivo | Estado |
|----|-------------|---------|--------|
| SEC-1 | Validación path traversal en `export_meeting` | `src-tauri/src/commands.rs` | ✅ |
| SEC-2 | SHA-256 en descargas de modelos (infraestructura lista, hashes TBD) | `src-tauri/src/commands.rs` | ✅ |
| SEC-3 | Sanitización HTML de snippets FTS5 server-side | `crates/echo-storage/src/sqlite.rs` | ✅ |
| SEC-4 | Eliminar `github.com` / `api.github.com` de CSP `connect-src` | `src-tauri/tauri.conf.json` | ✅ |
| SEC-5 | DB en `app_data_dir` en vez de workspace root | `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs` | ✅ |

### Pendiente menor (SEC-2):
- Poblar los hashes SHA-256 reales en `model_catalog()` ejecutando `shasum -a 256` sobre cada modelo descargado. Actualmente todos están en `None`.

---

## Fase 2 — Rendimiento

### PERF-1: Virtualizar LivePane con @tanstack/react-virtual

**Problema:** `LivePane` renderiza TODOS los transcript lines en un `<ul>` plano sin virtualización. Con sesiones largas (>200 líneas), cada re-render recorre todo el DOM.

**Archivos:**
- `src/features/live/LivePane.tsx` — L143-155: `lines.map((line) => <TranscriptRow .../>)`
- `src/features/live/TranscriptRow.tsx` — L6: `export function TranscriptRow` (sin `React.memo`)

**Implementación:**
1. Instalar `@tanstack/react-virtual`:
   ```bash
   pnpm add @tanstack/react-virtual
   ```
2. En `LivePane.tsx`, reemplazar el `<ul>` con un contenedor virtualizado:
   ```tsx
   import { useVirtualizer } from '@tanstack/react-virtual';

   const parentRef = useRef<HTMLDivElement>(null);
   const virtualizer = useVirtualizer({
     count: lines.length,
     getScrollElement: () => parentRef.current,
     estimateSize: () => 48, // altura estimada por fila
     overscan: 5,
   });
   ```
3. Envolver `TranscriptRow` en `React.memo`:
   ```tsx
   export const TranscriptRow = memo(function TranscriptRow({ line }: Props) {
     // ... sin cambios internos
   });
   ```
4. Auto-scroll al final cuando llegan nuevas líneas (mantener el comportamiento actual).

---

### PERF-2: Virtualizar MeetingDetail + extraer SegmentRow

**Problema:** `MeetingDetail` renderiza segmentos con `m.segments.map(...)` inline en un `<ol>`. No hay componente extraído, no hay memo, no hay virtualización. Reuniones con >500 segmentos son lentas.

**Archivos:**
- `src/features/meetings/MeetingDetail.tsx` — L64-90: inline `<ol>` con `.map()`

**Implementación:**
1. Extraer componente `SegmentRow` memoizado:
   ```tsx
   // src/features/meetings/SegmentRow.tsx
   export const SegmentRow = memo(function SegmentRow({
     segment, speaker, onRenameSpeaker
   }: Props) { ... });
   ```
2. Aplicar `useVirtualizer` al `<ol>` en `MeetingDetail`:
   ```tsx
   const virtualizer = useVirtualizer({
     count: m.segments.length,
     getScrollElement: () => scrollRef.current,
     estimateSize: () => 64,
     overscan: 8,
   });
   ```
3. Mantener el click-to-speaker-rename y scroll-to-segment funcionando con refs virtualizados.

---

### PERF-3: Cache del resampler (SincFixedIn)

**Problema:** `RubatoResamplerAdapter` crea un nuevo `SincFixedIn` con kernel sinc de 256 taps en CADA chunk de 5 segundos. La reconstrucción del filtro es innecesaria porque `(from_hz, to_hz)` nunca cambian dentro de una sesión.

**Archivos:**
- `crates/echo-audio/src/preprocess/resample.rs` — L115-133: `resample_mono()` crea `SincFixedIn::new(...)` en cada llamada
- `crates/echo-audio/src/preprocess/resample.rs` — L150-163: `RubatoResamplerAdapter` es stateless (`#[derive(Clone, Copy)]`)

**Implementación:**
1. Convertir `RubatoResamplerAdapter` en struct con estado:
   ```rust
   pub struct StatefulResampler {
       inner: Option<SincFixedIn<f32>>,
       from_hz: u32,
       to_hz: u32,
   }
   ```
2. Implementar `get_or_create()` que reutiliza el resampler si los parámetros coinciden, o crea uno nuevo si cambian.
3. Actualizar el trait `Resampler` en `echo-domain` si es necesario (cambiar `&self` → `&mut self` o usar interior mutability con `RefCell`).
4. Actualizar `AppState` en `commands.rs` — el campo `resampler` deberá reflejar el nuevo tipo.

**Nota:** Como la pipeline de streaming siempre usa el mismo sample rate, el resampler se construye una sola vez y se reutiliza ~N veces (N = duración_sesión / 5s).

---

### PERF-4: Pool de LlamaContext (opcional)

**Problema:** Cada request de chat/summary crea un nuevo `LlamaContext` short-lived. Para throughput concurrente esto está bien, pero el spin-up de context tiene overhead.

**Archivos:**
- `src-tauri/src/commands.rs` — L271-334: `ensure_llm_concrete()` + `ensure_llm()`

**Implementación (baja prioridad):**
1. Mantener el singleton `Arc<LlamaCppLlm>` actual para weights.
2. Añadir un pool de `LlamaContext` con tamaño configurable (default 2).
3. `checkout()` devuelve un context disponible o espera.
4. `checkin()` retorna el context al pool.

**Decisión:** Evaluar si el spin-up de context realmente es un cuello de botella medible antes de implementar. El patrón actual (per-request) es correcto para concurrencia.

---

### PERF-5: Cap en array de líneas en streaming

**Problema:** `useRecordingSession.ts` acumula líneas sin límite: `setLines((prev) => [...prev, newLine])`. Una sesión de 2 horas genera ~1,440 entries, cada re-render spread del array completo.

**Archivos:**
- `src/hooks/useRecordingSession.ts` — L65: `const [lines, setLines] = useState<StreamLine[]>([])`
- `src/hooks/useRecordingSession.ts` — L140-157: `setLines((prev) => [...prev, newLine])`

**Implementación:**
1. Definir `MAX_LIVE_LINES = 500` (configurable).
2. Cambiar el append:
   ```ts
   setLines((prev) => {
     const next = [...prev, newLine];
     return next.length > MAX_LIVE_LINES ? next.slice(-MAX_LIVE_LINES) : next;
   });
   ```
3. Opcionalmente mostrar un indicador "[N líneas anteriores ocultas]" en la UI.

**Nota:** El backend no necesita cap — los eventos se consumen via `mpsc` channel uno a uno, no se bufferean.

---

### Resumen Fase 2

| ID | Tarea | Prioridad | Complejidad |
|----|-------|-----------|-------------|
| PERF-1 | Virtualizar LivePane + memo TranscriptRow | 🔴 Alta | Media | ✅ |
| PERF-2 | Virtualizar MeetingDetail + extraer SegmentRow | 🔴 Alta | Media | ✅ |
| PERF-3 | Cache del resampler SincFixedIn | 🟡 Media | Baja | ✅ |
| PERF-4 | Pool de LlamaContext | 🟢 Baja | Alta | ⏭️ Diferido |
| PERF-5 | Cap en array de líneas live | 🟡 Media | Baja | ✅ |

**Orden sugerido:** PERF-5 → PERF-3 → PERF-1 → PERF-2 → PERF-4 (de simple a complejo)

---

## Fase 3 — Arquitectura

### ARCH-1: Dividir commands.rs en módulos

**Problema:** `commands.rs` tiene **1,783 líneas** — mezcla streaming, meetings CRUD, LLM, export, model management. Dificulta navegación y code review.

**Archivos:**
- `src-tauri/src/commands.rs` — 1,783 líneas

**Implementación:**
1. Crear módulos bajo `src-tauri/src/commands/`:
   ```
   src-tauri/src/commands/
   ├── mod.rs          # re-exports, AppState struct
   ├── streaming.rs    # start_streaming, stop_streaming, SessionEntry
   ├── meetings.rs     # list, get, delete, rename_speaker, search
   ├── llm.rs          # summarize, ask, ensure_llm, ensure_chat
   ├── export.rs       # export_meeting, render_markdown, render_plain_text
   └── models.rs       # model_catalog, get_model_status, download_model
   ```
2. `AppState` queda en `mod.rs` con los `ensure_*` helpers.
3. Cada módulo importa `AppState` y define sus `#[tauri::command]` functions.
4. `lib.rs` no cambia — sigue referenciando `commands::*`.

---

### ARCH-2: Extraer LazyModel<T> genérico

**Problema:** 4 campos de `AppState` repiten el patrón `AsyncMutex<Option<Arc<T>>>` + `spawn_blocking` + error mapping:
- `transcriber` (Whisper) — L393
- `llm` (LlamaCpp) — L271
- `vad` (SileroVad) — L344
- `build_diarizer()` — L361 (sin cache, per-session)

**Archivos:**
- `src-tauri/src/commands.rs` — `ensure_transcriber()` L393, `ensure_llm_concrete()` L271, `ensure_vad()` L344

**Implementación:**
```rust
pub struct LazyModel<T> {
    inner: AsyncMutex<Option<Arc<T>>>,
}

impl<T: Send + Sync + 'static> LazyModel<T> {
    pub fn new() -> Self {
        Self { inner: AsyncMutex::new(None) }
    }

    pub async fn get_or_init<E>(
        &self,
        init: impl FnOnce() -> Result<T, E> + Send + 'static,
    ) -> Result<Arc<T>, E>
    where
        E: Send + 'static,
    {
        let mut guard = self.inner.lock().await;
        if let Some(ref model) = *guard {
            return Ok(Arc::clone(model));
        }
        let model = tokio::task::spawn_blocking(init)
            .await
            .expect("spawn_blocking panicked")?;
        let arc = Arc::new(model);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }
}
```

Luego `AppState`:
```rust
transcriber: LazyModel<dyn Transcriber>,
llm: LazyModel<LlamaCppLlm>,
vad: LazyModel<SileroVad>,
```

---

### ARCH-3: Mover export a echo-app

**Problema:** La lógica de renderizado Markdown/TXT (`render_markdown`, `render_plain_text`, `render_summary_body_md`, `render_summary_body_txt`) es lógica de aplicación, no infraestructura de shell. Está en `commands.rs` L1136-1465.

**Archivos:**
- `src-tauri/src/commands.rs` — L1136 (`render_summary_body_md`), L1328 (`render_markdown`), L1387 (`render_plain_text`), L1440 (`render_summary_body_txt`)

**Implementación:**
1. Crear `crates/echo-app/src/use_cases/export.rs`.
2. Mover las 4 funciones de renderizado.
3. Crear un use case `ExportMeeting` que recibe `(Meeting, Option<MeetingSummary>, ExportFormat)` → `String`.
4. En `commands.rs`, `export_meeting` hace: validación de path + llama a `ExportMeeting::execute()` + escribe.

---

### ARCH-4: parking_lot::Mutex para sessions (opcional)

**Problema original reportado:** `std::sync::Mutex` tiene riesgo de "poison" si un thread panickea.

**Análisis actual:** El uso de `std::sync::Mutex` en `sessions` es **correcto** — locks breves, sin `.await` durante el lock, nunca cruza puntos de suspensión async. Poison solo ocurre si un thread panickea con el lock tomado, lo cual es improbable aquí.

**Implementación (baja prioridad):**
1. Agregar `parking_lot = "0.12"` a `src-tauri/Cargo.toml`.
2. Reemplazar `std::sync::Mutex` por `parking_lot::Mutex` en el campo `sessions`.
3. `parking_lot::Mutex` no tiene poison — `lock()` retorna `MutexGuard` directamente, sin `Result`.

**Decisión:** Solo implementar si hay preferencia por la ergonomía de `parking_lot`. No hay bug funcional actual.

---

### ARCH-5: Tests de integración para streaming pipeline

**Problema:** Solo existen unit tests inline en `crates/echo-app/src/use_cases/streaming/tests.rs` con mocks (`FakeStream`, `FakeTranscriber`). No hay tests que crucen la frontera Tauri IPC ni que usen adaptadores reales.

**Archivos:**
- `crates/echo-app/src/use_cases/streaming/tests.rs` — tests existentes (happy path, silence gate, sub-chunk, stop)

**Implementación:**
1. Crear `crates/echo-app/tests/streaming_integration.rs`.
2. Usar `RubatoResamplerAdapter` real + `SqliteMeetingStore` in-memory.
3. Mock solo el audio capture (`FakeCapture` que reproduce un `.wav` de fixtures).
4. Verificar end-to-end: audio → resample → transcribe → persist → query.
5. Agregar test de concurrencia: 2 sesiones simultáneas no interfieren.

---

### Resumen Fase 3

| ID | Tarea | Prioridad | Complejidad |
|----|-------|-----------|-------------|
| ARCH-1 | Dividir commands.rs en módulos | 🔴 Alta | Media | ✅ `91296da` |
| ARCH-2 | Extraer LazyModel\<T\> genérico | 🟡 Media | Baja | ✅ `91296da` |
| ARCH-3 | Mover export a echo-app | 🟡 Media | Media | ✅ `cd392b3` |
| ARCH-4 | parking_lot::Mutex para sessions | 🟢 Baja | Baja | ⏭️ Diferida |
| ARCH-5 | Tests integración streaming | 🟡 Media | Alta | ✅ `cd392b3` |

**Orden sugerido:** ARCH-2 → ARCH-1 → ARCH-3 → ARCH-5 → ARCH-4

---

## Contexto Técnico de Referencia

### Stack
- **Backend:** Rust 1.88+ / Tauri 2.10.3 / whisper-rs 0.16 / llama-cpp-2 0.1.144 / tract-onnx 0.22 / sqlx 0.8
- **Frontend:** React 18 / TypeScript 5.6.3 / Vite 5.4 / Tailwind / shadcn/ui / Zustand
- **Specta:** specta 2.0.0-rc.22 / tauri-specta 2.0.0-rc.21 / specta-typescript 0.0.9

### Convenciones
- Conventional commits obligatorios (sujeto ≤ 100 chars)
- Pre-commit: `cargo fmt --all --check` + `cargo clippy (workspace, -D warnings)` + `pnpm typecheck`
- `bindings.ts` auto-generado por tauri-specta → requiere `// @ts-nocheck` como primera línea
- Commit via `printf ... > /tmp/commit_msg.txt && git commit -F /tmp/commit_msg.txt`
- Git email: `albertomzcruz@gmail.com` / Branch: `develop`

### AppState (13 campos)
```
capture, resampler, transcriber, model_path, embed_model_path,
store, recorder, rename_speaker, sessions, llm_model_path,
llm, vad_model_path, vad
```
