# Sprint 1 — Memoria de desarrollo (handoff para día 10)

> **Autor:** Alberto Cruz
> **Última actualización:** 2026-04-20 (post-fix Silero VAD)
> **Branch de trabajo:** `develop` (al día con `origin/develop`)
> **Último tag estable:** `v0.1.0-sprint0` (final de Sprint 0)
> **Propósito:** documento vivo que captura el estado del proyecto al cierre
> del día 9 + trabajo en curso, para poder retomar Sprint 1 día 10 sin
> reconstruir contexto.

Si estás retomando el proyecto y solo puedes leer un archivo, léelo **tras**
`docs/SPRINT-0-RETRO.md`. Este doc asume que conoces la arquitectura base
descrita en `docs/ARCHITECTURE.md` y el plan completo en
`docs/DEVELOPMENT_PLAN.md`.

---

## 1. Dónde estamos (snapshot)

| Área | Estado | Dónde vive |
|---|---|---|
| Streaming mic + Whisper (macOS) | ✅ Producción | `echo-app::streaming` + `echo-asr::whisper_cpp` |
| Captura de audio del sistema macOS | ✅ Funcional (ScreenCaptureKit) | `echo-audio::capture::macos_system` |
| Routing mic vs system-output | ✅ Dos pistas independientes | `echo-audio::capture::routing` |
| VAD energético (RMS) | ✅ Fallback | `echo-app::streaming` |
| **VAD neural Silero v5.1.2** | ✅ **Funcional end-to-end (pendiente commit)** | `echo-audio::preprocess::silero_vad` + `scripts/simplify-silero-vad.py` |
| Diarización (ERes2Net + clustering online) | ✅ End-to-end (audio → DB → UI) | `echo-diarize` + UI |
| Persistencia SQLite + WAL + migraciones | ✅ | `echo-storage` |
| Búsqueda FTS5 en meetings | ✅ | `echo-storage` + sidebar UI |
| Resumen LLM local (Qwen 3 14B) | ✅ Template "general", on-demand | `echo-llm` + UI |
| Chat con la transcripción | ⏳ Siguiente | — |
| Windows / Linux system-audio capture | ⏳ No iniciado | — |
| Onboarding / perfiles Lite-Quality | ⏳ No iniciado | — |
| Encrypted-at-rest (SQLCipher) | ⏳ No iniciado | — |
| Bench matrix extendido | ⏳ Parcial (solo `base.en`) | `docs/benchmarks/PHASE-0.md` |

---

## 2. Qué se construyó en Sprint 1 (día por día)

Timeline de commits desde `v0.1.0-sprint0` (excluye refactors puros y fixes
triviales; la lista completa está en `git log v0.1.0-sprint0..HEAD`).

| Día | Commit(s) | Entrega |
|---|---|---|
| 1 | `e594e84` | UI: ErrorBoundary, capa de toasts, máquina de estados de grabación (`Idle → Recording → Stopping → Persisted`). |
| 2 | `c21c263` | Captura de audio del sistema en macOS vía ScreenCaptureKit (audio-only, sin bot). |
| 3 | `4c214f5` | `RoutingAudioCapture` — mic y system-output como **dos pistas separadas** (requisito de diarización por pista). |
| 4 | `c4e0b11` | Puerto `Vad` en `echo-domain` + adapter **Silero VAD v5** en `echo-audio` (adapter listo, aún no enganchado al pipeline). |
| 5 | `015e62b` | Puerto `Diarizer` + clustering online scaffolding (sin modelo real todavía). |
| 6 | `133cb34`, `dd3ecf6` | **ERes2Net speaker embedder** en ONNX + **ADR-0007** (`docs/adr/0007-diarization-via-onnx-eres2net.md`). Validado cross-language en es-ES y es-MX. |
| 7 | `f4b2ca7`, `cca261d`, `1d47f04` | Diarización end-to-end: Fase A wire al pipeline, Fase B persistir speakers en SQLite, Fase C exponer speakers + rename por Tauri + React. |
| 8 | `115c0d7`, `567c054` | FTS5: Fase A virtual table + migración 0003 + índice diacritic-insensitive. Fase B searchbox en sidebar con BM25 + snippets. Precedido por `f3d373d` smoke-test follow-ups. |
| 9 | `78a9f02` | Resúmenes LLM locales **on-demand** con Qwen 3 14B Q4_K_M (template "general"), IPC + botón "Generate summary" en detalle de reunión. |

Refactors y fixes significativos del medio (no son "features de día" pero son
relevantes para entender el estado del código):

- `c0dfcbe` → `10bfac7`: **reescritura del frontend** en feature-folders
  (`features/meetings`, `features/recording`, `features/transcript`),
  hooks orquestadores (`useMeetingDetail`, `useIpcAction`), `MeetingsContext`.
  El `App.tsx` quedó en ~109 líneas.
- `86a51be`, `e8f17d2`, `706cccf`, `e01a478`: **shutdown ordenado** del
  backend. El crash conocido de `ggml` al liberar Metal fuerza un
  `libc::_exit` final, pero antes hacemos drop de Whisper → checkpoint WAL
  → cerrar streams en orden. Orphan sessions se limpian al arrancar.
- `57a0a7d`: CI corre `pnpm test` también en macOS (antes solo Rust en macOS).
- `c4b2c7c`, `5d23d1c`: **pivote a stack Spanish-first**. Defaults pasaron
  de `base.en` + Qwen 2.5 a `large-v3-turbo` multilingüe + Qwen 3 14B.
  Reparadas URLs de Hugging Face que devolvían 401 con los nombres viejos.

---

## 3. Trabajo en curso **sin commitear** (lo que tendrá que entrar antes del día 10)

Archivos modificados en el working tree (ver `git status`):

```
M README.md
M crates/echo-app/src/use_cases/streaming/mod.rs
M crates/echo-app/src/use_cases/streaming/tests.rs
M crates/echo-asr/src/whisper_cpp.rs
M crates/echo-audio/src/preprocess/silero_vad.rs
M crates/echo-domain/src/entities/streaming.rs
M crates/echo-proto/src/main.rs
M scripts/download-models.sh
M src-tauri/src/commands.rs
?? scripts/simplify-silero-vad.py        # nuevo, ver §3.4
?? docs/SPRINT-1-STATUS.md               # este archivo
```

Contexto: tras pasar al stack Spanish-first (día 9.5), Whisper comenzó a
alucinar en silencio absoluto — principalmente `"Gracias."`,
`"Gracias indefinidamente."` y tokens meta tipo `[no speech]`. Se atacó en
tres capas acumulativas:

### 3.1 Hardening de Whisper `FullParams` — `echo-asr/whisper_cpp.rs`

Parámetros actuales al construir `FullParams`:

- `no_context = true` (no contaminar con chunk anterior, clave en streaming)
- `temperature = 0.0`
- `no_speech_thold = 0.5`
- `logprob_thold = -1.0`
- `entropy_thold = 2.4`
- `suppress_blank = true`
- `suppress_nst = true` (non-speech tokens)
- `SamplingStrategy::Greedy`

Además se añadió un **post-filter** `is_known_hallucination` que descarta
segmentos que sean 100% frases meta (`[música]`, `(Aplausos)`,
`Subtitulado por…`, YouTube outros) o exactamente `"Gracias."` /
`"Gracias indefinidamente."` en chunks cortos.

### 3.2 RMS gate más agresivo

El umbral RMS en `echo-app::streaming` pasó de **0.005 → 0.02**, y los
chunks por debajo emiten `TranscriptEvent::Skipped { reason: "silence" }`
sin invocar a Whisper.

### 3.3 Integración **Silero VAD v5.1.2** en el pipeline (lo nuevo de hoy)

El adapter ya existía del día 4 (`SileroVad` con estado LSTM + histéresis
voiced/silent). Hoy se **enganchó al `StreamingPipeline`** y se expuso en
las fachadas Tauri + CLI. Cambios clave:

- **`SileroVad::clone_for_new_session()`** (en `silero_vad.rs`):
  comparte el modelo optimizado (`Arc<TypedModel>` clone barato) pero
  resetea estado LSTM, carry buffer y histéresis. Evita re-optimizar el
  grafo de tract por cada sesión.
- **`StreamingPipeline::with_vad(Box<dyn Vad>)`** (builder nuevo en
  `echo-app::streaming`). Valida sample rate contra el esperado por el VAD
  al arrancar la sesión, llama `vad.reset()` al inicio de cada sesión, y
  en `process_chunk`, si hay VAD configurado, **bypasea el RMS gate**
  (Silero necesita *todos* los chunks para coherencia temporal) y solo
  transcribe si el VAD devuelve `VoiceState::Voiced`. Si devuelve
  `VoiceState::Silence`, se emite `TranscriptEvent::Skipped`.
- **`MockVad` + 4 tests nuevos** en `streaming/tests.rs`:
  `vad_silence_skips_chunks_even_when_rms_is_loud`,
  `vad_voiced_forwards_chunks_to_transcriber`,
  `vad_bypasses_rms_gate_for_silent_chunks`,
  `vad_is_reset_on_session_start`.
- **Tauri `AppState::ensure_vad()`** (`src-tauri/src/commands.rs`): carga
  perezosa y cacheada del modelo `./models/vad/silero_vad.onnx`. Devuelve
  `Ok(None)` si el archivo no existe, con log de warning (fallback a RMS).
  `start_streaming` lee `StartStreamingOptions { disable_neural_vad: Option<bool> }`
  y, salvo override, engancha una instancia per-sesión via
  `clone_for_new_session`.
- **CLI `echo-proto stream`**: flags `--vad-model <PATH>` (env
  `ECHO_VAD_MODEL`) y `--no-neural-vad`. Imprime `vad=silero` o `vad=rms`
  al arrancar para no tener dudas.

### 3.4 Pre-procesamiento del ONNX de Silero (la solución real, post-3 iteraciones)

> ⚠️ **Esta sección reemplaza una hipótesis previa equivocada**. El primer
> diagnóstico fue *"Silero v6 introdujo `If`, downgrade a v5.1.2"*. **Era
> falso**: todas las releases de v5.x (incluida v5.1.2) **también** tienen
> el operador `If`. Pinear la versión no resolvió nada — el archivo
> remoto del tag v5.1.2 pesa 2.27 MB y carga el mismo `If_0` que crasheaba
> tract.

**Causa raíz real**: el ONNX de Silero v5+ contiene un `If` ONNX que
despacha entre las sub-redes de 16 kHz y 8 kHz según el input `sr`.
`tract-onnx` (backend pure-Rust elegido en ADR-0007 para mantener el
binario sin runtimes nativos) **no implementa `If`** — devuelve
`optimize: Failed analyse for node #5 "If_0" If`. Tampoco soporta los
contrib ops de ONNX Runtime (`FusedConv`, `NchwcConv`, etc.).

**Solución**: `scripts/simplify-silero-vad.py` — un pre-procesador
en Python que se corre **una vez al descargar el modelo** y emite una
versión equivalente para 16 kHz pero sin operadores que tract no entienda:

1. **Inline manual del `If` externo**: como EchoNote siempre opera a
   16 kHz, el `then_branch` se inlinea directamente y el `Equal(sr, 16000)`
   + `If_0` se eliminan.
2. **Drop del input `sr`**: queda huérfano tras el inline.
3. **Lock de shapes estáticas**: `input=[1,512]`, `state=[2,1,128]`.
4. **`onnxruntime` a nivel `ORT_ENABLE_BASIC`** (no `ALL`, no `EXTENDED`):
   pliega los 3 `If` internos restantes que dependían de shape-inference,
   sin introducir contrib ops de ORT. Tabla empírica que justifica el
   nivel:

   | Nivel ORT | Nodos | `If` | `FusedConv` | Tract carga |
   |---|---|---|---|---|
   | `DISABLE_ALL` | 56 | 3 | 0 | ❌ por Ifs |
   | **`BASIC`** | **36** | **0** | **0** | ✅ |
   | `EXTENDED` | 31 | 0 | 5 | ❌ `Unimplemented(FusedConv)` |
   | `ALL` | 31 | 0 | 5 | ❌ `Unimplemented(FusedConv)` |

5. **Verificación de paridad numérica** contra el upstream cacheado en
   `models/vad/silero_vad.onnx.upstream`: las 3 entradas test (ruido,
   silencio, loud) producen Δ = 0.00e+00. Es bitwise-equivalente.

Resultado on-disk: 1.2 MB, 36 nodos, 0 `If`, 0 contrib ops. Las únicas
ops son `Conv, Relu, LSTM, Sigmoid, Sqrt, Pow, Pad, Slice, Add, Concat,
Squeeze, Unsqueeze, Gather, ReduceMean` — todas estándar y soportadas.

**Defensa contra regresiones futuras** (escrita en el script):
- `assert_no_contrib_ops()` aborta si alguien sube el nivel ORT.
- `already_simplified()` chequea **3** invariantes: sin `If`, sin `sr`,
  sin contrib ops (no solo size del archivo).
- Recovery automático desde `.upstream` si detecta un modelo en estado
  intermedio (ej: corrida anterior bug-eada que dejó `FusedConv`).

**Cambios en el adapter Rust** (`silero_vad.rs`):
- `with_input_fact(2, ...)` removido (ya no hay input `sr`).
- `tensor0::<i64>(SILERO_SAMPLE_RATE)` removido del `tvec!` de inferencia.
- Doc del módulo explica por qué el ONNX on-disk es un build modificado.

**Cambios en `download-models.sh`**:
- Cache check ya no confía en el tamaño (un modelo `FusedConv`-tainted
  pesa lo mismo que el correcto). Siempre invoca al simplificador
  (idempotente, ms-rápido en el caso ya-limpio).
- Auto-instala `onnx` + `onnxruntime` con `pip install --user` si faltan.
- Preserva el upstream raw en `.onnx.upstream` para auditoría y recovery.

### 3.5 README actualizado

Se añadió sección "Voice Activity Detection (VAD)" documentando el
sistema de dos niveles (Silero primario + RMS fallback), el flag CLI
`--no-neural-vad` y la opción Tauri `disableNeuralVad`. La tabla de
modelos ahora lista **Silero VAD v5.1.2 (~1.2 MB on disk tras pre-proceso,
~2.2 MB upstream)** y enlaza al script de simplificación.

### 3.6 Verificación hecha (post-fix)

- `cargo fmt --all` ✅
- `cargo clippy -p echo-audio -p echo-app --all-targets -- -D warnings` ✅
- `cargo test -p echo-audio -p echo-app --release` ✅ — **74 tests pasan**
  (31 audio + 43 app), incluidos los 6 específicos de Silero (`loads_…`,
  `pure_silence_stays_silent`, `pure_tone_is_not_classified_as_speech`,
  `detects_speech_in_meeting_fixture`, `reset_…`, `clone_for_new_session_…`)
  y los 4 nuevos de VAD-gated streaming.
- Smoke test del simplifier desde estado limpio + estado tainted ✅
- Tract carga el grafo BASIC sin errores ✅
- Paridad numérica vs upstream: Δ = 0.00e+00 ✅

**Pendiente explícito del usuario**: commitear este trabajo. Cuando se
retome, el primer paso del día 10 es decidir estrategia de commit. Yo
sugiero **tres commits temáticos** ahora que hay un script nuevo:

1. `feat(asr): harden Whisper params + drop known hallucinations`
   — solo `crates/echo-asr/src/whisper_cpp.rs`.
2. `feat(audio): pre-process Silero VAD ONNX for tract compatibility`
   — `scripts/simplify-silero-vad.py` + `scripts/download-models.sh` +
   `crates/echo-audio/src/preprocess/silero_vad.rs` (drop del input `sr`).
3. `feat(streaming): gate chunks through Silero VAD with RMS fallback`
   — `crates/echo-app/src/use_cases/streaming/{mod.rs,tests.rs}` +
   `crates/echo-domain/src/entities/streaming.rs` +
   `crates/echo-proto/src/main.rs` + `src-tauri/src/commands.rs` +
   `README.md` + `docs/SPRINT-1-STATUS.md`.

---

## 4. Estado de los riesgos de Sprint 0

La retro listó 6 riesgos; revisión al cierre del día 9 + trabajo en curso:

| # | Riesgo | Estado | Comentario |
|---|---|---|---|
| R1 | System-audio capture no wireado | ✅ Resuelto en macOS | WASAPI (Win) y PulseAudio loopback (Linux) siguen pendientes |
| R2 | Diarización ausente | ✅ Resuelto | ERes2Net + clustering + persistencia + UI de rename |
| R3 | No FTS5 | ✅ Resuelto | Migración 0003, diacritic-insensitive, BM25 + snippets |
| R4 | Bench gate solo en `base.en` | ⏳ Abierto | Se movió el default a `large-v3-turbo`, el bench *no* se actualizó. **Deuda** |
| R5 | Frontend sin ErrorBoundary | ✅ Resuelto día 1 | |
| R6 | Sin telemetría | ⏳ Abierto | `echo-telemetry` sigue dormido; probablemente Sprint 2 |

**Nuevos riesgos descubiertos en Sprint 1:**

| # | Riesgo | Severidad | Mitigación |
|---|---|---|---|
| R7 | `ggml` crashea al liberar Metal → se fuerza `libc::_exit` | Medio | Shutdown ordenado antes del `_exit` (checkpoint WAL, flush DB). Solución real requiere fix upstream en whisper-rs |
| R8 | `tract-onnx` no soporta `If` ni contrib ops de ORT | ✅ Resuelto | Pre-procesador `simplify-silero-vad.py` (§3.4) emite ONNX equivalente solo con ops estándar. Independiente de la versión de Silero — el mismo pipeline aplica si subimos a v6 |
| R9 | Whisper multilingual grande alucina en silencio más que `base.en` | **Alto** — afectaba UX real | Mitigado en 3 capas (VAD neural + RMS + hallucination filter). Verificar post-commit que no quedan residuos |
| R10 | Qwen 3 14B tarda ~30-60 s en resumir 30 min de audio en M1 Pro | Medio | Aceptable para MVP; Quality profile usará 30B-A3B MoE; Lite volverá a 8B |

---

## 5. Arquitectura — deltas relevantes desde Sprint 0

Sigue valiendo `docs/ARCHITECTURE.md`, pero incorpora estos ajustes:

1. **El dominio ganó un puerto `Vad`** (`echo-domain::ports::vad`) con enum
   `VoiceState::{Voiced, Silence}`. El streaming pipeline hoy tiene VAD
   como `Option<Box<dyn Vad>>` — NO es obligatorio para arrancar.
2. **Dos pistas reales**. El modelo mental de "una pista mono" del Sprint 0
   murió el día 3. Hoy `RoutingAudioCapture` entrega `TrackChunk { track:
   TrackId, samples: Vec<f32> }` y el pipeline cluster-iza por pista.
3. **`Diarizer` port** (`echo-domain::ports::diarizer`) con clustering
   online — mantiene un set de centroides por sesión, decide speaker_id
   por similitud coseno + umbral adaptativo.
4. **Resumen es un proyección**: no mutamos `Meeting`, se guarda en tabla
   aparte `meeting_summaries(meeting_id, template, content_json, created_at)`
   y se regenera a demanda (por eso el botón explícito de "Generate summary").
5. **FTS5 virtual table** `meetings_fts` se refresca vía triggers desde
   `segments` + `meetings.title` + `notes`. Migración 0003.
6. **Shutdown hook ordenado** en `src-tauri/src/setup.rs` garantiza:
   stop streams → drop Whisper/LLM contexts → checkpoint WAL → `_exit(0)`.

Pendiente de ADR formal: VAD neural (habría que escribir **ADR-0008**
para Silero cuando se commitee el trabajo de hoy). El ADR debe cubrir:
elección de Silero v5 sobre WebRTC-VAD/v6, decisión de mantener
`tract-onnx` (en lugar de saltar a `ort`) gracias al pre-procesador,
y el contrato de mantenimiento del script `simplify-silero-vad.py`
(qué nivel ORT, qué invariantes verifica).

---

## 6. Qué sigue — backlog propuesto para día 10 y en adelante

Priorizado. El orden asume que primero cerramos el trabajo en curso.

### 6.1 Limpiar lo pendiente (día 10 exacto)

- [ ] **Commit del trabajo de VAD + hallucinations** (sección 3 arriba).
      **Tres** commits sugeridos (asr / audio-onnx / streaming) + bump de
      README + este SPRINT-1-STATUS en el último.
- [ ] **ADR-0008: Silero VAD v5.1.2 + tract-onnx + simplify pipeline**.
      Documentar la triple decisión: (a) Silero sobre WebRTC-VAD por
      robustez al ruido, (b) `tract-onnx` sobre `ort` por binario sin
      runtime nativo, (c) pre-procesador `scripts/simplify-silero-vad.py`
      como pegamento (con la tabla empírica de niveles ORT del §3.4).
- [ ] **Smoke test real end-to-end**: grabar 2-3 min mitad silencio, mitad
      voz en español, verificar que:
  - Logs muestran `Silero VAD ready` en primer start.
  - En silencio hay `TranscriptEvent::Skipped { reason: "vad_silence" }`.
  - Ningún `Gracias.` fantasma en la UI.
  - `--no-neural-vad` revierte limpio al RMS gate (regresion test).

### 6.2 Features de Sprint 1 todavía abiertas (días 10-12)

- [ ] **Chat con la transcripción** (CU-05 del spec). Puerto nuevo
      `ChatAssistant` en dominio, implementación `LlamaCppChat` en
      `echo-llm`, IPC streaming de tokens a React. Citas a `segment_id`.
      Esto era el "siguiente" explícito del README.
- [ ] **Plantillas de resumen restantes** (1:1, sprint review, entrevista,
      sales, clase). Esqueleto ya existe; solo son prompts + schemas JSON.
- [ ] **Bench matrix en CI**: extender `bench.yml` a `small.en`,
      `medium.en` y `large-v3-turbo` (el default actual). Publicar en
      `docs/benchmarks/PHASE-1/`.

### 6.3 Hacia Sprint 2 (plataforma + onboarding)

- [ ] **WASAPI loopback en Windows** (requiere VM o máquina Windows en CI).
- [ ] **PulseAudio monitor en Linux** (más sencillo, pero necesita detectar
      PipeWire vs Pulse clásico).
- [ ] **Wizard de onboarding** con detección de hardware → perfil sugerido.
- [ ] **Perfiles Lite / Quality** reales (hoy todo corre "Balanced").
- [ ] **Atajos globales de teclado** (pause/resume).
- [ ] **Exportación MD/PDF/TXT/DOCX** (CU-08).

### 6.4 Deuda técnica acumulada

- [ ] Migrar el bench gate de `base.en` al modelo default actual.
- [ ] Despertar `echo-telemetry` (structured logs → archivo rotado;
      opt-in para crash reports).
- [ ] Documentar flujo de diarización + VAD como un diagrama de secuencia
      en `docs/ARCHITECTURE.md` (hoy está descrito en prosa).
- [ ] Evaluar migrar de `tract-onnx` a `ort` si necesitamos operadores
      modernos (GPT-style models), aceptando coste de runtime extra. Por
      ahora `simplify-silero-vad.py` cubre Silero; cualquier modelo nuevo
      con `If` o contrib ops nos forzaría a re-evaluar.
- [ ] Cubrir `simplify-silero-vad.py` con un test de CI (descarga en
      sandbox + ejecuta el script + verifica invariantes), para que un
      bump futuro de Silero no se nos pase de largo.

---

## 7. Cómo retomar mañana — checklist de 5 minutos

```bash
# 1. Sincronizar
git fetch --all --prune
git status                              # confirmar los 11 archivos (9 M + 2 ??)

# 2. Verificar que el build aún pasa
cargo fmt --check
cargo clippy -p echo-audio -p echo-app --all-targets -- -D warnings
cargo test -p echo-app -p echo-audio --release

# 3. Verificar que el modelo Silero está en disco Y simplificado
ls -la models/vad/silero_vad.onnx           # debe ser ~1.2 MB (no 2.2 MB)
ls -la models/vad/silero_vad.onnx.upstream  # backup raw ~2.2 MB
# Si ves 2.2 MB en el primero, el modelo está sin pre-procesar — corre:
bash scripts/download-models.sh vad         # idempotente: detecta y arregla

# 4. Run end-to-end
cargo run -p echo-shell                 # o: pnpm tauri dev
#    → en los logs debe aparecer "Silero VAD ready" la 1ª vez que pulses Start
#    → en silencio ya no debe aparecer "Gracias." ni "[no speech]"

# 5. CLI equivalente (más rápido para iterar)
cargo run -p echo-proto --release -- stream \
    --model models/whisper/ggml-large-v3-turbo.bin \
    --vad-model models/vad/silero_vad.onnx
```

Si algo falla, ordenar así la debugging:

1. **Silero no carga con `Failed analyse for node ... If`** → el modelo en
   disco no fue pre-procesado. Corre `bash scripts/download-models.sh vad`
   o, en última instancia, `python3 scripts/simplify-silero-vad.py
   models/vad/silero_vad.onnx`.
2. **Silero no carga con `Unimplemented(FusedConv)`** → un script previo
   bug-eado dejó el ONNX optimizado a `ORT_ENABLE_ALL`. El simplificador
   detecta el estado tainted y restaura desde `.upstream`; basta con
   re-correr el script.
3. **Cuelga al detener** → revisar que el shutdown hook ordenado sigue
   vivo (`src-tauri/src/setup.rs`).
4. **Hallucinations vuelven** → confirmar que los 3 layers (FullParams,
   RMS gate, filter) siguen intactos; el commit de VAD no debería haber
   tocado `whisper_cpp.rs`.
5. **Frontend en blanco** → mirar consola React y el `ErrorBoundary` (día 1).

---

## 8. Preguntas abiertas para decidir en día 10

1. **¿Convertimos `disable_neural_vad` en un setting persistente?** Hoy es
   solo un parámetro de la invocación (IPC/CLI). Debería vivir en
   `preferences.audio` para que el usuario lo configure desde la UI.
2. **¿Queremos un toggle de "modo estricto" en UI** que muestre los
   `Skipped` events (útil para depurar), o los ocultamos siempre?
3. **¿El chat consume el mismo contexto del LLM que el resumen**, o
   cargamos un modelo más pequeño para chat (latencia vs calidad)?
4. **¿Seguimos con `tract-onnx` o migramos a `ort`?** Decisión de ADR
   propio si migramos — afecta Windows/Linux portability.

---

_Este documento se actualiza al final de cada día de Sprint 1. Si estás
leyendo este archivo **después** del día 10, verifica la fecha del header
y contrasta con `git log --since=<fecha>`._
