# Proyecto Echo — Documento de arquitectura

**Versión:** 0.1
**Fecha:** 17 de abril de 2026
**Estado:** Borrador para revisión técnica

---

## Tabla de contenidos

1. [Principios arquitectónicos](#1-principios-arquitectónicos)
2. [Vista de alto nivel](#2-vista-de-alto-nivel)
3. [Stack tecnológico y justificaciones](#3-stack-tecnológico-y-justificaciones)
4. [Arquitectura por capas](#4-arquitectura-por-capas)
5. [Estructura del workspace](#5-estructura-del-workspace)
6. [Patrones de comunicación](#6-patrones-de-comunicación)
7. [Gestión de estado](#7-gestión-de-estado)
8. [Persistencia y modelo de datos](#8-persistencia-y-modelo-de-datos)
9. [Seguridad y privacidad](#9-seguridad-y-privacidad)
10. [Observabilidad](#10-observabilidad)
11. [Build, distribución y actualización](#11-build-distribución-y-actualización)
12. [Estrategia de testing](#12-estrategia-de-testing)
13. [Escalabilidad y evolución](#13-escalabilidad-y-evolución)
14. [Decisiones arquitectónicas (ADRs)](#14-decisiones-arquitectónicas-adrs)

---

## 1. Principios arquitectónicos

Estos principios son la constitución del proyecto. Cuando una decisión técnica presente un dilema, se resuelve a favor del principio listado primero.

### 1.1 Principios ordenados por prioridad

1. **Privacidad por diseño.** El audio, las transcripciones y los resúmenes nunca salen del equipo del usuario por defecto. Toda ruta de salida externa es explícita, opt-in, y auditable.

2. **Portabilidad antes que optimización específica.** Las tres plataformas (Windows, macOS, Linux) son ciudadanas de primera clase. Ninguna feature se considera "terminada" hasta funcionar en las tres.

3. **Separación de responsabilidades (Clean Architecture).** El núcleo de dominio (transcripción, diarización, resumen) no conoce a Tauri, React, SQLite ni a whisper.cpp. Depende de abstracciones, no de implementaciones.

4. **Inmutabilidad del audio del usuario.** El audio crudo nunca se modifica después de capturarse. Los pipelines de procesamiento operan sobre copias o vistas; los errores de procesamiento no corrompen el original.

5. **Fallar visible, no silenciosamente.** Cada error tiene un código, un mensaje al usuario cuando aplica, y telemetría local (opt-in para envío remoto).

6. **Reversibilidad.** Cada acción del usuario (incluida eliminar reuniones) tiene un período de gracia. Ningún dato se borra de disco inmediatamente.

7. **Testabilidad antes que elegancia.** Si una abstracción no se puede testear sin un entorno complejo, se refactoriza. El módulo de dominio debe correr en cualquier máquina sin permisos especiales.

### 1.2 Anti-principios (qué evitamos explícitamente)

- **No sobre-ingeniería especulativa.** No construimos abstracciones para features de v2 hoy. Refactorizamos cuando la necesidad sea real.
- **No arquitectura distribuida prematura.** Esto es una app de escritorio, no un sistema de microservicios. La simplicidad del monolito modular gana.
- **No frameworks dentro de frameworks.** Si Tauri + React ya resuelven un problema, no metemos Zustand + Redux + Jotai. Una solución por problema.
- **No ocultar complejidad inherente.** Si una operación es costosa (cargar un modelo de 2 GB), la UI debe mostrarlo honestamente.

---

## 2. Vista de alto nivel

### 2.1 Diagrama de bloques

```
┌─────────────────────────────────────────────────────────────────┐
│                    Proyecto Echo (monolito modular)              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Capa de presentación                  │    │
│  │  React 18 + TypeScript + Tailwind + shadcn/ui + Zustand │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                    Tauri IPC (commands + events)                 │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐    │
│  │                Capa de aplicación (Rust)                 │    │
│  │    Orquestación de casos de uso · sesiones · flujos     │    │
│  └────┬──────────┬──────────┬──────────┬──────────┬───────┘    │
│       │          │          │          │          │              │
│  ┌────┴────┐┌───┴────┐┌────┴────┐┌───┴────┐┌───┴────┐           │
│  │ Audio   ││Transcr.││Diariz.  ││  LLM   ││Storage │           │
│  │ Capture ││ (ASR)  ││ Speaker ││Service ││(SQLite)│           │
│  └────┬────┘└───┬────┘└────┬────┘└───┬────┘└───┬────┘           │
│       │         │          │         │         │                 │
│  ┌────┴─────────┴──────────┴─────────┴─────────┴────────┐       │
│  │              Adaptadores a bibliotecas nativas        │       │
│  │    cpal · WASAPI · CoreAudio · PulseAudio ·          │       │
│  │    whisper.cpp · llama.cpp · ONNX Runtime · SQLite    │       │
│  └───────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                      Sistema operativo host
```

### 2.2 Flujo principal de datos

```
[Micrófono] ─┐
             ├─► Audio Engine ─► Preproceso ─► VAD ─► ASR ─► Diarización
[Sistema]   ─┘                                                     │
                                                                    ▼
                                                              Merge & Align
                                                                    │
                                                                    ▼
                                            ┌──────────────────────┴──┐
                                            ▼                         ▼
                                      SQLite (FTS5)            LLM (resumen)
                                            │                         │
                                            └────────┬────────────────┘
                                                     ▼
                                                  Frontend
```

---

## 3. Stack tecnológico y justificaciones

### 3.1 Resumen ejecutivo

| Capa | Tecnología | Versión mínima |
|---|---|---|
| Framework de app | Tauri | 2.0 |
| Backend | Rust | 1.75 (edición 2021) |
| Frontend | React + TypeScript | 18.3 + TS 5.5 |
| Estilos | Tailwind CSS + shadcn/ui | Tailwind 3.4 |
| Estado frontend | Zustand + TanStack Query | Zustand 4.5 + TQ 5 |
| Runtime ASR | whisper.cpp via `whisper-rs` | whisper-rs 0.13 |
| Runtime LLM | llama.cpp via `llama-cpp-rs` | 0.1.x |
| ML inference | ONNX Runtime via `ort` | ort 2.0 |
| Base de datos | SQLite + SQLCipher | 3.45 |
| Captura audio mic | cpal | 0.15 |
| Captura audio sistema | wasapi (Win) / screencapturekit (Mac) / libpulse (Linux) | — |
| IPC | Tauri commands & events | — |
| Build frontend | Vite | 5.x |
| Build backend | Cargo | estable |

### 3.2 Justificaciones por elección

#### 3.2.1 Tauri 2.x sobre Electron

**Contexto:** Necesitamos una app de escritorio multiplataforma con captura de audio nativa e integración con librerías ML escritas en C/C++.

**Decisión:** Tauri 2.x con frontend React.

**Justificación:**
- Bundle 10-100× más pequeño que Electron (2-10 MB vs 80-150 MB).
- Consumo de RAM en reposo 30-40 MB vs 200-300 MB de Electron — crítico para una app siempre activa.
- Rust como backend permite FFI directo a whisper.cpp, llama.cpp y ONNX Runtime sin capas adicionales.
- Modelo de permisos basado en ACL (allowlist por capability) facilita auditorías de seguridad.
- Tauri 2.0 soporta iOS y Android si en v2 queremos app móvil.

**Trade-offs aceptados:**
- Curva de aprendizaje de Rust para el equipo backend.
- Ecosistema de plugins menor que Electron (compensable con código Rust propio).
- Webview nativo por OS puede tener inconsistencias menores (mitigable con testing).

#### 3.2.2 React + TypeScript sobre Vue/Svelte/Solid

**Decisión:** React 18 con TypeScript estricto (`strict: true`).

**Justificación:**
- Ecosistema más grande para componentes de UI avanzados (editor rico, virtualización).
- shadcn/ui ofrece componentes de alta calidad copiables (no dependencia npm).
- TypeScript estricto reduce bugs en runtime en ~40% según estudios internos de adopción.
- React Server Components no son relevantes aquí (app de escritorio), así que quedamos en React clásico.

#### 3.2.3 Zustand + TanStack Query sobre Redux/MobX

**Decisión:** Zustand para estado UI, TanStack Query para datos del backend.

**Justificación:**
- Zustand: API mínima (un `create((set) => ...)`), zero-boilerplate, 1.2 KB gzipped.
- TanStack Query maneja cache, invalidación y sincronización de datos de Tauri automáticamente.
- Separación clara: UI state (Zustand) vs server state (Query) es un patrón probado.
- Redux sería over-engineering para esta app; no hay tiempo-viaje ni necesidades similares.

#### 3.2.4 Tailwind CSS + shadcn/ui sobre MUI/Chakra

**Decisión:** Tailwind 3.4 con componentes de shadcn/ui copiados al repositorio.

**Justificación:**
- shadcn/ui no es una librería npm; son componentes que copias y posees. Sin lock-in.
- Tailwind produce CSS más pequeño que librerías CSS-in-JS y sin runtime overhead.
- Radix UI (base de shadcn) tiene accesibilidad impecable por defecto.
- Diseño coherente fácil de personalizar sin pelear con estilos heredados.

#### 3.2.5 whisper.cpp sobre faster-whisper

**Decisión:** whisper.cpp vía `whisper-rs` para ASR.

**Justificación:**
- C++ puro, estáticamente enlazable. No requiere Python runtime.
- Soporta CoreML en Mac (Neural Engine), CUDA, Vulkan y Metal.
- Bindings Rust maduros en `whisper-rs`.
- Distribución limpia: un solo binario con todo lo necesario.

**Trade-off aceptado:** faster-whisper es 20-30% más rápido en CPU pura, pero el costo de empaquetar Python (PyOxidizer, venvs, compatibilidad de versiones) es prohibitivo para distribución masiva.

#### 3.2.6 llama.cpp sobre Ollama/vLLM/TGI

**Decisión:** llama.cpp vía `llama-cpp-rs` embebido.

**Justificación:**
- Mismo razonamiento que whisper.cpp: biblioteca C++ enlazable sin dependencias externas.
- Soporta GGUF, el formato dominante para modelos cuantizados.
- Permite que la app funcione sin Ollama instalado, pero puede usar Ollama del usuario si existe.
- Ollama es un excelente **frontend** de llama.cpp para desarrolladores, pero agrega una dependencia pesada para usuarios no técnicos.

#### 3.2.7 SQLite + FTS5 sobre Postgres embebido/DuckDB

**Decisión:** SQLite con extensión FTS5 y SQLCipher para cifrado opcional.

**Justificación:**
- Estándar de facto para apps de escritorio. Un archivo, cero configuración.
- FTS5 ofrece búsqueda full-text con ranking BM25 nativo, suficiente para nuestro caso.
- SQLCipher cifra el archivo completo con AES-256 si el usuario lo elige.
- Tooling masivo: migraciones con `sqlx`, browser en DB Browser, backups triviales.

### 3.3 Dependencias explícitamente rechazadas

| Biblioteca | Razón de rechazo |
|---|---|
| Electron | Huella de recursos inaceptable (ver 3.2.1) |
| Python (faster-whisper, pyannote) | Fricción de empaquetado multi-plataforma |
| Ollama (como dependencia obligatoria) | Agrega complejidad de instalación; lo soportamos como opcional |
| Redux | Over-engineering para el scope actual |
| Material UI | Bundle grande, difícil de personalizar |
| IndexedDB | No aplica fuera del browser; SQLite es superior para apps nativas |
| Prisma | Complejidad innecesaria sobre `sqlx` en Rust |

---

## 4. Arquitectura por capas

Seguimos un modelo inspirado en **Clean Architecture** y **Hexagonal Architecture**, adaptado al contexto de una app Tauri.

### 4.1 Las cuatro capas

```
┌─────────────────────────────────────────────────┐
│  1. Presentación (React)                         │
│     · Componentes, vistas, routing              │
│     · Estado UI local (Zustand)                 │
│     · Traducción de eventos/comandos a Rust     │
├─────────────────────────────────────────────────┤
│  2. Aplicación (Rust)                            │
│     · Casos de uso ("StartRecording", etc.)     │
│     · Orquestación entre servicios de dominio   │
│     · Gestión de sesiones                       │
├─────────────────────────────────────────────────┤
│  3. Dominio (Rust puro, sin deps externas)       │
│     · Entidades: Meeting, Segment, Speaker      │
│     · Reglas de negocio puras                   │
│     · Puertos (traits) para servicios externos  │
├─────────────────────────────────────────────────┤
│  4. Infraestructura (Rust + adapters)            │
│     · Implementaciones concretas de puertos     │
│     · whisper-rs, llama-cpp-rs, sqlx, cpal      │
│     · Específico de plataforma (Win/Mac/Linux)  │
└─────────────────────────────────────────────────┘

Regla de oro: las flechas de dependencia solo van hacia abajo.
Dominio NO conoce Aplicación. Aplicación NO conoce Presentación.
Infraestructura implementa las interfaces definidas por Dominio.
```

### 4.2 Ejemplo concreto: iniciar una grabación

```rust
// --- Capa de dominio (puro, sin deps) ---
// crates/echo-domain/src/ports/audio.rs

pub trait AudioCapturePort: Send + Sync {
    fn start(&mut self, config: CaptureConfig) -> Result<CaptureHandle, DomainError>;
    fn stop(&mut self, handle: CaptureHandle) -> Result<CaptureResult, DomainError>;
}

// --- Capa de aplicación ---
// crates/echo-app/src/use_cases/start_recording.rs

pub struct StartRecordingUseCase<A: AudioCapturePort, S: SessionStore> {
    audio: A,
    sessions: S,
}

impl<A, S> StartRecordingUseCase<A, S>
where
    A: AudioCapturePort,
    S: SessionStore,
{
    pub async fn execute(&mut self, input: StartRecordingInput)
        -> Result<SessionId, AppError>
    {
        let config = CaptureConfig::from(&input);
        let handle = self.audio.start(config)?;
        let session = Session::new(handle, input.participant_hints);
        self.sessions.save(&session).await?;
        Ok(session.id)
    }
}

// --- Capa de infraestructura ---
// crates/echo-infra/src/audio/windows.rs (solo en target Windows)

pub struct WasapiAudioCapture { /* estado interno */ }

impl AudioCapturePort for WasapiAudioCapture {
    fn start(&mut self, config: CaptureConfig) -> Result<CaptureHandle, DomainError> {
        // Usa wasapi-rs para abrir streams de mic y loopback
        // ...
    }
}

// --- Capa de presentación (Tauri command) ---
// src-tauri/src/commands/recording.rs

#[tauri::command]
pub async fn echo_start_recording(
    state: State<'_, AppState>,
    input: StartRecordingInput,
) -> Result<String, String> {
    state.use_cases
        .start_recording
        .execute(input)
        .await
        .map(|id| id.to_string())
        .map_err(|e| e.to_string())
}
```

### 4.3 Beneficios de esta separación

- **Testabilidad:** el dominio y la aplicación se testean sin tocar audio real, usando mocks de los puertos.
- **Portabilidad:** agregar soporte a una nueva plataforma = escribir una nueva implementación de `AudioCapturePort`. Cero cambios en capas superiores.
- **Evolución:** reemplazar whisper.cpp por un modelo nuevo = cambiar una implementación del puerto `TranscriberPort`. Cero cambios en casos de uso.

---

## 5. Estructura del workspace

### 5.1 Layout de directorios

```
echo/
├── Cargo.toml                      # Workspace raíz
├── rust-toolchain.toml             # Versión de Rust fija
├── package.json                    # Dependencias de frontend
├── pnpm-lock.yaml                  # Lock determinista (pnpm sobre npm)
├── README.md
├── CONTRIBUTING.md
├── LICENSE
│
├── crates/                         # Crates Rust del workspace
│   ├── echo-domain/                # Capa de dominio (puro)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── entities/
│   │       │   ├── meeting.rs
│   │       │   ├── segment.rs
│   │       │   └── speaker.rs
│   │       ├── ports/
│   │       │   ├── audio.rs
│   │       │   ├── transcriber.rs
│   │       │   ├── diarizer.rs
│   │       │   ├── llm.rs
│   │       │   └── storage.rs
│   │       └── errors.rs
│   │
│   ├── echo-app/                   # Casos de uso
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── use_cases/
│   │       │   ├── start_recording.rs
│   │       │   ├── stop_recording.rs
│   │       │   ├── generate_summary.rs
│   │       │   ├── chat_with_transcript.rs
│   │       │   └── rename_speaker.rs
│   │       └── services/
│   │           ├── session_manager.rs
│   │           └── event_bus.rs
│   │
│   ├── echo-audio/                 # Captura y preproceso
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── capture/
│   │       │   ├── mod.rs
│   │       │   ├── windows.rs      #[cfg(target_os = "windows")]
│   │       │   ├── macos.rs        #[cfg(target_os = "macos")]
│   │       │   └── linux.rs        #[cfg(target_os = "linux")]
│   │       ├── preprocess/
│   │       │   ├── resample.rs
│   │       │   ├── denoise.rs
│   │       │   └── vad.rs
│   │       └── buffer.rs           # Ring buffer
│   │
│   ├── echo-asr/                   # Whisper wrapper
│   ├── echo-diarize/               # Diarización
│   ├── echo-llm/                   # LLM wrapper
│   ├── echo-storage/               # Persistencia
│   └── echo-telemetry/             # Logs y métricas
│
├── src-tauri/                      # Binario Tauri principal
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── capabilities/               # Permisos Tauri v2
│   │   └── default.json
│   └── src/
│       ├── main.rs
│       ├── commands/               # Comandos IPC
│       │   ├── mod.rs
│       │   ├── recording.rs
│       │   ├── meetings.rs
│       │   ├── settings.rs
│       │   └── models.rs
│       ├── events/                 # Eventos emitidos
│       │   └── mod.rs
│       ├── state.rs                # AppState compartido
│       └── setup.rs                # Inyección de dependencias
│
├── src/                            # Frontend React
│   ├── main.tsx
│   ├── App.tsx
│   ├── routes/                     # TanStack Router
│   │   ├── index.tsx
│   │   ├── meeting/$id.tsx
│   │   └── settings.tsx
│   ├── features/                   # Por feature, no por tipo
│   │   ├── recording/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── stores/
│   │   │   └── api.ts
│   │   ├── transcription/
│   │   ├── summary/
│   │   ├── library/
│   │   └── settings/
│   ├── components/                 # Componentes compartidos
│   │   └── ui/                     # shadcn/ui components
│   ├── lib/
│   │   ├── tauri.ts                # Wrappers de invoke/listen
│   │   └── utils.ts
│   ├── stores/                     # Zustand stores globales
│   └── styles/
│       └── globals.css
│
├── models/                         # Gitignored. Solo en dev.
│   └── .gitkeep
│
├── tests/
│   ├── fixtures/                   # Audios de prueba
│   ├── e2e/                        # Playwright
│   └── integration/                # Tests que cruzan capas
│
└── .github/
    └── workflows/
        ├── ci.yml                  # Tests en matriz OS
        ├── release.yml             # Build y publish
        └── nightly.yml             # Builds nocturnos
```

### 5.2 Principios de organización

- **Features sobre tipos en frontend.** `features/recording/` en vez de `components/`, `hooks/`, `stores/` separados. Cada feature es autocontenida.
- **Crates pequeños y enfocados.** Cada crate tiene una responsabilidad clara. Facilita compilación incremental.
- **Un binario, múltiples crates.** `src-tauri` es el único binario; el resto son librerías.
- **Tests cerca del código.** Unit tests en `#[cfg(test)] mod tests` dentro de cada archivo. Integration tests en `tests/` separado.


---

## 6. Patrones de comunicación

### 6.1 Frontend ↔ Backend (Tauri IPC)

Usamos dos mecanismos de IPC según el patrón:

**Commands** (request/response síncrono):
```typescript
// Para operaciones discretas que esperan una respuesta
const session = await invoke<SessionInfo>("echo_start_recording", {
  participantHints: ["Ana", "Carlos"],
});
```

**Events** (streaming asíncrono):
```typescript
// Para flujos continuos que no deben bloquear
const unlisten = await listen<StreamingSegment>(
  "echo:streaming_segment",
  (event) => { updateTranscript(event.payload); }
);
```

**Reglas:**
- Los comandos devuelven en < 100 ms. Si una operación es más larga (procesar un audio), el comando retorna inmediatamente con un `job_id` y el progreso se emite por eventos.
- Los eventos se emiten con un prefijo consistente (`echo:audio_level`, `echo:streaming_segment`).
- Todos los payloads se validan con un schema TypeScript generado desde Rust (`ts-rs` o `specta`).

### 6.2 Comunicación entre capas Rust

**Dentro de un proceso: canales tokio y trait objects.**

```rust
// Canal multi-productor, multi-consumidor para eventos internos
use tokio::sync::broadcast;

pub struct EventBus {
    tx: broadcast::Sender<DomainEvent>,
}

// Cada subscriber recibe todos los eventos
let mut rx = event_bus.subscribe();
while let Ok(event) = rx.recv().await {
    match event {
        DomainEvent::SegmentTranscribed(segment) => { /* ... */ }
        DomainEvent::RefinementComplete(result) => { /* ... */ }
        _ => {}
    }
}
```

**Entre threads de trabajo: canales mpsc.**

```rust
// El capturador envía frames al procesador
let (audio_tx, audio_rx) = tokio::sync::mpsc::channel::<AudioFrame>(1024);
```

**Regla:** los canales tienen tamaños acotados siempre. `unbounded` está prohibido fuera de casos muy excepcionales — un productor rápido puede agotar memoria.

### 6.3 Manejo de errores

**En dominio:** enums de error exhaustivos con `thiserror`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("audio device unavailable: {0}")]
    AudioDeviceUnavailable(String),

    #[error("model not loaded: {0}")]
    ModelNotLoaded(ModelId),

    #[error("invalid session state: {0}")]
    InvalidSessionState(String),
}
```

**En aplicación:** wrap de errores de dominio más errores de orquestación.

```rust
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error(transparent)]
    Domain(#[from] DomainError),

    #[error("use case precondition failed: {0}")]
    Precondition(String),
}
```

**En la frontera Tauri:** los errores se serializan a un shape estable para el frontend.

```rust
#[derive(Serialize)]
pub struct ErrorResponse {
    code: String,        // "AUDIO_DEVICE_UNAVAILABLE"
    message: String,     // Mensaje humano
    recoverable: bool,
    details: Option<Value>,
}
```

**En frontend:** los errores van a un handler global que decide entre mostrar toast, modal o silencio según severidad.

---

## 7. Gestión de estado

### 7.1 Estado en frontend

Dividimos el estado en cuatro categorías:

| Categoría | Herramienta | Ejemplo |
|---|---|---|
| UI efímero (local) | `useState` | Toggle de sidebar, input en curso |
| UI compartido | Zustand store | Tema actual, usuario logueado |
| Estado del servidor (Rust) | TanStack Query | Lista de reuniones, configuración |
| Estado en tiempo real | Zustand + listeners | Nivel de audio, transcript en vivo |

### 7.2 Ejemplo de store Zustand

```typescript
// src/features/recording/stores/recording-store.ts
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface RecordingState {
  sessionId: string | null;
  status: 'idle' | 'recording' | 'stopping' | 'refining';
  micLevel: number;
  systemLevel: number;
  streamingSegments: StreamingSegment[];

  start: (hints?: string[]) => Promise<void>;
  stop: () => Promise<void>;
  appendSegment: (segment: StreamingSegment) => void;
}

export const useRecordingStore = create<RecordingState>()(
  subscribeWithSelector((set, get) => ({
    sessionId: null,
    status: 'idle',
    micLevel: 0,
    systemLevel: 0,
    streamingSegments: [],

    start: async (hints) => {
      set({ status: 'recording', streamingSegments: [] });
      const id = await invoke<string>('echo_start_recording', { hints });
      set({ sessionId: id });
    },

    stop: async () => {
      set({ status: 'stopping' });
      const id = get().sessionId!;
      await invoke('echo_stop_recording', { sessionId: id });
      set({ status: 'refining' });
    },

    appendSegment: (segment) => {
      set(state => ({
        streamingSegments: [...state.streamingSegments, segment],
      }));
    },
  }))
);
```

### 7.3 Estado en backend

**Regla:** el estado mutable vive en un solo lugar por tipo, detrás de un `Mutex` o `RwLock` específico.

```rust
// src-tauri/src/state.rs
pub struct AppState {
    pub use_cases: Arc<UseCases>,
    pub sessions: Arc<RwLock<SessionRegistry>>,
    pub config: Arc<RwLock<AppConfig>>,
    pub model_manager: Arc<ModelManager>,
}
```

`Arc<RwLock<T>>` para estado consultado frecuentemente, `Arc<Mutex<T>>` para estado mutado frecuentemente. `dashmap` para estructuras concurrentes de alto tráfico.

---

## 8. Persistencia y modelo de datos

### 8.1 Ubicación y formato

Los datos viven en el directorio estándar de la plataforma:

- **Windows:** `%APPDATA%\Echo\`
- **macOS:** `~/Library/Application Support/Echo/`
- **Linux:** `~/.local/share/Echo/` (respeta XDG_DATA_HOME)

Estructura:
```
Echo/
├── echo.db                 # SQLite principal
├── echo.db-wal             # Write-Ahead Log
├── models/                 # Modelos descargados
├── audio-cache/            # Audios temporales (TTL configurable)
├── logs/                   # Rotados, max 30 MB total
└── config.toml             # Configuración user-level
```

### 8.2 Esquema de base de datos

```sql
-- Tabla de reuniones
CREATE TABLE meetings (
    id TEXT PRIMARY KEY,                    -- UUID v7 (sortable)
    title TEXT NOT NULL,
    started_at INTEGER NOT NULL,            -- Unix epoch ms
    ended_at INTEGER,
    duration_ms INTEGER,
    language TEXT,                           -- 'es', 'en', 'auto'
    folder_id TEXT,
    template_id TEXT,                        -- 'one_on_one', 'sales_call'
    audio_retained INTEGER DEFAULT 0,        -- 0/1, si se conserva audio
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
);
CREATE INDEX idx_meetings_started_at ON meetings(started_at DESC);
CREATE INDEX idx_meetings_folder ON meetings(folder_id);

-- Segmentos de transcripción
CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    speaker_id TEXT,                         -- nullable hasta diarización
    track TEXT NOT NULL,                     -- 'mic' | 'system'
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL,
    is_refined INTEGER DEFAULT 0,            -- 0 = streaming, 1 = refined
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE SET NULL
);
CREATE INDEX idx_segments_meeting ON segments(meeting_id, start_ms);

-- Speakers detectados
CREATE TABLE speakers (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    cluster_label TEXT NOT NULL,             -- 'local_01', 'remote_02'
    display_name TEXT,                       -- Nombre asignado por usuario
    embedding BLOB,                          -- Para v1.1: reconocimiento
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

-- Notas del usuario
CREATE TABLE notes (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    content TEXT NOT NULL,                   -- Markdown
    ai_enhanced INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

-- Resúmenes generados
CREATE TABLE summaries (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    template_id TEXT NOT NULL,
    content TEXT NOT NULL,                   -- JSON estructurado
    model_used TEXT NOT NULL,
    generated_at INTEGER NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

-- Carpetas
CREATE TABLE folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES folders(id) ON DELETE CASCADE
);

-- FTS5 virtual table para búsqueda
CREATE VIRTUAL TABLE segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers para mantener FTS sincronizado
CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER segments_au AFTER UPDATE ON segments BEGIN
    UPDATE segments_fts SET text = new.text WHERE rowid = new.rowid;
END;
CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
    DELETE FROM segments_fts WHERE rowid = old.rowid;
END;

-- Versionado de esquema
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
```

### 8.3 Migraciones

Usamos `sqlx::migrate!` que lee de `crates/echo-storage/migrations/`:

```
migrations/
├── 20260101000000_initial.sql
├── 20260115000000_add_folders.sql
└── 20260201000000_add_speakers_embedding.sql
```

**Reglas:**
- Las migraciones son forward-only. Nunca editamos una migración aplicada.
- Cada migración es atómica. Si falla a medias, `sqlx` hace rollback.
- Los esquemas se versionan; al arrancar la app, se aplican migraciones pendientes automáticamente.

### 8.4 Respaldo y recuperación

- Backup automático en cada cierre limpio de la app (copia del `.db` a `echo.db.backup`).
- Export manual desde Settings: genera un ZIP con DB + notas + audios si se conservan.
- Import: valida el ZIP, verifica schema, y merge de reuniones con IDs nuevos si hay conflicto.

---

## 9. Seguridad y privacidad

### 9.1 Amenazas consideradas

| Amenaza | Mitigación |
|---|---|
| Malware lee audio del mic sin permiso | Uso de APIs del OS que requieren consentimiento explícito |
| Usuario pierde equipo con reuniones sensibles | SQLCipher opcional con passphrase en keychain del OS |
| App envía datos sin consentimiento | Network allowlist estricto; auditable en código |
| Actualizaciones maliciosas | Firma de código con cert EV; verificación SHA256 en updater |
| Modelos descargados maliciosos | Checksum verificado contra manifiesto firmado por Echo |
| XSS vía transcripción con código | Escapado estricto en el renderer; sanitización con DOMPurify |

### 9.2 Red

La app solo hace tráfico de red en estos casos:

1. **Descarga de modelos** desde `https://models.echo-app.io/` (CDN propio + mirror HF).
2. **Check de actualizaciones** a `https://updates.echo-app.io/`.
3. **Telemetría de crashes** a `https://crash.echo-app.io/` (opt-in, sin PII).

**Todo lo demás está bloqueado** a nivel de Tauri capabilities:

```json
// src-tauri/capabilities/default.json
{
  "identifier": "default",
  "permissions": [
    "core:default",
    "http:default",
    {
      "identifier": "http:allow-fetch",
      "allow": [
        { "url": "https://models.echo-app.io/**" },
        { "url": "https://updates.echo-app.io/**" },
        { "url": "https://crash.echo-app.io/**" }
      ]
    }
  ]
}
```

### 9.3 Permisos del sistema operativo

**macOS** requiere los siguientes entitlements:
- `com.apple.security.device.audio-input` — micrófono
- `com.apple.security.device.microphone` — redundante por TCC
- `NSMicrophoneUsageDescription` en Info.plist
- `NSScreenCaptureDescription` en Info.plist — ScreenCaptureKit pide esto aunque solo usemos audio

**Windows:**
- `microphone` en el manifest de la app (Windows 10+).
- El loopback de WASAPI no requiere permiso explícito.

**Linux:**
- Generalmente sin prompts; PulseAudio/PipeWire permite por defecto.
- Detectamos si el usuario está en el grupo `audio` y lo sugerimos si no.

### 9.4 Cifrado en reposo

- **Por defecto:** SQLite sin cifrado (el usuario no paga costo de rendimiento si no lo necesita).
- **Opcional:** el usuario habilita cifrado en Settings → se genera una passphrase, se guarda en keychain del OS (`keyring-rs`), se migra la DB con `PRAGMA rekey`.
- Los modelos y audios nunca se cifran (no hay PII ahí).

### 9.5 Telemetría

**Principio:** "lo mínimo para que Echo mejore, y el usuario siempre puede decir no."

Enviamos (opt-in, solicitado al primer crash):
- Tipo de crash y stack trace (con paths absolutos redactados)
- Versión de Echo, versión de OS, arquitectura
- Perfil de hardware elegido (Lite/Balanced/Quality)

NUNCA enviamos:
- Contenido de transcripciones, notas o resúmenes
- Nombres de archivos de audio
- Identificadores personales (email, nombre del usuario)
- Dirección IP (proxy de Anthropic/Sentry lo anonimiza)

---

## 10. Observabilidad

### 10.1 Logging

Usamos `tracing` (Rust) con tres sinks:

1. **Consola** (solo en dev, con `RUST_LOG=debug`).
2. **Archivo rotado** (`~/Library/Application Support/Echo/logs/echo.log`, max 30 MB, 3 archivos).
3. **Sentry-like remoto** solo para errores nivel `ERROR` y si el usuario aceptó telemetría.

```rust
use tracing::{info, error, instrument};

#[instrument(skip(self))]
pub async fn execute(&mut self, input: StartRecordingInput) -> Result<...> {
    info!("starting recording session");
    match self.audio.start(config) {
        Ok(handle) => {
            info!(?handle, "capture started");
            Ok(handle)
        }
        Err(e) => {
            error!(error = %e, "capture failed");
            Err(e.into())
        }
    }
}
```

### 10.2 Métricas locales

Contadores y histogramas guardados en SQLite para el panel de diagnóstico del usuario:

- Tiempo de transcripción por minuto de audio
- RAM pico durante refinamiento
- Latencia de streaming
- Tasa de reinicios de captura (indicador de problemas de driver)

### 10.3 Health checks

En Settings → Diagnóstico, un panel ejecuta:

- Verifica modelos descargados (integridad SHA256)
- Prueba captura de audio (2 segundos de mic + sistema)
- Prueba inferencia de whisper con audio de 5 s
- Prueba inferencia de LLM con prompt corto
- Reporta "todo en orden" o problemas específicos

---

## 11. Build, distribución y actualización

### 11.1 Build local

```bash
# Desarrollo
pnpm install
pnpm tauri dev

# Producción
pnpm tauri build
```

Produce binarios en `src-tauri/target/release/bundle/`:
- Windows: `.msi`, `.exe` (NSIS)
- macOS: `.dmg`, `.app`
- Linux: `.deb`, `.rpm`, `.AppImage`

### 11.2 CI/CD

**GitHub Actions** con matriz en los tres OS:

```yaml
# .github/workflows/release.yml (resumen)
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-22.04
            target: x86_64-unknown-linux-gnu
          - os: ubuntu-22.04
            target: aarch64-unknown-linux-gnu
          - os: macos-14
            target: universal-apple-darwin    # Apple Silicon + Intel
          - os: windows-latest
            target: x86_64-pc-windows-msvc
```

Cada release:
1. Etiqueta de versión semver (`v1.2.3`) dispara el workflow.
2. Build en paralelo en los 3 OS.
3. Firma de código (certificado EV en Windows, Developer ID en Mac).
4. Notarización en macOS.
5. Publicación en GitHub Releases + CDN propio.

### 11.3 Auto-update

Usamos el updater de Tauri 2.x:

```json
// tauri.conf.json
{
  "plugins": {
    "updater": {
      "active": true,
      "endpoints": [
        "https://updates.echo-app.io/{{target}}/{{current_version}}"
      ],
      "pubkey": "...",
      "windows": { "installMode": "passive" }
    }
  }
}
```

El servidor de updates devuelve un manifiesto firmado que Tauri verifica antes de aplicar. Updates son opcionales por defecto; el usuario elige "automático", "manual" o "solo críticos" en Settings.

### 11.4 Distribución de modelos

Los modelos **no** se empaquetan con la app (crecería a 2-5 GB). Se descargan bajo demanda:

1. Al primer arranque, wizard detecta hardware y propone perfil.
2. Descarga desde `https://models.echo-app.io/{profile}/{model}.{variant}.gguf`.
3. Verificación SHA256 contra manifiesto firmado.
4. Progreso visible al usuario, pausable y resumible.

---

## 12. Estrategia de testing

### 12.1 Pirámide

```
        ╱ E2E (Playwright) ╲       10%   — Flujos completos críticos
       ╱────────────────────╲
      ╱  Integration (Rust)  ╲    30%   — Casos de uso con infra real
     ╱────────────────────────╲
    ╱      Unit (Rust + TS)    ╲  60%   — Funciones puras, componentes
   ╱────────────────────────────╲
```

### 12.2 Cobertura por capa

| Capa | Tipo de test | Herramienta | Meta cobertura |
|---|---|---|---|
| Dominio | Unit puro | `cargo test` | > 90% |
| Aplicación | Unit con mocks | `cargo test` + `mockall` | > 80% |
| Infraestructura | Integration | `cargo test` con fixtures | > 60% |
| Tauri commands | Integration | `tauri-test` | > 50% |
| Frontend utils | Unit | Vitest | > 70% |
| Frontend componentes | Component | Vitest + Testing Library | > 50% |
| Flujos críticos | E2E | Playwright | 5 flujos clave |

### 12.3 Fixtures de audio

20 audios etiquetados en `tests/fixtures/audio/`:
- 10 en español (variantes: clean/noisy, 1-5 speakers, 5-15 min)
- 10 en inglés (mismas variantes)
- 2 de code-switching ES-EN
- 2 con interrupciones y solapamientos

Cada uno con `.json` de ground truth (transcripción esperada, speakers, idioma).

### 12.4 Tests críticos de regresión

Una suite que NUNCA debe fallar para considerar un release listo:

```rust
#[test]
fn wer_spanish_clean_single_speaker_below_10pct() { ... }
#[test]
fn wer_english_clean_single_speaker_below_8pct() { ... }
#[test]
fn der_two_speakers_below_20pct() { ... }
#[test]
fn no_audio_loss_in_30min_recording() { ... }
#[test]
fn refinement_completes_in_under_90s_for_30min_audio() { ... }
```

---

## 13. Escalabilidad y evolución

### 13.1 Qué está diseñado para escalar

- **Nuevas plataformas:** agregar ARM Windows, ChromeOS = nueva implementación de `AudioCapturePort`.
- **Nuevos modelos ASR:** implementar `TranscriberPort` con la nueva librería. Cero cambios arriba.
- **Nuevos LLMs:** mismo patrón con `LlmPort`. Soporte de Ollama, LM Studio, etc. como adaptadores.
- **Móvil (v2.0):** Tauri 2.x soporta iOS/Android. UI se adapta con responsive; core Rust se recompila para ARM.
- **Modo team (v2.0):** sincronización opcional vía servidor auto-hosted. Agregar `SyncPort`.

### 13.2 Qué NO está diseñado para escalar

- **Multi-tenant SaaS:** Echo es app de escritorio. Si se quiere SaaS, es un producto diferente.
- **Miles de reuniones por día:** SQLite se comporta bien hasta ~100 GB; más allá requiere re-diseño.
- **Streaming a múltiples consumidores simultáneos:** el event bus es in-process.

### 13.3 Roadmap arquitectónico

| Versión | Capacidad arquitectónica añadida |
|---|---|
| v1.0 | Todo lo descrito en este documento |
| v1.1 | Reconocimiento cross-sesión de speakers (activa `speaker embedding registry`) |
| v1.2 | Integración con Google Calendar / Outlook como plugin |
| v2.0 | App móvil con sync opcional |
| v2.5 | Modo "cloud-assisted" opcional con proveedores LLM externos |

---

## 14. Key Architecture Decisions

- Tauri 2.x over Electron
- Rust + React as the base stack
- whisper.cpp over faster-whisper
- llama.cpp over Ollama as embedded runtime
- Separate audio track capture (mic and system)
- Hybrid streaming + refinement pipeline
- Speaker diarization via ONNX embeddings (3D-Speaker) over pyannote
- Zustand + TanStack Query over Redux
- SQLite + FTS5 with optional SQLCipher
- Clean Architecture with ports and adapters

---

## Apéndice A — Glosario

| Término | Definición |
|---|---|
| ADR | Architecture Decision Record |
| ASR | Automatic Speech Recognition |
| Clean Architecture | Patrón de Robert C. Martin con inversión de dependencias |
| DER | Diarization Error Rate |
| Diarización | Proceso de determinar "quién habló cuándo" |
| FFI | Foreign Function Interface |
| FTS | Full-Text Search |
| GGUF | Formato de cuantización de llama.cpp |
| IPC | Inter-Process Communication |
| Puerto | Interfaz abstracta definida por el dominio |
| RTF | Real-Time Factor (tiempo_proceso / tiempo_audio) |
| VAD | Voice Activity Detection |
| WER | Word Error Rate |

---

**Este documento es un living document.** Cambios significativos requieren un ADR. Cambios menores (correcciones, clarificaciones) pueden hacerse por PR directo.
