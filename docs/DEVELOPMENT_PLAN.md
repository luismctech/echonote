# Proyecto Echo — Plan de desarrollo y especificación

**Versión:** 0.1
**Fecha:** 17 de abril de 2026
**Estado:** Borrador para revisión

---

## Tabla de contenidos

1. [Visión del producto](#1-visión-del-producto)
2. [Alcance del MVP](#2-alcance-del-mvp)
3. [Especificación funcional](#3-especificación-funcional)
4. [Especificación no-funcional](#4-especificación-no-funcional)
5. [Plan de desarrollo por fases](#5-plan-de-desarrollo-por-fases)
6. [Desglose de trabajo (WBS)](#6-desglose-de-trabajo-wbs)
7. [Criterios de aceptación](#7-criterios-de-aceptación)
8. [Equipo y roles](#8-equipo-y-roles)
9. [Gestión de riesgos](#9-gestión-de-riesgos)
10. [Definiciones de "listo"](#10-definiciones-de-listo)
11. [Proceso de desarrollo](#11-proceso-de-desarrollo)

---

## 1. Visión del producto

### 1.1 Statement

> Echo es la alternativa privada a Granola: una app de escritorio multiplataforma que transcribe y resume reuniones con IA que corre 100% en tu equipo. Sin nubes. Sin suscripciones. Sin bots.

### 1.2 Problema que resolvemos

Profesionales que viven en reuniones necesitan capturar y revisar conversaciones sin escribir manualmente, pero las soluciones actuales:

- **Envían audio a servidores externos** (bloqueador para abogados, consultores, personal de salud y cualquiera con NDAs estrictos).
- **Cobran por minuto o por asiento**, costo que escala mal.
- **Se unen como bots a la llamada**, rompiendo el flujo y requiriendo permisos explícitos en cada reunión.

### 1.3 Solución

Echo captura audio directamente del dispositivo (sin unirse a la llamada), transcribe y resume localmente con modelos open source cuantizados. El resultado es una app:

- **Privada por diseño**: datos nunca salen del equipo.
- **Portable**: un binario nativo pequeño, funciona offline.
- **Multiplataforma desde día uno**: Windows, macOS, Linux.
- **Un solo pago** (o gratis en su versión core, por definir).

### 1.4 Métricas de éxito

| Métrica | Horizonte | Meta |
|---|---|---|
| Downloads activos en los primeros 6 meses | 6 meses | > 5,000 |
| Retention semana 4 | 6 meses | > 35% |
| NPS | 6 meses | > 40 |
| WER en español conversacional | Release | < 10% |
| WER en inglés conversacional | Release | < 8% |
| Crash rate | Continuo | < 0.5% sesiones |
| Tiempo a primera transcripción funcional (onboarding) | Release | < 5 minutos |

---

## 2. Alcance del MVP

### 2.1 Lo que entra (in-scope)

El MVP cubre el flujo crítico completo de captura → transcripción → resumen → consulta.

**Funcionalidades core:**
- Captura dual (mic + audio del sistema) en Windows, macOS y Linux.
- Transcripción híbrida (streaming en vivo + refinamiento al terminar) en español e inglés.
- Diarización por pista con clustering local.
- Detección de voces múltiples en micrófono y sistema.
- Pre-asignación opcional de participantes pegando una lista.
- Resumen estructurado al terminar la reunión.
- Chat con la transcripción usando LLM local.
- Biblioteca de reuniones con búsqueda full-text.
- Exportación a Markdown, PDF, TXT, DOCX.
- Tres perfiles de hardware (Lite/Balanced/Quality) con wizard inicial.
- Auto-update.

**Features de UX:**
- Onboarding guiado que detecta hardware y configura automáticamente.
- Notas manuales del usuario durante la reunión, mezcladas con IA al final.
- Plantillas de resumen: general, 1:1, sprint review, entrevista, llamada de ventas, clase.
- Editor post-generación del resumen.
- Atajos de teclado globales (pausa/reanudar grabación).
- Tema claro y oscuro, con auto-detección del sistema.

### 2.2 Lo que NO entra (out-of-scope para MVP)

Estas funcionalidades quedan planeadas para versiones posteriores:

- Reconocimiento cross-sesión de speakers (v1.1).
- Integración con Google Calendar / Outlook (v1.1).
- App móvil iOS/Android (v2.0).
- Sincronización entre dispositivos (v2.0).
- Colaboración team-wide (v2.0).
- Plugins / API pública (v2.0).
- Integración con CRMs (Salesforce, HubSpot) (v1.2+).
- Modo cloud-assisted con LLMs externos (v2.5).
- Soporte multilingüe amplio (>2 idiomas) (v1.2).
- Traducción simultánea (fuera de roadmap por ahora).

### 2.3 Supuestos del MVP

- El usuario tiene permisos de admin en su equipo para instalar software.
- El usuario tiene al menos 8 GB de RAM y 8 GB de espacio libre.
- El usuario entiende qué es una transcripción y un resumen generado por IA.
- Las reuniones duran típicamente 15-90 minutos.
- La mayoría de usuarios tienen al menos conexión a Internet para descargar modelos inicialmente (aunque luego funcionan offline).

---

## 3. Especificación funcional

### 3.1 Casos de uso principales

#### CU-01: Iniciar una grabación

**Actor:** Usuario

**Precondición:** App instalada, permisos de mic concedidos.

**Flujo principal:**
1. Usuario abre Echo o usa atajo global.
2. Usuario opcionalmente pega una lista de participantes en un campo de texto.
3. Usuario presiona "Grabar".
4. Sistema valida permisos de audio (mic + sistema).
5. Sistema carga modelos si no están en memoria (típicamente ya cargados).
6. Sistema abre streams de captura de mic y sistema.
7. Sistema muestra indicador de grabación activo en la ventana principal y en la barra de sistema.
8. Sistema muestra niveles de audio de ambas pistas en tiempo real.

**Flujos alternativos:**
- 4a. Permisos no concedidos: Sistema muestra dialog explicativo con botón "Abrir ajustes".
- 5a. Modelos no descargados: Sistema muestra progress bar de descarga con opción de cancelar.
- 6a. Dispositivo de audio no disponible: Sistema muestra mensaje con lista de dispositivos disponibles.

**Postcondición:** Sesión de grabación activa, transcripción en streaming visible.

---

#### CU-02: Transcripción en vivo

**Actor:** Sistema (automatizado durante grabación).

**Precondición:** Grabación en curso.

**Flujo principal:**
1. Sistema recibe frames de audio a 16 kHz mono de ambas pistas.
2. Silero VAD filtra silencios por pista.
3. Por cada chunk de 5 segundos con voz, Whisper small procesa y emite texto.
4. Sistema marca el segmento como "tentativo" (tono visual más suave).
5. Sistema emite evento `echo:streaming_segment` al frontend.
6. Frontend añade el segmento a la vista de transcripción.

**Criterios de rendimiento:**
- Latencia desde palabra dicha hasta aparición en UI: < 4 segundos.
- RTF (Real-Time Factor) en CPU de 8 núcleos: < 0.3.

---

#### CU-03: Finalizar grabación y refinar

**Actor:** Usuario + Sistema.

**Precondición:** Grabación en curso.

**Flujo principal:**
1. Usuario presiona "Detener".
2. Sistema cierra streams de captura.
3. Sistema pasa audio completo a la rama de refinamiento.
4. Sistema ejecuta Whisper medium (o small en perfil Lite) sobre cada pista completa.
5. Sistema aplica diarización por embeddings en cada pista.
6. Sistema mergea resultados por timestamp.
7. Sistema aplica matching con `participant_hints` si fueron provistos.
8. Sistema muestra transcripción refinada con speakers etiquetados.
9. Sistema automáticamente dispara generación de resumen (CU-04).

**Criterios de rendimiento:**
- Tiempo total desde stop hasta transcripción refinada: < 90 s para 30 min de audio.
- Progreso visible al usuario con porcentaje estimado.

---

#### CU-04: Generar resumen

**Actor:** Sistema (automatizado post-grabación).

**Precondición:** Transcripción refinada disponible.

**Flujo principal:**
1. Sistema selecciona plantilla (por defecto "general" salvo que se haya elegido otra antes).
2. Sistema construye prompt con transcripción + instrucciones del template.
3. Sistema invoca LLM local con el prompt.
4. Sistema parsea respuesta JSON estructurada.
5. Sistema muestra resumen con secciones: resumen breve, puntos clave, decisiones, acciones, preguntas pendientes.
6. Sistema guarda el resumen en la DB.

**Criterios de rendimiento:**
- Tiempo de generación: < 45 s para una transcripción de 30 min de audio.
- Si falla JSON parsing, reintentar 1 vez; si vuelve a fallar, mostrar resumen en texto libre.

---

#### CU-05: Chat con transcripción

**Actor:** Usuario.

**Precondición:** Reunión finalizada con transcripción refinada.

**Flujo principal:**
1. Usuario escribe pregunta en el campo "Pregunta a Echo".
2. Sistema construye prompt con: la transcripción + historial de chat previo + pregunta actual.
3. Sistema invoca LLM local.
4. Sistema streamea la respuesta al usuario (tokens aparecen incrementalmente).
5. Respuesta se guarda en el historial de chat de esa reunión.

---

#### CU-06: Buscar en reuniones

**Actor:** Usuario.

**Precondición:** Al menos una reunión guardada.

**Flujo principal:**
1. Usuario escribe en el campo de búsqueda de la biblioteca.
2. Sistema consulta SQLite con FTS5 sobre texto de segmentos + notas + resúmenes.
3. Sistema devuelve resultados ranqueados por BM25.
4. Sistema muestra resultados con snippet del contexto y link a la reunión.

**Criterios de rendimiento:**
- Resultados visibles en < 200 ms para bibliotecas de hasta 1000 reuniones.

---

#### CU-07: Renombrar speakers

**Actor:** Usuario.

**Precondición:** Reunión finalizada con speakers detectados.

**Flujo principal:**
1. Usuario hace clic en un label de speaker ("remoto_01").
2. Sistema muestra un input editable con sugerencias de los `participant_hints` no usados.
3. Usuario escribe o selecciona un nombre.
4. Sistema actualiza todos los segmentos de ese cluster con el nuevo nombre.

---

#### CU-08: Exportar reunión

**Actor:** Usuario.

**Precondición:** Reunión con contenido.

**Flujo principal:**
1. Usuario hace clic en "Exportar" y elige formato (MD, PDF, TXT, DOCX).
2. Sistema genera el documento con: título, fecha, participantes, resumen, transcripción completa.
3. Sistema abre diálogo de guardar archivo.
4. Usuario elige ubicación.

---

### 3.2 Plantillas de resumen

El MVP incluye 6 plantillas con prompts específicos. Cada plantilla produce un JSON con campos distintos.

#### 3.2.1 General (default)

```json
{
  "summary": "Resumen en 2-3 oraciones.",
  "key_points": ["Punto clave 1", "Punto clave 2"],
  "decisions": ["Decisión con contexto"],
  "action_items": [
    { "task": "...", "owner": "Ana", "due": "lunes" }
  ],
  "open_questions": ["Pregunta sin respuesta"]
}
```

#### 3.2.2 1:1

```json
{
  "summary": "...",
  "wins": ["Logros mencionados"],
  "blockers": ["Obstáculos"],
  "growth_feedback": ["Feedback de crecimiento"],
  "next_steps": [...],
  "follow_up_topics": ["Temas para próximo 1:1"]
}
```

#### 3.2.3 Sprint review

```json
{
  "summary": "...",
  "completed_items": [...],
  "carry_over": [...],
  "risks": [...],
  "next_sprint_priorities": [...]
}
```

#### 3.2.4 Entrevista (user research / hiring)

```json
{
  "summary": "...",
  "quotes": [
    { "speaker": "Ana", "quote": "...", "context": "..." }
  ],
  "themes": ["Tema recurrente"],
  "pain_points": [...],
  "opportunities": [...]
}
```

#### 3.2.5 Llamada de ventas

```json
{
  "summary": "...",
  "customer_context": "...",
  "pain_points": [...],
  "interest_signals": [...],
  "objections": [...],
  "next_steps": [...],
  "deal_stage_indicator": "discovery | evaluation | proposal | negotiation"
}
```

#### 3.2.6 Clase / lección

```json
{
  "summary": "...",
  "concepts_covered": [...],
  "definitions": [ { "term": "...", "definition": "..." } ],
  "examples": [...],
  "homework_or_next": [...]
}
```

### 3.3 Configuración y preferencias

Settings expone las siguientes categorías:

**Audio:**
- Dispositivo de mic (dropdown).
- Dispositivo de loopback (dropdown, avanzado).
- Sensibilidad del VAD (slider).
- Reducir ruido de fondo (toggle).

**Transcripción:**
- Idioma por defecto (auto / español / inglés).
- Modelo ASR (según perfil, con override manual).
- Guardar audio después de grabar (toggle).

**Resumen y chat:**
- Plantilla por defecto (dropdown).
- Modelo LLM (según perfil, con override).
- Temperature (slider, avanzado).

**Interfaz:**
- Tema (claro / oscuro / sistema).
- Idioma de la UI (español / inglés).
- Atajos de teclado (customizables).
- Mostrar transcripción en vivo (toggle).

**Privacidad:**
- Cifrar base de datos (toggle → pide passphrase).
- Telemetría de errores (toggle, off por defecto).
- Retención de audio (nunca / 7 días / 30 días / siempre).

**Avanzado:**
- Ruta de modelos.
- Ruta de base de datos.
- Exportar configuración.
- Restablecer de fábrica.

---

## 4. Especificación no-funcional

### 4.1 Rendimiento

| Métrica | Perfil Lite | Perfil Balanced | Perfil Quality |
|---|---|---|---|
| RAM en reposo | < 250 MB | < 350 MB | < 500 MB |
| RAM activa (recording) | < 2 GB | < 4 GB | < 8 GB |
| Binario instalador | < 50 MB | < 50 MB | < 50 MB |
| Tamaño total con modelos | ~1.1 GB | ~4.5 GB | ~15 GB |
| Latencia streaming | < 5 s | < 4 s | < 3 s |
| Tiempo de refinamiento (30 min audio) | < 120 s | < 90 s | < 45 s |
| Tiempo generación resumen | < 60 s | < 45 s | < 25 s |

### 4.2 Calidad

| Métrica | Meta |
|---|---|
| WER español (clean, 1-2 speakers) | < 10% |
| WER inglés (clean, 1-2 speakers) | < 8% |
| DER (2 speakers) | < 20% |
| DER (3-5 speakers) | < 30% |
| Crash rate | < 0.5% sesiones |
| Data loss rate | 0% |

### 4.3 Usabilidad

- **Time to first value:** < 5 minutos desde descarga hasta ver primer resumen generado.
- **Accesibilidad:** WCAG 2.1 AA en la UI principal. Navegación completa por teclado. Lectores de pantalla soportados.
- **Internacionalización:** UI en español e inglés. Fechas y números respetan locale del OS.
- **Responsive:** funcional en ventanas desde 900×600 hasta pantalla completa 4K.

### 4.4 Confiabilidad

- **Recuperación ante fallo de audio:** si el dispositivo se desconecta, la app muestra alerta y ofrece seguir grabando con pista disponible.
- **Recuperación ante fallo de OS:** si la app crashea durante grabación, al reabrir se ofrece recuperar la sesión desde el audio cacheado.
- **Backup automático:** base de datos respaldada en cada cierre limpio.

### 4.5 Seguridad

Ver sección 9 del documento de arquitectura. Resumen:
- Red allowlisted (solo dominios de Echo).
- Cifrado AES-256 opcional de la DB.
- Firma de código en todas las plataformas.
- Notarización en macOS.

### 4.6 Compatibilidad

| Plataforma | Versión mínima soportada |
|---|---|
| Windows | 10 (build 1809+) 64-bit |
| macOS | 12.3 (Monterey) |
| Linux | Ubuntu 22.04, Fedora 38, o equivalente con glibc 2.35+ |

Arquitecturas soportadas en release: x86_64 en todas, ARM64 en macOS y Windows.

---

## 5. Plan de desarrollo por fases

### 5.1 Overview de fases

| Fase | Duración | Entregable |
|---|---|---|
| Fase 0 — Discovery | 4-6 semanas | Prototipo CLI + benchmarks + validación con usuarios |
| Fase 1 — Alpha interna | 8-10 semanas | App funcional, solo perfil Balanced, 10-20 testers |
| Fase 2 — Beta pública | 6-8 semanas | MVP completo, 500-1000 usuarios beta |
| Fase 3 — Release v1.0 | 4 semanas | Lanzamiento público |
| Fase 4 — Iteración v1.1 | 12 semanas | Reconocimiento speakers, calendarios |

Tiempo total estimado desde inicio hasta v1.0: **22-28 semanas** (5.5-7 meses).

### 5.2 Fase 0 — Discovery (semanas 1-6)

**Objetivo:** validar viabilidad técnica y problema-solución antes de escribir código de producción.

**Entregables:**
- 5-10 entrevistas con usuarios objetivo (consultores, abogados, investigadores).
- Prototipo CLI en Linux: graba audio dual, transcribe con faster-whisper, genera resumen con un LLM local.
- Benchmarks documentados de 3 modelos ASR candidatos y 3 LLMs.
- POC de captura de loopback en las 3 plataformas (puede ser código sucio).
- Validación de permisos de macOS (el más restrictivo).
- Decisión final de stack congelada.

**Criterios de éxito:**
- Al menos 7 de 10 usuarios entrevistados confirman el problema y el value prop.
- Prototipo demuestra WER < 15% en español e inglés con Whisper small.
- POC de captura funciona en las 3 plataformas.

### 5.3 Fase 1 — Alpha interna (semanas 7-16)

**Objetivo:** construir la app funcional con el stack definitivo, un perfil (Balanced), y probarla internamente.

**Milestones semanales:**

- **Semana 7:** setup del monorepo, CI básica, esqueleto Tauri + React + Rust workspace.
- **Semana 8-9:** módulo `echo-audio` funcional en Linux (captura dual + preproceso + VAD).
- **Semana 10:** módulo `echo-asr` integrado (Whisper small + medium).
- **Semana 11:** módulo `echo-diarize` con clustering local.
- **Semana 12-13:** port de captura a Windows.
- **Semana 14-15:** port de captura a macOS.
- **Semana 16:** primera UI funcional integrada, usada por equipo interno.

**Criterios de éxito:**
- 10 miembros del equipo pueden grabar una reunión y obtener resumen en las 3 plataformas.
- Tests de WER y DER pasan en CI.
- Crash rate < 5% en uso interno.

### 5.4 Fase 2 — Beta pública (semanas 17-24)

**Objetivo:** completar el MVP, incorporar perfiles Lite y Quality, y validar con 500+ usuarios beta.

**Milestones:**

- **Semana 17:** agregar perfiles Lite y Quality, wizard de onboarding.
- **Semana 18:** módulo `echo-llm` con templates de resumen.
- **Semana 19:** chat con transcripción implementado.
- **Semana 20:** biblioteca de reuniones, búsqueda FTS5, exportaciones.
- **Semana 21:** pulimento de UX, tema oscuro, atajos de teclado.
- **Semana 22:** landing page, sistema de distribución de modelos, auto-update.
- **Semana 23:** beta privada con 100 usuarios invitados.
- **Semana 24:** beta pública abierta con 500-1000 usuarios.

**Criterios de éxito:**
- Crash rate < 1% sesiones.
- NPS inicial > 30.
- Feedback recogido y triaged para v1.0.

### 5.5 Fase 3 — Release v1.0 (semanas 25-28)

**Objetivo:** estabilizar, documentar y lanzar.

**Milestones:**

- **Semana 25:** bug bash interno, fix de issues de beta P0/P1.
- **Semana 26:** documentación completa (user docs + docs de contribución).
- **Semana 27:** release candidate, test de regresión completo.
- **Semana 28:** lanzamiento en Product Hunt, Hacker News, comunidades target.

**Criterios de éxito:**
- 5,000+ downloads en primer mes.
- Retention semana 1 > 60%, semana 4 > 35%.
- NPS > 40.

---

## 6. Desglose de trabajo (WBS)

### 6.1 Epics principales

| Epic | Subtareas | Estimación |
|---|---|---|
| E1. Setup técnico | Monorepo, CI/CD, firma de código, distribución | 2 semanas |
| E2. Módulo audio | Captura dual × 3 OS, preproceso, VAD | 6 semanas |
| E3. Módulo ASR | whisper.cpp integration, streaming + refine | 3 semanas |
| E4. Módulo diarización | Embeddings + clustering | 2 semanas |
| E5. Módulo LLM | llama.cpp + templates + chat | 3 semanas |
| E6. Módulo storage | SQLite schema, FTS5, migraciones | 1 semana |
| E7. UI principal | Onboarding, grabación, biblioteca, settings | 6 semanas |
| E8. Perfiles y modelos | Wizard, descargas, gestión de modelos | 2 semanas |
| E9. Exportación y compartir | MD/PDF/TXT/DOCX | 1 semana |
| E10. Testing integral | Fixtures, E2E, platform tests | 3 semanas (en paralelo) |
| E11. Observabilidad | Logging, diagnóstico, telemetría opt-in | 1 semana |
| E12. Documentación | User docs, architecture docs, contribución | 2 semanas |

**Total estimado:** ~32 semanas-persona. Con 3 personas en paralelo, ~10-12 semanas calendario para MVP.

### 6.2 Stories por epic (muestra — E2 módulo audio)

**E2.1** Como desarrollador, necesito un trait `AudioCapturePort` en `echo-domain` que defina la API de captura.

**E2.2** Como desarrollador, necesito una implementación Linux usando `cpal` + `libpulse-binding` que capture mic y loopback simultáneamente.

**E2.3** Como desarrollador, necesito una implementación Windows usando `wasapi-rs` con flag de loopback.

**E2.4** Como desarrollador, necesito una implementación macOS usando `screencapturekit` en modo audio-only.

**E2.5** Como desarrollador, necesito un preprocesador que resamplee a 16 kHz mono usando `rubato`.

**E2.6** Como desarrollador, necesito un módulo VAD con Silero cargado vía `ort`.

**E2.7** Como desarrollador, necesito un ring buffer thread-safe con capacidad de 30 segundos de audio.

**E2.8** Como usuario, quiero que la app detecte automáticamente mis dispositivos de audio y los muestre en un dropdown.

**E2.9** Como usuario, quiero ver el nivel de audio de mic y sistema en tiempo real mientras grabo.

**E2.10** Como desarrollador, necesito tests de integración del módulo audio en CI para las 3 plataformas.

---

## 7. Criterios de aceptación

### 7.1 Criterios generales del MVP

Para considerar el MVP "listo para release":

- [ ] Todos los casos de uso principales (CU-01 a CU-08) funcionan en las 3 plataformas.
- [ ] Métricas de rendimiento del apartado 4.1 se cumplen en los 3 perfiles.
- [ ] Métricas de calidad del apartado 4.2 se cumplen.
- [ ] Cobertura de tests cumple metas del apartado 12.2 del doc de arquitectura.
- [ ] 0 bugs P0 abiertos, < 5 P1 abiertos.
- [ ] Documentación de usuario completa (getting started, FAQ, troubleshooting).
- [ ] Documentación técnica completa (ARCHITECTURE.md, CONTRIBUTING.md).
- [ ] Firma de código configurada en las 3 plataformas.
- [ ] Notarización macOS aprobada.
- [ ] Sistema de auto-update probado end-to-end.
- [ ] Página de descarga con checksums publicada.

### 7.2 Criterios específicos por módulo

#### Audio
- [ ] Captura de mic y sistema funciona en Windows 10/11, macOS 12.3+, Ubuntu 22.04+.
- [ ] 0 frames perdidos en grabaciones de 60 minutos (medido por gaps en timestamps).
- [ ] Cambio de dispositivo mid-sesión recupera sin crashear.

#### ASR
- [ ] Whisper small ES: WER < 10% en fixtures de prueba.
- [ ] Whisper small EN: WER < 8%.
- [ ] Whisper medium mejora WER en > 2 puntos en ambos idiomas.
- [ ] Streaming emite segmentos con latencia < 4 s.

#### Diarización
- [ ] 2 hablantes: DER < 20%.
- [ ] 3-5 hablantes: DER < 30%.
- [ ] No confunde speakers entre pista A y pista B.

#### LLM
- [ ] Resúmenes JSON válidos en > 95% de casos (sin retry).
- [ ] Todas las 6 plantillas producen output estructurado.
- [ ] Chat mantiene contexto de hasta 10 turnos.

#### UX
- [ ] Onboarding < 5 min end-to-end.
- [ ] Navegación completa por teclado.
- [ ] Tema claro y oscuro funcionales.
- [ ] Responsive desde 900×600.

---

## 8. Equipo y roles

### 8.1 Roles necesarios para el MVP

| Rol | Dedicación | Responsabilidades |
|---|---|---|
| Tech Lead | 100% | Arquitectura, decisiones técnicas, code review |
| Rust Engineer | 100% | Módulos backend (audio, ASR, LLM, storage) |
| Frontend Engineer | 100% | UI React, Tauri IPC, UX de flujos |
| ML Engineer (part-time) | 50% | Optimización de modelos, benchmarks, fine-tuning |
| Designer | 50% | UI/UX, sistema de diseño, mockups |
| QA / Test Engineer | 50% | Fixtures, tests de regresión, bug bashes |
| Product Manager | 50% | Discovery, beta program, feedback loop |

**Equipo mínimo viable:** 3 personas full-time (Tech Lead, Rust Eng, Frontend Eng) con ayuda part-time del resto.

### 8.2 Responsabilidades cruzadas

- **Tech Lead + Rust Engineer** colaboran en módulos críticos (audio, ASR).
- **Frontend Engineer + Designer** colaboran en sistema de diseño y componentes.
- **ML Engineer + Rust Engineer** colaboran en integración de modelos y optimización.
- **QA + Todos** en el proceso de bug bash y tests.

---

## 9. Gestión de riesgos

### 9.1 Matriz de riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Captura de audio en macOS más compleja de lo estimado | Alta | Alto | Atacar macOS primero si da sospechas; tener fallback con BlackHole |
| Calidad de Whisper insuficiente en español conversacional | Media | Alto | Tener Canary Qwen como plan B en perfil Quality |
| LLMs pequeños alucinan en resúmenes | Alta | Medio | Prompts bien diseñados, validación con citations, botón de regenerar |
| Hardware de usuarios más débil que asumido | Media | Medio | Perfil Lite muy conservador, modo "solo ASR" sin LLM |
| Fricción de permisos hunde onboarding | Alta | Alto | Onboarding con explicaciones claras; videos en docs |
| Google Calendar rechaza nuestra app en OAuth | Baja | Medio (v1.1) | v1.1 usa Calendar-like APIs genéricas |
| Litigios por grabación sin consentimiento | Media | Alto | Disclaimers claros; plantilla legal en docs; feature de aviso al iniciar |
| Competencia local (Granola lanza modo offline) | Media | Medio | Velocidad de ejecución; diferenciador open source |
| Problemas de rendimiento en CPUs ARM antiguas | Media | Medio | Testing en M1/M2/M3 desde el inicio |
| Descarga de modelos falla en redes restringidas | Alta | Bajo | Resumibilidad, mirrors, opción de instalar modelos manualmente |

### 9.2 Plan de contingencia

**Si en fase 0 descubrimos que Whisper es insuficiente:**
- Pivot a Canary Qwen 2.5B como default (aunque más pesado).
- Aceptar perfil Lite con WER peor y comunicarlo honestamente.

**Si en fase 1 descubrimos que 3 OS en paralelo toma 2× el tiempo:**
- Lanzar beta primero en los 2 OS más listos.
- Completar el tercero en 4-6 semanas post-beta.

**Si en fase 2 la retención beta es baja:**
- Entrevistar a abandonadores (> 20 sesiones).
- Priorizar el issue más repetido antes de lanzamiento público.

---

## 10. Definiciones de "listo"

### 10.1 Definition of Ready (DoR) — para empezar una story

- [ ] Story escrita en formato "Como X, quiero Y, para Z".
- [ ] Criterios de aceptación definidos.
- [ ] Dependencias identificadas y resueltas o planificadas.
- [ ] Diseño / mockup disponible si aplica.
- [ ] Estimación hecha por el equipo.

### 10.2 Definition of Done (DoD) — para cerrar una story

- [ ] Código implementado pasa linters (clippy para Rust, ESLint para TS).
- [ ] Unit tests escritos y pasando.
- [ ] Integration tests pasando en CI en las 3 plataformas.
- [ ] Code review aprobado por al menos 1 persona.
- [ ] Criterios de aceptación verificados manualmente.
- [ ] Documentación actualizada si aplica.
- [ ] Merged a `main` sin breaking changes.

### 10.3 Definition of Release (DoR-2) — para liberar una versión

- [ ] Todos los criterios del apartado 7.1 cumplidos.
- [ ] Changelog actualizado.
- [ ] Release notes escritas.
- [ ] Binarios firmados generados en CI.
- [ ] Smoke tests manuales en las 3 plataformas pasados.
- [ ] Tag de versión creado en Git.

---

## 11. Proceso de desarrollo

### 11.1 Control de versiones

**Git flow simplificado:**
- `main` — rama siempre desplegable.
- `develop` — integración de features en curso.
- `feature/E2.3-windows-wasapi-capture` — una rama por story.
- `release/v1.0.0` — rama de estabilización antes de tag.
- `hotfix/xxx` — correcciones urgentes en producción.

**Convención de commits:** Conventional Commits.

```
feat(audio): add Windows WASAPI loopback capture
fix(asr): prevent race condition in streaming buffer
docs(arch): update ADR-005 with final decision
```

### 11.2 Code review

- Cada PR requiere 1 aprobación mínimo, 2 si toca código crítico (audio, seguridad).
- Linters y tests deben pasar antes de review.
- Reviewer valida: correctness, tests, docs, performance impact.
- PR grandes (> 500 líneas) deben partirse o justificarse.

### 11.3 CI/CD

**Por cada PR:**
- Compila en las 3 plataformas.
- Corre unit tests + integration tests.
- Corre linters (clippy, ESLint, Prettier).
- Verifica cobertura no decrece.
- Comenta con resultados en el PR.

**Por cada merge a `develop`:**
- Genera build nightly para los 3 OS.
- Publica en canal "nightly" para testers internos.

**Por cada tag `v*`:**
- Build firmado y notarizado.
- Sube a GitHub Releases y CDN.
- Notifica a canal de Discord/Slack de releases.

### 11.4 Ceremonias

**Daily standup** (15 min): lo de ayer, lo de hoy, bloqueos.

**Weekly planning** (60 min): priorización de stories de la semana.

**Biweekly demo** (30 min): showcase de lo terminado, feedback del equipo.

**Monthly retrospective** (60 min): qué funcionó, qué no, acciones.

**Architecture review** (ad-hoc): cuando se propone un ADR nuevo.

### 11.5 Comunicación

- **Slack / Discord**: día a día, asíncrono primero.
- **GitHub Issues**: tracking de bugs y features.
- **GitHub Projects**: board de sprint.
- **Notion / Docs**: documentación viva (user docs, ADRs, specs).

---

## Apéndice A — Cronograma visual resumido

```
Semana:     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
Fase 0:     █  █  █  █  █  █
Fase 1:                       █  █  █  █  █  █  █  █  █  █
Fase 2:                                                     █  █  █  █  █  █  █  █
Fase 3:                                                                               █  █  █  █

Milestones:
W6:   ▲ Prototipo CLI + validación usuarios
W12:  ▲ Audio + ASR funcional en Linux
W16:  ▲ Alpha interna en 3 OS
W22:  ▲ Feature-complete beta
W24:  ▲ Beta pública 500+ users
W28:  ▲ Release v1.0 🚀
```

---

## Apéndice B — Documentos relacionados

- **PRD.md** — Product Requirements Document v0.1
- **ARCHITECTURE.md** — Documento de arquitectura técnica
- **DESIGN.md** — Sistema de diseño y UI/UX (próximo)
- **CONTRIBUTING.md** — Guía de contribución (durante fase 1)

---

**Este documento evoluciona con el proyecto.** Actualizaciones significativas requieren discusión y aprobación del Tech Lead + PM.
