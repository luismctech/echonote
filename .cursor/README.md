# Catálogo de Skills y Agents — Proyecto Echo

Este directorio contiene **skills** (capacidades reutilizables tipo prompt+scripts) y **agents** (subagentes especializados) curados para el desarrollo de Proyecto Echo. Están versionados con el proyecto para que todo el equipo use las mismas guías.

## Cómo se organiza

```
.cursor/
├── skills/     # Skills oficiales (Anthropic + Vercel Labs)
└── agents/     # Subagents de wshobson/agents
```

---

## Skills instaladas

### De [anthropics/skills](https://github.com/anthropics/skills) (oficial Anthropic)

Las skills son conjuntos de instrucciones + scripts que Claude/Cursor cargan bajo demanda cuando la tarea coincide con su descripción.

| Skill | Cuándo se activa | Utilidad para Echo |
|---|---|---|
| `frontend-design` | Al construir componentes web, páginas, UIs | Diseñar la UI de React + Tailwind + shadcn/ui (vistas `home`, `meeting`, `recording` de los mockups) con criterio estético profesional |
| `webapp-testing` | Para testear apps web locales con Playwright | Implementar los 5 flujos E2E críticos del plan de testing (sección 12) |
| `skill-creator` | Para crear o mejorar skills | Meta-skill: crear skills propias de Echo (p.ej. `whisper-cpp-integration`, `tauri-ipc-patterns`, `asr-fixtures`) |
| `doc-coauthoring` | Al redactar documentación, specs, ADRs | Escribir los 10 ADRs iniciales listados en la sección 14 del ARCHITECTURE.md |
| `mcp-builder` | Al construir servidores MCP | Opcional: si en v2 se expone Echo vía MCP para integraciones externas |
| `pdf` | Al generar/editar PDF | Export a PDF de reuniones (sección 2.1 del Development Plan) |
| `docx` | Al generar/editar DOCX | Export a DOCX de reuniones (sección 2.1 del Development Plan) |

### De [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) (oficial Vercel)

Skills de Vercel Engineering para React. Son las únicas guías React mantenidas por un fabricante grande (creadores de Next.js y empleadores de varios miembros del React core team). Foco en performance, arquitectura de componentes y a11y.

| Skill | Cuándo se activa | Utilidad para Echo |
|---|---|---|
| `vercel-react-best-practices` | Al escribir, refactorizar o revisar componentes React | 72 reglas oficiales de Vercel agrupadas por impacto (waterfalls, bundle, re-renders, rendering, JS perf, advanced). Aplicar a todo `src/features/*` y vistas `home`/`meeting`/`recording` |
| `vercel-composition-patterns` | Al diseñar APIs de componentes, refactorizar props booleanos, crear compound components | Guía la arquitectura de los componentes shadcn que se compongan (modales de meeting, drawer de recording, etc.) — evita explosión de props |
| `vercel-web-design-guidelines` | Al pedir "review my UI", "check accessibility", "audit design" | Auditoría automática contra 100+ reglas (a11y, focus, forms, animation, tipografía, perf, dark mode, i18n). Usar antes de cada PR de UI |
| `vercel-react-view-transitions` | Al añadir animaciones entre rutas/estados, shared elements, list reorder | Implementar transiciones suaves entre `home → meeting → recording` con la API nativa `<ViewTransition>`. **Nota:** requiere React 19/canary; útil como referencia para cuando upgradees desde React 18 |

## Subagents instalados (de [wshobson/agents](https://github.com/wshobson/agents))

Los subagents son configuraciones de Claude Code/Cursor especializadas en un rol. Se invocan delegándoles una tarea concreta.

| Agent | Rol | Dónde encaja en Echo |
|---|---|---|
| `rust-pro` | Experto Rust 1.75+, async/Tokio, sistemas | Todo lo de `crates/echo-*/`, `src-tauri/`, FFI con whisper.cpp/llama.cpp |
| `frontend-developer` | React moderno, accesibilidad, performance | Todo `src/features/*` y componentes shadcn |
| `typescript-pro` | TypeScript estricto, tipos avanzados | `strict: true` en todo el frontend; tipos IPC generados con `specta` |
| `test-automator` | Estrategia de testing multinivel | Implementar la pirámide (unit/integration/E2E) descrita en sección 12 |
| `code-reviewer` | Review de calidad, seguridad, estilo | PRs antes de merge; aplicar antes de tocar ramas de release |
| `architect-review` | Evaluar decisiones arquitectónicas | Revisar adhesión a Clean Architecture (capas 1-4) y escribir/auditar ADRs |
| `sql-pro` | SQL avanzado, optimización, schema | Diseño de las migraciones SQLite + FTS5 (sección 8.2/8.3) |
| `performance-engineer` | Profiling, optimización, benchmarks | Cumplir metas: WER <10%/8%, RTF, refinamiento <90s para 30min audio |
| `security-auditor` | Amenazas, cifrado, permisos | Revisar la matriz de amenazas (9.1), capabilities Tauri, SQLCipher, entitlements |

---

## Cómo usarlos desde Cursor o Claude Code

**Skills** (se activan automáticamente por su `description`):
- Cursor lee los `SKILL.md` si los declara en su configuración de agent skills. Para cargarlas de forma explícita, puedes referenciarlas por su path absoluto en el chat:
  `"lee y aplica .cursor/skills/frontend-design/SKILL.md"`
- También puedes copiarlas a `~/.claude/skills/` para disponibilidad global.

**Agents** (se delegan con la herramienta Task):
- En Claude Code: `"usa el subagent rust-pro para refactorizar este módulo"`.
- En Cursor, mover a `.claude/agents/` si usas Claude Code junto al proyecto, o referenciarlos como contexto: `"lee .cursor/agents/rust-pro.md y actúa como ese experto"`.

---

## Mantenimiento

- **Orígenes oficiales:**
  - https://github.com/anthropics/skills — skills oficiales de Anthropic
  - https://github.com/vercel-labs/agent-skills — skills oficiales de Vercel (React/Next/UI)
  - https://github.com/wshobson/agents — agents (comunidad, MIT)
- **Actualización:** periódicamente re-clonar los tres repos y copiar versiones nuevas de los items listados aquí.
- **Extensiones propias:** usar `skill-creator` para crear skills específicas del dominio (ej: `tauri-capabilities`, `whisper-rs-wrapper`, `onnx-diarization`).
- **Licencias:**
  - `anthropics/skills`: ver `LICENSE.txt` de cada skill (Anthropic license).
  - `vercel-labs/agent-skills`: MIT (ver [LICENSE](https://github.com/vercel-labs/agent-skills/blob/main/LICENSE)).
  - `wshobson/agents`: MIT (ver [LICENSE](https://github.com/wshobson/agents/blob/main/LICENSE)).

---

## Aviso sobre fabricantes oficiales

- **Skills oficiales React/UI:** las únicas mantenidas por un fabricante grande son las de **Vercel Labs** (`vercel-labs/agent-skills`). El equipo React de Meta **no publica** skills/agents oficiales propios (su guía oficial es la documentación en react.dev).
- **Skills oficiales generales:** las de **Anthropic** (`anthropics/skills`) — incluyen `frontend-design` y `webapp-testing` aplicables a React.
- **Subagents:** Anthropic **no mantiene un repositorio oficial**. `wshobson/agents` es comunidad pero es la colección más popular (miles de estrellas, activamente mantenida).
- **Nota Next.js:** Si en algún momento migras a Next.js, las skills `vercel-react-best-practices` y `vercel-react-view-transitions` ya cubren patrones específicos del App Router (Server Components, Server Actions, `next/link transitionTypes`).
