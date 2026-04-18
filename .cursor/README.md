# Catálogo de Skills y Agents — Proyecto Echo

Este directorio contiene **skills** (capacidades reutilizables tipo prompt+scripts) y **agents** (subagentes especializados) curados para el desarrollo de Proyecto Echo. Están versionados con el proyecto para que todo el equipo use las mismas guías.

## Cómo se organiza

```
.cursor/
├── skills/     # Skills oficiales de Anthropic (anthropics/skills)
└── agents/     # Subagents de wshobson/agents
```

---

## Skills instaladas (de [anthropics/skills](https://github.com/anthropics/skills))

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

- **Origen oficial:** https://github.com/anthropics/skills (skills) y https://github.com/wshobson/agents (agents).
- **Actualización:** periódicamente re-clonar ambos repos y copiar versiones nuevas de los items listados aquí.
- **Extensiones propias:** usar `skill-creator` para crear skills específicas del dominio (ej: `tauri-capabilities`, `whisper-rs-wrapper`, `onnx-diarization`).
- **Licencias:**
  - `anthropics/skills`: ver `LICENSE.txt` de cada skill (Anthropic license).
  - `wshobson/agents`: MIT (ver [LICENSE](https://github.com/wshobson/agents/blob/main/LICENSE)).

---

## Aviso

Anthropic **no mantiene un repositorio oficial de subagents**. `wshobson/agents` es de comunidad pero es la colección más popular (miles de estrellas, activamente mantenida). Las **skills sí son oficiales** de Anthropic.
