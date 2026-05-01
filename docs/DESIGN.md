# Proyecto Echo — Diseño UI/UX

**Versión:** 0.2
**Fecha:** 1 de mayo de 2026
**Estado:** Implementado (Phase 1 — Foundation)

---

## Tabla de contenidos

1. [Filosofía de diseño](#1-filosofía-de-diseño)
2. [Análisis de Granola (competencia)](#2-análisis-de-granola-competencia)
3. [Dirección estética](#3-dirección-estética)
4. [Sistema de diseño](#4-sistema-de-diseño)
5. [Arquitectura de información](#5-arquitectura-de-información)
6. [Flujos principales](#6-flujos-principales)
7. [Pantallas clave](#7-pantallas-clave)
8. [Componentes](#8-componentes)
9. [Estados y feedback](#9-estados-y-feedback)
10. [Accesibilidad](#10-accesibilidad)
11. [Motion y micro-interacciones](#11-motion-y-micro-interacciones)

---

## 1. Filosofía de diseño

### 1.1 Tres principios de diseño

**1. Presencia antes que producto.**
La interfaz nunca compite por atención con la reunión. Durante la grabación, Echo desaparece; trabaja. Granola lo hizo bien y lo honramos: la ventana principal durante una reunión se parece más a un cuaderno abierto que a un dashboard de SaaS.

**2. El humano manda, la IA asiste.**
Aplicamos el concepto de "autonomy slider" de Karpathy. El usuario toma notas si quiere, o las delega totalmente a la IA. Ambos caminos son de primera clase. Las adiciones de IA son visualmente distintas a las del humano — nunca pretendemos que el usuario escribió algo que no escribió.

**3. Táctil y editorial, no genérico ni corporativo.**
Rechazamos el estilo "SaaS moderno" de gradientes morados, íconos rellenos redondos y copywriting emocionado. Echo se siente más cercano a un cuaderno de Moleskine digital que a un producto Figma típico. Tipografía editorial, paleta terrosa, asimetrías suaves, tacto analógico.

### 1.2 Anti-patrones explícitos

Cosas que **no** haremos:

- Gradientes morado-rosa característicos del "AI slop".
- Emojis en la interfaz principal (excepto donde el usuario los escribe).
- Iconografía redonda con colores brillantes genéricos.
- Copy del estilo "🚀 Supercharge your meetings with AI!".
- Notificaciones intrusivas o sonidos sin propósito.
- Dashboards con 8 KPIs que nadie lee.
- Onboarding con pop-ups que saltan unos sobre otros.

---

## 2. Análisis de Granola (competencia)

### 2.1 Fortalezas de Granola que adoptamos

| Fortaleza | Aplicación en Echo |
|---|---|
| **IA invisible** — sin chrome durante la reunión | Vista de grabación minimalista tipo cuaderno |
| **Notas humano vs IA diferenciadas visualmente** | Notas humano en primario, IA en tono más suave + indicador |
| **Split-screen: notas + transcripción** | Layout similar pero con mejor balance visual |
| **Recipes / Templates de resumen** | 6 templates MVP, editor visual de templates en v1.1 |
| **Chat contextual con atajo rápido** | Cmd/Ctrl+J abre chat global con la reunión activa |
| **Sin bot en la llamada** | Captura de dispositivo, idéntico patrón |
| **"Enhance notes" al terminar** | Botón equivalente post-stop |

### 2.2 Debilidades de Granola que resolvemos

| Debilidad | Cómo lo resolvemos |
|---|---|
| **"Gris sobre gris, parece Windows 95"** | Paleta terrosa cálida con acento ámbar; alma editorial |
| **Confusión sobre dónde está transcript vs notas** | Layout con etiquetas claras y divisores visuales fuertes |
| **Organización pobre (solo folders)** | Folders + tags + filtros por speaker, fecha, duración, plantilla |
| **Speaker identification débil** | Diarización explícita mic/sistema + clustering; UI para renombrar fácil |
| **Dependencia de Google Workspace** | Funciona standalone; calendar opcional |
| **Audio descartado sin opción** | Retención configurable (nunca / 7d / 30d / siempre) |
| **Sin tema oscuro bien pensado** | Tema claro + oscuro + sistema, tratados como ciudadanos de primera |
| **Windows tratado como segundón** | Las 3 plataformas con paridad visual desde día 1 |

### 2.3 Oportunidades que Granola no explota

- **Visualización del audio que se sienta orgánica** (no las "dancing bars" genéricas).
- **Biblioteca con búsqueda semántica**, no solo full-text (v1.1 con embeddings).
- **Editor de transcripción con word-level highlighting al reproducir** (si retiene audio).
- **Modo "focus" durante la grabación**: oculta el resto de la UI, solo el cuaderno.
- **Exportación a formatos editoriales serios**: DOCX con tipografía propia, PDF con layout de revista.

---

## 3. Dirección estética

### 3.1 Mood

**"El cuaderno del pensador moderno."**

Imagínate una mesa de madera con un cuaderno Moleskine abierto, una pluma estilográfica al lado, café cerca, y en el cuaderno hay garabatos, flechas, y notas ordenadas. Ahora pon ese cuaderno dentro de una app. Esa es la sensación.

Referencias:
- Readwise Reader (cuando oscila hacia editorial)
- iA Writer (tipografía y calma)
- Craft (sistema de documentos minimalista)
- The Browser Company / Arc (detalles táctiles, motion con alma)
- Working Copy (densidad bien manejada)

Lo que **no** es:
- Notion (demasiado plano, genérico)
- Linear (demasiado frío, SaaS-y)
- Figma (demasiado herramientoso)

### 3.2 Personalidad de marca en interfaz

- **Serio pero no rígido** — como un periódico bien diseñado.
- **Cálido pero no casual** — como una librería de barrio próspera.
- **Técnico bajo la superficie, humano arriba** — el usuario no ve la complejidad hasta que la pide.
- **Silencioso cuando debe, enfático cuando importa** — una decisión capturada por IA merece un tratamiento tipográfico cuidado.

---

## 4. Sistema de diseño

### 4.1 Tipografía

> **Referencia de alineación:** macOS Human Interface Guidelines (built-in text styles).
> La escala UI de Echo está calibrada para que los tamaños se correspondan con los
> roles tipográficos nativos de macOS, garantizando proporciones familiares en una
> app de escritorio.

**Sans-serif (toda la UI, títulos incluidos):**
- `Inter` — sans-serif workhorse para títulos, body y toda la interfaz
- Fallback: `system-ui`, `-apple-system`, sans-serif

**Monospace (código, timestamps, técnico):**
- `JetBrains Mono` — excelente ligaduras, neutral, legible
- Fallback: `SF Mono`, `Consolas`, monospace

**Escalas tipográficas (alineadas a macOS HIG):**

```css
/* Display — solo para momentos hero (onboarding, empty states) */
--font-display-xl: 48px / 1.05 / Inter;
--font-display-lg: 36px / 1.1  / Inter;
--font-display-md: 28px / 1.15 / Inter;

/* UI — escala principal de interfaz (alineada a macOS) */
--font-ui-lg: 15px / 1.45 / Inter;      /* ≈ macOS Title 3 — headings de panel */
--font-ui-md: 13px / 1.4  / Inter;      /* ≈ macOS Body — texto primario, botones, chat */
--font-ui-sm: 12px / 1.35 / Inter;      /* ≈ macOS Callout — inputs, labels, secondary */
--font-ui-xs: 11px / 1.3  / Inter;      /* ≈ macOS Subheadline — tags, section headers */

/* Reading — para transcripciones y notas largas */
--font-reading-lg: 18px / 1.65 / Inter;  /* notas expandidas */
--font-reading-md: 16px / 1.7  / Inter;  /* transcripción expandida */

/* Mono */
--font-mono-md: 12px / 1.5 / JetBrains Mono;

/* Micro — badges, status indicators */
--font-micro: 10px / 1.3 / Inter;
```

**Correspondencia con macOS HIG:**

| Token Echo | Tamaño | macOS equivalent | Rol |
|---|---|---|---|
| `display-md` | 28px | Large Title (26pt) | Títulos hero |
| `ui-lg` | 15px | Title 3 (15pt) | Headings de panel |
| `ui-md` | 13px | Body / Headline (13pt) | Texto primario, botones |
| `ui-sm` | 12px | Callout (12pt) | Inputs, labels funcionales |
| `ui-xs` | 11px | Subheadline (11pt) | Section headers, tags |
| `micro` | 10px | Footnote / Caption (10pt) | Badges, indicadores |

**Pesos utilizados:** 400 (regular), 500 (medium), 600 (semibold). Evitar 700+ (demasiado pesado contra body).

### 4.2 Paleta de colores

**Tema claro (base):**

```css
/* Superficies */
--bg-base: #F8F5F0;          /* off-white cálido, papel */
--bg-elevated: #FFFFFF;      /* cards, popovers */
--bg-sunken: #EAE4DA;        /* hovers, sidebars */
--bg-inset: #DED6C8;         /* inputs, wells */

/* Texto */
--text-primary: #161410;     /* casi negro cálido, alto contraste */
--text-secondary: #3A3630;   /* cuerpo legible */
--text-tertiary: #5C564E;    /* metadata, headers de sección */
--text-placeholder: #948C82;

/* Bordes */
--border-subtle: rgba(22, 20, 16, 0.08);
--border-default: rgba(22, 20, 16, 0.18);
--border-strong: rgba(22, 20, 16, 0.32);

/* Acento principal — ámbar quemado */
--accent-50: #FDF7ED;
--accent-100: #FAE8C7;
--accent-400: #E89938;
--accent-600: #C2410C;        /* el primario */
--accent-700: #9A330A;
--accent-900: #5A1F06;

/* Semánticos */
--success: #4F7A3C;           /* verde oliva */
--warning: #B8860B;           /* ámbar oscuro */
--danger: #9C2A28;            /* rojo terracota */
--info: #3A5C7A;              /* azul profundo */

/* Speakers (hasta 8 distinguibles) */
--speaker-1: #C2410C;         /* ámbar — usuario */
--speaker-2: #4A6B7A;         /* azul pizarra */
--speaker-3: #6B4A7A;         /* ciruela */
--speaker-4: #4F7A3C;         /* verde oliva */
--speaker-5: #9C6B3C;         /* cobre */
--speaker-6: #7A4A5C;         /* mauve */
--speaker-7: #3C6B7A;         /* teal apagado */
--speaker-8: #7A6B4A;         /* mostaza */
```

**Tema oscuro:**

```css
--bg-base: #10121B;           /* navy profundo azulado */
--bg-elevated: #181A26;
--bg-sunken: #0A0C14;
--bg-inset: #06070E;

--text-primary: #E8ECF4;      /* blanco con matiz azul */
--text-secondary: #A8B0C4;
--text-tertiary: #767E96;
--text-placeholder: #444C64;

--border-subtle: rgba(200, 210, 235, 0.07);
--border-default: rgba(200, 210, 235, 0.14);
--border-strong: rgba(200, 210, 235, 0.26);

--accent-50: #28201C;
--accent-100: #44301C;
--accent-400: #E89938;
--accent-600: #F2A552;        /* más claro en dark */
--accent-700: #F7BA7C;
--accent-900: #FAD8AC;
```

### 4.3 Espaciado y grid

Escala basada en 4px:

```
--space-1:  4px
--space-2:  8px
--space-3:  12px
--space-4:  16px
--space-5:  20px
--space-6:  24px
--space-8:  32px
--space-10: 40px
--space-12: 48px
--space-16: 64px
--space-20: 80px
--space-24: 96px
```

**Grid de página principal:** 12 columnas con gutter de 24px. Max width 1400px (desktop amplio), pero el contenido usualmente vive entre 800-1200px.

**Layout de app:**

```
┌──────────────────────────────────────────────────┐
│  Sidebar  │                                        │
│   240px   │          Canvas principal               │
│           │          (fluido)                        │
│           │                                          │
│           │                                          │
└──────────────────────────────────────────────────┘
```

### 4.4 Esquinas y formas

- **Radius escala:** 4 / 8 / 12 / 16 / 24 / pill
- **Cards principales:** 12px
- **Botones:** 8px (o pill para toggles)
- **Inputs:** 8px
- **Modales:** 16px
- **Avatar / chips:** pill

**Principio:** esquinas más generosas en elementos grandes, más pequeñas en controles. Nada filoso, nada demasiado redondeado.

### 4.5 Elevación

Usamos sombras muy sutiles, más como "lifting" que "floating":

```css
--shadow-sm: 0 1px 2px rgba(26, 24, 20, 0.04),
             0 1px 3px rgba(26, 24, 20, 0.06);
--shadow-md: 0 2px 4px rgba(26, 24, 20, 0.04),
             0 4px 12px rgba(26, 24, 20, 0.08);
--shadow-lg: 0 4px 8px rgba(26, 24, 20, 0.06),
             0 12px 32px rgba(26, 24, 20, 0.12);
--shadow-xl: 0 8px 16px rgba(26, 24, 20, 0.08),
             0 24px 48px rgba(26, 24, 20, 0.16);
```

En tema oscuro, sombras más profundas + un posible highlight superior sutil para la sensación de capas.

### 4.6 Iconografía

- **Set principal:** Lucide (open source, bien diseñado, consistente).
- **Tamaños:** 14 / 16 / 20 / 24 px.
- **Stroke:** 1.5 px por defecto, 2 px para iconos pequeños.
- **Nunca rellenos saturados.** Si un icono necesita peso visual, se usa un background circular sutil, no un fill sólido.

---

## 5. Arquitectura de información

### 5.1 Mapa de la app

```
Echo
├── Home / Library (default)
│   ├── Reuniones recientes
│   ├── Filtros (fecha, carpeta, speaker, template)
│   └── Búsqueda
│
├── Recording (activo solo durante grabación)
│   ├── Canvas de notas
│   ├── Transcripción live (panel lateral)
│   └── Controles de audio
│
├── Meeting (vista de reunión cerrada)
│   ├── Resumen (default)
│   ├── Transcripción
│   ├── Notas
│   ├── Chat
│   └── Metadata (speakers, audio si retenido, etc.)
│
├── Settings
│   ├── Audio
│   ├── Transcripción
│   ├── Resumen & Chat
│   ├── Interfaz
│   ├── Privacidad
│   └── Avanzado
│
└── Global
    ├── Command palette (Cmd/Ctrl+K)
    └── Chat global (Cmd/Ctrl+J)
```

### 5.2 Navegación

**Sidebar izquierdo fijo** (240px, collapsable a 64px):

```
┌─────────────────────────┐
│  Echo      ⌘K  │
├─────────────────────────┤
│  ⏺  Nueva grabación     │  ← Botón primario
├─────────────────────────┤
│  📚  Biblioteca         │
│  ⭐  Favoritos          │
│  🗂  Carpetas           │
│     ├─ Clientes         │
│     ├─ Equipo           │
│     └─ Personal         │
├─────────────────────────┤
│  🏷  Tags               │
│     #producto          │
│     #ventas            │
├─────────────────────────┤
│  ⚙  Ajustes           │
└─────────────────────────┘
```

(Los iconos son placeholders; en la UI real son Lucide stroke, no emojis)

### 5.3 Jerarquía de acciones

**Primarias** (una por pantalla, visualmente destacada):
- Nueva grabación (en Home)
- Detener grabación (en Recording)
- Enhance notes (post-recording)

**Secundarias:** acciones frecuentes pero no únicas (Exportar, Compartir, Renombrar).

**Terciarias:** acciones avanzadas, viven en menús contextuales (Cambiar modelo, Ver detalles técnicos).

---

## 6. Flujos principales

### 6.1 Primer uso (onboarding)

```
[Welcome] → [Privacy promise] → [Permissions] → [Hardware detect] →
[Model download] → [Test recording] → [Home]
```

**Duración objetivo:** < 5 minutos, con la descarga de modelos como paso más largo (depende de red).

**Detalles por pantalla:**

1. **Welcome:** mensaje breve, logo grande, 1 botón "Empezar". Sin video ni cards.
2. **Privacy promise:** una sola frase tipográficamente fuerte: *"Todo se queda en tu equipo. Siempre."* Y bullets pequeños explicativos.
3. **Permissions:** explica por qué necesita cada permiso, con botones para abrir ajustes del OS. Si ya están concedidos, avanza automáticamente.
4. **Hardware detect:** "Detectando tu equipo..." con animación sutil. Después muestra el perfil recomendado (Lite/Balanced/Quality) con detalles de cada uno. Usuario puede cambiar.
5. **Model download:** barra de progreso honesta con velocidad actual, tiempo estimado, y tamaño total. Permite cancelar y reanudar.
6. **Test recording:** "Grabemos 10 segundos de prueba" con botón rojo. Transcripción en vivo aparece debajo. Confirma que todo funciona.
7. **Home:** pequeña celebración ("¡Listo!"), luego fade al home vacío con CTA grande.

### 6.2 Flujo de grabación (CU-01, CU-02, CU-03)

```
[Home] → [Start Recording] → [Recording active] → [Stop] →
[Refining with progress] → [Meeting view with summary]
```

**Tiempos:**
- Click a start → captura comienza: < 500 ms
- Texto streaming aparece: < 4 s desde palabra dicha
- Stop → resumen listo (30 min audio): 60-90 s

### 6.3 Flujo de búsqueda (CU-06)

```
[Cmd+K o click search] → [Type query] → [Results ranked] → [Click → Meeting]
```

Resultados agrupados:
- Reuniones (por título / metadata)
- Segmentos (match en transcripción, con snippet)
- Notas (match en notas manuales)

### 6.4 Flujo de exportación (CU-08)

```
[Meeting view] → [Export button] → [Format picker + options] → [Save dialog] → [Done toast]
```

Opciones por formato:
- **MD/TXT:** solo transcripción / solo resumen / ambos; con o sin timestamps.
- **PDF:** layout editorial con tipografía Echo, o simple. Logo y footer personalizables.
- **DOCX:** plantilla Echo o plantilla minimal.

---

## 7. Pantallas clave

A continuación se describen las pantallas más importantes. Los mockups interactivos se entregan en archivos separados (ver apéndice).

### 7.1 Home / Library

**Propósito:** aterrizaje principal, punto de partida para todas las acciones.

**Composición:**
- Header minimalista con saludo contextual por hora ("Buenas tardes, [nombre]").
- CTA principal gigante: "Nueva grabación" — con icono de círculo de grabación pulsando sutilmente.
- Debajo: "Continuar" con las 3 últimas reuniones como cards compactas.
- Sección "Todas las reuniones" con filtros arriba y lista debajo.
- Empty state: ilustración simple dibujada a mano, texto corto, CTA.

**Interacciones:**
- Hover en card muestra preview del resumen a la derecha.
- Click en card → abre meeting view.
- Right-click → menú contextual (mover, exportar, eliminar, etc.)
- Cmd/Ctrl+N → nueva grabación
- Cmd/Ctrl+K → command palette.

### 7.2 Recording (grabación activa)

**Propósito:** no distraer. Máxima presencia, mínima interfaz.

**Composición (layout asimétrico):**

```
┌────────────────────────────────────────────────────┐
│  ◼ En vivo · 12:34    [Idioma: ES]    [Min] [Exp]  │ ← top bar minimalista
├────────────────────────────────────────────────────┤
│                                         │           │
│                                         │  Live     │
│    Tu cuaderno                          │  ─────    │
│    ─────────────                        │  Tú:      │
│                                         │  Hola a   │
│    # Reunión con cliente X              │  todos,   │
│                                         │  gracias  │
│    - Revisar propuesta de mayo          │  por      │
│    - Confirmar timeline                 │  venir... │
│    |                                    │           │
│                                         │  Ana:     │
│                                         │  Claro,   │
│                                         │  déjame   │
│                                         │  compar-  │
│                                         │  tir...   │
│                                         │           │
├────────────────────────────────────────────────────┤
│  ∿∿∿∿∿∿∿∿∿∿∿∿  [●  Detener ]  🎤 ▓▓░  🔊 ▓▓▓       │ ← waveform + controls
└────────────────────────────────────────────────────┘
```

- **Panel izquierdo 60%:** canvas de notas con tipografía Reading. Título editable arriba, cursor parpadeando, listo para escribir.
- **Panel derecho 40%:** transcripción live. Colapsable con botón o atajo.
- **Bottom bar:** waveform sismograma de ambas pistas (colores diferentes), botón STOP prominente en ámbar, niveles de mic y sistema.
- **Top bar:** status, timer, controles mínimos.

**Estado "Focus Mode"** (atajo Cmd/Ctrl+Shift+F):
- Oculta sidebar, colapsa panel de transcripción, maximiza notas.
- Reduce UI a lo esencial: notas + STOP flotante.

### 7.3 Meeting view (post-recording)

**Propósito:** consumir, editar y operar sobre una reunión terminada.

**Composición (tabs horizontales):**

```
┌────────────────────────────────────────────────────┐
│  ← Biblioteca   [Título editable]    Exp  Share  ⋯ │
│                                                     │
│  15 abr 2026 · 34 min · Ana, Carlos, Tú            │
├────────────────────────────────────────────────────┤
│  [Resumen]  [Transcripción]  [Notas]  [Chat]       │ ← tabs
├────────────────────────────────────────────────────┤
│                                                     │
│    Tus notas                                        │
│    ─────                                            │
│    - Revisar propuesta de mayo                      │
│    - Confirmar timeline                             │
│                                                     │
│    ↗ Resumen IA                    [Regenerar]     │
│    ─────────                                        │
│    TL;DR                                            │
│    Ana y Carlos revisaron la propuesta de mayo      │
│    y acordaron ajustar el timeline para entregar    │
│    en la segunda semana de junio.                   │
│                                                     │
│    Decisiones                                       │
│    ● Aceptar propuesta con enmiendas → [ver cita]   │
│    ● Timeline final: 13 de junio → [ver cita]       │
│                                                     │
│    Acciones                                         │
│    □ [Carlos] Preparar draft final    lun 19 abr    │
│    □ [Ana] Confirmar presupuesto      mar 20 abr    │
│    □ [Tú] Enviar contrato             mié 21 abr    │
│                                                     │
└────────────────────────────────────────────────────┘
```

**Tabs:**
1. **Resumen** (default): tus notas + resumen IA en tipografía editorial.
2. **Transcripción:** timeline con speakers etiquetados por color, timestamps clicables, búsqueda interna.
3. **Notas:** editor markdown para tus notas crudas.
4. **Chat:** conversación con la reunión como contexto.

**Jerarquía visual notas vs IA:**
- Tus notas en `--text-primary` con typography estándar.
- Resumen IA en contenedor sutilmente diferenciado (fondo `--bg-sunken`, borde izquierdo `--accent-600` de 2px).
- Cada bullet del resumen tiene `[ver cita]` que resalta el segmento en la transcripción.

### 7.4 Command palette (Cmd/Ctrl+K)

**Propósito:** acceso rápido a todo.

Modal centrado, 600×500px aprox. Input de búsqueda arriba, resultados debajo categorizados:
- Acciones (Nueva grabación, Ajustes, Tema oscuro...)
- Reuniones (match por título, fecha, contenido)
- Segmentos (match en transcripciones con snippet)
- Speakers conocidos (para futuro v1.1)

Navegación completa por teclado. ESC cierra.

### 7.5 Chat (Cmd/Ctrl+J)

**Propósito:** preguntar sobre la reunión actual (o seleccionada).

Panel lateral derecho o modal inferior según preferencia. Conversación con el LLM, cada respuesta con citas [1], [2] que enlazan a segmentos específicos.

```
┌─────────────────────────────────────┐
│  Chat                          [×]  │
├─────────────────────────────────────┤
│                                     │
│  tú                                 │
│  ¿qué decidimos sobre el timeline?  │
│                                     │
│  echo                               │
│  Acordaron entregar en la segunda   │
│  semana de junio, específicamente   │
│  el 13 de junio [1]. Carlos se com- │
│  prometió a preparar el draft       │
│  final para el 19 de abril [2].     │
│                                     │
│  [1] 00:18:42 — Ana                │
│  [2] 00:24:15 — Carlos             │
│                                     │
├─────────────────────────────────────┤
│  [Pregunta algo...]          [→]   │
└─────────────────────────────────────┘
```

### 7.6 Settings

**Propósito:** configurar sin abrumar.

Sidebar izquierdo con categorías, contenido a la derecha. Cada sección tiene descripción breve arriba y controles debajo, bien espaciados. Ningún toggle sin explicación.

Ejemplo de sección "Privacidad":

```
Privacidad

Controla qué datos se guardan en tu equipo y cómo.

Retención de audio
────────────────
Qué pasa con el audio después de grabar.
○ Descartar inmediatamente (recomendado)
● Guardar 7 días
○ Guardar 30 días
○ Guardar siempre

Cifrar base de datos
────────────────
Tus reuniones se cifran con AES-256.
La contraseña se guarda en el keychain de tu sistema.
[ Activar cifrado ]

Telemetría de errores
────────────────
Ayúdanos a mejorar enviando stack traces cuando algo crashea.
Nunca enviamos contenido de tus reuniones.
[ toggle OFF ]
```

---

## 8. Componentes

### 8.1 Primitivas (de shadcn/ui, adaptadas)

- **Button** (variants: primary, secondary, ghost, danger)
- **Input** (text, search, textarea)
- **Select / Combobox**
- **Checkbox / Radio / Switch**
- **Tabs**
- **Dialog / Modal**
- **Sheet** (slide-over desde lado)
- **Toast**
- **Tooltip**
- **Dropdown menu**
- **Context menu**
- **Popover**
- **Command** (base para command palette)

### 8.2 Componentes compuestos (específicos de Echo)

#### MeetingCard

Card de reunión en listas. Muestra título, fecha relativa, duración, 2-3 participantes con avatares, primera línea de resumen si existe. Hover eleva y revela acciones rápidas.

#### Waveform

Visualización de audio estilo sismograma. 2 tracks apiladas (mic en ámbar, sistema en azul pizarra), tiempo horizontal, amplitud vertical. Durante grabación es live; post-recording es scrubbeable si hay audio retenido.

#### TranscriptSegment

Un segmento de transcripción con: avatar/color de speaker, nombre, timestamp clicable, texto. Hover muestra acciones (reproducir desde aquí si hay audio, copiar, saltar al resumen).

#### SummaryBlock

Bloque de resumen con título de sección (Decisiones, Acciones, etc.), items con linkage al transcript. Editable inline.

#### SpeakerChip

Chip con color del speaker, nombre, pequeño indicador de pista (mic/sistema). Clicable para renombrar.

#### AudioLevel

Indicador de nivel de mic/sistema. Barras verticales con animación suave, color según intensidad.

#### ModelStatus

Pequeño badge en settings que muestra modelo activo, tamaño en disco, estado (cargado/no cargado).

---

## 9. Estados y feedback

### 9.1 Estados de grabación

| Estado | Indicador visual |
|---|---|
| Idle | Botón "Nueva grabación" ámbar prominente |
| Iniciando | Spinner sutil, "Preparando..." |
| Recording | Timer pulsante, waveform animado, indicador rojo en título de ventana |
| Paused | Timer estático en gris, waveform congelado, botón "Reanudar" |
| Stopping | "Finalizando..." con spinner |
| Refining | Progress bar "Afinando transcripción... 23%" + "Generando resumen... 67%" |
| Ready | Transición suave a Meeting view con fade |

### 9.2 Empty states

Cada lista vacía tiene:
- Ilustración sutil dibujada a mano (estilo pluma/tinta, paleta neutra).
- Título corto.
- Descripción de 1 línea.
- CTA si aplica.

Ejemplo de Home vacío:

```
        ~  ~  ~
       ~   ✎   ~
        ~  ~  ~

  Aún no has grabado nada.

  Empieza tu primera reunión y verás
  tus notas aparecer aquí.

        [ Nueva grabación ]
```

### 9.3 Error states

Errores siempre con:
- Código claro (`AUDIO_DEVICE_NOT_FOUND`)
- Mensaje humano ("No encontramos tu micrófono. Verifica que esté conectado.")
- Acción de recuperación ("Abrir ajustes de audio" / "Reintentar" / "Cancelar")

Ningún error sin salida.

### 9.4 Loading states

- **Operaciones < 200 ms:** sin indicador.
- **200 ms - 2 s:** skeleton (no spinner).
- **2 s - 10 s:** skeleton + mensaje contextual.
- **> 10 s:** progress bar con porcentaje real si es medible, con mensaje y opción de cancelar si aplica.

---

## 10. Accesibilidad

### 10.1 Metas

- WCAG 2.1 AA en toda la UI principal.
- AAA en texto largo (transcripciones, notas).

### 10.2 Implementación

- Todos los controles navegables por teclado.
- Focus rings visibles y consistentes (`--accent-600` con outline-offset).
- Contraste mínimo 4.5:1 para texto normal, 3:1 para texto grande y controles.
- `prefers-reduced-motion` respetado en todas las animaciones.
- `prefers-color-scheme` para tema automático.
- Screen readers: todos los iconos tienen `aria-label`, landmarks correctos, live regions para el streaming.
- Internacionalización: todas las strings externalizadas en JSON (ES + EN en MVP).

### 10.3 Teclado — atajos globales

| Atajo | Acción |
|---|---|
| Cmd/Ctrl+N | Nueva grabación |
| Cmd/Ctrl+Shift+N | Nueva grabación con hints de participantes |
| Cmd/Ctrl+K | Command palette |
| Cmd/Ctrl+J | Chat con reunión actual |
| Cmd/Ctrl+F | Buscar en vista actual |
| Cmd/Ctrl+, | Abrir ajustes |
| Cmd/Ctrl+Shift+F | Focus mode (durante grabación) |
| Space | Pausa/reanuda grabación (cuando está en Recording) |
| Esc | Cerrar modal / salir de focus mode |

---

## 11. Motion y micro-interacciones

### 11.1 Principios de motion

- **Propósito primero:** cada animación comunica algo (cambio de estado, relación entre elementos, feedback).
- **Rápido y limpio:** la mayoría de transiciones son 150-250 ms. Más de 400 ms se siente lento.
- **Easing natural:** `cubic-bezier(0.32, 0.72, 0, 1)` como default. Nada de linear ni rebotes exagerados.
- **Respeta `prefers-reduced-motion`:** cuando está activo, las transiciones se reducen a fades de 100 ms o se eliminan.

### 11.2 Micro-interacciones clave

**Waveform en grabación:**
Estilo sismograma orgánico. Dos pistas apiladas, cada una con una "línea de tiempo" que dibuja amplitud. No los clásicos "bars pumping" de Granola — algo más cercano a una pluma sobre papel.

**Botón de grabación:**
En idle, un pulso muy sutil (3 s de periodo, 10% scale) que respira.
Al hacer click, click táctil con un hold de 100 ms antes de iniciar.
Durante grabación, cambia a cuadrado de stop, con un anillo que indica que está activo.

**Refinamiento post-stop:**
Progress bar segmentada — "Afinando transcripción" (primer segmento), "Detectando speakers" (segundo), "Generando resumen" (tercero). Cada segmento se ilumina a medida que avanza.

**Entrada de texto en streaming:**
Nuevas palabras aparecen con un fade-in de 150 ms, sin slide ni pop. Como tinta apareciendo.

**Cambio entre tabs de Meeting view:**
Crossfade de 200 ms entre contenidos. El indicador activo se mueve con spring suave.

**Hover en MeetingCard:**
Elevación + scale 1.01 + border ligeramente más fuerte. 150 ms.

**Toast:**
Entra desde abajo con slide + fade. Auto-dismiss en 4 s. Slide out al hacer swipe.

---

## Apéndice A — Pantallas principales

Las pantallas clave de la aplicación:

1. **Home / Library** — Lista de reuniones y búsqueda
2. **Recording** — Vista de grabación activa
3. **Meeting view** — Transcripción con resumen
4. **Onboarding** — Flujo de configuración inicial
5. **Settings** — Preferencias de la aplicación

## Apéndice B — Referencias visuales

- **Tipografía editorial:** NYT, Readwise, iA Writer
- **Paleta terrosa:** Typewolf recent picks, Are.na editorial moodboards
- **Motion minimal:** Linear, Arc, Raycast
- **Density bien manejada:** Linear, Height, Pitch

## Apéndice C — Qué falta en este documento

Para una v1.0 de diseño completo faltará:
- Spec detallada de iOS/Android (v2.0).
- Plantillas de emails de compartir (si se implementa).
- Brand guidelines completas (logo variants, usos incorrectos, etc.).
- Illustrations para empty states (producción).
- Design tokens exportados como JSON (Figma → código).
- Component library completa en Figma o Storybook.

---

**Este documento es un punto de partida.** Evoluciona con cada iteración de mockup real y con feedback de usuarios beta.
