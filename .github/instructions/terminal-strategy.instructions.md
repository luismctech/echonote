---
applyTo: "**"
---

# Estrategia Obligatoria de Ejecución en Terminal

## Principio Fundamental

**SIEMPRE** verifica la salida real de un comando antes de tomar decisiones basadas en él. **NUNCA** asumas que un comando tuvo éxito sin leer su output. Esta regla aplica a TODOS los CLIs sin excepción: `az`, `terraform`, `bicep`, `git`, `npm`, `node`, `docker`, `kubectl`, `gh`, `curl`, `sonar-scanner`, y cualquier otro ejecutable.

## Requisito Previo: Carga de Herramientas

Antes de tu primera interacción con terminal en una conversación, carga las herramientas diferidas necesarias:
- `send_to_terminal` — para enviar comandos a terminales existentes del usuario.
- `terminal_last_command` — para verificar el último comando ejecutado y su exit code.

Si `send_to_terminal` no está disponible después de cargar, usa `run_in_terminal` como alternativa.

## Descubrimiento de Terminal Activa

Antes de enviar el primer comando, determina el `terminalId` correcto:

1. **Revisa el contexto del turno** — VS Code incluye información de terminales activas en `<context>` con el `terminalId`, último comando, cwd y exit code. Usa este dato primero.
2. **Si no hay contexto**, usa `get_terminal_output(terminalId: 1)` como intento inicial. Si falla, incrementa el ID.
3. **`terminal_last_command` NO sirve para descubrir terminales** — devuelve "No active terminal instance found" si no hay terminal registrada. Solo úsalo para verificar exit codes de comandos ya ejecutados.
4. **Si no hay terminal activa**, usa `run_in_terminal` para crear una y luego cambia a `send_to_terminal` + `get_terminal_output` para los comandos siguientes.

**IMPORTANTE:** Una vez que descubras el `terminalId` correcto, reutilízalo para toda la conversación. No lo busques de nuevo salvo que recibas un error de terminal no encontrada.

## Por Qué `run_in_terminal` NO Es Confiable para Captura de Output

> **Evidencia probada:** En pruebas reales, `run_in_terminal(mode=sync)` devolvió SOLO el prompt del shell (sin output del comando) en **3 de 3 intentos** consecutivos. El mismo comando ejecutado con `send_to_terminal` + `get_terminal_output` capturó el **100% del output** en todos los casos.

`run_in_terminal` tiene un race condition: retorna antes de que el output del comando llegue al buffer del terminal. Esto lo hace **inadecuado** para cualquier comando cuyo output necesites leer. Úsalo SOLO para:
- Crear una terminal nueva cuando no existe ninguna.
- Procesos long-running en `mode=async` (servers, watchers).
- Como último fallback si `send_to_terminal` no está disponible.

## Reglas Anti-Paging (OBLIGATORIAS)

La causa principal de pérdida de output es el **paging**. SIEMPRE agrega flags anti-paging a estos CLIs:

**Regla simple: SIEMPRE agrega `| cat` al final de cualquier comando.** Es el safety net universal que previene paging sin efectos secundarios.

| CLI | Flag obligatorio | Ejemplo |
|---|---|---|
| `az` | `\| cat` al final | `az account show \| cat` |
| `git` | `\| cat` (preferido por simplicidad) | `git log --oneline -10 \| cat` |
| `gh` | `\| cat` | `gh pr list \| cat` |
| `terraform` | `-no-color \| cat` | `terraform plan -no-color \| cat` |
| `kubectl` | `\| cat` | `kubectl get pods \| cat` |
| `docker` | `\| cat` | `docker ps \| cat` |
| `less` / `more` | Reemplazar con `cat` | Usar `cat archivo.txt` |
| Cualquier CLI | `\| cat` siempre | `<comando> \| cat` |

> **Nota:** `--no-pager` en `git` y `gh` también funciona, pero `| cat` es universal y más simple de recordar como regla única.

## Selección de Herramienta según Tipo de Comando

| Escenario | Herramienta | Razón |
|---|---|---|
| **Comando estándar (< 30s)** | `send_to_terminal` → `get_terminal_output` | Captura completa del buffer del terminal del usuario |
| **Comando que necesita entorno/variables del usuario** | `send_to_terminal` → `get_terminal_output` | Reutiliza sesión, env vars y cwd del usuario |
| **Comando interactivo (prompts)** | `send_to_terminal` para cada respuesta, `get_terminal_output` entre cada una | Una respuesta por envío |
| **Proceso largo (server, watch, build > 60s)** | `run_in_terminal` con `mode=async` | Tiene gestión de timeout integrada |
| **Fallback (si `send_to_terminal` no disponible)** | `run_in_terminal` con `mode=sync`, timeout=30000 | Alternativa con espera integrada |

## Flujo Principal: send_to_terminal (3 pasos)

### Paso 1: Enviar el comando
Usa `send_to_terminal` con el `terminalId` de la terminal activa del usuario.
- Si no conoces el `terminalId`, usa `get_terminal_output` sin parámetros o `terminal_last_command` para descubrir la terminal activa.
- NO hardcodees `terminalId: 1` — el ID puede variar. Usa el ID real de la terminal visible.

```
send_to_terminal(terminalId: <id_real>, command: "az account show | cat")
```

### Paso 2: Leer la salida
Llama a `get_terminal_output` con el mismo `terminalId` para obtener el resultado completo.

```
get_terminal_output(terminalId: <id_real>)
```

### Paso 3: Validar el resultado
Antes de proceder al siguiente comando:
- ✅ Verifica que la salida contiene lo esperado (JSON válido, mensaje de éxito, etc.)
- ✅ Busca indicadores de error: `ERROR`, `FATAL`, `failed`, `denied`, `not found`
- ✅ Si la salida parece incompleta o vacía, re-lee con `get_terminal_output`
- ✅ Si el comando falló, diagnostica antes de reintentar

## Flujo Alternativo: run_in_terminal

Usar SOLO cuando `send_to_terminal` no esté disponible o para procesos long-running:

```
run_in_terminal(
  command: "terraform plan -no-color | cat",
  explanation: "Genera plan de Terraform para revisar cambios",
  goal: "Terraform plan",
  mode: "sync",
  timeout: 30000
)
```

Para servidores o procesos continuos:
```
run_in_terminal(
  command: "npm start",
  explanation: "Inicia el servidor de desarrollo",
  goal: "Start dev server",
  mode: "async",
  timeout: 10000
)
```

## Reglas Estrictas

### PROHIBIDO
- ❌ Ejecutar el siguiente comando sin leer la salida del anterior con `get_terminal_output`.
- ❌ Asumir que un comando tuvo éxito sin verificar su output.
- ❌ Usar CLIs con paging (`az`, `git`, `gh`, `terraform`, `kubectl`) sin flags anti-paging.
- ❌ Encadenar con `&&` múltiples comandos cuando cada resultado importa para decisiones.
- ❌ Ignorar mensajes de error en la salida.
- ❌ Hardcodear `terminalId: 1` sin verificar que existe.

### OBLIGATORIO
- ✅ Cargar `send_to_terminal` con `tool_search` antes de su primer uso.
- ✅ Aplicar flags anti-paging a TODOS los CLIs listados en la tabla.
- ✅ Leer y validar la salida de CADA comando antes de ejecutar el siguiente.
- ✅ Verificar el directorio de trabajo (`pwd`) antes de comandos sensibles al path (`terraform`, `npm`, `cd`).
- ✅ Usar `terminal_last_command` cuando necesites verificar el exit code del último comando.

## Manejo de Errores

| Situación | Acción |
|---|---|
| Salida vacía | Esperar y re-leer con `get_terminal_output`. Si persiste, reintentar el comando. |
| `command not found` | Verificar con `which <comando>` antes de reintentar. |
| `permission denied` | Informar al usuario. NO usar `sudo` sin autorización explícita. |
| Timeout / comando colgado | Verificar si el proceso sigue corriendo. Considerar si falta flag anti-paging. |
| Output potencialmente grande | **Primero mide**: `<comando> 2>&1 \| wc -l`. Si >100 líneas, filtra con `\| head -50`, `\| grep <patrón>`, o `\| tail -30`. |
| Terminal no encontrada | Usar `run_in_terminal` como fallback para crear una nueva sesión. |
| Error de autenticación (`az`, `gh`) | Informar al usuario que necesita autenticarse manualmente. |

## Directorio de Trabajo

Antes de ejecutar comandos sensibles al path, verifica el directorio actual.

**Patrón atómico obligatorio para cambiar de directorio:**

SIEMPRE combina `cd` con `pwd` en un solo comando para confirmar el cambio en una sola operación:

```
send_to_terminal(terminalId: <id>, command: "cd devops-frontend && pwd")
get_terminal_output(terminalId: <id>) → confirmar que el output termina en ".../devops-frontend"
```

**NUNCA** hagas `cd` solo sin `&& pwd` — no tendrás confirmación visual de que el cambio fue exitoso.

Comandos que REQUIEREN verificación de directorio antes de ejecutarse:
- `terraform init/plan/apply/destroy`
- `npm install/test/run/build`
- `ng serve/build/test`
- `az bicep build`
- `git` (para confirmar que estás en el repo correcto)

## Ejemplos

### ✅ Correcto: Comando estándar con anti-paging
```
1. send_to_terminal(terminalId: 2, command: "az account show | cat")
2. get_terminal_output(terminalId: 2) → verificar JSON con subscription activa
3. Usar los datos del resultado
```

### ✅ Correcto: Multi-step con verificación entre cada paso
```
1. send_to_terminal(terminalId: 2, command: "cd terraform && pwd")
2. get_terminal_output(terminalId: 2) → confirmar "…/terraform"
3. send_to_terminal(terminalId: 2, command: "terraform init -no-color | cat")
4. get_terminal_output(terminalId: 2) → verificar "Terraform has been successfully initialized"
5. send_to_terminal(terminalId: 2, command: "terraform validate -no-color | cat")
6. get_terminal_output(terminalId: 2) → verificar "Success!"
```

### ✅ Correcto: Comando interactivo
```
1. send_to_terminal(terminalId: 2, command: "npm init")
2. get_terminal_output(terminalId: 2) → leer primer prompt
3. send_to_terminal(terminalId: 2, command: "mi-paquete")
4. get_terminal_output(terminalId: 2) → leer siguiente prompt
5. [continuar una respuesta por turno hasta finalizar]
```

### ✅ Correcto: Proceso long-running
```
1. run_in_terminal(command: "npm start", mode: "async", timeout: 10000, ...)
2. [el agente recibe notificación cuando hay output o el proceso está idle]
3. get_terminal_output(id: "<uuid-retornado>") → verificar que el servidor inició
```

### ✅ Correcto: Output grande (medir antes de filtrar)
```
1. send_to_terminal(terminalId: 2, command: "npm list --all 2>&1 | wc -l")
2. get_terminal_output(terminalId: 2) → lee el conteo (ej: "1130")
3. send_to_terminal(terminalId: 2, command: "npm list --all 2>&1 | head -30 | cat")
4. get_terminal_output(terminalId: 2) → lee las primeras 30 líneas
```

### ✅ Correcto: Descubrimiento de terminal desde contexto
```
1. Leer <context> del turno → Terminal: zsh, terminalId visible
2. get_terminal_output(terminalId: <id_del_contexto>) → confirmar que responde
3. Proceder con send_to_terminal usando ese terminalId
```

### ❌ Incorrecto
```
❌ send_to_terminal(command: "az group list")              ← Falta | cat
❌ run_in_terminal(command: "git log")                     ← Falta | cat, se cuelga
❌ run_in_terminal(command: "terraform plan") y usar su output  ← Output probablemente vacío
❌ Ejecutar terraform apply sin leer el output del plan
❌ Asumir que npm install fue exitoso sin verificar la salida
❌ send_to_terminal(terminalId: 1, ...) sin verificar que el terminal 1 existe
❌ cd terraform (sin && pwd) → no hay confirmación de que el cambio funcionó
❌ npm list --all | cat → output de 1000+ líneas sin filtrar, posible truncado
```
