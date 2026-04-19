# EchoNote frontend — folder layout & layering

The React frontend follows a small clean-architecture split. The
folders below are layered top-to-bottom; **each layer may import from
layers above it but never the other way around**. There is no lint
plugin enforcing this; the rule is short enough to keep in your head.

```
types/        ← pure domain shapes (no React, no IPC)
ipc/          ← Tauri adapter (the only callers of `invoke`)
state/        ← reducers + Context providers (recording, meetings)
hooks/        ← application-layer orchestrators
lib/          ← pure presentation utilities (formatters, palettes…)
components/   ← reusable, prop-driven UI primitives
features/     ← view-level components grouped by feature folder
App.tsx       ← shell that composes providers + layout
main.tsx      ← entry point: provider stack + ReactDOM.createRoot
```

## What lives where

### `types/`

Domain entities that mirror the Rust backend (`HealthStatus`,
`Meeting`, `Speaker`, `TranscriptEvent`) plus pure UI-state types
shared across multiple components (`Probe`, `StreamLine`, `MainView`).

Rule: zero imports of React, of `@tauri-apps/api`, or of any other
file in this tree. Pure structural types only.

### `ipc/`

The adapter boundary. **The only files in the entire frontend that
may import from `@tauri-apps/api/core`.**

- `client.ts` — typed wrappers over `invoke()` (`healthCheck`,
  `startStreaming`, `listMeetings`, …).
- `isTauri.ts` — environment guard so the app degrades gracefully
  in `pnpm dev` (no IPC available).
- `useIpcAction.ts` — DRY hook + pure `runIpcAction` helper that
  collapses the "try IPC, push toast on failure" pattern.

### `state/`

Reducers and React Context providers that own non-trivial cross-cutting
state.

- `recording.ts` — finite-state machine for the live session
  (`idle → starting → recording → stopping → persisted | error`)
  plus selectors. Pure: no React, fully unit-tested.
- `useMeetingsStore.tsx` — `MeetingsProvider` + `useMeetings()`. Owns
  the meetings list, the right-pane `view`, the search input/hits/
  loading/error, and the on-mount refresh effect. Composes the
  meeting-detail actions internally so consumers get
  `goToMeeting` / `deleteMeeting` / `renameSpeaker` ready to use.

### `hooks/`

Application-layer orchestrators. Each one knows how to *use* the IPC
adapter and the toast API to fulfil one use case; views consume the
returned values + callbacks without ever calling `invoke()`.

- `useHealthProbe` — owns the on-mount `health_check`.
- `useRecordingSession` — owns the recording reducer + lines + stats
  + auto-scroll ref + `handleEvent` translator + `start`/`stop`/
  `dismissError`/`reset` actions + the dedup'd error→toast effect.
- `useMeetingDetail` — owns `openMeeting` / `renameSpeakerAction` /
  `deleteMeetingAction`. Receives `view` / `setView` / `refresh` /
  `setMeetingsError` from outside so it slots into either App.tsx
  or `MeetingsProvider` unchanged.

### `lib/`

Pure presentation utilities.

- `format.ts` — `formatTimestamp`, `formatDate`, `formatDurationMs`.
- `speakers.ts` — palette, `paletteFor(slot)`, `displayName`,
  `shortTag`, `indexSpeakers`. Unit-tested.
- `useDebouncedValue.ts` — generic debounced-value hook.

### `components/`

Reusable, prop-driven UI primitives. **Never call `invoke()` and
never read `useMeetings()`.** Anything cross-cutting flows in via
props.

- `ErrorBoundary` — class boundary for top-of-tree crashes.
- `Toaster` — toast API (`useToast`) + renderer.
- `SpeakerChip`, `Stat`, `StatsBar` — visual atoms.

### `features/`

View-level components grouped by feature folder. They may consume
hooks and state contexts (that's the point of feature containers),
but they still must not call `invoke()` directly.

```
features/live/        LivePane, TranscriptRow, HealthProbe
features/meetings/    MeetingDetail, SpeakersPanel, SpeakerEditor
features/sidebar/     Sidebar (container), MeetingsList,
                      MeetingsSearchBox, SearchResults
```

The rail-level container `<Sidebar />` reads from `useMeetings()` so
the shell does not have to thread a dozen props through; the leaf
components stay prop-driven so they remain testable in isolation.

### `App.tsx`

The shell. Composes hooks (`useHealthProbe`, `useRecordingSession`)
with the meetings context, owns the user-pref toggles (`language`,
`diarize`) that need to outlive view switches, and lays out the
header / sidebar / main pane. Should stay around 100 lines.

### `main.tsx`

Entry point. Wires the provider stack:

```tsx
<ErrorBoundary>
  <ToastProvider>
    <MeetingsProvider>
      <App />
    </MeetingsProvider>
  </ToastProvider>
</ErrorBoundary>
```

Provider order matters: each child reads APIs declared by its
ancestors (`MeetingsProvider` consumes `useToast()` via
`useIpcAction`).

## Naming conventions

- **Components**: `PascalCase.tsx`, one named export per file.
- **Hooks**: `useCamelCase.ts(x)`, one named export per file.
- **Pure utilities**: `camelCase.ts`, named exports only.
- **No default exports anywhere** — they break re-export grep'ability
  and IDE rename refactors.

## Adding a new feature

1. Add backend Rust command + types.
2. Add the matching TypeScript types in `types/<domain>.ts`.
3. Add the IPC wrapper in `ipc/client.ts`.
4. If the feature has non-trivial orchestration, add a
   `hooks/use<Feature>.ts` — keep components dumb.
5. Build the views in `features/<feature>/`. Reuse primitives from
   `components/` and formatters from `lib/`.
6. Wire it into `App.tsx` (or extend `MeetingsProvider` if it touches
   meetings/view/search state).

If you find yourself importing `invoke` outside `ipc/`, stop and add
a wrapper in `ipc/client.ts` instead. That's the boundary.
