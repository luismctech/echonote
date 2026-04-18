# ADR-0002: Rust + React + TypeScript as the base stack

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** Tech Lead, Frontend Engineer
- **Technical story:** Sprint 0 — confirm language choices for backend and
  frontend before any crate or component is written.

## Context and problem statement

Given the decision to ship on Tauri (see [ADR-0001](./0001-tauri-over-electron.md)),
we must pick exactly one backend language and one frontend stack. Hesitating
on this choice costs compound time — every additional library, test harness
and CI step is written for whichever pair we commit to today.

## Decision drivers

- **Native FFI** with whisper.cpp, llama.cpp, ONNX Runtime and SQLite with no
  Python or JVM in the loop.
- **Memory safety** in the component that handles raw audio buffers and
  concurrent streams.
- **Hiring pool.** Both layers must be staffable from a small founding team
  and, later, from the open-source community.
- **Type safety across the IPC boundary** so we can generate contracts from
  one side and consume them on the other.
- **Design-system fit** with the editorial, notebook-like direction set in
  `docs/DESIGN.md` — requires a framework with a mature headless component
  ecosystem.
- **Accessibility.** WCAG 2.1 AA out of the box is non-negotiable; the UI
  framework must help, not hinder.

## Considered options

### Backend

1. **Rust** (1.88+) — the language Tauri is written in.
2. **Go** with CGO bindings to whisper.cpp / llama.cpp.
3. **C++** directly with a thin Tauri Rust wrapper.

### Frontend

1. **React 18 + TypeScript** with Tailwind + shadcn/ui.
2. **Svelte 5 + TypeScript** with Tailwind.
3. **SolidJS + TypeScript**.
4. **Vue 3 + TypeScript**.

## Decision outcome

**Chosen options:**

- Backend: **Rust 1.88+** with strict `#![warn(rust_2018_idioms, clippy::all)]`
  and `#![forbid(unsafe_code)]` on domain and storage crates.
- Frontend: **React 18 + TypeScript 5.5 (`strict: true`)** with Tailwind 3.4
  and shadcn/ui components copied into the repository, state split between
  Zustand (UI) and TanStack Query (server state) — to be revisited in
  [ADR-0008](./0008-zustand-over-redux.md).

## Consequences

### Positive

- Tauri already mandates Rust in `src-tauri`; choosing Rust for every other
  crate removes the ceremony of crossing language boundaries internally.
- The domain crate can compile with `no_std` compatible dependencies only,
  which keeps it trivially portable and testable.
- React 18's concurrent features (`useDeferredValue`, Suspense for transitions)
  are well-suited to our live transcript UI.
- TypeScript contracts between the UI and Rust are generated via `specta` or
  `ts-rs`, eliminating shape drift at the IPC boundary.
- shadcn/ui gives us accessibility-correct primitives that we own in-tree;
  no npm dependency to audit repeatedly.

### Negative

- Rust compile times with heavy native crates (whisper-rs, llama-cpp-rs) will
  reach minutes on cold builds. Mitigated by `sccache`, `Swatinem/rust-cache`
  in CI and staged dependency introduction across Sprint 0.
- The combination of Rust + React doubles the toolchain setup on developer
  machines. Mitigated by `rust-toolchain.toml`, pinned `pnpm` version and a
  `scripts/bootstrap.sh` helper (to be added in Sprint 0 day 3).

### Neutral

- Introducing React 19 is an explicit Sprint 2+ decision: we want server
  components out of scope for a desktop app, and the concurrent model we
  need is fully stable on 18.

## Pros and cons of the options

### Backend

#### Rust (chosen)

- **Pros.** Memory safety with zero-cost abstractions; Tauri-native; mature
  ecosystem for audio (`cpal`), ML bindings (`whisper-rs`, `llama-cpp-rs`,
  `ort`) and SQL (`sqlx`); trait-based ports fit Clean Architecture cleanly.
- **Cons.** Learning curve; long compile times on heavy crates.

#### Go + CGO

- **Pros.** Lower ramp-up; fast compile; good concurrency.
- **Cons.** CGO boundary penalizes hot audio paths; Tauri integration
  requires an extra process; memory safety story weaker than Rust for the
  buffers we juggle.

#### C++

- **Pros.** Absolute control; direct FFI to every native dep.
- **Cons.** No memory safety; slow iteration; undermines the whole privacy
  posture the app is built on.

### Frontend

#### React 18 + TypeScript (chosen)

- **Pros.** Biggest component ecosystem; shadcn/ui alignment; mature testing
  tools (Vitest, Testing Library, Playwright); large hiring pool.
- **Cons.** Heavier bundle than alternatives; more implicit renders; needs
  discipline with `useMemo`/`useCallback` in streaming UI.

#### Svelte 5

- **Pros.** Smaller bundles; simpler reactive model; excellent DX.
- **Cons.** Smaller pool of accessible component libraries; shadcn-Svelte
  port lags behind upstream; Tauri integrations and testing tooling not as
  battle-tested.

#### SolidJS

- **Pros.** Minimal overhead; fine-grained reactivity; React-like mental
  model.
- **Cons.** Ecosystem too young for the design system and accessibility
  guarantees we require; limited hiring pool.

#### Vue 3

- **Pros.** Strong DX; gentle learning curve; decent ecosystem.
- **Cons.** Accessibility coverage weaker than Radix/shadcn; pulls us away
  from headless primitives we want to own.

## References

- `docs/ARCHITECTURE.md` §3.2.2–3.2.4 — original rationale summary.
- `docs/DESIGN.md` §4, §8 — design system and component palette that bake in
  shadcn-style primitives.
- TypeScript strict handbook — https://www.typescriptlang.org/docs/handbook/2/basic-types.html
