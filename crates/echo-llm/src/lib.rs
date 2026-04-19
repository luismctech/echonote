//! # echo-llm
//!
//! Local Large Language Model adapter. Wraps llama.cpp (via `llama-cpp-rs`)
//! to serve two distinct workloads:
//!
//! - **Summarization**: one-shot prompts that return JSON matching one of
//!   the six templates in DESIGN.md §3.2.
//! - **Chat**: multi-turn conversation about a meeting's transcript with
//!   citation back to segments.
//!
//! Default model: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (Balanced profile).

#![warn(rust_2018_idioms, clippy::all)]
