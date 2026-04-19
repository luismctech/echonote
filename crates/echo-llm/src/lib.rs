//! # echo-llm
//!
//! Local Large Language Model adapter. Wraps llama.cpp (via
//! [`llama-cpp-2`]) to serve two distinct workloads:
//!
//! - **Summarization**: one-shot prompts that return JSON matching one
//!   of the templates in DESIGN.md §3.2 (Sprint 1 day 9 ships the
//!   "general" template only; the rest follow in Sprint 2).
//! - **Chat**: multi-turn conversation about a meeting's transcript
//!   with citation back to segments (Sprint 2 / CU-05).
//!
//! Default model: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (Balanced profile
//! per `docs/DEVELOPMENT_PLAN.md`). Smaller alternatives can be swapped
//! in by pointing `ECHO_LLM_MODEL` at a different `.gguf` file.

#![warn(rust_2018_idioms, clippy::all)]

mod llama_cpp;

pub use llama_cpp::{LlamaCppLlm, LoadOptions};
