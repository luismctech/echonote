//! # echo-llm
//!
//! Local Large Language Model adapter. Wraps llama.cpp (via
//! [`llama-cpp-2`]) to serve two distinct workloads off the same
//! loaded model:
//!
//! - **Summarization** ([`LlamaCppLlm`]): one-shot prompts that return
//!   JSON matching one of the templates in `docs/DESIGN.md` §3.2
//!   (Sprint 1 day 9 ships the "general" template only; the rest
//!   follow in Sprint 2).
//! - **Chat** ([`LlamaCppChat`]): multi-turn conversation about a
//!   meeting's transcript with citation back to segments
//!   (Sprint 1 day 10 — CU-05). Streams tokens incrementally via
//!   [`echo_domain::ChatAssistant::ask`] so the React UI renders the
//!   reply as it's decoded.
//!
//! Both adapters can share a single loaded model — see
//! [`LlamaCppLlm::chat_handle`] for the canonical wiring.
//!
//! Default model: `Qwen3-14B-Instruct-Q4_K_M.gguf` (Quality profile per
//! `docs/DEVELOPMENT_PLAN.md`). Smaller alternatives can be swapped in
//! by pointing `ECHO_LLM_MODEL` at a different `.gguf` file.

#![warn(rust_2018_idioms, clippy::all)]

mod backend;
mod llama_cpp;
mod llama_cpp_chat;
mod shared;

pub use llama_cpp::{LlamaCppLlm, LoadOptions};
pub use llama_cpp_chat::LlamaCppChat;
