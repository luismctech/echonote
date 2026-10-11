//! llama.cpp implementation of the worker's [`Engine`]: one persistent
//! `LlamaContext` plus the single decode loop shared by every entry point
//! (one-shot generation, raw streaming and chat).
//!
//! ## Prefix reuse
//!
//! The engine records the exact tokens resident in the KV cache (the
//! evaluated prompt plus every generated token that was decoded into it).
//! A new request keeps the longest common prefix with that record, drops
//! the rest of the cache with a partial removal and decodes only the
//! remaining prompt tokens, so a follow-up chat turn does not re-evaluate
//! the system prompt and transcript again (see [`crate::prefix`] for the
//! policy). At least the last prompt token is always decoded so logits
//! exist for the first sample. Reuse leaves the model's input unchanged, so
//! greedy output is expected to match a cold run; it is not guaranteed to
//! be bitwise identical, because a backend may compute a position with
//! different kernels depending on the batch it was decoded in (a reused
//! prefix was evaluated under another batch shape than a cold run would
//! use), which can flip a near-tie token. Any decode error clears the cache and the record, so a later
//! request never trusts a half-written state; dropping the engine (idle
//! release, rebuild after a panic) discards both.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tracing::{debug, warn};

use echo_domain::DomainError;

use crate::prefix::{prepare_cache, KvCache};
use crate::shared::LoadedModel;
use crate::worker::{Engine, EngineSource, Flow, PieceSink, SamplingParams};

/// Seed used when the caller does not pin one.
const DEFAULT_SEED: u32 = 1234;

/// [`EngineSource`] over a loaded model; builds the persistent context.
pub(crate) struct LlamaSource {
    loaded: Arc<LoadedModel>,
    /// Where each engine this source builds reports its reuse.
    reused: Arc<AtomicUsize>,
}

impl LlamaSource {
    /// `reused` receives, per request, how many prompt tokens were found
    /// already evaluated in the KV cache. It belongs to one model, so
    /// engines of other models cannot race it.
    pub(crate) fn new(loaded: Arc<LoadedModel>, reused: Arc<AtomicUsize>) -> Self {
        Self { loaded, reused }
    }
}

impl EngineSource for LlamaSource {
    fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError> {
        let loaded: &LoadedModel = &self.loaded;
        let n_ctx = NonZeroU32::new(loaded.n_ctx)
            .ok_or_else(|| DomainError::Invariant("n_ctx must be > 0".into()))?;

        // `swa_full` stays at llama.cpp's default (true) on purpose: sliding-
        // window models (Gemma 4) then keep the whole prompt in the KV cache,
        // which is what makes the partial removal behind prefix reuse (see
        // `crate::prefix`) safe. With `swa_full = false` the window layers
        // drop old positions and a reused prefix could miss them.
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(loaded.n_ctx);
        if let Some(threads) = loaded.n_threads {
            ctx_params = ctx_params.with_n_threads(threads);
            ctx_params = ctx_params.with_n_threads_batch(threads);
        }

        let ctx = loaded
            .model
            .new_context(loaded.backend, ctx_params)
            .map_err(|e| DomainError::LlmFailed(format!("create context: {e}")))?;
        debug!(n_ctx = loaded.n_ctx, "created llama.cpp context");
        Ok(Box::new(LlamaEngine {
            loaded,
            ctx,
            resident: Vec::new(),
            reused: Arc::clone(&self.reused),
        }))
    }
}

/// A live context bound to the worker thread that created it.
struct LlamaEngine<'a> {
    loaded: &'a LoadedModel,
    ctx: LlamaContext<'a>,
    /// Tokens currently in the KV cache, by position (sequence 0).
    resident: Vec<LlamaToken>,
    /// Reuse of the latest request, shared with the owning model.
    reused: Arc<AtomicUsize>,
}

/// [`KvCache`] over the context's sequence 0.
impl KvCache for LlamaContext<'_> {
    fn remove_from(&mut self, from: usize) -> bool {
        let Ok(from) = u32::try_from(from) else {
            return false;
        };
        matches!(self.clear_kv_cache_seq(Some(0), Some(from), None), Ok(true))
    }

    fn clear(&mut self) {
        self.clear_kv_cache();
    }
}

impl LlamaEngine<'_> {
    /// Empty the cache and the record of what it holds.
    fn reset_cache(&mut self) {
        self.ctx.clear_kv_cache();
        self.resident.clear();
    }

    /// Decode `tokens` at absolute positions `start..`, in `n_batch`-sized
    /// chunks, asking for logits only on the very last one, and record each
    /// chunk as resident once the context accepted it. Returns the size of
    /// the last chunk (where the logits of the final token live).
    fn decode_prompt(&mut self, tokens: &[LlamaToken], start: usize) -> Result<i32, DomainError> {
        let chunk = (self.ctx.n_batch() as usize).max(1);
        let mut batch = LlamaBatch::new(chunk.min(tokens.len()).max(1), 1);
        let mut position = start;
        let mut last_chunk = 0;
        for part in tokens.chunks(chunk) {
            batch.clear();
            for (i, token) in part.iter().enumerate() {
                let is_last = position + i + 1 == start + tokens.len();
                batch
                    .add(*token, (position + i) as i32, &[0], is_last)
                    .map_err(|e| DomainError::LlmFailed(format!("batch.add prompt: {e}")))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| DomainError::LlmFailed(format!("decode prompt: {e}")))?;
            self.resident.extend_from_slice(part);
            position += part.len();
            last_chunk = part.len() as i32;
        }
        Ok(last_chunk)
    }
}

impl Engine for LlamaEngine<'_> {
    fn serve(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn PieceSink,
    ) -> Result<(), DomainError> {
        let result = self.serve_inner(prompt, params, sink);
        if result.is_err() {
            // A failed decode may have left positions half written.
            self.reset_cache();
        }
        result
    }
}

impl LlamaEngine<'_> {
    fn serve_inner(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn PieceSink,
    ) -> Result<(), DomainError> {
        let loaded = self.loaded;

        // The prompt was rendered with the model's chat template (see
        // `crate::template`), which writes any BOS marker it needs as text;
        // tokenizing with special-token parsing turns that text into the
        // BOS token, so none is added here: `AddBos::Never` avoids a
        // duplicate.
        let prompt_tokens = loaded
            .model
            .str_to_token(prompt, AddBos::Never)
            .map_err(|e| DomainError::LlmFailed(format!("tokenize prompt: {e}")))?;

        if prompt_tokens.is_empty() {
            return Err(DomainError::LlmFailed("prompt has no tokens".into()));
        }
        let n_prompt = prompt_tokens.len() as i32;
        let n_max_total = n_prompt.saturating_add(params.max_tokens.min(i32::MAX as u32) as i32);
        if (n_max_total as u32) > loaded.n_ctx {
            return Err(DomainError::LlmFailed(format!(
                "prompt ({} tok) + max_tokens ({}) exceeds context window ({})",
                n_prompt, params.max_tokens, loaded.n_ctx
            )));
        }

        // Keep what the cache already holds of this prompt; the rest of the
        // previous request's tokens must not leak into this one.
        let reused = prepare_cache(&mut self.ctx, &mut self.resident, &prompt_tokens);
        self.reused.store(reused, Ordering::Relaxed);
        debug!(
            reused,
            decoded = prompt_tokens.len() - reused,
            "prompt prefix reuse"
        );
        let last_chunk = self.decode_prompt(&prompt_tokens[reused..], reused)?;

        // Sampler state (penalty window, RNG) is per request.
        let mut sampler = build_sampler(params);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut batch = LlamaBatch::new(1, 1);
        let mut n_cur = n_prompt;
        // Logits live at the last index of the batch decoded most recently:
        // the final prompt chunk first, then one-token generation batches.
        let mut logits_index = last_chunk - 1;
        let mut n_decoded: u32 = 0;
        let max_new = params.max_tokens;

        debug!(
            n_prompt,
            n_ctx = loaded.n_ctx,
            max_new,
            temp = params.temperature,
            top_p = params.top_p,
            "starting llama decode loop"
        );

        while n_decoded < max_new {
            // A consumer that left (even while a `<think>` block keeps
            // output suppressed) must not keep the queue busy.
            if sink.is_closed() {
                debug!(n_decoded, "consumer dropped; aborting decode");
                return Ok(());
            }

            let token = sampler.sample(&self.ctx, logits_index);
            sampler.accept(token);

            if loaded.model.is_eog_token(token) {
                debug!(n_decoded, "stopped on EOG token");
                return Ok(());
            }

            // `lstrip = None` matches the upstream `simple` example:
            // llama.cpp only inserts leading whitespace when the tokenizer
            // originally produced it.
            let piece = loaded
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| DomainError::LlmFailed(format!("detokenize: {e}")))?;
            n_decoded += 1;

            if sink.on_piece(&piece) == Flow::Stop {
                debug!(n_decoded, "stopped by the sink");
                return Ok(());
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| DomainError::LlmFailed(format!("batch.add gen: {e}")))?;
            n_cur += 1;

            self.ctx
                .decode(&mut batch)
                .map_err(|e| DomainError::LlmFailed(format!("decode step: {e}")))?;
            self.resident.push(token);
            logits_index = 0;
        }

        warn!(max_new, "hit max_tokens without an EOG/stop token");
        Ok(())
    }
}

/// Greedy when the caller asks for it (`temperature == 0`), otherwise a
/// top-p + temperature stack. `dist` is always last so it draws the final
/// token from the distribution the earlier samplers shaped.
fn build_sampler(params: &SamplingParams) -> LlamaSampler {
    let seed = params.seed.map_or(DEFAULT_SEED, |s| s as u32);
    let temp = params.temperature.clamp(0.0, 2.0);
    let top_p = params.top_p.clamp(0.0, 1.0);
    if temp <= f32::EPSILON {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }
    let mut chain = Vec::with_capacity(4);
    if params.penalize_repeats {
        // Penalise recently generated tokens to avoid degenerate
        // repetition loops (look back 64 tokens, repeat penalty 1.1, no
        // frequency/presence penalty).
        chain.push(LlamaSampler::penalties(64, 1.1, 0.0, 0.0));
    }
    chain.push(LlamaSampler::top_p(top_p, 1));
    chain.push(LlamaSampler::temp(temp));
    chain.push(LlamaSampler::dist(seed));
    LlamaSampler::chain_simple(chain)
}
