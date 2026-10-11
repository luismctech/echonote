//! llama.cpp implementation of the worker's [`Engine`]: one persistent
//! `LlamaContext` plus the single decode loop shared by every entry point
//! (one-shot generation, raw streaming and chat).

use std::num::NonZeroU32;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use tracing::{debug, warn};

use echo_domain::DomainError;

use crate::shared::LoadedModel;
use crate::worker::{Engine, EngineSource, Flow, PieceSink, SamplingParams};

/// Initial size of the decoding batch. Resized on demand when the prompt
/// is larger.
const INITIAL_BATCH_TOKENS: usize = 512;

/// Seed used when the caller does not pin one.
const DEFAULT_SEED: u32 = 1234;

/// [`EngineSource`] over a loaded model; builds the persistent context.
pub(crate) struct LlamaSource {
    loaded: Arc<LoadedModel>,
}

impl LlamaSource {
    pub(crate) fn new(loaded: Arc<LoadedModel>) -> Self {
        Self { loaded }
    }
}

impl EngineSource for LlamaSource {
    fn create(&self) -> Result<Box<dyn Engine + '_>, DomainError> {
        let loaded: &LoadedModel = &self.loaded;
        let n_ctx = NonZeroU32::new(loaded.n_ctx)
            .ok_or_else(|| DomainError::Invariant("n_ctx must be > 0".into()))?;

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
        Ok(Box::new(LlamaEngine { loaded, ctx }))
    }
}

/// A live context bound to the worker thread that created it.
struct LlamaEngine<'a> {
    loaded: &'a LoadedModel,
    ctx: LlamaContext<'a>,
}

impl Engine for LlamaEngine<'_> {
    fn serve(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn PieceSink,
    ) -> Result<(), DomainError> {
        let loaded = self.loaded;

        // The use-case layer wraps the prompt with the model's chat
        // template (including any BOS marker it needs), so no BOS is
        // added here: `AddBos::Never` avoids a duplicate.
        let prompt_tokens = loaded
            .model
            .str_to_token(prompt, AddBos::Never)
            .map_err(|e| DomainError::LlmFailed(format!("tokenize prompt: {e}")))?;

        let n_prompt = prompt_tokens.len() as i32;
        let n_max_total = n_prompt.saturating_add(params.max_tokens.min(i32::MAX as u32) as i32);
        if (n_max_total as u32) > loaded.n_ctx {
            return Err(DomainError::LlmFailed(format!(
                "prompt ({} tok) + max_tokens ({}) exceeds context window ({})",
                n_prompt, params.max_tokens, loaded.n_ctx
            )));
        }

        // Every request starts from an empty cache: the previous request's
        // tokens must not leak into this one.
        self.ctx.clear_kv_cache();

        let batch_capacity = prompt_tokens.len().max(INITIAL_BATCH_TOKENS);
        let mut batch = LlamaBatch::new(batch_capacity, 1);
        let last_index = (prompt_tokens.len() as i32) - 1;
        for (i, token) in (0_i32..).zip(prompt_tokens) {
            let is_last = i == last_index;
            batch
                .add(token, i, &[0], is_last)
                .map_err(|e| DomainError::LlmFailed(format!("batch.add prompt: {e}")))?;
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| DomainError::LlmFailed(format!("decode prompt: {e}")))?;

        let mut sampler = build_sampler(params);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
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

            let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);
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
