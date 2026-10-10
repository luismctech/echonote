//! Token budgeting and map-reduce condensing for the summary use case.
//!
//! The summary never truncates the transcript. When the whole
//! transcript plus the template prompt fits the model's context
//! window it is sent as is ([`Input::Whole`]). Otherwise the
//! transcript is split on segment boundaries, every chunk is turned
//! into faithful plain-text notes (map), and the ordered notes replace
//! the transcript in the template prompt (reduce). If the notes alone
//! still overflow the reduce budget they are condensed again in groups,
//! for a bounded number of passes; running out of passes or of
//! progress is a typed error, never a silent cut.
//!
//! ## Budget
//!
//! `window - SUMMARY_MAX_TOKENS - safety margin - prompt overhead`,
//! where the overhead is the rendered prompt with an empty transcript
//! (template, language instruction, user notes). The safety margin is
//! `max(256, 5% of the window)`: it absorbs what the budget cannot
//! measure exactly (chat-template tokens added at tokenization time,
//! joining newlines, the parser feedback of a retry prompt).

use echo_domain::LlmModel;

use super::{generate_opts, SummarizeMeetingError, SUMMARY_MAX_TOKENS};
use crate::sanitize::strip_chat_tokens;
use crate::transcript::chunk_lines;

/// Smallest chunk or reduce budget worth running a map-reduce with.
/// Below it the model would receive scraps of context per call, so the
/// request fails with [`SummarizeMeetingError::ContextTooSmall`] instead.
const MIN_BUDGET_TOKENS: usize = 256;

/// Floor of the safety margin, in tokens. See the module docs.
const MIN_SAFETY_MARGIN_TOKENS: usize = 256;

/// Maximum number of times the notes are condensed again when they do
/// not fit the reduce budget. Each pass costs LLM calls, so the loop is
/// bounded even though every pass must also shrink the note count.
const MAX_COLLAPSE_PASSES: usize = 4;

/// Sentence prepended to the reduce input so the model treats the
/// notes as the whole meeting rather than as a transcript fragment.
const REDUCE_INTRO: &str = "The transcript below is not verbatim: it is a sequence of notes, \
     in chronological order, that together cover the whole meeting. \
     Treat the notes as the full transcript.";

/// What the template prompt receives as its transcript.
#[derive(Debug)]
pub(super) enum Input {
    /// The full transcript fits the budget; use it unchanged.
    Whole(String),
    /// The transcript must be condensed first; see [`condense`].
    Condense(Condense),
}

impl Input {
    /// The text to use as the transcript of the final template prompt:
    /// the transcript itself, or the result of [`condense`].
    ///
    /// # Errors
    ///
    /// Same as [`condense`].
    pub(super) async fn resolve(
        self,
        llm: &dyn LlmModel,
        language_instruction: &str,
    ) -> Result<String, SummarizeMeetingError> {
        match self {
            Self::Whole(transcript) => Ok(transcript),
            Self::Condense(plan) => condense(llm, plan, language_instruction).await,
        }
    }
}

/// Pre-computed map phase: the chunks to map and the budgets that bound
/// the later passes.
#[derive(Debug)]
pub(super) struct Condense {
    chunks: Vec<String>,
    chunk_budget: usize,
    reduce_budget: usize,
}

fn safety_margin(window: usize) -> usize {
    MIN_SAFETY_MARGIN_TOKENS.max(window / 20)
}

/// Decide between a single pass and map-reduce for `lines`.
///
/// `empty_prompt` is the final template prompt rendered with an empty
/// transcript; its size is the fixed overhead of the reduce call.
///
/// # Errors
///
/// [`SummarizeMeetingError::ContextTooSmall`] when the window leaves
/// less than [`MIN_BUDGET_TOKENS`] for chunks or for the reduce input,
/// or when a single segment alone exceeds the chunk budget.
pub(super) fn plan(
    llm: &dyn LlmModel,
    lines: &[String],
    language_instruction: &str,
    empty_prompt: &str,
) -> Result<Input, SummarizeMeetingError> {
    let window = llm.context_window() as usize;
    let base = window.saturating_sub(SUMMARY_MAX_TOKENS as usize + safety_margin(window));
    let single_budget = base.saturating_sub(llm.count_tokens(empty_prompt));

    let whole = lines.join("\n");
    if llm.count_tokens(&whole) <= single_budget {
        return Ok(Input::Whole(whole));
    }

    let reduce_budget = single_budget.saturating_sub(llm.count_tokens(REDUCE_INTRO) + 2);
    let chunk_budget =
        base.saturating_sub(llm.count_tokens(&map_prompt(language_instruction, "", 1, 1)));
    if reduce_budget < MIN_BUDGET_TOKENS || chunk_budget < MIN_BUDGET_TOKENS {
        return Err(SummarizeMeetingError::ContextTooSmall(format!(
            "a {window}-token context window leaves {chunk_budget} tokens per chunk and \
             {reduce_budget} for the final prompt; at least {MIN_BUDGET_TOKENS} are required"
        )));
    }

    let count = |text: &str| llm.count_tokens(text);
    let chunks = chunk_lines(lines, chunk_budget, &count);
    ensure_chunks_fit(&chunks, chunk_budget, &count)?;
    Ok(Input::Condense(Condense {
        chunks,
        chunk_budget,
        reduce_budget,
    }))
}

/// Run the map phase and return the text to use as the transcript of
/// the final template prompt (the reduce input).
///
/// Calls are sequential and non-streamed: a local model serves one
/// generation at a time, so parallelism would only add contention.
///
/// # Errors
///
/// [`SummarizeMeetingError::Llm`] when a map call fails;
/// [`SummarizeMeetingError::ContextTooSmall`] when condensing makes no
/// progress or exhausts [`MAX_COLLAPSE_PASSES`].
pub(super) async fn condense(
    llm: &dyn LlmModel,
    plan: Condense,
    language_instruction: &str,
) -> Result<String, SummarizeMeetingError> {
    let count = |text: &str| llm.count_tokens(text);
    let mut notes = map_all(llm, &plan.chunks, language_instruction).await?;

    for pass in 0..=MAX_COLLAPSE_PASSES {
        let labelled = label_notes(&notes);
        let rendered = labelled.join("\n\n");
        if count(&rendered) <= plan.reduce_budget {
            return Ok(format!("{REDUCE_INTRO}\n\n{rendered}"));
        }
        if pass == MAX_COLLAPSE_PASSES {
            break;
        }

        let groups = chunk_lines(&labelled, plan.chunk_budget, &count);
        if groups.len() >= notes.len() {
            return Err(SummarizeMeetingError::ContextTooSmall(format!(
                "{} partial notes do not fit {} tokens and cannot be grouped any further",
                notes.len(),
                plan.reduce_budget
            )));
        }
        ensure_chunks_fit(&groups, plan.chunk_budget, &count)?;
        notes = map_all(llm, &groups, language_instruction).await?;
    }

    Err(SummarizeMeetingError::ContextTooSmall(format!(
        "partial notes still exceed {} tokens after {MAX_COLLAPSE_PASSES} condensing passes",
        plan.reduce_budget
    )))
}

/// `chunk_lines` emits a line that alone exceeds the budget as its own
/// chunk; sending it would overflow the window, so refuse instead.
fn ensure_chunks_fit(
    chunks: &[String],
    budget: usize,
    count: &dyn Fn(&str) -> usize,
) -> Result<(), SummarizeMeetingError> {
    match chunks.iter().map(|c| count(c)).find(|&n| n > budget) {
        Some(tokens) => Err(SummarizeMeetingError::ContextTooSmall(format!(
            "one unit of {tokens} tokens exceeds the {budget}-token chunk budget"
        ))),
        None => Ok(()),
    }
}

async fn map_all(
    llm: &dyn LlmModel,
    chunks: &[String],
    language_instruction: &str,
) -> Result<Vec<String>, SummarizeMeetingError> {
    let total = chunks.len();
    let mut notes = Vec::with_capacity(total);
    for (i, chunk) in chunks.iter().enumerate() {
        let prompt = map_prompt(language_instruction, chunk, i + 1, total);
        let response = llm
            .generate(&prompt, &generate_opts())
            .await
            .map_err(SummarizeMeetingError::Llm)?;
        notes.push(strip_chat_tokens(&response).trim().to_string());
    }
    Ok(notes)
}

fn label_notes(notes: &[String]) -> Vec<String> {
    let total = notes.len();
    notes
        .iter()
        .enumerate()
        .map(|(i, note)| format!("Part {}/{total}:\n{note}", i + 1))
        .collect()
}

/// Prompt of one map call. `excerpt` is a transcript chunk, or the
/// labelled notes of a previous pass when condensing again.
fn map_prompt(language_instruction: &str, excerpt: &str, position: usize, total: usize) -> String {
    let safe_excerpt = strip_chat_tokens(excerpt);
    format!(
        "<|im_start|>system\n\
         You are a meticulous meeting note-taker. {language_instruction}\n\
         You will receive one part of a long meeting, either as a transcript excerpt or as \
         notes already taken from one. Write faithful plain-text notes of that part only: \
         topics discussed, decisions, action items with their owners, numbers, dates and \
         names, and who said what. Do not invent anything, do not add an introduction or a \
         conclusion, and do not use JSON or markdown fences.<|im_end|>\n\
         <|im_start|>user\n\
         Part {position} of {total}:\n---\n{safe_excerpt}\n---\n/no_think<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}
