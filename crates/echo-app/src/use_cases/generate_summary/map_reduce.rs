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
//! Reduce input: `window - SUMMARY_MAX_TOKENS - safety margin - prompt
//! overhead`, where the overhead is the rendered prompt with an empty
//! transcript (template, language instruction, user notes). The safety
//! margin is `max(256, 5% of the window)`: it absorbs what the budget
//! cannot measure exactly (chat-template tokens added at tokenization
//! time, joining newlines, the parser feedback of a retry prompt).
//!
//! Map call: with `avail = window - safety margin - map prompt overhead`,
//! the note may use `map_max_tokens = min(avail / 2, MAP_OUTPUT_CEILING)`
//! and the chunk gets `min(avail - map_max_tokens, map_max_tokens)`.
//! Chunk and note budgets are equal because small models write notes
//! that are close to verbatim; this way a note that is as long as its
//! chunk still fits its own output budget, and `prompt + chunk + note`
//! stays within the window. For an 8,192-token window that is about
//! 3,072 tokens each. The port exposes no finish reason, so a note whose
//! length reaches `map_max_tokens` minus [`CAP_DETECTION_MARGIN_TOKENS`]
//! is taken as cut off; its chunk is then split in two and mapped again
//! (at most [`MAX_SPLIT_DEPTH`] times), never kept silently.

use echo_domain::{GenerateOptions, LlmModel};
use tracing::warn;

use super::{generate_opts, SummarizeMeetingError, SUMMARY_MAX_TOKENS};
use crate::sanitize::strip_chat_tokens;
use crate::transcript::{chunk_lines, safety_margin};

/// Smallest chunk or reduce budget worth running a map-reduce with.
/// Below it the model would receive scraps of context per call, so the
/// request fails with [`SummarizeMeetingError::ContextTooSmall`] instead.
const MIN_BUDGET_TOKENS: usize = 256;

/// Upper bound of one map note, in tokens. Longer notes would only make
/// every map call slower; chunks shrink with it (see the module docs).
const MAP_OUTPUT_CEILING_TOKENS: usize = 3_072;

/// A map note within this many tokens of its output cap is assumed to
/// have been cut off by the cap (the port reports no finish reason).
const CAP_DETECTION_MARGIN_TOKENS: usize = 8;

/// How many times a truncated map chunk is halved and mapped again. At
/// depth 3 a chunk is an eighth of the original, so a model that keeps
/// filling its output budget costs at most 15 calls per chunk.
const MAX_SPLIT_DEPTH: usize = 3;

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
    map_max_tokens: usize,
    reduce_budget: usize,
}

/// Decide between a single pass and map-reduce for `lines`.
///
/// `empty_prompt` is the final template prompt rendered with an empty
/// transcript; its size is the fixed overhead of the reduce call.
///
/// # Errors
///
/// [`SummarizeMeetingError::ContextTooSmall`] when the window leaves
/// less than [`MIN_BUDGET_TOKENS`] for chunks or for the reduce input.
/// A segment larger than a chunk is split by [`chunk_lines`], not rejected.
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
    let map_avail = window
        .saturating_sub(safety_margin(window))
        .saturating_sub(llm.count_tokens(&map_prompt(language_instruction, "", 1, 1)));
    let map_max_tokens = (map_avail / 2).min(MAP_OUTPUT_CEILING_TOKENS);
    let chunk_budget = map_avail.saturating_sub(map_max_tokens).min(map_max_tokens);
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
        map_max_tokens,
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
    let mut notes = map_all(llm, &plan.chunks, language_instruction, plan.map_max_tokens).await?;

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
        notes = map_all(llm, &groups, language_instruction, plan.map_max_tokens).await?;
    }

    Err(SummarizeMeetingError::ContextTooSmall(format!(
        "partial notes still exceed {} tokens after {MAX_COLLAPSE_PASSES} condensing passes",
        plan.reduce_budget
    )))
}

/// `chunk_lines` splits oversized lines, so a chunk over budget means the
/// budget cannot hold even one character after the speaker label (or the
/// token counter is not monotonic); sending it would overflow the
/// window, so refuse instead.
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

/// Generation options of a map call: the summary defaults with the
/// map-specific output budget.
fn map_generate_opts(map_max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens: u32::try_from(map_max_tokens).unwrap_or(u32::MAX),
        ..generate_opts()
    }
}

/// Map every chunk into a note, in order. A note that reached its output
/// cap is not kept: its chunk is halved and mapped again, up to
/// [`MAX_SPLIT_DEPTH`] times. A note still capped at that depth is kept
/// (nothing better is available) and reported with a `warn!`.
async fn map_all(
    llm: &dyn LlmModel,
    chunks: &[String],
    language_instruction: &str,
    map_max_tokens: usize,
) -> Result<Vec<String>, SummarizeMeetingError> {
    let total = chunks.len();
    let opts = map_generate_opts(map_max_tokens);
    let cap = map_max_tokens.saturating_sub(CAP_DETECTION_MARGIN_TOKENS);
    let count = |text: &str| llm.count_tokens(text);
    let mut notes = Vec::with_capacity(total);
    for (i, chunk) in chunks.iter().enumerate() {
        // LIFO stack; pieces are pushed reversed so notes stay in order.
        let mut pending = vec![(chunk.clone(), 0usize)];
        while let Some((text, depth)) = pending.pop() {
            let prompt = map_prompt(language_instruction, &text, i + 1, total);
            let response = llm
                .generate(&prompt, &opts)
                .await
                .map_err(SummarizeMeetingError::Llm)?;
            // Measured raw: tokens spent on stripped content still count
            // against the output cap.
            let note = strip_chat_tokens(&response).trim().to_string();
            if count(&response) < cap {
                notes.push(note);
                continue;
            }
            match (depth < MAX_SPLIT_DEPTH)
                .then(|| split_in_two(&text, &count))
                .filter(|pieces| pieces.len() > 1)
            {
                Some(pieces) => {
                    pending.extend(pieces.into_iter().rev().map(|piece| (piece, depth + 1)));
                }
                None => {
                    warn!(
                        chunk = i + 1,
                        depth, "map note reached its output cap and cannot be split further"
                    );
                    notes.push(note);
                }
            }
        }
    }
    Ok(notes)
}

/// Split a chunk's lines into about two halves. The budget is half the
/// chunk plus one token per line to cover the joining newlines that
/// `chunk_lines` charges, so the result is two pieces, rarely three.
fn split_in_two(chunk: &str, count: &dyn Fn(&str) -> usize) -> Vec<String> {
    let lines: Vec<String> = chunk.split('\n').map(str::to_owned).collect();
    let budget = count(chunk).div_ceil(2) + lines.len();
    chunk_lines(&lines, budget, count)
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
         names, and who said what. Condense: paraphrase in short bullet points instead of \
         copying sentences, so the notes are roughly a quarter of the length of the part, \
         but never drop a figure, amount, date or name. Do not invent anything, do not add an introduction or a \
         conclusion, and do not use JSON or markdown fences.<|im_end|>\n\
         <|im_start|>user\n\
         Part {position} of {total}:\n---\n{safe_excerpt}\n---\n/no_think<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use echo_domain::DomainError;

    use super::*;

    /// LLM that answers every call with `response` and counts the calls.
    struct FixedLlm {
        window: u32,
        response: String,
        calls: Mutex<usize>,
    }

    #[async_trait]
    impl LlmModel for FixedLlm {
        fn model_id(&self) -> &str {
            "fixed"
        }
        fn context_window(&self) -> u32 {
            self.window
        }
        async fn generate(&self, _: &str, _: &GenerateOptions) -> Result<String, DomainError> {
            *self.calls.lock().unwrap() += 1;
            Ok(self.response.clone())
        }
    }

    fn fixed(window: u32, response: String) -> FixedLlm {
        FixedLlm {
            window,
            response,
            calls: Mutex::new(0),
        }
    }

    fn lines(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("L{i:03} {}", "palabra ".repeat(30)))
            .collect()
    }

    #[test]
    fn chunk_and_note_budgets_fit_the_window_for_any_window() {
        for window in [2_600u32, 4_096, 8_192, 32_768] {
            let llm = fixed(window, String::new());
            let plan = match plan(&llm, &lines(2_000), "Always respond in English.", "x").unwrap() {
                Input::Condense(plan) => plan,
                Input::Whole(_) => panic!("{window}: expected map-reduce"),
            };
            let overhead = llm.count_tokens(&map_prompt("Always respond in English.", "", 1, 1));
            let used =
                overhead + plan.chunk_budget + plan.map_max_tokens + safety_margin(window as usize);
            assert!(used <= window as usize, "{window}: map call needs {used}");
            assert!(plan.chunk_budget <= plan.map_max_tokens, "{window}");
            assert!(plan.map_max_tokens <= MAP_OUTPUT_CEILING_TOKENS, "{window}");
        }
    }

    #[test]
    fn the_map_output_budget_grows_with_the_window_up_to_the_ceiling() {
        let budget = |window| {
            let llm = fixed(window, String::new());
            match plan(&llm, &lines(2_000), "x", "x").unwrap() {
                Input::Condense(plan) => plan.map_max_tokens,
                Input::Whole(_) => unreachable!(),
            }
        };
        assert!(budget(4_096) < budget(8_192));
        assert_eq!(budget(8_192), MAP_OUTPUT_CEILING_TOKENS);
        assert_eq!(budget(32_768), MAP_OUTPUT_CEILING_TOKENS);
    }

    #[tokio::test]
    async fn a_model_that_always_hits_the_cap_stops_after_the_depth_bound() {
        // 16 lines, every answer cap-sized: 1 + 2 + 4 + 8 calls, then the
        // 8 deepest notes are kept with a warning.
        let llm = fixed(8_192, "nota ".repeat(4_000));
        let chunk = lines(16).join("\n");

        let notes = map_all(&llm, &[chunk], "x", 3_072).await.unwrap();

        assert_eq!(*llm.calls.lock().unwrap(), 15);
        assert_eq!(notes.len(), 8);
    }

    #[tokio::test]
    async fn cap_detection_measures_the_raw_response_not_the_stripped_note() {
        // The budget was spent on control tokens that are stripped from
        // the note: the response is capped even though the note is short.
        let llm = fixed(8_192, format!("{}ok", "<|im_end|>".repeat(1_000)));
        let chunk = lines(4).join("\n");

        map_all(&llm, &[chunk], "x", 3_072).await.unwrap();

        assert!(*llm.calls.lock().unwrap() > 1, "the capped chunk was kept");
    }

    #[tokio::test]
    async fn map_calls_use_the_map_output_cap() {
        struct Recorder(Mutex<Vec<u32>>);
        #[async_trait]
        impl LlmModel for Recorder {
            fn model_id(&self) -> &str {
                "rec"
            }
            async fn generate(&self, _: &str, o: &GenerateOptions) -> Result<String, DomainError> {
                self.0.lock().unwrap().push(o.max_tokens);
                Ok("short".into())
            }
        }
        let llm = Recorder(Mutex::new(Vec::new()));

        map_all(&llm, &["a".to_string(), "b".to_string()], "x", 1_500)
            .await
            .unwrap();

        assert_eq!(*llm.0.lock().unwrap(), vec![1_500, 1_500]);
    }
}
