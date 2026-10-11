//! Real-model acceptance test for the chat use case on a long meeting.
//!
//! Gated like `summary_real_model.rs`: `#[ignore]`d, and it also skips
//! cleanly when `ECHO_LLM_MODEL` is unset. Locally:
//!
//! ```sh
//! ECHO_LLM_MODEL="$HOME/Library/Application Support/app.echonote.desktop/models/llm/Qwen3-8B-Q4_K_M.gguf" \
//!   cargo test -p echo-app --test chat_real_model -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The transcript is far larger than the 8,192-token window, so the answer
//! about a fact planted in its middle can only be right if the chat
//! retrieved the right segment and cited its real id.

use std::time::Instant;

use futures::StreamExt;

use echo_app::{AskAboutMeeting, AskAboutMeetingEvent};
use echo_domain::{MeetingStore, SegmentRetriever};
use echo_llm::LlamaCppLlm;

mod common;
use common::{model_from_env, seed_meeting, synthetic_dialogue, PLANTED_NUMBER, PLANTED_SENTENCE};

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn long_meeting_chat_answers_from_a_fact_in_the_middle_with_a_valid_citation() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };
    let count = 330;
    let utterances = synthetic_dialogue(count, Some(count / 2));
    let (id, store) = seed_meeting(&utterances).await;
    let planted = store
        .get(id)
        .await
        .unwrap()
        .unwrap()
        .segments
        .into_iter()
        .find(|s| s.text == PLANTED_SENTENCE)
        .expect("planted segment")
        .id;

    let llm = LlamaCppLlm::load(&model_path).expect("load gguf");
    let retriever: std::sync::Arc<dyn SegmentRetriever> = store.clone();
    let usecase = AskAboutMeeting::new(std::sync::Arc::new(llm.chat_handle()), store, retriever);

    let started = Instant::now();
    let events: Vec<AskAboutMeetingEvent> = usecase
        .execute(
            id,
            vec![],
            "¿Cuál es el presupuesto final del proyecto Atlas?".into(),
        )
        .await
        .expect("chat must start")
        .collect()
        .await;
    eprintln!("elapsed: {:.1}s", started.elapsed().as_secs_f64());

    let Some(AskAboutMeetingEvent::Finished {
        text,
        citations,
        had_citations,
    }) = events.last()
    else {
        panic!("chat did not finish: {:?}", events.last());
    };
    eprintln!("answer: {text}\ncitations: {citations:?}");
    assert!(text.contains(PLANTED_NUMBER), "fact missing from: {text}");
    // The fixture meeting has no language tag: the answer must follow the
    // Spanish question instead of defaulting to English.
    assert!(
        text.to_lowercase().contains("presupuesto"),
        "answer is not in the question's language: {text}"
    );
    assert!(*had_citations, "answer carries no valid citation: {text}");
    assert!(
        citations.contains(&planted),
        "citations {citations:?} do not include the planted segment {planted:?}"
    );
}
