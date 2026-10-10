//! Real-model acceptance tests for the summary use case.
//!
//! Gated behind a GGUF model on disk (like `echo-llm/tests/llama_smoke.rs`):
//! the tests are `#[ignore]`d and also skip cleanly when `ECHO_LLM_MODEL`
//! is unset. Locally:
//!
//! ```sh
//! ECHO_LLM_MODEL=$(pwd)/models/llm/Qwen3-8B-Q4_K_M.gguf \
//!   cargo test -p echo-app --test summary_real_model -- --ignored --nocapture --test-threads=1
//! ```
//!
//! They prove what the scripted unit tests cannot: that map-reduce over a
//! long meeting fits the production 8,192-token window with the model's own
//! tokenizer, yields the structured General template, and keeps a fact that
//! sits in the middle of the transcript.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::BoxStream;

use echo_app::SummarizeMeeting;
use echo_domain::{
    AudioFormat, CreateMeeting, DomainError, GenerateOptions, LlmModel, MeetingId, MeetingStore,
    Segment, SegmentId, Speaker, SummaryContent,
};
use echo_llm::LlamaCppLlm;
use echo_storage::SqliteMeetingStore;

/// Number the planted fact must surface as in the summary.
const PLANTED_NUMBER: &str = "437";
const PLANTED_SENTENCE: &str =
    "Quiero dejar claro que el presupuesto final del proyecto Atlas es de 437 mil euros.";

/// Decorator counting how many completions the use case asks the model for.
struct CountingLlm {
    inner: Arc<LlamaCppLlm>,
    calls: AtomicUsize,
}

#[async_trait]
impl LlmModel for CountingLlm {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn context_window(&self) -> u32 {
        self.inner.context_window()
    }
    fn count_tokens(&self, text: &str) -> usize {
        self.inner.count_tokens(text)
    }
    async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, DomainError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate(prompt, options).await
    }
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<BoxStream<'static, Result<String, DomainError>>, DomainError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate_stream(prompt, options).await
    }
}

fn model_from_env() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("ECHO_LLM_MODEL").ok()?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("ECHO_LLM_MODEL points to {path:?} but it does not exist");
        None
    }
}

/// Deterministic pseudo-random index so the dialogue is varied but stable.
fn pick(seed: &mut u64, len: usize) -> usize {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as usize) % len
}

/// Builds `count` varied Spanish meeting utterances rotating over four
/// speakers; the planted fact replaces the one at `plant_at`.
fn synthetic_dialogue(count: usize, plant_at: Option<usize>) -> Vec<String> {
    const TOPICS: [&str; 10] = [
        "la migración de la base de datos",
        "el módulo de facturación",
        "la campaña de captación de clientes",
        "el plan de formación del equipo",
        "la integración con el proveedor de pagos",
        "el rediseño de la aplicación móvil",
        "la auditoría de seguridad",
        "el calendario de lanzamientos",
        "la atención al cliente en horario nocturno",
        "el informe trimestral para la dirección",
    ];
    const OPENERS: [&str; 8] = [
        "Sobre {t}, creo que deberíamos revisar los plazos antes de comprometernos.",
        "Respecto a {t}, el equipo ha avanzado más de lo previsto esta semana.",
        "Me preocupa {t} porque todavía hay dependencias sin resolver.",
        "Hemos hablado con el cliente acerca de {t} y quiere ver una demostración pronto.",
        "En cuanto a {t}, propongo que Marta prepare un borrador para el jueves.",
        "Tengo dudas sobre {t}; necesitamos más datos antes de decidir nada.",
        "Para {t} haría falta reforzar el equipo con una persona más durante un mes.",
        "Veo bien {t}, aunque convendría documentar mejor los riesgos asociados.",
    ];
    const FOLLOW_UPS: [&str; 8] = [
        "Estoy de acuerdo, pero habría que validarlo con el área legal.",
        "Entonces lo dejamos anotado y lo retomamos en la próxima reunión.",
        "Puedo encargarme yo de hacer el seguimiento durante las dos próximas semanas.",
        "No me parece realista con los recursos que tenemos ahora mismo.",
        "Perfecto, así evitamos retrasos con el resto de entregables.",
        "Deberíamos medir el impacto antes de ampliar el alcance.",
        "Si nadie se opone, avanzamos con esa propuesta y revisamos el viernes.",
        "Eso nos obligaría a mover la fecha de entrega, ¿lo ven razonable?",
    ];

    let mut seed = 0x5EED_u64;
    (0..count)
        .map(|i| {
            if plant_at == Some(i) {
                return PLANTED_SENTENCE.to_string();
            }
            let topic = TOPICS[pick(&mut seed, TOPICS.len())];
            let opener = OPENERS[pick(&mut seed, OPENERS.len())].replace("{t}", topic);
            let follow = FOLLOW_UPS[pick(&mut seed, FOLLOW_UPS.len())];
            // Alternate single sentences and two-sentence turns.
            if i % 3 == 0 {
                opener
            } else {
                format!("{opener} {follow}")
            }
        })
        .collect()
}

/// Persists `utterances` as a meeting and returns its id plus the store.
async fn seed_meeting(utterances: &[String]) -> (MeetingId, Arc<SqliteMeetingStore>) {
    let store = Arc::new(SqliteMeetingStore::open_in_memory().await.unwrap());
    let id = MeetingId::new();
    store
        .create(CreateMeeting {
            id,
            title: "Reunión de seguimiento".into(),
            input_format: AudioFormat::WHISPER,
        })
        .await
        .unwrap();

    let speakers: Vec<Speaker> = ["Ana", "Luis", "Marta", "Pablo"]
        .iter()
        .enumerate()
        .map(|(slot, name)| Speaker::anonymous(slot as u32).renamed(*name))
        .collect();
    for speaker in &speakers {
        store.upsert_speaker(id, speaker).await.unwrap();
    }

    let mut clock_ms = 0_u32;
    let segments: Vec<Segment> = utterances
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let start_ms = clock_ms;
            clock_ms += 4_000 + (text.len() as u32) * 60;
            Segment {
                id: SegmentId::new(),
                start_ms,
                end_ms: clock_ms,
                text: text.clone(),
                speaker_id: Some(speakers[i % speakers.len()].id),
                confidence: None,
            }
        })
        .collect();
    store.append_segments(id, &segments).await.unwrap();
    (id, store)
}

/// Runs the use case and returns `(summary text blob, calls, seconds)`,
/// asserting the structured General template came back.
async fn run_summary(utterances: &[String]) -> (String, usize, f64) {
    let model_path = model_from_env().expect("guarded by caller");
    let llm = Arc::new(CountingLlm {
        inner: Arc::new(LlamaCppLlm::load(&model_path).expect("load gguf")),
        calls: AtomicUsize::new(0),
    });
    let (id, store) = seed_meeting(utterances).await;
    let chars: usize = utterances.iter().map(String::len).sum();
    eprintln!(
        "transcript: {} utterances, {chars} chars, window {} tokens",
        utterances.len(),
        llm.context_window()
    );

    let usecase = SummarizeMeeting::new(llm.clone(), store);
    let started = Instant::now();
    let summary = usecase
        .execute(id, "general", false, Some("es"))
        .await
        .expect("summary must succeed");
    let elapsed = started.elapsed().as_secs_f64();
    let calls = llm.calls.load(Ordering::SeqCst);

    let json = serde_json::to_string_pretty(&summary.content).unwrap();
    eprintln!("elapsed: {elapsed:.1}s, LLM calls: {calls}\n{json}");

    match &summary.content {
        SummaryContent::FreeText { text } => {
            panic!("expected structured General summary, got free text:\n{text}")
        }
        SummaryContent::General { .. } => {}
        other => panic!("expected General template, got {other:?}"),
    }
    (json, calls, elapsed)
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn long_meeting_is_map_reduced_and_keeps_a_mid_transcript_fact() {
    if model_from_env().is_none() {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    }
    // ~40k chars; the fact sits in the middle third.
    let count = 330;
    let utterances = synthetic_dialogue(count, Some(count / 2));
    let chars: usize = utterances.iter().map(String::len).sum();
    assert!(chars >= 35_000, "fixture too short: {chars} chars");

    let (json, calls, _) = run_summary(&utterances).await;

    // >= 2 map calls + 1 reduce/final call.
    assert!(calls >= 3, "expected map-reduce (>= 3 calls), got {calls}");
    assert!(
        json.contains(PLANTED_NUMBER),
        "planted fact ({PLANTED_NUMBER}) missing from summary:\n{json}"
    );
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn short_meeting_is_summarised_in_a_single_pass() {
    if model_from_env().is_none() {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    }
    let utterances = synthetic_dialogue(14, Some(7));
    let (_, calls, _) = run_summary(&utterances).await;
    assert_eq!(calls, 1, "a meeting that fits must take exactly one pass");
}
