//! Latency and memory bench for the llama.cpp adapter, gated behind a real
//! GGUF model on disk.
//!
//! It records evidence (printed with `--nocapture`), not pass/fail
//! thresholds: the only assertions are sanity checks on the output.
//!
//! ```sh
//! ECHO_LLM_MODEL="$HOME/Library/Application Support/app.echonote.desktop/models/llm/Qwen3-8B-Q4_K_M.gguf" \
//!   cargo test -p echo-llm --test llama_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! What is measured, with the default [`LoadOptions`] (8,192-token window):
//!
//! * **context cost** - a one-token completion of a tiny prompt, run twice.
//!   The prompt evaluation is negligible, so the first call shows model
//!   warm-up plus context creation and the second shows what a request
//!   costs once any reusable state exists.
//! * **prompt-eval throughput** - a ~6,000-token ChatML prompt with
//!   `max_tokens = 1`, so the time is dominated by evaluating the prompt.
//! * **chat turn** - the same long prompt with a 64-token greedy answer,
//!   twice in a row: time to first token and total time.
//! * **RSS** - resident set size of this process (via `ps`) before load,
//!   after load, and after the first and later requests.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use echo_domain::{
    ChatAssistant, ChatMessage, ChatOptions, ChatRequest, GenerateOptions, LlmModel,
};
use echo_llm::{LlamaCppLlm, LoadOptions};
use futures::StreamExt;

const TARGET_PROMPT_TOKENS: usize = 6_000;

fn model_from_env() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("ECHO_LLM_MODEL").ok()?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("ECHO_LLM_MODEL points to {path:?} but it does not exist");
        None
    }
}

/// Resident set size of this process in MiB, or `None` when `ps` is
/// unavailable or its output is not a number.
fn rss_mib() -> Option<f64> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kib / 1024.0)
}

fn report_rss(label: &str) {
    match rss_mib() {
        Some(mib) => eprintln!("[bench] rss {label}: {mib:.0} MiB"),
        None => eprintln!("[bench] rss {label}: unavailable"),
    }
}

/// Synthetic Spanish meeting transcript of at least `target` tokens under
/// the model's own tokenizer.
fn synthetic_transcript(llm: &LlamaCppLlm, target: usize) -> String {
    const LINES: [&str; 6] = [
        "Ana: Revisé la planificación del trimestre y creo que deberíamos reorganizar las prioridades según la capacidad del equipo.",
        "Luis: De acuerdo, pero antes necesitamos confirmar el presupuesto de infraestructura y la aprobación de la migración a producción.",
        "Marta: El cliente pidió lanzar la versión nueva el lunes por la mañana; todavía falta cerrar las pruebas de regresión.",
        "Pedro: Propongo dividir el trabajo en dos entregas y asignar a cada responsable una fecha límite concreta para la revisión.",
        "Ana: Perfecto. Quedamos en que Luis envía el informe de costes el jueves y Marta coordina la ventana de despliegue.",
        "Luis: Anotado. También conviene documentar los riesgos del proveedor externo y definir un plan de reversión antes del lanzamiento.",
    ];
    let mut text = String::new();
    let mut i = 0_usize;
    while llm.count_tokens(&text) < target {
        text.push_str(&format!("[{:02}:{:02}] {}\n", i / 60, i % 60, LINES[i % 6]));
        i += 1;
    }
    text
}

fn chatml(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn greedy(max_tokens: u32) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(7),
        stop: vec!["<|im_end|>".into()],
    }
}

#[tokio::test]
#[ignore = "requires ECHO_LLM_MODEL to point at a .gguf file"]
async fn llama_latency_and_memory_baseline() {
    let Some(model_path) = model_from_env() else {
        eprintln!("skipping: ECHO_LLM_MODEL not set");
        return;
    };

    report_rss("before load");
    let started = Instant::now();
    let llm = LlamaCppLlm::load_with(&model_path, LoadOptions::default().with_model_id("bench"))
        .expect("load gguf");
    eprintln!("[bench] model load: {} ms", started.elapsed().as_millis());
    report_rss("after load");
    eprintln!("[bench] context window: {}", llm.context_window());

    // (a) context cost: tiny prompt, one token, twice.
    let tiny = chatml("Answer with one word.", "Say OK. /no_think");
    for call in 1..=2 {
        let started = Instant::now();
        let out = llm.generate(&tiny, &greedy(1)).await.expect("tiny");
        eprintln!(
            "[bench] tiny request #{call} (1 token, tiny prompt): {} ms (output {out:?})",
            started.elapsed().as_millis()
        );
        report_rss(&format!("after tiny request #{call}"));
    }

    // (b) prompt-eval throughput on a ~6k-token prompt.
    let transcript = synthetic_transcript(&llm, TARGET_PROMPT_TOKENS);
    let system = format!("You answer questions about this meeting transcript.\n\n{transcript}");
    let question = "¿Qué se acordó sobre el informe de costes? Responde en una frase. /no_think";
    let prompt = chatml(&system, question);
    let n_prompt = llm.count_tokens(&prompt);
    let started = Instant::now();
    let _ = llm
        .generate(&prompt, &greedy(1))
        .await
        .expect("long prompt");
    let eval = started.elapsed();
    eprintln!(
        "[bench] prompt eval: {n_prompt} tokens in {} ms = {:.1} tok/s (includes 1 generated token)",
        eval.as_millis(),
        n_prompt as f64 / eval.as_secs_f64()
    );
    report_rss("after long-prompt request");

    // (c) chat turn: time to first token and total, twice on the same prompt.
    let chat = llm.chat_handle();
    for turn in 1..=2 {
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system(system.clone()),
                ChatMessage::user(question),
            ],
            options: ChatOptions {
                max_tokens: 64,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(7),
                stop: vec!["<|im_end|>".into()],
            },
        };
        let started = Instant::now();
        let mut stream = chat.ask(&request).await.expect("ask");
        let mut first: Option<Duration> = None;
        let mut answer = String::new();
        while let Some(item) = stream.next().await {
            let token = item.expect("stream item");
            first.get_or_insert_with(|| started.elapsed());
            answer.push_str(&token.delta);
        }
        let total = started.elapsed();
        eprintln!(
            "[bench] chat turn #{turn}: first token {} ms, total {} ms, {} chars: {:?}",
            first.map_or(0, |d| d.as_millis()),
            total.as_millis(),
            answer.len(),
            answer.chars().take(80).collect::<String>()
        );
        assert!(!answer.trim().is_empty(), "chat turn #{turn} was empty");
    }
    report_rss("after chat turns");
}
