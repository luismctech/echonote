//! # echo-proto
//!
//! CLI prototype for EchoNote's Phase 0 (Discovery). The binary stitches
//! the capture, ASR and LLM crates together to prove the end-to-end
//! pipeline on macOS before any UI work begins.
//!
//! Subcommands (incrementally wired during Sprints 0 and 1):
//!
//! - `record --duration N` — capture N seconds of dual audio to WAV.
//!   (Wired Sprint 0 day 5.)
//! - `transcribe FILE` — run Whisper on a WAV and print segments.
//!   (Wired Sprint 0 day 6.)
//! - `summarize FILE` — feed a transcript to the LLM and print JSON.
//!   (Wired Sprint 0 day 7.)
//! - `run --duration N` — full end-to-end record → transcribe → summarize.
//!   (Wired Sprint 0 day 8.)
//! - `stream --duration N` — live capture → resample → whisper streaming,
//!   optionally diarized; persists into the SQLite store the desktop
//!   shell shares. (Wired Sprint 0 day 9; diarize flag added Sprint 1.)
//! - `meetings list|get|delete` — inspect / clean up the SQLite store
//!   from the terminal without launching the UI.
//!   (Wired Sprint 0 day 10.)
//! - `bench wer` / `bench llm` — Phase 0 benchmarks.
//!   (Wired Sprint 0 day 9-10.)
//!
//! Every subcommand is a thin adapter that builds the relevant use case
//! from `echo-app` and executes it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use echo_app::{
    compute_wer, FrameSink, MeetingRecorder, RecordToSink, StreamingPipeline, SummarizeMeeting,
    SummarizeMeetingError, WerStats,
};
use echo_asr::WhisperCppTranscriber;
use echo_audio::{
    resample_to_whisper, CpalMicrophoneCapture, RoutingAudioCapture, RubatoResamplerAdapter,
    SileroVad, WavSink, WriteOptions, WHISPER_SAMPLE_RATE,
};
use echo_diarize::{Eres2NetEmbedder, OnlineDiarizer};
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, CaptureSpec, Diarizer, DomainError,
    LlmModel, MeetingId, MeetingStore, Resampler, StreamingOptions, SummaryContent,
    TranscribeOptions, Transcriber, TranscriptEvent,
};
use echo_llm::{LlamaCppLlm, LoadOptions as LlamaLoadOptions};
use echo_storage::SqliteMeetingStore;

/// EchoNote CLI prototype.
#[derive(Parser, Debug)]
#[command(
    name = "echo-proto",
    version,
    about = "EchoNote Phase 0 prototype — records, transcribes, summarizes on your machine",
    long_about = None,
    propagate_version = true
)]
struct Cli {
    /// Verbose mode (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Record N seconds of microphone audio to a WAV file.
    ///
    /// Sprint 0 day 5 captures the microphone only. System audio
    /// (loopback) lands in Sprint 0 day 6 alongside ScreenCaptureKit.
    Record {
        /// Duration of the capture, in seconds.
        #[arg(long, default_value_t = 5)]
        duration: u64,
        /// Output WAV file path. Parent directories are created.
        #[arg(long, short, default_value = "./recordings/mic.wav")]
        output: PathBuf,
        /// Optional input device name. Use `record-devices` to discover.
        #[arg(long)]
        device: Option<String>,
    },

    /// List the input devices the host exposes.
    RecordDevices,

    /// Transcribe a WAV file with the local whisper.cpp adapter.
    ///
    /// The file is decoded, downmixed to mono and resampled to 16 kHz
    /// `f32` before being fed to Whisper.
    Transcribe {
        /// Path to any RIFF/WAV file (mono or stereo, any sample rate).
        input: PathBuf,
        /// Path to a `ggml-*.bin` Whisper model. Defaults to the
        /// `ECHO_ASR_MODEL` env var, then the largest installed
        /// multilingual ggml model under `./models/asr/`.
        #[arg(long, env = "ECHO_ASR_MODEL")]
        model: Option<PathBuf>,
        /// ISO-639-1 language hint (e.g. `en`, `es`). Auto-detect if
        /// omitted.
        #[arg(long)]
        language: Option<String>,
        /// Translate to English instead of transcribing in source.
        #[arg(long)]
        translate: bool,
        /// Emit the full transcript as JSON instead of plain text.
        #[arg(long)]
        json: bool,
    },

    /// Live capture → resample → whisper streaming. Prints transcript
    /// events to stdout as they arrive.
    ///
    /// Useful as a head-less smoke test of the same pipeline that the
    /// Tauri shell drives in the UI. With `--source system-output` it
    /// captures the system audio mix on macOS via ScreenCaptureKit
    /// (Sprint 1 day 3) — Screen Recording permission required.
    Stream {
        /// Capture duration, in seconds. The pipeline stops cleanly
        /// after the deadline.
        #[arg(long, default_value_t = 30)]
        duration: u64,
        /// Path to a `ggml-*.bin` Whisper model. Defaults to the
        /// `ECHO_ASR_MODEL` env var, then the largest installed
        /// multilingual ggml model under `./models/asr/`.
        #[arg(long, env = "ECHO_ASR_MODEL")]
        model: Option<PathBuf>,
        /// Capture source. `microphone` is portable; `system-output`
        /// requires macOS 13+ with Screen Recording permission.
        #[arg(long, value_enum, default_value_t = SourceArg::Microphone)]
        source: SourceArg,
        /// Optional input device name. Use `record-devices` to discover.
        /// Ignored when `--source system-output`.
        #[arg(long)]
        device: Option<String>,
        /// ISO-639-1 language hint. Auto-detect if omitted.
        #[arg(long)]
        language: Option<String>,
        /// Chunk size in milliseconds. Defaults to 5 s — Whisper's sweet
        /// spot for streaming.
        #[arg(long, default_value_t = 5_000)]
        chunk_ms: u32,
        /// Skip transcription of chunks below this RMS. `0.0` disables
        /// the gate. Default 0.02 (~ -34 dBFS); raise it if room noise
        /// is sneaking past, lower it for very quiet speakers.
        #[arg(long, default_value_t = 0.02)]
        silence_threshold: f32,
        /// Attach a speaker diarizer to the pipeline. Each chunk is
        /// also fed through ERes2Net + online clustering and the
        /// resulting speaker is printed alongside the transcript.
        #[arg(long, default_value_t = false)]
        diarize: bool,
        /// Path to the ERes2Net ONNX export. Defaults to the
        /// `ECHO_EMBED_MODEL` env var or
        /// `./models/embedder/eres2net_en_voxceleb.onnx`.
        /// Only consulted when `--diarize` is set.
        #[arg(long, env = "ECHO_EMBED_MODEL")]
        embed_model: Option<PathBuf>,
        /// Path to the Silero VAD ONNX. Defaults to the
        /// `ECHO_VAD_MODEL` env var or `./models/vad/silero_vad.onnx`.
        /// When the file is present (and `--no-neural-vad` is not
        /// passed) the pipeline gates Whisper behind Silero — sharply
        /// reducing hallucinations on silent / noisy chunks.
        #[arg(long, env = "ECHO_VAD_MODEL")]
        vad_model: Option<PathBuf>,
        /// Disable the neural VAD and rely only on the energy-based
        /// RMS gate. Useful for benchmarking the old behaviour or
        /// when a soft speaker is being misclassified as silence.
        #[arg(long, default_value_t = false)]
        no_neural_vad: bool,
    },

    /// Summarize a stored meeting using the local LLM (Sprint 1 day 9).
    ///
    /// Loads the persisted meeting, prompts the LLM with the structured
    /// transcript, parses the JSON payload (with one corrective retry
    /// and free-text fallback), and upserts the resulting `Summary`.
    /// The summary is also printed to stdout — JSON when `--json`,
    /// a plain text rendering otherwise.
    Summarize {
        /// Meeting UUIDv7 (positional form).
        #[arg(value_name = "ID")]
        id_positional: Option<String>,
        /// Meeting UUIDv7 (flag form).
        #[arg(long = "id", value_name = "ID", conflicts_with = "id_positional")]
        id_flag: Option<String>,
        /// Path to the GGUF LLM model. Defaults to `ECHO_LLM_MODEL`
        /// or the highest-priority installed Qwen GGUF under
        /// `models/llm/` (Qwen 3 first, Qwen 2.5 as legacy fallback).
        #[arg(long, env = "ECHO_LLM_MODEL")]
        model: Option<PathBuf>,
        /// Emit the full `Summary` as JSON instead of plain text.
        #[arg(long)]
        json: bool,
    },

    /// Full end-to-end pipeline: record → transcribe → summarize.
    Run {
        /// Duration of the capture, in seconds.
        #[arg(long, default_value_t = 30)]
        duration: u64,
    },

    /// Phase 0 benchmarks (WER, LLM quality, latency).
    Bench {
        #[command(subcommand)]
        kind: BenchKind,
    },

    /// Inspect persisted meetings (Sprint 0 day 8).
    Meetings {
        #[command(subcommand)]
        kind: MeetingsKind,
    },
}

/// CLI mirror of [`echo_domain::AudioSource`]. Kept separate so the
/// `clap` value parser can derive friendly kebab-case names without
/// coupling the domain enum to clap.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SourceArg {
    /// Default microphone input via cpal.
    #[default]
    Microphone,
    /// System audio loopback. macOS 13+ only (ScreenCaptureKit).
    SystemOutput,
}

impl From<SourceArg> for AudioSource {
    fn from(value: SourceArg) -> Self {
        match value {
            SourceArg::Microphone => AudioSource::Microphone,
            SourceArg::SystemOutput => AudioSource::SystemOutput,
        }
    }
}

#[derive(Subcommand, Debug)]
enum MeetingsKind {
    /// List meetings, newest first.
    List {
        /// Maximum rows to return. `0` = unlimited.
        #[arg(long, default_value_t = 50)]
        limit: u32,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
    /// Show one meeting (header + segments).
    ///
    /// Accept the meeting id either as a positional arg
    /// (`meetings show 019d…`) or via the `--id` flag
    /// (`meetings show --id 019d…`) — both work, exactly one is required.
    Show {
        /// Meeting UUIDv7 (positional form).
        #[arg(value_name = "ID")]
        id_positional: Option<String>,
        /// Meeting UUIDv7 (flag form).
        #[arg(long = "id", value_name = "ID", conflicts_with = "id_positional")]
        id_flag: Option<String>,
        /// Emit JSON instead of plain text.
        #[arg(long)]
        json: bool,
    },
    /// Delete a meeting and its segments.
    ///
    /// Accept the meeting id either as a positional arg or via `--id`.
    Delete {
        /// Meeting UUIDv7 (positional form).
        #[arg(value_name = "ID")]
        id_positional: Option<String>,
        /// Meeting UUIDv7 (flag form).
        #[arg(long = "id", value_name = "ID", conflicts_with = "id_positional")]
        id_flag: Option<String>,
    },
    /// Full-text search over segment text. Returns one hit per
    /// meeting, ordered by FTS5 BM25 rank (best match first).
    Search {
        /// Search query. Quote multi-word phrases at the shell level
        /// (`echo-proto meetings search "design review"`); FTS5
        /// operators in the input are stripped before matching.
        #[arg(value_name = "QUERY")]
        query: String,
        /// Maximum hits to return. `0` = unlimited.
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum BenchKind {
    /// Word Error Rate benchmark over fixture audios.
    ///
    /// Discovers `<fixtures>/audio/<name>.wav` paired with
    /// `<fixtures>/transcripts/<name>.txt`, runs Whisper on each, and
    /// reports per-clip + global WER along with RTF stats. Exits non-zero
    /// when global WER exceeds `--max-wer`.
    Wer {
        /// Path to the fixtures root (must contain `audio/` and `transcripts/`).
        #[arg(long, default_value = "./fixtures")]
        fixtures: PathBuf,
        /// Path to a `ggml-*.bin` Whisper model. Defaults to the
        /// `ECHO_ASR_MODEL` env var, then the largest installed
        /// multilingual ggml model under `./models/asr/`.
        #[arg(long, env = "ECHO_ASR_MODEL")]
        model: Option<PathBuf>,
        /// ISO-639-1 hint passed to whisper.
        #[arg(long, default_value = "en")]
        language: String,
        /// Fail the run if global WER exceeds this fraction (0..=1).
        #[arg(long, default_value_t = 0.25)]
        max_wer: f64,
        /// Optional path to write a JSON report. Parent dirs created.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// LLM summary benchmark over gold transcripts.
    ///
    /// Scaffolding only — wires the contract (input set, prompt
    /// template, tokens/s + latency reporting). Exits with a clear
    /// message until the `echo-llm` adapter lands in Sprint 1.
    Llm {
        /// Path to the fixtures root (uses `transcripts/`).
        #[arg(long, default_value = "./fixtures")]
        fixtures: PathBuf,
        /// Optional path to a GGUF model (Qwen2.5-3B etc.).
        #[arg(long, env = "ECHO_LLM_MODEL")]
        model: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    echo_telemetry::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Record {
            duration,
            output,
            device,
        } => run_record(duration, output, device).await,
        Command::RecordDevices => run_list_devices().await,
        Command::Transcribe {
            input,
            model,
            language,
            translate,
            json,
        } => run_transcribe(input, model, language, translate, json).await,
        Command::Stream {
            duration,
            model,
            source,
            device,
            language,
            chunk_ms,
            silence_threshold,
            diarize,
            embed_model,
            vad_model,
            no_neural_vad,
        } => {
            run_stream(
                duration,
                model,
                source,
                device,
                language,
                chunk_ms,
                silence_threshold,
                diarize,
                embed_model,
                vad_model,
                no_neural_vad,
            )
            .await
        }
        Command::Summarize {
            id_positional,
            id_flag,
            model,
            json,
        } => run_summarize(id_positional, id_flag, model, json).await,
        Command::Run { duration } => {
            tracing::info!(
                duration_secs = duration,
                "run subcommand not yet wired (Sprint 0 day 8)"
            );
            Ok(())
        }
        Command::Bench { kind } => match kind {
            BenchKind::Wer {
                fixtures,
                model,
                language,
                max_wer,
                report,
            } => run_bench_wer(fixtures, model, language, max_wer, report).await,
            BenchKind::Llm { fixtures, model } => run_bench_llm(fixtures, model).await,
        },
        Command::Meetings { kind } => run_meetings(kind).await,
    }
}

// ---------------------------------------------------------------------------
// `record` subcommand
// ---------------------------------------------------------------------------

async fn run_record(duration_secs: u64, output: PathBuf, device: Option<String>) -> Result<()> {
    let capture = Arc::new(CpalMicrophoneCapture::new());
    let spec = CaptureSpec {
        source: AudioSource::Microphone,
        device_id: device,
        preferred_format: AudioFormat::WHISPER,
    };

    // Probe once to learn the negotiated format so the WAV header matches.
    let probe = capture
        .start(spec.clone())
        .await
        .context("failed to open the input device for probing")?;
    let format = probe.format();
    drop(probe); // release the device immediately

    tracing::info!(
        sample_rate_hz = format.sample_rate_hz,
        channels = format.channels,
        path = %output.display(),
        duration_secs,
        "negotiated capture format"
    );

    let sink = WavFrameSink::create(&output, format)?;
    let report = RecordToSink::new(capture, sink)
        .execute(spec, Duration::from_secs(duration_secs))
        .await
        .context("recording failed")?;

    println!(
        "wrote {bytes}-sample wav to {path}\n  duration: {dur:.2} s\n  format:   {rate} Hz × {ch} ch\n  frames:   {frames}",
        bytes = report.samples,
        path = output.display(),
        dur = report.elapsed.as_secs_f64(),
        rate = report.format.sample_rate_hz,
        ch = report.format.channels,
        frames = report.frames,
    );
    Ok(())
}

async fn run_list_devices() -> Result<()> {
    let capture = CpalMicrophoneCapture::new();
    let devices = capture.list_devices(AudioSource::Microphone).await?;
    if devices.is_empty() {
        println!("no input devices reported by the host");
        return Ok(());
    }
    println!("input devices ({n}):", n = devices.len());
    for d in &devices {
        let marker = if d.is_default { "*" } else { " " };
        println!("  {marker} {} ({})", d.name, d.id);
    }
    Ok(())
}

/// Adapter that wraps [`WavSink`] as an [`echo_app::FrameSink`].
struct WavFrameSink {
    inner: Option<WavSink>,
}

impl WavFrameSink {
    fn create(path: &Path, format: AudioFormat) -> Result<Self> {
        let inner = WavSink::create(path.to_path_buf(), format, WriteOptions::default())
            .with_context(|| format!("failed to create wav sink at {}", path.display()))?;
        Ok(Self { inner: Some(inner) })
    }
}

impl FrameSink for WavFrameSink {
    fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
        let sink = self
            .inner
            .as_mut()
            .ok_or_else(|| DomainError::Invariant("wav sink already finalized".into()))?;
        sink.write_frame(frame)
            .map_err(|e| DomainError::Invariant(format!("wav write: {e}")))
    }

    fn finish(&mut self) -> Result<(), DomainError> {
        if let Some(sink) = self.inner.take() {
            sink.finalize()
                .map(|_| ())
                .map_err(|e| DomainError::Invariant(format!("wav finalize: {e}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// `transcribe` subcommand
// ---------------------------------------------------------------------------

async fn run_transcribe(
    input: PathBuf,
    model: Option<PathBuf>,
    language: Option<String>,
    translate: bool,
    json: bool,
) -> Result<()> {
    let model_path = resolve_asr_model(model)?;

    let (samples, source_format) = load_wav_as_pcm(&input)
        .with_context(|| format!("failed to read wav: {}", input.display()))?;
    tracing::info!(
        path = %input.display(),
        samples = samples.len(),
        source.rate = source_format.sample_rate_hz,
        source.ch = source_format.channels,
        "loaded wav"
    );

    let pcm16k = resample_to_whisper(&samples, source_format)
        .map_err(DomainError::from)
        .context("resample to 16 kHz mono failed")?;
    tracing::info!(
        samples_in = samples.len(),
        samples_out = pcm16k.len(),
        target_rate = WHISPER_SAMPLE_RATE,
        "resampled to whisper format"
    );

    let started = std::time::Instant::now();
    let transcriber = WhisperCppTranscriber::load(&model_path)
        .map_err(anyhow::Error::from)
        .context("failed to load whisper model")?;
    tracing::info!(
        model = %model_path.display(),
        load_ms = started.elapsed().as_millis() as u64,
        "whisper context ready"
    );

    let opts = TranscribeOptions {
        language,
        initial_prompt: None,
        translate,
        threads: None,
    };
    let started = std::time::Instant::now();
    let transcript = transcriber.transcribe(&pcm16k, &opts).await?;
    let elapsed = started.elapsed();

    let audio_secs = pcm16k.len() as f64 / f64::from(WHISPER_SAMPLE_RATE);
    let rtf = elapsed.as_secs_f64() / audio_secs.max(1e-6);

    if json {
        println!("{}", serde_json::to_string_pretty(&transcript)?);
    } else {
        println!(
            "{}\n\n--\nlanguage: {}\nsegments: {}\naudio:    {:.2} s\nelapsed:  {:.2} s (rtf={:.3})",
            transcript.full_text(),
            transcript.language.as_deref().unwrap_or("?"),
            transcript.segments.len(),
            audio_secs,
            elapsed.as_secs_f64(),
            rtf,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `stream` subcommand
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_stream(
    duration_secs: u64,
    model: Option<PathBuf>,
    source_arg: SourceArg,
    device: Option<String>,
    language: Option<String>,
    chunk_ms: u32,
    silence_threshold: f32,
    diarize: bool,
    embed_model: Option<PathBuf>,
    vad_model: Option<PathBuf>,
    no_neural_vad: bool,
) -> Result<()> {
    let model_path = resolve_asr_model(model)?;

    let load_started = std::time::Instant::now();
    let transcriber = WhisperCppTranscriber::load(&model_path)
        .map_err(anyhow::Error::from)
        .context("failed to load whisper model")?;
    tracing::info!(
        model = %model_path.display(),
        load_ms = load_started.elapsed().as_millis() as u64,
        "whisper context ready"
    );

    // The router holds both adapters and dispatches per source. Same
    // facade serves the cpal mic and the macOS ScreenCaptureKit
    // loopback adapter without any branching at the call site.
    let capture: Arc<dyn AudioCapture> = Arc::new(RoutingAudioCapture::with_default_adapters());
    let resampler: Arc<dyn Resampler> = Arc::new(RubatoResamplerAdapter);
    let transcriber: Arc<dyn Transcriber> = Arc::new(transcriber);

    let source: AudioSource = source_arg.into();
    // ScreenCaptureKit identifies the loopback target by display, not
    // by named device. Surfacing `--device` for it would be misleading.
    let device_id = match source {
        AudioSource::Microphone => device,
        AudioSource::SystemOutput => {
            if device.is_some() {
                tracing::warn!("--device is ignored when --source system-output");
            }
            None
        }
    };

    let spec = CaptureSpec {
        source,
        device_id,
        preferred_format: AudioFormat::WHISPER,
    };
    let options = StreamingOptions {
        language,
        chunk_ms,
        silence_rms_threshold: silence_threshold,
    };

    let store: Arc<dyn MeetingStore> = Arc::new(open_cli_store().await?);
    let recorder = MeetingRecorder::with_default_title(store.clone());

    let mut pipeline = StreamingPipeline::new(capture, resampler, transcriber);
    if diarize {
        let embed_path = embed_model
            .or_else(|| {
                // Matches what `scripts/download-models.sh embed` writes.
                Some(PathBuf::from("./models/embedder/eres2net_en_voxceleb.onnx"))
            })
            .filter(|p| p.exists())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no speaker embedder found. Set --embed-model or ECHO_EMBED_MODEL, or run \
                     `scripts/download-models.sh embed` to fetch the default."
                )
            })?;
        let embed_started = std::time::Instant::now();
        let embedder = Eres2NetEmbedder::new(&embed_path)
            .map_err(anyhow::Error::from)
            .context("failed to load ERes2Net embedder")?;
        tracing::info!(
            model = %embed_path.display(),
            load_ms = embed_started.elapsed().as_millis() as u64,
            "speaker embedder ready"
        );
        let diarizer: Box<dyn Diarizer> =
            Box::new(OnlineDiarizer::with_defaults(Box::new(embedder)));
        pipeline = pipeline.with_diarizer(diarizer);
    }

    let vad_active = if no_neural_vad {
        tracing::info!("--no-neural-vad set; using RMS gate only");
        false
    } else {
        let vad_path = vad_model
            .or_else(|| Some(PathBuf::from("./models/vad/silero_vad.onnx")))
            .filter(|p| p.exists());
        match vad_path {
            Some(path) => {
                let started = std::time::Instant::now();
                let vad = SileroVad::for_meetings(&path)
                    .map_err(anyhow::Error::from)
                    .context("failed to load Silero VAD")?;
                tracing::info!(
                    model = %path.display(),
                    load_ms = started.elapsed().as_millis() as u64,
                    "Silero VAD ready"
                );
                pipeline = pipeline.with_vad(Box::new(vad));
                true
            }
            None => {
                tracing::warn!(
                    "Silero VAD model not found at ./models/vad/silero_vad.onnx; \
                     falling back to RMS gate. Run `scripts/download-models.sh vad` \
                     for sharper voice/non-voice discrimination."
                );
                false
            }
        }
    };

    let mut handle = pipeline
        .start_with_spec(spec, options)
        .await
        .context("failed to start streaming pipeline")?;

    println!(
        "▶ streaming session {} — source={:?} — {}s window {}ms — diarize={} — vad={} — Ctrl-C or wait to stop",
        handle.session_id(),
        source,
        duration_secs,
        chunk_ms,
        diarize,
        if vad_active { "silero" } else { "rms" },
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(duration_secs);
    let mut total_chunks: u32 = 0;
    let mut total_skipped: u32 = 0;
    let mut stopping = false;

    loop {
        let evt = if stopping {
            handle.next_event().await
        } else {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    handle.stop().await.context("failed to stop pipeline")?;
                    stopping = true;
                    continue;
                }
                evt = handle.next_event() => evt,
            }
        };

        if let Some(ref evt) = evt {
            if let Err(e) = recorder.record(evt).await {
                tracing::warn!(error = %e, "recorder.record failed");
            }
        }

        match evt {
            Some(TranscriptEvent::Started { input_format, .. }) => {
                println!(
                    "  · started: {} Hz × {} ch",
                    input_format.sample_rate_hz, input_format.channels,
                );
            }
            Some(TranscriptEvent::Chunk {
                chunk_index,
                offset_ms,
                segments,
                language,
                rtf,
                speaker_slot,
                ..
            }) => {
                total_chunks += 1;
                let text: String = segments
                    .iter()
                    .map(|s| s.text.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                let lang = language.as_deref().unwrap_or("?");
                let body = if text.is_empty() {
                    "<no speech>"
                } else {
                    &text
                };
                let speaker_tag = match speaker_slot {
                    Some(slot) => format!("S{}", slot + 1),
                    None if diarize => "S?".to_string(),
                    None => "  ".to_string(),
                };
                // RTF is meaningless when whisper produced no segments
                // (e.g. <no speech> on a tail chunk of ~10ms): the
                // ratio explodes because the divisor is the *audio*
                // duration, not the wall clock. Render `rtf= --` in
                // that case so the column stays aligned but doesn't
                // mislead the eye.
                let rtf_cell = if text.is_empty() {
                    "rtf=  --".to_string()
                } else {
                    format!("rtf={rtf:.2}")
                };
                println!(
                    "  [{chunk_index:>2}] +{offset_ms:>5} ms  {rtf_cell}  {speaker_tag}  {lang} → {body}"
                );
            }
            Some(TranscriptEvent::Skipped {
                chunk_index,
                offset_ms,
                rms,
                ..
            }) => {
                total_skipped += 1;
                println!("  [{chunk_index:>2}] +{offset_ms:>5} ms  silence (rms={rms:.4})");
            }
            Some(TranscriptEvent::Stopped {
                total_segments,
                total_audio_ms,
                ..
            }) => {
                let secs = (total_audio_ms as f64) / 1_000.0;
                println!(
                    "■ stopped: {total_chunks} chunks ({total_skipped} skipped), {total_segments} segments, {secs:.2} s of audio",
                );
                break;
            }
            Some(TranscriptEvent::Failed { message, .. }) => {
                anyhow::bail!("pipeline failed: {message}");
            }
            None => break,
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// `summarize` subcommand (Sprint 1 day 9)
// ---------------------------------------------------------------------------

/// Resolve the GGUF model path: `--model`, then `ECHO_LLM_MODEL`,
/// then the highest-priority installed Qwen GGUF under `models/llm/`.
/// Mirrors the resolution order the Tauri shell uses (Qwen 3 first,
/// Qwen 2.5 as legacy fallback).
fn resolve_llm_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            anyhow::bail!("LLM model not found at {}", p.display());
        }
        return Ok(p);
    }
    // Qwen 3 filenames mirror the official Qwen team's HF naming
    // convention (`Qwen3-<size>-Q4_K_M.gguf`, capital `Q`, no
    // `-Instruct-` infix). Keep the legacy lowercase Qwen 2.5 paths
    // for back-compat with day-9 setups.
    const CANDIDATES: &[&str] = &[
        "./models/llm/Qwen3-30B-A3B-Q4_K_M.gguf",
        "./models/llm/Qwen3-14B-Q4_K_M.gguf",
        "./models/llm/Qwen3-8B-Q4_K_M.gguf",
        "./models/llm/qwen2.5-7b-instruct-q4_k_m.gguf",
        "./models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
    ];
    for rel in CANDIDATES {
        let p = PathBuf::from(rel);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no GGUF LLM model found. Set --model or ECHO_LLM_MODEL, or run \
         `scripts/download-models.sh llm` to fetch the default Qwen 3 14B."
    )
}

/// Resolve the Whisper ggml model path used by `transcribe`, `stream`
/// and `bench wer`. Same priority as the Tauri shell's
/// `preferred_asr_model`: Spanish fine-tune > multilingual (turbo →
/// large → medium → small → base → tiny) > English-only fallbacks.
/// Returning `Result` (instead of falling back silently) keeps the
/// CLI honest about missing setup.
fn resolve_asr_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            anyhow::bail!("ASR model not found at {}", p.display());
        }
        return Ok(p);
    }
    const CANDIDATES: &[&str] = &[
        "./models/asr/ggml-large-v3-turbo-es.bin",
        "./models/asr/ggml-large-v3-turbo.bin",
        "./models/asr/ggml-large-v3.bin",
        "./models/asr/ggml-medium.bin",
        "./models/asr/ggml-small.bin",
        "./models/asr/ggml-base.bin",
        "./models/asr/ggml-tiny.bin",
        "./models/asr/ggml-base.en.bin",
        "./models/asr/ggml-small.en.bin",
        "./models/asr/ggml-tiny.en.bin",
    ];
    for rel in CANDIDATES {
        let p = PathBuf::from(rel);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no Whisper ggml model found. Set --model or ECHO_ASR_MODEL, or run \
         `scripts/download-models.sh` to fetch the default multilingual large-v3-turbo."
    )
}

async fn run_summarize(
    id_positional: Option<String>,
    id_flag: Option<String>,
    model: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let meeting_id = resolve_meeting_id(id_positional, id_flag)?;
    let model_path = resolve_llm_model(model)?;

    // Load model on the blocking pool (mmap + ggml init are sync and
    // expensive). The Tauri shell uses the exact same pattern.
    println!("loading LLM: {}", model_path.display());
    let load_started = std::time::Instant::now();
    let llm = tokio::task::spawn_blocking({
        let p = model_path.clone();
        move || LlamaCppLlm::load_with(&p, LlamaLoadOptions::default())
    })
    .await
    .context("LLM load task panicked")?
    .map_err(anyhow::Error::from)
    .with_context(|| format!("failed to load LLM at {}", model_path.display()))?;
    tracing::info!(
        model = %model_path.display(),
        load_ms = load_started.elapsed().as_millis() as u64,
        "llm ready"
    );
    let llm: Arc<dyn LlmModel> = Arc::new(llm);

    let store: Arc<dyn MeetingStore> = Arc::new(open_cli_store().await?);
    let use_case = SummarizeMeeting::new(llm, store);

    let started = std::time::Instant::now();
    let summary = use_case
        .execute(meeting_id, "general")
        .await
        .map_err(|e| match e {
            SummarizeMeetingError::NotFound(id) => {
                anyhow::anyhow!("meeting {id} not found")
            }
            SummarizeMeetingError::EmptyTranscript(id) => {
                anyhow::anyhow!("meeting {id} has no segments to summarise")
            }
            SummarizeMeetingError::InvalidTemplate(t) => {
                anyhow::anyhow!("invalid template: {t}")
            }
            SummarizeMeetingError::Llm(err) => anyhow::anyhow!("llm: {err}"),
            SummarizeMeetingError::Storage(err) => anyhow::anyhow!("storage: {err}"),
        })?;
    let elapsed = started.elapsed();

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }

    println!(
        "\nsummary {} (model={} · {:.2}s)",
        summary.id.0,
        summary.model,
        elapsed.as_secs_f64(),
    );
    if let Some(lang) = &summary.language {
        println!("language: {lang}");
    }
    println!();

    match &summary.content {
        SummaryContent::General {
            summary: text,
            key_points,
            decisions,
            action_items,
            open_questions,
        } => {
            println!("## Summary\n{text}\n");
            print_section("Key points", key_points);
            print_section("Decisions", decisions);
            if !action_items.is_empty() {
                println!("## Action items");
                for it in action_items {
                    let owner = it.owner.as_deref().unwrap_or("-");
                    let due = it.due.as_deref().unwrap_or("-");
                    println!("  · {} (owner={owner}, due={due})", it.task);
                }
                println!();
            }
            print_section("Open questions", open_questions);
        }
        SummaryContent::FreeText { text } => {
            println!("## Summary (free text — JSON parse failed)\n{text}");
        }
        // SummaryContent is `#[non_exhaustive]`; future templates will
        // need an explicit branch here, but for now we match all known
        // variants and serde would have caught any unknown one earlier.
        _ => {
            println!("(unknown summary template — re-run with --json to inspect)");
        }
    }
    Ok(())
}

fn print_section(title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    println!("## {title}");
    for it in items {
        println!("  · {it}");
    }
    println!();
}

// ---------------------------------------------------------------------------
// `meetings` subcommand
// ---------------------------------------------------------------------------

async fn open_cli_store() -> Result<SqliteMeetingStore> {
    let path = std::env::var("ECHO_DB_PATH").unwrap_or_else(|_| "./echonote.db".to_string());
    SqliteMeetingStore::open(&path)
        .await
        .map_err(anyhow::Error::from)
        .with_context(|| format!("open meeting store at {path}"))
}

async fn run_meetings(kind: MeetingsKind) -> Result<()> {
    let store = open_cli_store().await?;
    match kind {
        MeetingsKind::List { limit, json } => {
            let rows = store.list(limit).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
                return Ok(());
            }
            if rows.is_empty() {
                println!("(no meetings yet)");
                return Ok(());
            }
            println!(
                "{:<38}  {:<25}  {:>8}  {:>5}  title",
                "id", "started_at", "dur_s", "segs"
            );
            for m in &rows {
                let started = m
                    .started_at
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default();
                let dur = (m.duration_ms as f64) / 1_000.0;
                println!(
                    "{:<38}  {:<25}  {:>8.2}  {:>5}  {}",
                    m.id, started, dur, m.segment_count, m.title
                );
            }
        }
        MeetingsKind::Show {
            id_positional,
            id_flag,
            json,
        } => {
            let id = resolve_meeting_id(id_positional, id_flag)?;
            let Some(meeting) = store.get(id).await? else {
                anyhow::bail!("meeting {id} not found");
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&meeting)?);
                return Ok(());
            }
            let s = &meeting.summary;
            println!("# {}\nid: {}\nstarted: {}\nended:   {}\nlanguage: {}\nduration: {:.2} s\nsegments: {}\nspeakers: {}\nformat:   {} Hz × {} ch\n",
                s.title,
                s.id,
                s.started_at.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
                s.ended_at.map(|d| d.format(&time::format_description::well_known::Rfc3339).unwrap_or_default()).unwrap_or_else(|| "-".into()),
                s.language.as_deref().unwrap_or("?"),
                (s.duration_ms as f64) / 1_000.0,
                s.segment_count,
                meeting.speakers.len(),
                meeting.input_format.sample_rate_hz,
                meeting.input_format.channels,
            );
            if !meeting.speakers.is_empty() {
                println!("Speakers:");
                for sp in &meeting.speakers {
                    println!("  S{}  {}  ({})", sp.slot + 1, sp.display_name(), sp.id);
                }
                println!();
            }
            // Index speakers by id so each segment line can render
            // its tag in O(1) without a linear scan per row.
            let speaker_tags: std::collections::HashMap<_, _> = meeting
                .speakers
                .iter()
                .map(|sp| (sp.id, format!("S{}", sp.slot + 1)))
                .collect();
            for seg in &meeting.segments {
                let tag = seg
                    .speaker_id
                    .and_then(|sid| speaker_tags.get(&sid))
                    .map(|s| s.as_str())
                    .unwrap_or("  ");
                println!(
                    "  [{:>6}-{:>6} ms] {}  {}",
                    seg.start_ms,
                    seg.end_ms,
                    tag,
                    seg.text.trim()
                );
            }
        }
        MeetingsKind::Delete {
            id_positional,
            id_flag,
        } => {
            let id = resolve_meeting_id(id_positional, id_flag)?;
            let removed = store.delete(id).await?;
            if removed {
                println!("deleted {id}");
            } else {
                println!("not found: {id}");
            }
        }
        MeetingsKind::Search { query, limit, json } => {
            let hits = store.search(&query, limit).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
                return Ok(());
            }
            if hits.is_empty() {
                println!("(no matches for {query:?})");
                return Ok(());
            }
            println!(
                "{:<38}  {:>9}  {:<25}  title / snippet",
                "id", "rank", "started_at"
            );
            for hit in &hits {
                let started = hit
                    .meeting
                    .started_at
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default();
                // Strip the <mark> markers for the plain CLI; the JSON
                // path keeps them so downstream tools can still highlight.
                let snippet = hit
                    .snippet
                    .replace("<mark>", "")
                    .replace("</mark>", "")
                    .replace('\n', " ");
                println!(
                    "{:<38}  {:>9.3}  {:<25}  {}\n{:<76}{}",
                    hit.meeting.id, hit.rank, started, hit.meeting.title, "", snippet,
                );
            }
        }
    }
    Ok(())
}

fn parse_meeting_id(s: &str) -> Result<MeetingId> {
    let uuid = uuid::Uuid::parse_str(s).with_context(|| format!("invalid meeting id: {s:?}"))?;
    Ok(MeetingId(uuid))
}

/// Accept the meeting id either as a positional value or via the `--id`
/// flag. `clap` already enforces `conflicts_with`, so at most one of the
/// two is `Some`; here we just collapse them and surface a friendly
/// error when both are absent.
fn resolve_meeting_id(positional: Option<String>, flag: Option<String>) -> Result<MeetingId> {
    let raw = positional.or(flag).ok_or_else(|| {
        anyhow::anyhow!("missing meeting id: pass it as the positional arg or with `--id <UUID>`")
    })?;
    parse_meeting_id(&raw)
}

// ---------------------------------------------------------------------------
// `bench wer` subcommand
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize)]
struct WerClipReport {
    name: String,
    reference_words: u32,
    hypothesis_words: u32,
    substitutions: u32,
    deletions: u32,
    insertions: u32,
    wer: f64,
    audio_seconds: f64,
    elapsed_seconds: f64,
    rtf: f64,
    detected_language: Option<String>,
    hypothesis: String,
}

#[derive(Debug, serde::Serialize)]
struct WerBenchReport {
    model: String,
    language: String,
    clips: Vec<WerClipReport>,
    global_wer: f64,
    rtf_p50: f64,
    rtf_p95: f64,
    total_audio_seconds: f64,
    total_elapsed_seconds: f64,
    max_wer_threshold: f64,
}

async fn run_bench_wer(
    fixtures: PathBuf,
    model: Option<PathBuf>,
    language: String,
    max_wer: f64,
    report_path: Option<PathBuf>,
) -> Result<()> {
    if !(0.0..=1.0).contains(&max_wer) {
        anyhow::bail!("--max-wer must be in [0, 1], got {max_wer}");
    }

    let model_path = resolve_asr_model(model)?;

    let pairs = discover_fixture_pairs(&fixtures)?;
    if pairs.is_empty() {
        anyhow::bail!(
            "no fixtures found under {}. Run scripts/build-fixtures.sh first.",
            fixtures.display()
        );
    }

    println!(
        "loading whisper model: {} ({} fixture pairs)",
        model_path.display(),
        pairs.len()
    );
    let load_started = std::time::Instant::now();
    let transcriber = WhisperCppTranscriber::load(&model_path)
        .map_err(anyhow::Error::from)
        .context("failed to load whisper model")?;
    tracing::info!(
        load_ms = load_started.elapsed().as_millis() as u64,
        "whisper context ready"
    );

    let mut clips = Vec::with_capacity(pairs.len());
    for (name, wav, txt) in pairs {
        let reference = std::fs::read_to_string(&txt)
            .with_context(|| format!("read reference {}", txt.display()))?;
        let (samples, source_format) =
            load_wav_as_pcm(&wav).with_context(|| format!("read wav {}", wav.display()))?;

        let pcm16k = resample_to_whisper(&samples, source_format)
            .map_err(DomainError::from)
            .context("resample to 16 kHz failed")?;
        let audio_seconds = pcm16k.len() as f64 / f64::from(WHISPER_SAMPLE_RATE);

        let opts = TranscribeOptions {
            language: Some(language.clone()),
            initial_prompt: None,
            translate: false,
            threads: None,
        };
        let started = std::time::Instant::now();
        let transcript = transcriber.transcribe(&pcm16k, &opts).await?;
        let elapsed = started.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let rtf = elapsed_seconds / audio_seconds.max(1e-6);

        let hypothesis = transcript.full_text();
        let stats = compute_wer(&reference, &hypothesis);

        println!(
            "  {name:<28}  ref={ref_w:>4}  hyp={hyp_w:>4}  S={s:>2} D={d:>2} I={i:>2}  WER={wer:>6.2}%  rtf={rtf:.2}",
            name = name,
            ref_w = stats.reference_words,
            hyp_w = stats.hypothesis_words,
            s = stats.substitutions,
            d = stats.deletions,
            i = stats.insertions,
            wer = stats.wer() * 100.0,
            rtf = rtf,
        );

        clips.push(WerClipReport {
            name,
            reference_words: stats.reference_words,
            hypothesis_words: stats.hypothesis_words,
            substitutions: stats.substitutions,
            deletions: stats.deletions,
            insertions: stats.insertions,
            wer: stats.wer(),
            audio_seconds,
            elapsed_seconds,
            rtf,
            detected_language: transcript.language.clone(),
            hypothesis,
        });
    }

    // Aggregate across clips.
    let global_stats = clips
        .iter()
        .map(|c| WerStats {
            reference_words: c.reference_words,
            hypothesis_words: c.hypothesis_words,
            substitutions: c.substitutions,
            deletions: c.deletions,
            insertions: c.insertions,
        })
        .fold(WerStats::default(), |a, b| a.merge(b));
    let global_wer = global_stats.wer();
    let total_audio: f64 = clips.iter().map(|c| c.audio_seconds).sum();
    let total_elapsed: f64 = clips.iter().map(|c| c.elapsed_seconds).sum();
    let mut rtfs: Vec<f64> = clips.iter().map(|c| c.rtf).collect();
    rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rtf_p50 = percentile(&rtfs, 0.50);
    let rtf_p95 = percentile(&rtfs, 0.95);

    println!();
    println!(
        "global  WER={:.2}%  ({} clips, {:.1}s audio, {:.1}s elapsed, rtf p50={:.2} p95={:.2})",
        global_wer * 100.0,
        clips.len(),
        total_audio,
        total_elapsed,
        rtf_p50,
        rtf_p95,
    );
    println!("threshold: {:.2}%", max_wer * 100.0);

    let report = WerBenchReport {
        model: model_path.display().to_string(),
        language: language.clone(),
        clips,
        global_wer,
        rtf_p50,
        rtf_p95,
        total_audio_seconds: total_audio,
        total_elapsed_seconds: total_elapsed,
        max_wer_threshold: max_wer,
    };

    if let Some(path) = report_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create report parent dir {}", parent.display()))?;
            }
        }
        std::fs::write(&path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("write report {}", path.display()))?;
        println!("wrote report → {}", path.display());
    }

    if global_wer > max_wer {
        anyhow::bail!(
            "global WER {:.2}% exceeds gate {:.2}%",
            global_wer * 100.0,
            max_wer * 100.0
        );
    }

    Ok(())
}

// Discover (basename, audio_path, transcript_path) triples. Skips
// transcripts without a matching audio (common after pulling new
// fixtures.txt without regenerating WAVs).
fn discover_fixture_pairs(root: &Path) -> Result<Vec<(String, PathBuf, PathBuf)>> {
    let txt_dir = root.join("transcripts");
    let wav_dir = root.join("audio");
    if !txt_dir.is_dir() {
        anyhow::bail!("missing fixtures dir: {}", txt_dir.display());
    }
    if !wav_dir.is_dir() {
        anyhow::bail!(
            "missing fixtures dir: {} (run scripts/build-fixtures.sh)",
            wav_dir.display()
        );
    }
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&txt_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("invalid filename: {}", path.display()))?
            .to_string();
        let wav = wav_dir.join(format!("{stem}.wav"));
        if !wav.exists() {
            tracing::warn!(
                fixture = %stem,
                "skipping fixture: matching wav not found at {}",
                wav.display()
            );
            continue;
        }
        pairs.push((stem, wav, path));
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(pairs)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// `bench llm` subcommand (scaffolding)
// ---------------------------------------------------------------------------

async fn run_bench_llm(fixtures: PathBuf, model: Option<PathBuf>) -> Result<()> {
    let txt_dir = fixtures.join("transcripts");
    if !txt_dir.is_dir() {
        anyhow::bail!("missing fixtures dir: {}", txt_dir.display());
    }
    let mut transcripts = Vec::new();
    for entry in std::fs::read_dir(&txt_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            transcripts.push(path);
        }
    }
    transcripts.sort();
    println!("bench llm contract:");
    println!("  · would summarize {} transcripts", transcripts.len());
    println!(
        "  · model: {}",
        model
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<unset>".into())
    );
    println!("  · metrics: tokens/s, time-to-first-token, total latency");
    println!();
    println!("  not yet runnable — `echo-llm` adapter lands in Sprint 1 (chat use case).");
    println!("  this subcommand intentionally exits 0 so CI can wire the job ahead of time.");
    Ok(())
}

// ---------------------------------------------------------------------------
// shared WAV loader
// ---------------------------------------------------------------------------

/// Decode any RIFF/WAV file (16-bit int, 24-bit int or 32-bit float;
/// any sample rate, any channel count) into interleaved `f32` samples
/// in `[-1.0, 1.0]` plus the source [`AudioFormat`].
fn load_wav_as_pcm(path: &Path) -> Result<(Vec<f32>, AudioFormat)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("decoding f32 wav samples")?,
        hound::SampleFormat::Int => {
            let max = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()
                .context("decoding integer wav samples")?
        }
    };
    let format = AudioFormat {
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
    };
    Ok((samples, format))
}
