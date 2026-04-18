//! # echo-proto
//!
//! CLI prototype for EchoNote's Phase 0 (Discovery). The binary stitches
//! the capture, ASR and LLM crates together to prove the end-to-end
//! pipeline on macOS before any UI work begins.
//!
//! Subcommands (incrementally wired during Sprint 0):
//!
//! - `record --duration N` — capture N seconds of dual audio to WAV.
//!   (Wired Sprint 0 day 5.)
//! - `transcribe FILE` — run Whisper on a WAV and print segments.
//!   (Wired Sprint 0 day 6.)
//! - `summarize FILE` — feed a transcript to the LLM and print JSON.
//!   (Wired Sprint 0 day 7.)
//! - `run --duration N` — full end-to-end record → transcribe → summarize.
//!   (Wired Sprint 0 day 8.)
//! - `bench wer` / `bench llm` — Phase 0 benchmarks.
//!   (Wired Sprint 0 day 9-10.)
//!
//! Every subcommand is a thin adapter that builds the relevant use case
//! from `echo-app` and executes it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use echo_app::{FrameSink, RecordToSink, StreamingPipeline};
use echo_asr::WhisperCppTranscriber;
use echo_audio::{
    resample_to_whisper, CpalMicrophoneCapture, RubatoResamplerAdapter, WavSink, WriteOptions,
    WHISPER_SAMPLE_RATE,
};
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, CaptureSpec, DomainError, Resampler,
    StreamingOptions, TranscribeOptions, Transcriber, TranscriptEvent,
};

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
        /// `ECHO_ASR_MODEL` env var or `./models/asr/ggml-base.en.bin`.
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

    /// Live mic → resample → whisper streaming. Prints transcript
    /// events to stdout as they arrive.
    ///
    /// Useful as a head-less smoke test of the same pipeline that the
    /// Tauri shell drives in the UI.
    Stream {
        /// Capture duration, in seconds. The pipeline stops cleanly
        /// after the deadline.
        #[arg(long, default_value_t = 30)]
        duration: u64,
        /// Path to a `ggml-*.bin` Whisper model. Defaults to the
        /// `ECHO_ASR_MODEL` env var or `./models/asr/ggml-base.en.bin`.
        #[arg(long, env = "ECHO_ASR_MODEL")]
        model: Option<PathBuf>,
        /// Optional input device name. Use `record-devices` to discover.
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
        /// the gate.
        #[arg(long, default_value_t = 0.005)]
        silence_threshold: f32,
    },

    /// Summarize a transcript using the local LLM.
    Summarize {
        /// Path to a plain-text transcript.
        input: String,
        /// Template id (general, one_on_one, sprint_review, ...).
        #[arg(long, default_value = "general")]
        template: String,
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
}

#[derive(Subcommand, Debug)]
enum BenchKind {
    /// Word Error Rate benchmark over fixture audios.
    Wer,
    /// LLM summary benchmark over gold transcripts.
    Llm,
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
            device,
            language,
            chunk_ms,
            silence_threshold,
        } => {
            run_stream(
                duration,
                model,
                device,
                language,
                chunk_ms,
                silence_threshold,
            )
            .await
        }
        Command::Summarize { input, template } => {
            tracing::info!(
                input = %input,
                template = %template,
                "summarize subcommand not yet wired (Sprint 0 day 7)"
            );
            Ok(())
        }
        Command::Run { duration } => {
            tracing::info!(
                duration_secs = duration,
                "run subcommand not yet wired (Sprint 0 day 8)"
            );
            Ok(())
        }
        Command::Bench { kind } => {
            tracing::info!(?kind, "bench subcommand not yet wired (Sprint 0 day 9-10)");
            Ok(())
        }
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
    let model_path = model
        .or_else(|| Some(PathBuf::from("./models/asr/ggml-base.en.bin")))
        .filter(|p| p.exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model found. Set --model or ECHO_ASR_MODEL, or run \
                 `scripts/download-models.sh base.en` to fetch the default."
            )
        })?;

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

async fn run_stream(
    duration_secs: u64,
    model: Option<PathBuf>,
    device: Option<String>,
    language: Option<String>,
    chunk_ms: u32,
    silence_threshold: f32,
) -> Result<()> {
    let model_path = model
        .or_else(|| Some(PathBuf::from("./models/asr/ggml-base.en.bin")))
        .filter(|p| p.exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model found. Set --model or ECHO_ASR_MODEL, or run \
                 `scripts/download-models.sh base.en` to fetch the default."
            )
        })?;

    let load_started = std::time::Instant::now();
    let transcriber = WhisperCppTranscriber::load(&model_path)
        .map_err(anyhow::Error::from)
        .context("failed to load whisper model")?;
    tracing::info!(
        model = %model_path.display(),
        load_ms = load_started.elapsed().as_millis() as u64,
        "whisper context ready"
    );

    let capture: Arc<dyn AudioCapture> = Arc::new(CpalMicrophoneCapture::new());
    let resampler: Arc<dyn Resampler> = Arc::new(RubatoResamplerAdapter);
    let transcriber: Arc<dyn Transcriber> = Arc::new(transcriber);

    let spec = CaptureSpec {
        source: AudioSource::Microphone,
        device_id: device,
        preferred_format: AudioFormat::WHISPER,
    };
    let options = StreamingOptions {
        language,
        chunk_ms,
        silence_rms_threshold: silence_threshold,
    };

    let pipeline = StreamingPipeline::new(capture, resampler, transcriber);
    let mut handle = pipeline
        .start_with_spec(spec, options)
        .await
        .context("failed to start streaming pipeline")?;

    println!(
        "▶ streaming session {} — {}s window {}ms — Ctrl-C or wait to stop",
        handle.session_id(),
        duration_secs,
        chunk_ms
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
                println!("  [{chunk_index:>2}] +{offset_ms:>5} ms  rtf={rtf:.2}  {lang} → {body}");
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
