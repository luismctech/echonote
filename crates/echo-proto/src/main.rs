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

use echo_app::{FrameSink, RecordToSink};
use echo_audio::{CpalMicrophoneCapture, WavSink, WriteOptions};
use echo_domain::{AudioCapture, AudioFormat, AudioFrame, AudioSource, CaptureSpec, DomainError};

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

    /// Transcribe a previously recorded WAV file.
    Transcribe {
        /// Path to a 16 kHz mono WAV file (or it will be resampled).
        input: String,
        /// ASR model path. Defaults to the environment-detected model.
        #[arg(long)]
        model: Option<String>,
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
        Command::Transcribe { input, model } => {
            tracing::info!(
                input = %input,
                model = ?model,
                "transcribe subcommand not yet wired (Sprint 0 day 6)"
            );
            Ok(())
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
