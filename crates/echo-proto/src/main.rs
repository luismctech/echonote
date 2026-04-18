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

use anyhow::Result;
use clap::{Parser, Subcommand};

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
    /// Record N seconds of dual audio (mic + system) to WAV files.
    Record {
        /// Duration of the capture, in seconds.
        #[arg(long, default_value_t = 30)]
        duration: u64,
        /// Output directory for the resulting WAV files.
        #[arg(long, default_value = "./recordings")]
        out: String,
    },

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
        Command::Record { duration, out } => {
            tracing::info!(
                duration_secs = duration,
                out_dir = %out,
                "record subcommand not yet wired (Sprint 0 day 5)"
            );
            Ok(())
        }
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
