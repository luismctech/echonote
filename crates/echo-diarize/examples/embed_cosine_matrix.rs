//! Embed a list of WAV files with the active `Eres2NetEmbedder` and
//! print the pairwise cosine-similarity matrix. Useful for sanity-
//! checking embedder quality across languages, accents, or new
//! checkpoints without writing a one-off test.
//!
//! Usage:
//!
//! ```bash
//! ECHO_EMBED_MODEL=models/embedder/eres2net_en_voxceleb.onnx \
//!   cargo run --example embed_cosine_matrix -- \
//!     fixture_a.wav fixture_b.wav fixture_c.wav
//! ```
//!
//! All inputs must be 16 kHz mono PCM (the script asserts this so you
//! don't accidentally feed it the wrong sample rate). The first column
//! of the printed matrix shows the file's basename for readability.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use echo_diarize::{cosine_similarity, Eres2NetEmbedder, SpeakerEmbedder, ERES2NET_SAMPLE_RATE};
use hound::WavReader;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: ECHO_EMBED_MODEL=<path-to.onnx> \\\n  \
             cargo run --example embed_cosine_matrix -- file1.wav file2.wav ..."
        );
        return ExitCode::from(2);
    }

    let model_path = match env::var("ECHO_EMBED_MODEL") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!("ECHO_EMBED_MODEL env var must point to the ONNX checkpoint");
            return ExitCode::from(2);
        }
    };

    let mut embedder = match Eres2NetEmbedder::new(&model_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("failed to load model {}: {err}", model_path.display());
            return ExitCode::FAILURE;
        }
    };

    args.sort();
    let mut labels: Vec<String> = Vec::with_capacity(args.len());
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(args.len());

    for path in &args {
        let p = Path::new(path);
        let label = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();
        let samples = match load_wav_16k_mono(p) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("skip {}: {err}", p.display());
                continue;
            }
        };
        match embedder.embed(&samples) {
            Ok(Some(embedding)) => {
                labels.push(label);
                embeddings.push(embedding);
            }
            Ok(None) => eprintln!("skip {}: clip too short for embedder", p.display()),
            Err(err) => eprintln!("skip {} (embed error): {err}", p.display()),
        }
    }

    if embeddings.len() < 2 {
        eprintln!("need at least 2 valid fixtures, got {}", embeddings.len());
        return ExitCode::FAILURE;
    }

    let max_label_len = labels.iter().map(|s| s.len()).max().unwrap_or(8).max(8);

    print!("{:width$}", "", width = max_label_len + 2);
    for label in &labels {
        print!(" {label:>10}");
    }
    println!();

    for (i, row_label) in labels.iter().enumerate() {
        print!("{:width$}", row_label, width = max_label_len + 2);
        for j in 0..labels.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!(" {sim:>10.3}");
        }
        println!();
    }

    ExitCode::SUCCESS
}

fn load_wav_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("open: {e}"))?;
    let spec = reader.spec();
    if spec.sample_rate != ERES2NET_SAMPLE_RATE {
        return Err(format!(
            "sample rate {} != {ERES2NET_SAMPLE_RATE}",
            spec.sample_rate
        ));
    }
    if spec.channels != 1 {
        return Err(format!("channels {} != 1 (mono required)", spec.channels));
    }
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / f32::from(i16::MAX)))
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };
    samples.map_err(|e| format!("decode: {e}"))
}
