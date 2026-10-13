//! `bench embed-dump`: speaker embeddings of a list of wav files, written
//! to one binary file for offline model fitting (the PLDA behind the
//! VBx re-segmentation, `scripts/fit-plda.py`).
//!
//! This is developer tooling for public training corpora. The app never
//! writes embeddings anywhere (ADR 0001, D1); only this command does, and
//! only to the path it is given.
//!
//! Output format (little-endian): the 8-byte magic `ECHOEMB1`, `u32` row
//! count, `u32` dimension, then `rows x dim` `f32` values in row order. A
//! sidecar `<out>.labels.txt` holds one class label per row. Files the
//! embedder declines (shorter than its minimum) are skipped and counted.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use echo_diarize::EmbedderFamily;

/// Magic bytes at the start of an embedding dump.
pub(crate) const MAGIC: &[u8; 8] = b"ECHOEMB1";

/// Options of `echo-proto bench embed-dump`.
#[derive(Debug, Clone)]
pub(crate) struct EmbedDumpArgs {
    /// List file: one `<wav> <label>` pair per line, `#` comments,
    /// relative paths resolved against the list's directory.
    pub list: PathBuf,
    /// Speaker embedder model file.
    pub embedder: PathBuf,
    /// Catalog id of the embedder; `None` guesses the family from the
    /// file name.
    pub embedder_id: Option<String>,
    /// Output file.
    pub out: PathBuf,
}

/// Parse a list file body into `(wav, label)` pairs.
fn parse_list(text: &str, base: &Path) -> Result<Vec<(PathBuf, String)>> {
    let mut items = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 {
            bail!("list line {}: expected `<wav> <label>`", i + 1);
        }
        items.push((base.join(fields[0]), fields[1].to_string()));
    }
    Ok(items)
}

/// Serialise `rows` (all of length `dim`) in the dump format.
pub(crate) fn encode(rows: &[Vec<f32>], dim: usize) -> Result<Vec<u8>> {
    ensure!(
        rows.iter().all(|r| r.len() == dim),
        "every embedding must have dimension {dim}"
    );
    let mut out = Vec::with_capacity(16 + rows.len() * dim * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&u32::try_from(rows.len())?.to_le_bytes());
    out.extend_from_slice(&u32::try_from(dim)?.to_le_bytes());
    for x in rows.iter().flatten() {
        out.extend_from_slice(&x.to_le_bytes());
    }
    Ok(out)
}

/// Entry point of `echo-proto bench embed-dump`.
pub(crate) fn run(args: &EmbedDumpArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.list)
        .with_context(|| format!("read {}", args.list.display()))?;
    let items = parse_list(&text, args.list.parent().unwrap_or(Path::new(".")))?;
    let family = match args.embedder_id.as_deref() {
        Some(id) => EmbedderFamily::from_model_id(id)
            .with_context(|| format!("unknown embedder id `{id}`"))?,
        None => EmbedderFamily::from_path(&args.embedder),
    };
    let mut embedder = family.load(&args.embedder)?;
    let rate = embedder.sample_rate_hz();
    let (mut rows, mut labels, mut skipped) = (Vec::new(), Vec::new(), 0_usize);
    for (wav, label) in &items {
        let (samples, format) =
            super::load_wav_as_pcm(wav).with_context(|| format!("read {}", wav.display()))?;
        ensure!(
            format.channels == 1 && format.sample_rate_hz == rate,
            "{}: expected mono {rate} Hz, got {} ch {} Hz",
            wav.display(),
            format.channels,
            format.sample_rate_hz
        );
        match embedder.embed(&samples)? {
            Some(e) if e.iter().all(|x| x.is_finite()) => {
                rows.push(e);
                labels.push(label.as_str());
            }
            _ => skipped += 1,
        }
    }
    let dim = embedder.dim();
    std::fs::write(&args.out, encode(&rows, dim)?)
        .with_context(|| format!("write {}", args.out.display()))?;
    let mut sidecar = args.out.clone().into_os_string();
    sidecar.push(".labels.txt");
    let mut f = std::fs::File::create(&sidecar)?;
    for label in &labels {
        writeln!(f, "{label}")?;
    }
    eprintln!(
        "{} embeddings (dim {dim}) written, {skipped} files skipped",
        rows.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_writes_magic_header_and_row_major_floats() {
        let bytes = encode(&[vec![1.0, 2.0], vec![3.0, -4.0]], 2).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 2);
        let floats: Vec<f32> = bytes[16..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0, -4.0]);
    }

    #[test]
    fn encode_rejects_ragged_rows() {
        assert!(encode(&[vec![1.0], vec![1.0, 2.0]], 1).is_err());
    }

    #[test]
    fn list_lines_resolve_against_the_base_and_skip_comments() {
        let items = parse_list(
            "# header\na.wav spk1\n\n b.wav  spk2 # note\n",
            Path::new("/d"),
        )
        .unwrap();
        assert_eq!(
            items,
            vec![
                (PathBuf::from("/d/a.wav"), "spk1".to_string()),
                (PathBuf::from("/d/b.wav"), "spk2".to_string()),
            ]
        );
        assert!(parse_list("only-one-field\n", Path::new("/d")).is_err());
    }
}
