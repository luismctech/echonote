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
//! sidecar `<out>.labels.txt` holds one class label per row. Both files
//! are written to temporary names first, then the labels and finally the
//! dump are renamed into place (see `write_pair`). Files that
//! cannot be read or have the wrong format are skipped and listed; clips
//! the embedder declines (shorter than its minimum) and non-finite
//! embeddings are skipped and counted separately.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use echo_diarize::EmbedderFamily;
use echo_domain::AudioFormat;

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

/// Embeddings gathered from a list, with what was skipped.
#[derive(Debug, Default)]
struct Collected<'a> {
    rows: Vec<Vec<f32>>,
    labels: Vec<&'a str>,
    /// Clips the embedder declined (too short).
    declined: usize,
    /// Clips whose embedding had a NaN or infinite value.
    non_finite: usize,
    /// Files that could not be read or have the wrong format, with why.
    bad_files: Vec<(PathBuf, String)>,
}

/// Embed every listed file. `load` reads a wav, `embed` runs the
/// embedder; both are parameters so the skip rules are testable without
/// a model.
///
/// # Errors
///
/// An embedder error, or an embedding whose length is not `dim` (named
/// with its file); a bad file is skipped, not an error.
fn collect<'a>(
    items: &'a [(PathBuf, String)],
    rate: u32,
    dim: usize,
    mut load: impl FnMut(&Path) -> Result<(Vec<f32>, AudioFormat)>,
    mut embed: impl FnMut(&[f32]) -> Result<Option<Vec<f32>>>,
) -> Result<Collected<'a>> {
    let mut c = Collected::default();
    for (wav, label) in items {
        let samples = match load(wav) {
            Ok((s, f)) if f.channels == 1 && f.sample_rate_hz == rate => s,
            Ok((_, f)) => {
                let why = format!(
                    "expected mono {rate} Hz, got {} ch {} Hz",
                    f.channels, f.sample_rate_hz
                );
                c.bad_files.push((wav.clone(), why));
                continue;
            }
            Err(e) => {
                c.bad_files.push((wav.clone(), format!("{e:#}")));
                continue;
            }
        };
        match embed(&samples).with_context(|| format!("embed {}", wav.display()))? {
            None => c.declined += 1,
            Some(e) if e.len() != dim => bail!(
                "{}: embedding has {} values, the embedder reports {dim}",
                wav.display(),
                e.len()
            ),
            Some(e) if !e.iter().all(|x| x.is_finite()) => c.non_finite += 1,
            Some(e) => {
                c.rows.push(e);
                c.labels.push(label.as_str());
            }
        }
    }
    Ok(c)
}

/// Temporary name next to `path` (`<path>.tmp`).
fn temp_path(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

/// Rename `tmp` over `path`, removing `tmp` when that fails.
fn rename(tmp: &Path, path: &Path) -> Result<()> {
    std::fs::rename(tmp, path).with_context(|| {
        let _ = std::fs::remove_file(tmp);
        format!("rename {} to {}", tmp.display(), path.display())
    })
}

/// Write the labels sidecar and the dump as a pair: both temporaries are
/// written before anything is renamed (a failed write replaces neither
/// file), then the labels are renamed and the dump last. Only a crash
/// between the two renames can leave new labels beside an old dump, and
/// `scripts/fit-plda.py` rejects a pair whose row counts disagree.
fn write_pair(labels: (&Path, &[u8]), dump: (&Path, &[u8])) -> Result<()> {
    let (labels_tmp, dump_tmp) = (temp_path(labels.0), temp_path(dump.0));
    let written = std::fs::write(&labels_tmp, labels.1)
        .with_context(|| format!("write {}", labels_tmp.display()))
        .and_then(|()| {
            std::fs::write(&dump_tmp, dump.1)
                .with_context(|| format!("write {}", dump_tmp.display()))
        });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&labels_tmp);
        let _ = std::fs::remove_file(&dump_tmp);
        return Err(e);
    }
    rename(&labels_tmp, labels.0).inspect_err(|_| {
        let _ = std::fs::remove_file(&dump_tmp);
    })?;
    rename(&dump_tmp, dump.0)
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
    let (rate, dim) = (embedder.sample_rate_hz(), embedder.dim());
    let c = collect(
        &items,
        rate,
        dim,
        |wav| super::load_wav_as_pcm(wav).with_context(|| format!("read {}", wav.display())),
        |samples| Ok(embedder.embed(samples)?),
    )?;
    for (wav, why) in &c.bad_files {
        eprintln!("skipped {}: {why}", wav.display());
    }
    let mut sidecar = args.out.clone().into_os_string();
    sidecar.push(".labels.txt");
    let labels: String = c.labels.iter().map(|l| format!("{l}\n")).collect();
    write_pair(
        (Path::new(&sidecar), labels.as_bytes()),
        (&args.out, &encode(&c.rows, dim)?),
    )?;
    eprintln!(
        "{} embeddings (dim {dim}) written; skipped: {} bad files, {} declined clips, {} non-finite embeddings",
        c.rows.len(),
        c.bad_files.len(),
        c.declined,
        c.non_finite
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

    fn mono(rate: u32) -> AudioFormat {
        AudioFormat {
            sample_rate_hz: rate,
            channels: 1,
        }
    }

    #[test]
    fn bad_files_are_skipped_and_non_finite_counted_apart_from_declined() {
        let items: Vec<(PathBuf, String)> = ["ok", "broken", "stereo", "short", "nan", "ok2"]
            .iter()
            .map(|n| (PathBuf::from(format!("/d/{n}.wav")), format!("spk_{n}")))
            .collect();
        let load = |p: &Path| -> Result<(Vec<f32>, AudioFormat)> {
            match p.file_stem().and_then(|s| s.to_str()) {
                Some("broken") => bail!("not a wav"),
                Some("stereo") => Ok((
                    vec![0.0; 4],
                    AudioFormat {
                        channels: 2,
                        ..mono(16_000)
                    },
                )),
                Some("short") => Ok((vec![0.0; 1], mono(16_000))),
                Some("nan") => Ok((vec![f32::NAN; 4], mono(16_000))),
                _ => Ok((vec![0.5; 4], mono(16_000))),
            }
        };
        let embed =
            |x: &[f32]| -> Result<Option<Vec<f32>>> { Ok((x.len() > 1).then(|| vec![x[0], 1.0])) };
        let c = collect(&items, 16_000, 2, load, embed).unwrap();
        assert_eq!(c.rows, vec![vec![0.5, 1.0], vec![0.5, 1.0]]);
        assert_eq!(c.labels, vec!["spk_ok", "spk_ok2"]);
        assert_eq!(c.declined, 1);
        assert_eq!(c.non_finite, 1);
        let bad: Vec<&Path> = c.bad_files.iter().map(|(p, _)| p.as_path()).collect();
        assert_eq!(
            bad,
            vec![Path::new("/d/broken.wav"), Path::new("/d/stereo.wav")]
        );
    }

    #[test]
    fn an_embedding_of_the_wrong_dimension_is_an_error_naming_the_file() {
        let items = vec![(PathBuf::from("/d/a.wav"), "x".to_string())];
        let load =
            |_: &Path| -> Result<(Vec<f32>, AudioFormat)> { Ok((vec![0.1; 4], mono(16_000))) };
        let embed = |_: &[f32]| -> Result<Option<Vec<f32>>> { Ok(Some(vec![1.0; 3])) };
        let err = collect(&items, 16_000, 2, load, embed)
            .unwrap_err()
            .to_string();
        assert!(err.contains("/d/a.wav"), "{err}");
    }

    #[test]
    fn a_failed_dump_write_leaves_the_old_labels_in_place() {
        let dir = std::env::temp_dir().join(format!("echo-embed-pair-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let labels = dir.join("dump.bin.labels.txt");
        std::fs::write(&labels, b"old\n").unwrap();
        // The dump's directory does not exist, so its temporary write fails.
        let dump = dir.join("no-such-dir").join("dump.bin");
        let err = write_pair((&labels, b"new\n"), (&dump, b"x")).unwrap_err();
        assert!(format!("{err:#}").contains("no-such-dir"), "{err:#}");
        assert_eq!(std::fs::read(&labels).unwrap(), b"old\n");
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("dump.bin.labels.txt")]);
        std::fs::remove_file(&labels).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_pair_write_replaces_both_files_and_leaves_no_temporary() {
        let dir = std::env::temp_dir().join(format!("echo-embed-both-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (labels, dump) = (dir.join("d.bin.labels.txt"), dir.join("d.bin"));
        std::fs::write(&labels, b"old").unwrap();
        std::fs::write(&dump, b"old").unwrap();
        write_pair((&labels, b"a\n"), (&dump, b"rows")).unwrap();
        assert_eq!(std::fs::read(&labels).unwrap(), b"a\n");
        assert_eq!(std::fs::read(&dump).unwrap(), b"rows");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
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
