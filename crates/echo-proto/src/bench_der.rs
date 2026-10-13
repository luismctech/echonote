//! `bench der`: diarization error rate of the app's streaming
//! diarization over a manifest of `<wav> <rttm>` pairs.
//!
//! Each recording is fed through the real [`StreamingPipeline`] (file
//! capture, rubato resampler, Silero VAD with speech-aligned chunks,
//! [`OnlineDiarizer`] with the app's per-embedder preset) with a no-op
//! transcriber, so chunk boundaries and diarizer calls are exactly the
//! ones a live session makes. A recording decorator around the diarizer
//! captures every chunk's speaker spans; `Chunk` events supply each
//! chunk's session offset and `SpeakersMerged` the merge-only offline
//! refinement. A re-segmenting refine (`--refine-mode resegment`) is
//! captured from the diarizer's span relabel plan, because the no-op
//! transcriber emits no segments for `SpeakersReassigned` to move. Both
//! are applied to the spans before scoring.
//!
//! The hypothesis is the diarizer's span timeline. The app persists one
//! speaker per Whisper segment (dominant span), so app-visible DER also
//! depends on ASR segmentation; this harness isolates the diarizer.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use serde::Serialize;

use echo_app::use_cases::streaming::DEFAULT_REFINE_TIMEOUT;
use echo_app::StreamingPipeline;
use echo_audio::{RubatoResamplerAdapter, SileroVad};
use echo_diarize::{
    parse_rttm, score_der, write_rttm, DerConfig, DerReport, EmbedderFamily, OfflineRefineMode,
    OnlineDiarizer, Plda, PyannoteSegmenter, RttmSegment, VbxConfig,
};
use echo_domain::{
    AudioCapture, AudioFormat, AudioFrame, AudioSource, AudioStream, CaptureSpec, ChunkContext,
    DeviceInfo, Diarizer, DomainError, RefineBudget, RefineOutcome, Resampler, Sample, SpanRef,
    Speaker, SpeakerId, SpeakerSpan, StreamingOptions, TranscribeOptions, Transcriber, Transcript,
    TranscriptEvent,
};

/// Diarizer sample rate: spans are in 16 kHz samples.
const SPAN_RATE_HZ: f64 = 16_000.0;
/// Device frames per capture callback (20 ms at 16 kHz), like a live mic.
const FRAME_MS: u32 = 20;
/// Extra stop-time wait per `--vbx-grid` setting.
const GRID_WAIT_PER_SPEC: std::time::Duration = std::time::Duration::from_secs(60);

/// Options of `echo-proto bench der`.
#[derive(Debug, Clone)]
pub(crate) struct DerArgs {
    /// Manifest file: one `<wav> <rttm>` pair per line.
    pub manifest: PathBuf,
    /// Speaker embedder model file.
    pub embedder: PathBuf,
    /// Catalog id of the embedder (`embedder-*`); `None` guesses the
    /// family from the file name.
    pub embedder_id: Option<String>,
    /// Optional pyannote segmenter model (whole-chunk embedding when absent).
    pub segmenter: Option<PathBuf>,
    /// Override of the preset similarity threshold.
    pub threshold: Option<f32>,
    /// Override of the preset online speaker cap.
    pub max_speakers: Option<usize>,
    /// Override of the preset offline refinement merge threshold.
    pub refine_threshold: Option<f32>,
    /// End-of-session refinement; `None` uses the embedder preset.
    pub refine_mode: Option<OfflineRefineMode>,
    /// Override of the preset VBx settings (a [`parse_vbx_spec`] string).
    pub vbx: Option<String>,
    /// Log dense window embeddings for the re-segmentation.
    pub dense_windows: bool,
    /// PLDA model file replacing the family's bundled one.
    pub plda: Option<PathBuf>,
    /// Extra VBx settings, one spec per line, each scored on the same
    /// diarized sessions besides the main run (calibration grids).
    pub vbx_grid: Option<PathBuf>,
    /// Silero VAD model; `None` runs fixed chunks behind the RMS gate.
    pub vad: Option<PathBuf>,
    /// Speech-aligned chunk bounds (ignored without a VAD).
    pub min_chunk_ms: u32,
    /// Speech-aligned chunk bounds (ignored without a VAD).
    pub max_chunk_ms: u32,
    /// Fixed chunk length when no VAD is used.
    pub chunk_ms: u32,
    /// Scoring collar in seconds on each side of reference boundaries.
    pub collar: f64,
    /// Exclude reference overlap from scoring.
    pub skip_overlap: bool,
    /// Directory to write hypothesis RTTMs into.
    pub hyp_dir: Option<PathBuf>,
    /// Print a JSON report instead of the text table.
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct FileReport {
    name: String,
    audio_s: f64,
    der: f64,
    missed: f64,
    false_alarm: f64,
    confusion: f64,
    scored_s: f64,
    ref_speakers: usize,
    hyp_speakers: usize,
    chunks: usize,
    merges: usize,
    /// Spans whose speaker the re-segmenting refine changed.
    relabelled_spans: usize,
    /// Wall time of the end-of-session refine, in milliseconds.
    refine_ms: f64,
    elapsed_s: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    embedder: String,
    family: String,
    similarity_threshold: f32,
    max_speakers: usize,
    segmenter: Option<String>,
    refine: bool,
    /// `Off`, `MergeOnly` or `Resegment`.
    refine_mode: String,
    /// Offline merge threshold; `None` when the refinement is skipped.
    refine_threshold: Option<f32>,
    vad: Option<String>,
    min_chunk_ms: u32,
    max_chunk_ms: u32,
    collar: f64,
    skip_overlap: bool,
    files: Vec<FileReport>,
    pooled: PooledReport,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    grid: Vec<GridReport>,
}

/// Pooled score of one `--vbx-grid` setting.
#[derive(Debug, Serialize)]
struct GridReport {
    spec: String,
    der: f64,
    missed: f64,
    false_alarm: f64,
    confusion: f64,
    scored_s: f64,
    mean_hyp_speakers: f64,
    mean_ref_speakers: f64,
    exact_speaker_count: usize,
    /// Files whose re-segmentation failed (scored with the online labels).
    failed: usize,
    /// Slowest re-segmentation of the setting, in milliseconds.
    max_refine_ms: f64,
    /// Per-file DER, in manifest order.
    file_der: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct PooledReport {
    der: f64,
    missed: f64,
    false_alarm: f64,
    confusion: f64,
    scored_s: f64,
    /// Mean of |hyp_speakers - ref_speakers| over files.
    mean_abs_speaker_count_error: f64,
}

/// Parse `--refine-mode`: `off`, `merge-only` (or `merge`), `resegment`
/// (or `reseg`).
///
/// # Errors
///
/// Returns a message naming the accepted values.
pub(crate) fn parse_refine_mode(raw: &str) -> Result<OfflineRefineMode, String> {
    OfflineRefineMode::parse(raw)
        .ok_or_else(|| format!("`{raw}` is not a refine mode (off, merge-only, resegment)"))
}

/// Parse `--max-speakers`: a speaker cap of at least one.
///
/// # Errors
///
/// Returns a message for non-integers and for `0`.
pub(crate) fn parse_max_speakers(raw: &str) -> Result<usize, String> {
    let cap: usize = raw
        .parse()
        .map_err(|e| format!("`{raw}` is not a speaker count: {e}"))?;
    if cap == 0 {
        return Err("the speaker cap must be at least 1".to_string());
    }
    Ok(cap)
}

/// Parse a cosine-similarity threshold (`--refine-threshold`): a finite
/// value in `-1..=1`.
///
/// # Errors
///
/// Returns a message for non-numbers, NaN, infinities and values outside
/// `-1..=1`.
pub(crate) fn parse_cosine_threshold(raw: &str) -> Result<f32, String> {
    let value: f32 = raw
        .parse()
        .map_err(|e| format!("`{raw}` is not a number: {e}"))?;
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(format!(
            "`{raw}` is not a cosine similarity (a finite value in -1..=1)"
        ));
    }
    Ok(value)
}

/// One VBx setting: the configuration and the AHC cut of its
/// initialisation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VbxSpec {
    /// The spec string, as given.
    pub label: String,
    pub config: VbxConfig,
    /// Fixed AHC cut (or the calibration fallback).
    pub init_threshold: f32,
}

/// Parse a VBx spec on top of `base`: comma-separated `key=value` pairs
/// with keys `fa`, `fb`, `loop` (loopP), `init` (`cal:<offset>` for the
/// calibrated AHC cut, or a fixed cosine similarity), `lda`, `smooth`,
/// `maxspk` and `iters`. An empty spec is `base` unchanged.
///
/// # Errors
///
/// Returns a message for unknown keys and out-of-range values.
pub(crate) fn parse_vbx_spec(
    base: VbxConfig,
    base_threshold: f32,
    spec: &str,
) -> Result<VbxSpec, String> {
    let mut config = base;
    let mut init_threshold = base_threshold;
    let num = |k: &str, v: &str| -> Result<f64, String> {
        let x: f64 = v
            .parse()
            .map_err(|e| format!("`{k}={v}` is not a number: {e}"))?;
        if x.is_finite() {
            Ok(x)
        } else {
            Err(format!("`{k}={v}` must be finite"))
        }
    };
    let count = |k: &str, v: &str| -> Result<usize, String> {
        match v.parse::<usize>() {
            Ok(n) if n > 0 => Ok(n),
            _ => Err(format!("`{k}={v}` must be a positive integer")),
        }
    };
    for pair in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once('=')
            .ok_or_else(|| format!("`{pair}` is not key=value"))?;
        match k.trim() {
            "fa" => config.params.fa = num(k, v)?,
            "fb" => config.params.fb = num(k, v)?,
            "loop" => config.params.loop_p = num(k, v)?,
            "smooth" => config.init_smoothing = num(k, v)?,
            "lda" => config.lda_dim = count(k, v)?,
            "maxspk" => config.max_speakers = count(k, v)?,
            "iters" => config.params.max_iters = count(k, v)?,
            "init" => match v.strip_prefix("cal:") {
                Some(offset) => config.init_offset = Some(num(k, offset)? as f32),
                None => {
                    config.init_offset = None;
                    init_threshold = parse_cosine_threshold(v)?;
                }
            },
            other => return Err(format!("unknown VBx key `{other}`")),
        }
    }
    if !(config.params.fa > 0.0 && config.params.fb > 0.0) {
        return Err("fa and fb must be positive".to_string());
    }
    if !(0.0..=1.0).contains(&config.params.loop_p) {
        return Err("loop must be within 0..=1".to_string());
    }
    Ok(VbxSpec {
        label: spec.trim().to_string(),
        config,
        init_threshold,
    })
}

/// Embedder family to benchmark: the catalog id when given (as the app
/// dispatches), else the file-name fallback for `embedder`.
fn embedder_family(embedder: &Path, embedder_id: Option<&str>) -> Result<EmbedderFamily> {
    match embedder_id {
        Some(id) => EmbedderFamily::from_model_id(id).with_context(|| {
            let known: Vec<&str> = EmbedderFamily::ALL.iter().map(|f| f.model_id()).collect();
            format!("unknown embedder id `{id}` (known: {})", known.join(", "))
        }),
        None => Ok(EmbedderFamily::from_path(embedder)),
    }
}

/// Entry point of `echo-proto bench der`.
pub(crate) async fn run(args: DerArgs) -> Result<()> {
    let pairs = read_manifest(&args.manifest)?;
    ensure!(
        !pairs.is_empty(),
        "manifest {} lists no pairs",
        args.manifest.display()
    );
    ensure!(
        args.embedder.exists(),
        "embedder {} not found",
        args.embedder.display()
    );
    let family = embedder_family(&args.embedder, args.embedder_id.as_deref())?;
    let mut cluster = family.cluster_config(args.threshold);
    if let Some(max_speakers) = args.max_speakers {
        cluster.max_speakers = max_speakers;
    }
    let mut refine = family.refine_config();
    let mut resegment_threshold = family.resegment_threshold();
    if let Some(threshold) = args.refine_threshold {
        refine.merge_threshold = threshold;
        resegment_threshold = threshold;
    }
    let main_vbx = parse_vbx_spec(
        family.vbx_config(),
        resegment_threshold,
        args.vbx.as_deref().unwrap_or(""),
    )
    .map_err(anyhow::Error::msg)?;
    let grid: Vec<VbxSpec> = match &args.vbx_grid {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .filter(|l| !l.is_empty())
            .map(|l| parse_vbx_spec(main_vbx.config, main_vbx.init_threshold, l))
            .collect::<Result<_, _>>()
            .map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    let plda = match &args.plda {
        Some(path) => Some(Arc::new(
            Plda::from_bytes(
                &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
            )
            .with_context(|| format!("parse {}", path.display()))?,
        )),
        None => family.plda(),
    };
    let refine_mode = args.refine_mode.unwrap_or_else(|| family.refine_mode());
    let refine_on = refine_mode != OfflineRefineMode::Off;
    ensure!(
        grid.is_empty() || (refine_on && plda.is_some()),
        "--vbx-grid needs a refine mode other than off and an embedder with a PLDA model"
    );
    let mut grid_pooled = vec![DerReport::default(); grid.len()];
    let mut grid_stats = vec![(0_usize, 0_usize, 0.0_f64, Vec::new()); grid.len()];
    let der_config = DerConfig {
        collar: args.collar,
        skip_overlap: args.skip_overlap,
    };

    let mut files = Vec::with_capacity(pairs.len());
    let mut pooled = DerReport::default();
    let names = unique_names(&pairs);
    for ((wav, rttm), name) in pairs.iter().zip(names) {
        let reference = parse_rttm(
            &std::fs::read_to_string(rttm).with_context(|| format!("read {}", rttm.display()))?,
        )
        .with_context(|| format!("parse {}", rttm.display()))?;
        let reference = reference_for(reference, &file_stem(wav))
            .with_context(|| format!("select reference in {}", rttm.display()))?;

        let mut diarizer = OnlineDiarizer::new(family.load(&args.embedder)?, cluster)
            .with_offline_refinement_config(refine)
            .with_resegment_threshold(main_vbx.init_threshold)
            .with_vbx_config(main_vbx.config)
            .with_dense_windows(args.dense_windows)
            .with_refine_mode(refine_mode);
        if let Some(plda) = &plda {
            diarizer = diarizer.with_plda(Arc::clone(plda));
        }
        if let Some(seg) = &args.segmenter {
            diarizer = diarizer.with_segmenter(Box::new(
                PyannoteSegmenter::new(seg).context("load pyannote segmenter")?,
            ));
        }
        let started = Instant::now();
        let run = diarize_file(wav, Box::new(diarizer), &grid, &args).await?;
        let elapsed_s = started.elapsed().as_secs_f64();

        for (i, result) in run.grid.iter().enumerate() {
            let (map, failed) = match &result.relabel {
                Some(m) => (m.clone(), 0),
                None => (HashMap::new(), 1),
            };
            let (chunks, _) = relabel_spans(run.chunks.clone(), &map);
            let hyp = spans_to_segments(&name, &chunks, &HashMap::new());
            let r = score_der(&reference, &hyp, &der_config);
            grid_pooled[i] += r;
            let stats = &mut grid_stats[i];
            stats.0 += usize::from(r.hyp_speakers == r.ref_speakers);
            stats.1 += failed;
            stats.2 = stats.2.max(result.elapsed_ms);
            stats.3.push(r.der());
        }
        let (chunks, relabelled_spans) = relabel_spans(run.chunks, &run.relabel);
        let hypothesis = spans_to_segments(&name, &chunks, &resolve_merges(&run.merges));
        if let Some(dir) = &args.hyp_dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(format!("{name}.rttm")), write_rttm(&hypothesis))?;
        }
        let r = score_der(&reference, &hypothesis, &der_config);
        pooled += r;
        let file = FileReport {
            name,
            audio_s: run.audio_s,
            der: r.der(),
            missed: r.missed,
            false_alarm: r.false_alarm,
            confusion: r.confusion,
            scored_s: r.total,
            ref_speakers: r.ref_speakers,
            hyp_speakers: r.hyp_speakers,
            chunks: chunks.len(),
            merges: run.merges.len(),
            relabelled_spans,
            refine_ms: run.refine_ms,
            elapsed_s,
        };
        if !args.json {
            eprintln!(
                "{:<16} DER {:>6.2}%  miss {:>6.1}s  FA {:>6.1}s  conf {:>6.1}s  spk {}/{}  chunks {}  {:.1}s",
                file.name,
                file.der * 100.0,
                file.missed,
                file.false_alarm,
                file.confusion,
                file.hyp_speakers,
                file.ref_speakers,
                file.chunks,
                file.elapsed_s
            );
        }
        files.push(file);
    }

    let count_err = files
        .iter()
        .map(|f| f.hyp_speakers.abs_diff(f.ref_speakers) as f64)
        .sum::<f64>()
        / files.len() as f64;
    let report = Report {
        embedder: args.embedder.display().to_string(),
        family: format!("{family:?}"),
        similarity_threshold: cluster.similarity_threshold,
        max_speakers: cluster.max_speakers,
        segmenter: args.segmenter.as_ref().map(|p| p.display().to_string()),
        refine: refine_on,
        refine_mode: format!("{refine_mode:?}"),
        refine_threshold: refine_on.then_some(refine.merge_threshold),
        vad: args.vad.as_ref().map(|p| p.display().to_string()),
        min_chunk_ms: args.min_chunk_ms,
        max_chunk_ms: args.max_chunk_ms,
        collar: args.collar,
        skip_overlap: args.skip_overlap,
        pooled: PooledReport {
            der: pooled.der(),
            missed: pooled.missed,
            false_alarm: pooled.false_alarm,
            confusion: pooled.confusion,
            scored_s: pooled.total,
            mean_abs_speaker_count_error: count_err,
        },
        grid: grid
            .iter()
            .zip(grid_pooled)
            .zip(grid_stats)
            .map(
                |((spec, p), (exact, failed, max_ms, file_der))| GridReport {
                    spec: spec.label.clone(),
                    der: p.der(),
                    missed: p.missed,
                    false_alarm: p.false_alarm,
                    confusion: p.confusion,
                    scored_s: p.total,
                    mean_hyp_speakers: p.hyp_speakers as f64 / files.len() as f64,
                    mean_ref_speakers: p.ref_speakers as f64 / files.len() as f64,
                    exact_speaker_count: exact,
                    failed,
                    max_refine_ms: max_ms,
                    file_der,
                },
            )
            .collect(),
        files,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let p = &report.pooled;
        println!(
            "pooled ({} files, {:.0} s scored): DER {:.2}%  miss {:.2}%  FA {:.2}%  conf {:.2}%  |Δspk| {:.2}",
            report.files.len(),
            p.scored_s,
            p.der * 100.0,
            100.0 * p.missed / p.scored_s.max(f64::MIN_POSITIVE),
            100.0 * p.false_alarm / p.scored_s.max(f64::MIN_POSITIVE),
            100.0 * p.confusion / p.scored_s.max(f64::MIN_POSITIVE),
            p.mean_abs_speaker_count_error
        );
        for g in &report.grid {
            println!(
                "grid [{}]: DER {:.2}%  spk {:.2}/{:.2}  exact {}  failed {}  max refine {:.0} ms",
                g.spec,
                g.der * 100.0,
                g.mean_hyp_speakers,
                g.mean_ref_speakers,
                g.exact_speaker_count,
                g.failed,
                g.max_refine_ms
            );
        }
    }
    Ok(())
}

/// Parse a manifest: one `<wav> <rttm>` pair per non-blank line, `#`
/// starts a comment, relative paths resolve against the manifest's
/// directory.
fn read_manifest(path: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_manifest(&text, path.parent().unwrap_or(Path::new(".")))
}

fn parse_manifest(text: &str, base: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 {
            bail!("manifest line {}: expected `<wav> <rttm>`", i + 1);
        }
        pairs.push((base.join(fields[0]), base.join(fields[1])));
    }
    Ok(pairs)
}

/// File stem of `path`, the RTTM file id a recording is matched by.
fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("recording")
        .to_string()
}

/// One report name per pair: the wav stem, suffixed `_2`, `_3`, … when
/// two recordings share a stem, so hypothesis RTTMs never overwrite
/// each other.
fn unique_names(pairs: &[(PathBuf, PathBuf)]) -> Vec<String> {
    let mut used: HashSet<String> = HashSet::new();
    pairs
        .iter()
        .map(|(wav, _)| {
            let stem = file_stem(wav);
            let mut name = stem.clone();
            let mut n = 1;
            // A generated `stem_n` may equal a later (or earlier) real
            // stem, so keep counting until the name is unused.
            while used.contains(&name) {
                n += 1;
                name = format!("{stem}_{n}");
            }
            used.insert(name.clone());
            name
        })
        .collect()
}

/// Reference turns of recording `file_id`. An RTTM that lists several
/// recordings is filtered by file id (and must contain `file_id`); a
/// single-recording RTTM must carry `file_id` too, so a manifest line
/// that pairs a wav with the wrong RTTM fails instead of scoring
/// against another recording's turns.
fn reference_for(segments: Vec<RttmSegment>, file_id: &str) -> Result<Vec<RttmSegment>> {
    let first = segments.first().map(|s| s.file_id.clone());
    if let Some(only) = first.filter(|f| segments.iter().all(|s| &s.file_id == f)) {
        ensure!(
            only == file_id,
            "single-recording RTTM is for file id `{only}`, not `{file_id}` (the wav stem)"
        );
        return Ok(segments);
    }
    let selected: Vec<RttmSegment> = segments
        .into_iter()
        .filter(|s| s.file_id == file_id)
        .collect();
    ensure!(
        !selected.is_empty(),
        "multi-recording RTTM has no turns for file id `{file_id}`"
    );
    Ok(selected)
}

/// Follow `(absorbed, surviving)` chains to each absorbed speaker's
/// final survivor.
fn resolve_merges(merges: &[(SpeakerId, SpeakerId)]) -> HashMap<SpeakerId, SpeakerId> {
    let direct: HashMap<SpeakerId, SpeakerId> = merges.iter().copied().collect();
    direct
        .keys()
        .map(|&start| {
            let mut id = start;
            // Bounded walk: a cycle would be a refine bug, not a hang.
            for _ in 0..=direct.len() {
                match direct.get(&id) {
                    Some(&next) if next != id => id = next,
                    _ => break,
                }
            }
            (start, id)
        })
        .collect()
}

/// Apply a re-segmentation plan to the recorded spans. Returns the
/// `(offset ms, spans)` chunks and how many spans changed speaker.
fn relabel_spans(
    chunks: Vec<RecordedChunk>,
    relabel: &HashMap<SpanRef, SpeakerId>,
) -> (Vec<(u32, Vec<SpeakerSpan>)>, usize) {
    let mut changed = 0;
    let out = chunks
        .into_iter()
        .map(|c| {
            let spans = c
                .spans
                .into_iter()
                .enumerate()
                .map(|(i, mut span)| {
                    let at = SpanRef {
                        chunk_index: c.chunk_index,
                        span_index: u16::try_from(i).unwrap_or(u16::MAX),
                    };
                    if let Some(&id) = relabel.get(&at) {
                        changed += usize::from(id != span.speaker_id);
                        span.speaker_id = id;
                    }
                    span
                })
                .collect();
            (c.offset_ms, spans)
        })
        .collect();
    (out, changed)
}

/// Turn per-chunk spans into session-time RTTM segments, relabelled
/// through `merges`.
fn spans_to_segments(
    file_id: &str,
    chunks: &[(u32, Vec<SpeakerSpan>)],
    merges: &HashMap<SpeakerId, SpeakerId>,
) -> Vec<RttmSegment> {
    let mut segments = Vec::new();
    for (offset_ms, spans) in chunks {
        let offset = f64::from(*offset_ms) / 1_000.0;
        for span in spans {
            let id = merges
                .get(&span.speaker_id)
                .copied()
                .unwrap_or(span.speaker_id);
            segments.push(RttmSegment {
                file_id: file_id.to_string(),
                speaker: id.to_string(),
                start: offset + span.start_sample as f64 / SPAN_RATE_HZ,
                duration: span.len_samples() as f64 / SPAN_RATE_HZ,
            });
        }
    }
    segments
}

/// One diarized chunk as the bench saw it.
#[derive(Clone)]
struct RecordedChunk {
    chunk_index: u32,
    offset_ms: u32,
    spans: Vec<SpeakerSpan>,
}

struct FileRun {
    audio_s: f64,
    /// Every diarized chunk, in order.
    chunks: Vec<RecordedChunk>,
    merges: Vec<(SpeakerId, SpeakerId)>,
    /// Span relabelling of a re-segmenting refine (empty otherwise).
    relabel: HashMap<SpanRef, SpeakerId>,
    /// Wall time of the main refine, in milliseconds.
    refine_ms: f64,
    /// One result per `--vbx-grid` setting.
    grid: Vec<GridResult>,
}

/// Re-segmentation of one session with one grid setting.
#[derive(Clone)]
struct GridResult {
    /// Span relabelling (empty when unchanged); `None` when it failed.
    relabel: Option<HashMap<SpanRef, SpeakerId>>,
    elapsed_ms: f64,
}

/// A diarizer the bench can also re-segment with other VBx settings.
trait GridDiarizer: Diarizer {
    /// Re-segment the session once per spec. The default (a diarizer
    /// without that ability) returns nothing.
    fn resegment_grid(&self, _specs: &[VbxSpec]) -> Vec<GridResult> {
        Vec::new()
    }
}

impl GridDiarizer for OnlineDiarizer {
    fn resegment_grid(&self, specs: &[VbxSpec]) -> Vec<GridResult> {
        specs
            .iter()
            .map(|spec| {
                let started = Instant::now();
                let budget = RefineBudget {
                    deadline: Instant::now() + std::time::Duration::from_secs(60),
                };
                let relabel = match self.resegment_with(spec.config, spec.init_threshold, budget) {
                    Ok(RefineOutcome::Relabel(plan)) => Some(plan.spans.into_iter().collect()),
                    Ok(_) => Some(HashMap::new()),
                    Err(_) => None,
                };
                GridResult {
                    relabel,
                    elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                }
            })
            .collect()
    }
}

async fn diarize_file(
    wav: &Path,
    diarizer: Box<dyn GridDiarizer>,
    grid: &[VbxSpec],
    args: &DerArgs,
) -> Result<FileRun> {
    let (samples, format) =
        super::load_wav_as_pcm(wav).with_context(|| format!("read {}", wav.display()))?;
    let audio_s = samples.len() as f64
        / f64::from(format.channels.max(1))
        / f64::from(format.sample_rate_hz.max(1));
    let span_log: SpanLog = Arc::default();
    let plan_slot: PlanSlot = Arc::default();
    let refine_slot: RefineSlot = Arc::default();
    let capture: Arc<dyn AudioCapture> = Arc::new(FileCapture {
        samples: Arc::new(samples),
        format,
    });
    let resampler: Arc<dyn Resampler> = Arc::new(RubatoResamplerAdapter::new());
    // A calibration grid re-segments the session once per setting inside
    // the stop-time refine; the pipeline must wait for all of them. The
    // main refine keeps the app's deadline (`RecordingDiarizer`).
    let refine_wait = if grid.is_empty() {
        DEFAULT_REFINE_TIMEOUT
    } else {
        DEFAULT_REFINE_TIMEOUT + GRID_WAIT_PER_SPEC * u32::try_from(grid.len()).unwrap_or(u32::MAX)
    };
    let mut pipeline = StreamingPipeline::new(capture, resampler, Arc::new(NullTranscriber))
        .with_refine_timeout(refine_wait)
        .with_diarizer(Box::new(RecordingDiarizer {
            inner: diarizer,
            log: Arc::clone(&span_log),
            plan: Arc::clone(&plan_slot),
            grid: grid.to_vec(),
            refine: Arc::clone(&refine_slot),
        }));
    if let Some(vad) = &args.vad {
        pipeline = pipeline.with_vad(Box::new(
            SileroVad::for_meetings(vad).context("load Silero VAD")?,
        ));
    }
    let options = StreamingOptions {
        chunk_ms: args.chunk_ms,
        speech_aligned_chunks: args.vad.is_some(),
        min_chunk_ms: args.min_chunk_ms,
        max_chunk_ms: args.max_chunk_ms,
        ..StreamingOptions::default()
    };
    let mut handle = pipeline.start(options).await?;
    let mut offsets = Vec::new();
    let mut merges = Vec::new();
    while let Some(event) = handle.next_event().await {
        match event {
            TranscriptEvent::Chunk {
                chunk_index,
                offset_ms,
                ..
            } => offsets.push((chunk_index, offset_ms)),
            TranscriptEvent::SpeakersMerged { merges: m, .. } => {
                merges.extend(m.into_iter().map(|x| (x.absorbed, x.surviving)));
            }
            TranscriptEvent::Failed { message, .. } => bail!("pipeline failed: {message}"),
            TranscriptEvent::Stopped { .. } => break,
            _ => {}
        }
    }
    let spans = std::mem::take(&mut *span_log.lock().expect("span log poisoned"));
    ensure!(
        spans.len() == offsets.len(),
        "{} diarizer calls but {} chunk events",
        spans.len(),
        offsets.len()
    );
    let mut chunks = Vec::with_capacity(spans.len());
    for ((chunk_index, offset_ms), (logged_index, spans)) in offsets.into_iter().zip(spans) {
        ensure!(
            chunk_index == logged_index,
            "diarizer saw chunk {logged_index} where the pipeline emitted chunk {chunk_index}"
        );
        chunks.push(RecordedChunk {
            chunk_index,
            offset_ms,
            spans,
        });
    }
    let relabel = plan_slot
        .lock()
        .expect("plan slot poisoned")
        .take()
        .unwrap_or_default();
    let (refine_ms, grid) = std::mem::take(&mut *refine_slot.lock().expect("refine slot poisoned"));
    Ok(FileRun {
        audio_s,
        chunks,
        merges,
        relabel,
        refine_ms,
        grid,
    })
}

/// `(chunk index, spans)` per diarizer call, in call order.
type SpanLog = Arc<Mutex<Vec<(u32, Vec<SpeakerSpan>)>>>;
/// Span relabelling captured from the diarizer's refine plan.
type PlanSlot = Arc<Mutex<Option<HashMap<SpanRef, SpeakerId>>>>;
/// Main refine wall time (ms) and the grid re-segmentations.
type RefineSlot = Arc<Mutex<(f64, Vec<GridResult>)>>;

/// Forwards to the real diarizer, records each chunk's spans and keeps
/// a copy of a re-segmenting refine plan returned within its deadline.
struct RecordingDiarizer {
    inner: Box<dyn GridDiarizer>,
    log: SpanLog,
    plan: PlanSlot,
    /// Extra VBx settings to re-segment the session with at stop.
    grid: Vec<VbxSpec>,
    refine: RefineSlot,
}

#[async_trait]
impl Diarizer for RecordingDiarizer {
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    async fn assign(&mut self, samples: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
        self.inner.assign(samples).await
    }

    async fn assign_spans(&mut self, samples: &[Sample]) -> Result<Vec<SpeakerSpan>, DomainError> {
        self.inner.assign_spans(samples).await
    }

    async fn assign_spans_at(
        &mut self,
        samples: &[Sample],
        ctx: ChunkContext,
    ) -> Result<Vec<SpeakerSpan>, DomainError> {
        // Errors are logged as an empty chunk, matching the pipeline,
        // which treats a failed diarization as unlabelled.
        let result = self.inner.assign_spans_at(samples, ctx).await;
        let spans = result.as_ref().map(Clone::clone).unwrap_or_default();
        self.log
            .lock()
            .expect("span log poisoned")
            .push((ctx.chunk_index, spans));
        result
    }

    fn speakers(&self) -> Vec<Speaker> {
        self.inner.speakers()
    }

    fn rename(&mut self, id: SpeakerId, label: &str) -> Result<bool, DomainError> {
        self.inner.rename(id, label)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)> {
        self.inner.refine_offline()
    }

    fn refine_plan(&self, budget: RefineBudget) -> Result<RefineOutcome, DomainError> {
        let started = Instant::now();
        // With a grid the pipeline waits longer (see `diarize_file`), so
        // the main refine gets the app's deadline back here.
        let budget = RefineBudget {
            deadline: budget.deadline.min(started + DEFAULT_REFINE_TIMEOUT),
        };
        let outcome = self.inner.refine_plan(budget);
        let refine_ms = started.elapsed().as_secs_f64() * 1_000.0;
        // The port forbids a late plan. Enforce it here so the pipeline
        // discards a refine that missed its deadline (the online result
        // stands) and the bench never scores a plan the app would not
        // have applied.
        let outcome = outcome.and_then(|o| budget.check().map(|()| o));
        if let Ok(RefineOutcome::Relabel(plan)) = &outcome {
            *self.plan.lock().expect("plan slot poisoned") =
                Some(plan.spans.iter().copied().collect());
        }
        // Grid settings run after the main refine and outside its
        // deadline: they only feed the calibration report.
        let grid = self.inner.resegment_grid(&self.grid);
        *self.refine.lock().expect("refine slot poisoned") = (refine_ms, grid);
        outcome
    }
}

/// Transcriber that returns no text: the bench only needs the
/// pipeline's chunking and diarizer calls.
struct NullTranscriber;

#[async_trait]
impl Transcriber for NullTranscriber {
    async fn transcribe(
        &self,
        samples: &[Sample],
        _options: &TranscribeOptions,
    ) -> Result<Transcript, DomainError> {
        Ok(Transcript {
            segments: Vec::new(),
            language: None,
            duration_ms: (samples.len() as f64 * 1_000.0 / SPAN_RATE_HZ) as u32,
        })
    }
}

/// Capture that serves a decoded file in 20 ms frames as fast as the
/// pipeline pulls them, then ends the stream.
struct FileCapture {
    samples: Arc<Vec<f32>>,
    format: AudioFormat,
}

#[async_trait]
impl AudioCapture for FileCapture {
    async fn list_devices(&self, _source: AudioSource) -> Result<Vec<DeviceInfo>, DomainError> {
        Ok(Vec::new())
    }

    async fn start(&self, _spec: CaptureSpec) -> Result<Box<dyn AudioStream>, DomainError> {
        let frame = (self.format.sample_rate_hz * FRAME_MS / 1_000) as usize
            * usize::from(self.format.channels.max(1));
        Ok(Box::new(FileStream {
            samples: Arc::clone(&self.samples),
            format: self.format,
            frame: frame.max(1),
            pos: 0,
        }))
    }
}

struct FileStream {
    samples: Arc<Vec<f32>>,
    format: AudioFormat,
    frame: usize,
    pos: usize,
}

#[async_trait]
impl AudioStream for FileStream {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_frame(&mut self) -> Option<AudioFrame> {
        // Yield so the pipeline task interleaves like a live callback.
        tokio::task::yield_now().await;
        if self.pos >= self.samples.len() {
            return None;
        }
        let end = (self.pos + self.frame).min(self.samples.len());
        let samples = self.samples[self.pos..end].to_vec();
        self.pos = end;
        Some(AudioFrame {
            samples,
            format: self.format,
            captured_at_ns: 0,
        })
    }

    async fn stop(&mut self) -> Result<(), DomainError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_resolves_relative_paths_and_skips_comments() {
        let pairs = parse_manifest(
            "# eval set\n\na.wav a.rttm\n/abs/b.wav  b.rttm # trailing\n",
            Path::new("/data"),
        )
        .unwrap();
        assert_eq!(
            pairs,
            vec![
                (PathBuf::from("/data/a.wav"), PathBuf::from("/data/a.rttm")),
                (PathBuf::from("/abs/b.wav"), PathBuf::from("/data/b.rttm")),
            ]
        );
        assert!(parse_manifest("only-one-field\n", Path::new(".")).is_err());
    }

    fn seg(file_id: &str, speaker: &str) -> RttmSegment {
        RttmSegment {
            file_id: file_id.to_string(),
            speaker: speaker.to_string(),
            start: 0.0,
            duration: 1.0,
        }
    }

    #[test]
    fn multi_recording_rttm_is_filtered_by_file_id() {
        let all = vec![seg("a", "x"), seg("b", "y"), seg("a", "z")];
        let got = reference_for(all.clone(), "a").unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|s| s.file_id == "a"));
        assert!(reference_for(all, "missing").is_err());
    }

    #[test]
    fn single_recording_rttm_must_name_the_recording() {
        let one = vec![seg("rec", "x"), seg("rec", "y")];
        assert_eq!(reference_for(one, "rec").unwrap().len(), 2);
        let other = vec![seg("other", "x"), seg("other", "y")];
        let err = reference_for(other, "rec").unwrap_err().to_string();
        assert!(err.contains("other") && err.contains("rec"), "{err}");
    }

    #[test]
    fn duplicate_stems_get_distinct_names() {
        let pair = |w: &str| (PathBuf::from(w), PathBuf::from("r.rttm"));
        let names = unique_names(&[pair("/x/rec.wav"), pair("/y/rec.wav"), pair("/z/b.wav")]);
        assert_eq!(names, vec!["rec", "rec_2", "b"]);
    }

    #[test]
    fn generated_names_never_collide_with_real_stems() {
        let pair = |w: &str| (PathBuf::from(w), PathBuf::from("r.rttm"));
        let names = unique_names(&[
            pair("/x/rec.wav"),
            pair("/y/rec.wav"),
            pair("/z/rec_2.wav"),
            pair("/w/rec.wav"),
        ]);
        let distinct: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(distinct.len(), names.len(), "{names:?}");
        assert_eq!(names[0], "rec");
        assert_eq!(names[2], "rec_2_2", "{names:?}");
    }

    #[test]
    fn speaker_cap_must_be_positive() {
        assert_eq!(parse_max_speakers("1").unwrap(), 1);
        assert_eq!(parse_max_speakers("8").unwrap(), 8);
        for bad in ["0", "-1", "x", ""] {
            assert!(parse_max_speakers(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn refine_threshold_must_be_a_finite_cosine() {
        for ok in ["-1", "0", "0.5", "1"] {
            assert!(parse_cosine_threshold(ok).is_ok(), "{ok}");
        }
        for bad in ["NaN", "inf", "-inf", "1.01", "-1.5", "x"] {
            assert!(parse_cosine_threshold(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn embedder_id_wins_over_the_file_name() {
        let path = Path::new("/m/custom_model.onnx");
        assert_eq!(
            embedder_family(path, Some("embedder-campplus-multi")).unwrap(),
            EmbedderFamily::CamPlusPlusMultilingual
        );
        assert_eq!(
            embedder_family(path, None).unwrap(),
            EmbedderFamily::Eres2Net
        );
        assert!(embedder_family(path, Some("embedder-nope")).is_err());
    }

    #[test]
    fn merge_chains_resolve_to_the_final_survivor() {
        let (a, b, c) = (SpeakerId::new(), SpeakerId::new(), SpeakerId::new());
        let resolved = resolve_merges(&[(a, b), (b, c)]);
        assert_eq!(resolved[&a], c);
        assert_eq!(resolved[&b], c);
        assert!(!resolved.contains_key(&c));
    }

    #[test]
    fn refine_mode_flag_accepts_the_documented_values() {
        assert_eq!(parse_refine_mode("off"), Ok(OfflineRefineMode::Off));
        assert_eq!(
            parse_refine_mode("merge-only"),
            Ok(OfflineRefineMode::MergeOnly)
        );
        assert_eq!(
            parse_refine_mode("resegment"),
            Ok(OfflineRefineMode::Resegment)
        );
        assert!(parse_refine_mode("vbx").is_err());
    }

    #[test]
    fn resegment_plan_relabels_recorded_spans_by_chunk_and_position() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let span = |id, s, e| SpeakerSpan {
            speaker_id: id,
            slot: 0,
            start_sample: s,
            end_sample: e,
        };
        let chunks = vec![
            RecordedChunk {
                chunk_index: 0,
                offset_ms: 0,
                spans: vec![span(a, 0, 100), span(b, 100, 200)],
            },
            // Chunk 1 was skipped as silence: index 2 follows index 0.
            RecordedChunk {
                chunk_index: 2,
                offset_ms: 9_000,
                spans: vec![span(b, 0, 100)],
            },
        ];
        let at = |chunk_index, span_index| SpanRef {
            chunk_index,
            span_index,
        };
        let relabel = HashMap::from([(at(0, 1), a), (at(2, 0), a), (at(0, 0), a)]);
        let (out, changed) = relabel_spans(chunks, &relabel);
        assert_eq!(changed, 2);
        assert_eq!(out[1].0, 9_000);
        assert!(out
            .iter()
            .flat_map(|(_, spans)| spans)
            .all(|s| s.speaker_id == a));
    }

    #[tokio::test]
    async fn recording_diarizer_forwards_context_and_captures_the_plan() {
        struct Inner;
        #[async_trait]
        impl Diarizer for Inner {
            fn sample_rate_hz(&self) -> u32 {
                16_000
            }
            async fn assign(&mut self, _: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
                Ok(None)
            }
            async fn assign_spans_at(
                &mut self,
                _: &[Sample],
                ctx: ChunkContext,
            ) -> Result<Vec<SpeakerSpan>, DomainError> {
                Ok(vec![SpeakerSpan {
                    speaker_id: SpeakerId(uuid::Uuid::from_u128(u128::from(ctx.offset_ms))),
                    slot: 0,
                    start_sample: 0,
                    end_sample: 1,
                }])
            }
            fn speakers(&self) -> Vec<Speaker> {
                vec![]
            }
            fn rename(&mut self, _: SpeakerId, _: &str) -> Result<bool, DomainError> {
                Ok(false)
            }
            fn reset(&mut self) {}
            fn refine_plan(&self, _: RefineBudget) -> Result<RefineOutcome, DomainError> {
                Ok(RefineOutcome::Relabel(echo_domain::SpanRelabelPlan {
                    spans: vec![(
                        SpanRef {
                            chunk_index: 4,
                            span_index: 0,
                        },
                        SpeakerId(uuid::Uuid::nil()),
                    )],
                    ..Default::default()
                }))
            }
        }
        impl GridDiarizer for Inner {}
        let (log, plan): (SpanLog, PlanSlot) = (Arc::default(), Arc::default());
        let mut d = RecordingDiarizer {
            inner: Box::new(Inner),
            log: Arc::clone(&log),
            plan: Arc::clone(&plan),
            grid: Vec::new(),
            refine: Arc::default(),
        };
        let ctx = ChunkContext {
            chunk_index: 4,
            offset_ms: 7,
            epoch: 0,
        };
        let spans = d.assign_spans_at(&[0.0], ctx).await.unwrap();
        assert_eq!(
            spans[0].speaker_id.0.as_u128(),
            7,
            "context reaches the inner diarizer"
        );
        assert_eq!(log.lock().unwrap()[0].0, 4);
        let budget = RefineBudget {
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(1),
        };
        assert!(matches!(
            d.refine_plan(budget).unwrap(),
            RefineOutcome::Relabel(_)
        ));
        assert_eq!(plan.lock().unwrap().as_ref().map(HashMap::len), Some(1));

        // A plan returned after the deadline is a failed refine: the
        // pipeline keeps the online result, so the bench must not score it.
        *plan.lock().unwrap() = None;
        let late = RefineBudget {
            deadline: std::time::Instant::now() - std::time::Duration::from_millis(1),
        };
        assert!(d.refine_plan(late).is_err());
        assert!(plan.lock().unwrap().is_none());
    }

    #[test]
    fn grid_runs_after_the_main_plan_and_never_spends_its_deadline() {
        struct Slow;
        #[async_trait]
        impl Diarizer for Slow {
            fn sample_rate_hz(&self) -> u32 {
                16_000
            }
            async fn assign(&mut self, _: &[Sample]) -> Result<Option<SpeakerId>, DomainError> {
                Ok(None)
            }
            fn speakers(&self) -> Vec<Speaker> {
                vec![]
            }
            fn rename(&mut self, _: SpeakerId, _: &str) -> Result<bool, DomainError> {
                Ok(false)
            }
            fn reset(&mut self) {}
            fn refine_plan(&self, _: RefineBudget) -> Result<RefineOutcome, DomainError> {
                Ok(RefineOutcome::Relabel(
                    echo_domain::SpanRelabelPlan::default(),
                ))
            }
        }
        impl GridDiarizer for Slow {
            fn resegment_grid(&self, specs: &[VbxSpec]) -> Vec<GridResult> {
                std::thread::sleep(std::time::Duration::from_millis(80));
                specs
                    .iter()
                    .map(|_| GridResult {
                        relabel: Some(HashMap::new()),
                        elapsed_ms: 80.0,
                    })
                    .collect()
            }
        }
        let (plan, refine): (PlanSlot, RefineSlot) = (Arc::default(), Arc::default());
        let d = RecordingDiarizer {
            inner: Box::new(Slow),
            log: Arc::default(),
            plan: Arc::clone(&plan),
            grid: vec![parse_vbx_spec(VbxConfig::default(), 0.5, "fa=0.4").unwrap()],
            refine: Arc::clone(&refine),
        };
        let budget = RefineBudget {
            deadline: std::time::Instant::now() + std::time::Duration::from_millis(40),
        };
        assert!(
            d.refine_plan(budget).is_ok(),
            "the grid ran past the deadline"
        );
        assert!(plan.lock().unwrap().is_some());
        assert_eq!(refine.lock().unwrap().1.len(), 1);
    }

    #[test]
    fn vbx_spec_overrides_only_the_named_keys() {
        let base = VbxConfig::default();
        let same = parse_vbx_spec(base, 0.5, "").unwrap();
        assert_eq!((same.config, same.init_threshold), (base, 0.5));

        let s = parse_vbx_spec(base, 0.5, "fa=0.4, fb=11,loop=0.7,lda=64,init=0.3").unwrap();
        assert_eq!(s.config.params.fa, 0.4);
        assert_eq!(s.config.params.fb, 11.0);
        assert_eq!(s.config.params.loop_p, 0.7);
        assert_eq!(s.config.lda_dim, 64);
        assert_eq!(s.config.init_offset, None);
        assert_eq!(s.init_threshold, 0.3);
        assert_eq!(s.config.init_smoothing, base.init_smoothing);

        let cal = parse_vbx_spec(base, 0.5, "init=cal:0.1").unwrap();
        assert_eq!(cal.config.init_offset, Some(0.1));
        assert_eq!(cal.init_threshold, 0.5, "fallback keeps the base cut");
    }

    #[test]
    fn vbx_spec_rejects_bad_keys_and_values() {
        let base = VbxConfig::default();
        for bad in [
            "nope=1", "fa", "fa=x", "fa=0", "loop=1.5", "lda=0", "init=2", "fb=inf",
        ] {
            assert!(parse_vbx_spec(base, 0.5, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn spans_are_offset_and_relabelled() {
        let (a, b) = (SpeakerId::new(), SpeakerId::new());
        let span = |id, s, e| SpeakerSpan {
            speaker_id: id,
            slot: 0,
            start_sample: s,
            end_sample: e,
        };
        let chunks = vec![
            (0, vec![span(a, 0, 16_000)]),
            (4_000, vec![span(b, 8_000, 24_000)]),
        ];
        let merges = HashMap::from([(b, a)]);
        let segs = spans_to_segments("rec", &chunks, &merges);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].speaker, a.to_string());
        assert!((segs[1].start - 4.5).abs() < 1e-9);
        assert!((segs[1].duration - 1.0).abs() < 1e-9);
    }
}
