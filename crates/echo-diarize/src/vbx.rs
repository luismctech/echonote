//! VBx: Bayesian HMM clustering of speaker-embedding sequences.
//!
//! Rust port of `VBx/VBx.py` (`VBx`, `forward_backward`) and
//! `twoGMMcalib_lin` from `VBx/diarization_lib.py` in BUTSpeechFIT VBx
//! (<https://github.com/BUTSpeechFIT/VBx>, commit `57466e6`):
//!
//! > Copyright 2021 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz,
//! > mireia@fit.vutbr.cz). Copyright 2013-2019 Lukas Burget, Mireia Diez.
//! > Licensed under the Apache License, Version 2.0; see
//! > `third_party/VBx-LICENSE` in this crate.
//!
//! Changes from the original (as required by Apache-2.0 §4(b)):
//!
//! - Ported from numpy to plain Rust with `f64` arithmetic.
//! - The transition matrix `loopP·I + (1 - loopP)·pi` is never built: the
//!   forward-backward pass uses its rank-one structure, `O(T·S)` instead
//!   of `O(T·S²)`, with the same `+1e-8` floor on every transition
//!   probability as the original.
//! - A sequence may hold several independent runs (capture epochs split
//!   by pause/resume, ADR 0001 D7). The first frame of each run is
//!   entered from the speaker priors, like frame 0, instead of through a
//!   speaker transition, and counts toward the prior update like frame 0.
//! - The reference-DER, plotting and random-initialisation options are
//!   dropped; the caller always supplies `gamma`.
//! - Cooperative deadline checks between iterations.
//!
//! Model (Landini et al., "Bayesian HMM clustering of x-vector sequences
//! (VBx) in speaker diarization", 2022): frames `x_t` (PLDA-projected
//! embeddings) are emitted by a hidden speaker `s` as `x_t ~ N(V·y_s, I)`
//! with `y_s ~ N(0, I)` and `V = diag(sqrt(Phi))`. `Fa` scales the
//! acoustic statistics, `Fb` regularises the speaker models (a larger
//! `Fb` leaves fewer speakers) and `loopP` is the probability of staying
//! with the same speaker between consecutive frames.

use std::sync::Arc;
use std::time::Instant;

use echo_domain::{DomainError, RefineBudget};

use crate::embedding::l2_normalize;
use crate::embedding_log::EmbeddingLogEntry;
use crate::plda::Plda;
use crate::resegment::{nearest, strided_ahc, AhcCut, OfflineReclusterer};

/// VB-HMM hyper-parameters (names as in the original).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbxParams {
    /// Scale of the acoustic sufficient statistics.
    pub fa: f64,
    /// Speaker-model regularisation; larger keeps fewer speakers.
    pub fb: f64,
    /// Probability of not switching speakers between frames.
    pub loop_p: f64,
    /// Maximum number of VB iterations.
    pub max_iters: usize,
    /// Stop once the ELBO improves by less than this.
    pub epsilon: f64,
}

/// Result of [`vbx`].
#[derive(Debug, Clone, PartialEq)]
pub struct VbxOutput {
    /// `T x S` per-frame speaker posteriors, row-major.
    pub gamma: Vec<f64>,
    /// Learned speaker priors; redundant speakers converge to zero.
    pub pi: Vec<f64>,
    /// ELBO after every iteration.
    pub elbo: Vec<f64>,
}

/// Floor added to every transition and initial probability before the
/// logarithm, as in the original `forward_backward`.
const EPS: f64 = 1e-8;

fn deadline_error() -> DomainError {
    DomainError::DiarizationFailed("offline refine deadline exceeded".into())
}

/// Run VB-HMM inference.
///
/// * `x` — `T x D` frames, row-major (PLDA-projected embeddings). Taken
///   by value: it is scaled in place into the statistics VBx needs, so
///   the frames are held once.
/// * `phi` — the `D` across-class variances.
/// * `gamma` — `T x S` initial responsibilities (rows sum to one).
/// * `run_starts` — indices of frames (besides 0, which always starts a
///   run) that start a new independent run: strictly increasing, each in
///   `1..T`.
///
/// # Errors
///
/// [`DomainError::DiarizationFailed`] once `deadline` has passed, when
/// the shapes disagree, for a non-finite frame, responsibility or
/// variance, a negative `phi` or `gamma` value, out-of-range parameters
/// (`fa` and `fb` must be finite and positive, `loop_p` within `0..=1`)
/// or `run_starts` outside its contract.
pub fn vbx(
    mut x: Vec<f64>,
    phi: &[f64],
    mut gamma: Vec<f64>,
    run_starts: &[usize],
    params: VbxParams,
    deadline: Instant,
) -> Result<VbxOutput, DomainError> {
    let d = phi.len();
    if d == 0 || !x.len().is_multiple_of(d) {
        return Err(DomainError::DiarizationFailed(
            "VBx: bad frame shape".into(),
        ));
    }
    let t_len = x.len() / d;
    if t_len == 0 || gamma.is_empty() || !gamma.len().is_multiple_of(t_len) {
        return Err(DomainError::DiarizationFailed(
            "VBx: bad gamma shape".into(),
        ));
    }
    let s = gamma.len() / t_len;
    validate(&x, phi, &gamma, params)?;
    let mut is_start = vec![false; t_len];
    is_start[0] = true;
    let mut previous = 0;
    for &i in run_starts {
        if i <= previous || i >= t_len {
            return Err(DomainError::DiarizationFailed(format!(
                "VBx: run start {i} is not increasing within 1..{t_len}"
            )));
        }
        is_start[i] = true;
        previous = i;
    }

    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let g: Vec<f64> = x
        .chunks_exact(d)
        .map(|row| -0.5 * (row.iter().map(|v| v * v).sum::<f64>() + d as f64 * ln_2pi))
        .collect();
    let v: Vec<f64> = phi.iter().map(|p| p.sqrt()).collect();
    // (18): rho = X * V, computed in place.
    for row in x.chunks_exact_mut(d) {
        for (a, b) in row.iter_mut().zip(&v) {
            *a *= b;
        }
    }
    let rho = x;

    let mut pi = vec![1.0 / s as f64; s];
    let mut elbo: Vec<f64> = Vec::new();
    let ratio = params.fa / params.fb;
    let mut log_p = vec![0.0_f64; t_len * s];
    let mut inv_l = vec![0.0_f64; s * d];
    let mut alpha = vec![0.0_f64; s * d];
    for ii in 0..params.max_iters {
        if Instant::now() >= deadline {
            return Err(deadline_error());
        }
        // (17), (16): speaker model posteriors.
        for k in 0..s {
            let occupancy: f64 = (0..t_len).map(|t| gamma[t * s + k]).sum();
            for j in 0..d {
                inv_l[k * d + j] = 1.0 / (1.0 + ratio * occupancy * phi[j]);
            }
            for j in 0..d {
                let stat: f64 = (0..t_len).map(|t| gamma[t * s + k] * rho[t * d + j]).sum();
                alpha[k * d + j] = ratio * inv_l[k * d + j] * stat;
            }
        }
        // (23): per-frame log-likelihoods.
        let penalty: Vec<f64> = (0..s)
            .map(|k| {
                (0..d)
                    .map(|j| (inv_l[k * d + j] + alpha[k * d + j].powi(2)) * phi[j])
                    .sum::<f64>()
            })
            .collect();
        for t in 0..t_len {
            let r = &rho[t * d..(t + 1) * d];
            for k in 0..s {
                let a = &alpha[k * d..(k + 1) * d];
                let dot: f64 = r.iter().zip(a).map(|(p, q)| p * q).sum();
                log_p[t * s + k] = params.fa * (dot - 0.5 * penalty[k] + g[t]);
            }
        }
        // (19)-(22).
        let fb = forward_backward(&log_p, s, &pi, params.loop_p, &is_start);
        gamma = fb.gamma;
        // (25).
        let reg: f64 = inv_l
            .iter()
            .zip(&alpha)
            .map(|(l, a)| l.ln() - l - a * a + 1.0)
            .sum();
        let value = fb.tll + params.fb * 0.5 * reg;
        // (24): speaker priors.
        let mut new_pi = vec![0.0_f64; s];
        for t in 0..t_len {
            if is_start[t] {
                for k in 0..s {
                    new_pi[k] += gamma[t * s + k];
                }
                continue;
            }
            let prev = log_sum_exp(&fb.lfw[(t - 1) * s..t * s]);
            for k in 0..s {
                let e = (prev + log_p[t * s + k] + fb.lbw[t * s + k] - fb.tll).exp();
                new_pi[k] += (1.0 - params.loop_p) * pi[k] * e;
            }
        }
        let total: f64 = new_pi.iter().sum();
        if total > 0.0 && total.is_finite() {
            pi = new_pi.into_iter().map(|p| p / total).collect();
        }
        let improved = elbo.last().map(|prev| value - prev);
        elbo.push(value);
        if ii > 0 && improved.is_some_and(|delta| delta < params.epsilon) {
            break;
        }
    }
    Ok(VbxOutput { gamma, pi, elbo })
}

/// Reject inputs the VB updates are not defined for, so a bad model or
/// setting fails here instead of producing NaN posteriors.
fn validate(x: &[f64], phi: &[f64], gamma: &[f64], params: VbxParams) -> Result<(), DomainError> {
    let fail = |what: &str| Err(DomainError::DiarizationFailed(format!("VBx: {what}")));
    if !x.iter().all(|v| v.is_finite()) {
        return fail("non-finite frame");
    }
    if !phi.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return fail("phi must be finite and non-negative");
    }
    if !gamma.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return fail("gamma must be finite and non-negative");
    }
    if !(params.fa.is_finite() && params.fa > 0.0) {
        return fail("fa must be finite and positive");
    }
    if !(params.fb.is_finite() && params.fb > 0.0) {
        return fail("fb must be finite and positive");
    }
    if !(0.0..=1.0).contains(&params.loop_p) {
        return fail("loop_p must be within 0..=1");
    }
    Ok(())
}

/// Forward-backward pass output.
struct ForwardBackward {
    gamma: Vec<f64>,
    tll: f64,
    lfw: Vec<f64>,
    lbw: Vec<f64>,
}

/// `ln(sum(exp(v)))`, `-inf` for an empty or all `-inf` slice.
fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// HMM forward-backward with transition matrix
/// `tr[j][k] = loop_p·δ(j,k) + (1 - loop_p)·pi[k]` (plus [`EPS`]) and
/// initial probabilities `pi` (plus [`EPS`]). A frame flagged in
/// `is_start` is entered from `pi`, as frame 0 is.
fn forward_backward(
    lls: &[f64],
    s: usize,
    pi: &[f64],
    loop_p: f64,
    is_start: &[bool],
) -> ForwardBackward {
    let t_len = lls.len() / s;
    let c: Vec<f64> = pi.iter().map(|p| (1.0 - loop_p) * p + EPS).collect();
    let log_ip: Vec<f64> = pi.iter().map(|p| (p + EPS).ln()).collect();
    let mut lfw = vec![f64::NEG_INFINITY; t_len * s];
    let mut lbw = vec![f64::NEG_INFINITY; t_len * s];
    let mut e = vec![0.0_f64; s];

    for t in 0..t_len {
        if is_start[t] {
            let carried = if t == 0 {
                0.0
            } else {
                log_sum_exp(&lfw[(t - 1) * s..t * s])
            };
            for k in 0..s {
                lfw[t * s + k] = lls[t * s + k] + carried + log_ip[k];
            }
            continue;
        }
        let prev = &lfw[(t - 1) * s..t * s];
        let m = prev.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for (ek, p) in e.iter_mut().zip(prev) {
            *ek = (p - m).exp();
            total += *ek;
        }
        for k in 0..s {
            lfw[t * s + k] = lls[t * s + k] + m + (loop_p * e[k] + c[k] * total).ln();
        }
    }

    for k in 0..s {
        lbw[(t_len - 1) * s + k] = 0.0;
    }
    for t in (0..t_len.saturating_sub(1)).rev() {
        let next: Vec<f64> = (0..s)
            .map(|k| lls[(t + 1) * s + k] + lbw[(t + 1) * s + k])
            .collect();
        if is_start[t + 1] {
            let entered: Vec<f64> = next.iter().zip(&log_ip).map(|(a, b)| a + b).collect();
            let value = log_sum_exp(&entered);
            for k in 0..s {
                lbw[t * s + k] = value;
            }
            continue;
        }
        let m = next.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut w = 0.0;
        for k in 0..s {
            e[k] = (next[k] - m).exp();
            w += c[k] * e[k];
        }
        for j in 0..s {
            lbw[t * s + j] = m + (loop_p * e[j] + w).ln();
        }
    }

    let tll = log_sum_exp(&lfw[(t_len - 1) * s..]);
    let gamma = lfw
        .iter()
        .zip(&lbw)
        .map(|(f, b)| (f + b - tll).exp())
        .collect();
    ForwardBackward {
        gamma,
        tll,
        lfw,
        lbw,
    }
}

/// Threshold separating the two components of a two-Gaussian mixture
/// with shared variance fitted to `scores` (`twoGMMcalib_lin`, 20 EM
/// iterations). VBx uses it as a per-recording AHC threshold over the
/// pairwise cosine similarities. `None` for fewer than two distinct
/// scores.
#[must_use]
pub fn two_gmm_threshold(scores: &[f32]) -> Option<f64> {
    let n = scores.len() as f64;
    if scores.len() < 2 {
        return None;
    }
    let mean = scores.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var0 = scores
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    if var0.is_nan() || var0 <= 0.0 {
        return None;
    }
    let std = var0.sqrt();
    let mut weights = [0.5_f64, 0.5];
    let mut means = [mean - std, mean + std];
    let mut var = var0;
    let mut threshold = f64::INFINITY;
    for _ in 0..20 {
        let mut cnt = [0.0_f64; 2];
        let mut first = [0.0_f64; 2];
        let mut second = [0.0_f64; 2];
        for &v in scores {
            let v = f64::from(v);
            let ll = [0, 1]
                .map(|i| weights[i].ln() - 0.5 * var.ln() - 0.5 * (v - means[i]).powi(2) / var);
            let m = ll[0].max(ll[1]);
            let z = (ll[0] - m).exp() + (ll[1] - m).exp();
            for i in 0..2 {
                let g = (ll[i] - m).exp() / z;
                cnt[i] += g;
                first[i] += v * g;
                second[i] += v * v * g;
            }
        }
        let total = cnt[0] + cnt[1];
        weights = [cnt[0] / total, cnt[1] / total];
        means = [first[0] / cnt[0], first[1] / cnt[1]];
        var = (0..2)
            .map(|i| (second[i] / cnt[i] - means[i].powi(2)) * weights[i])
            .sum();
        if var.is_nan() || var <= 0.0 || !means.iter().all(|m| m.is_finite()) {
            return None;
        }
        let a = [0, 1].map(|i| (weights[i].powi(2) / var).ln() - means[i].powi(2) / var);
        let b = [0, 1].map(|i| means[i] / var);
        threshold = -0.5 * (a[0] - a[1]) / (b[0] - b[1]);
    }
    threshold.is_finite().then_some(threshold)
}

/// Settings of the VBx re-segmentation ([`VbxReclusterer`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbxConfig {
    /// VB-HMM hyper-parameters.
    pub params: VbxParams,
    /// AHC initialisation cut: `Some(offset)` uses the per-recording
    /// two-Gaussian threshold plus `offset` (the VBx recipe), `None` a
    /// fixed similarity (the diarizer's re-segmentation threshold). Both
    /// cut cosine similarities in the PLDA's LDA space.
    pub init_offset: Option<f32>,
    /// Sharpness of the initial responsibilities: each frame starts with
    /// `softmax(smoothing * one_hot(ahc_label))`.
    pub init_smoothing: f64,
    /// PLDA dimensions VBx uses (at most the model's LDA dimension).
    pub lda_dim: usize,
    /// At most this many AHC clusters seed VBx; frames of smaller
    /// clusters start from the nearest kept one. Bounds the `T x S`
    /// matrices.
    pub max_speakers: usize,
}

impl Default for VbxConfig {
    /// The BUTSpeechFIT example recipe (`run_example.sh`): Fa 0.3, Fb 17,
    /// loopP 0.99, AHC offset -0.015, smoothing 5, LDA 128, 40 iterations.
    fn default() -> Self {
        Self {
            params: VbxParams {
                fa: 0.3,
                fb: 17.0,
                loop_p: 0.99,
                max_iters: 40,
                epsilon: 1e-6,
            },
            init_offset: Some(-0.015),
            init_smoothing: 5.0,
            lda_dim: 128,
            max_speakers: 20,
        }
    }
}

/// At most this many frames take part in the AHC initialisation itself
/// (a condensed f32 matrix of about 8.4 MB); the others start from the
/// nearest AHC centroid. VBx then revisits every frame, so a coarser
/// initialisation than the AHC-only bound
/// ([`crate::resegment::AHC_MAX_REPRESENTATIVES`]) keeps the stop-time
/// transient memory near the ADR 0001 (Q3) target.
pub const VBX_INIT_MAX_REPRESENTATIVES: usize = 2_048;

/// At most this many frames enter VBx; a longer log is an invariant
/// violation of the embedding-log caps and is rejected (fail-safe).
pub const VBX_MAX_FRAMES: usize = 32_768;

/// VB-HMM re-clustering of the embedding log, initialised by AHC.
#[derive(Debug, Clone)]
pub(crate) struct VbxReclusterer {
    /// PLDA model of the embedder that produced the log.
    pub plda: Arc<Plda>,
    /// VBx settings.
    pub config: VbxConfig,
    /// Fixed AHC cut when [`VbxConfig::init_offset`] is `None`, and the
    /// fallback when the calibration is impossible.
    pub init_threshold: f32,
}

/// One VBx frame: an embedding and the log entry it votes for.
struct Frame<'a> {
    entry: usize,
    epoch: u16,
    embedding: &'a [f32],
}

/// Frames of the log, in time order: an entry's dense windows when it
/// has any, else its span embedding.
fn frames(entries: &[EmbeddingLogEntry]) -> Vec<Frame<'_>> {
    let mut out = Vec::with_capacity(entries.len());
    for (entry, e) in entries.iter().enumerate() {
        if e.windows.is_empty() {
            out.push(Frame {
                entry,
                epoch: e.epoch,
                embedding: &e.embedding,
            });
        } else {
            out.extend(e.windows.iter().map(|w| Frame {
                entry,
                epoch: e.epoch,
                embedding: w,
            }));
        }
    }
    out
}

impl OfflineReclusterer for VbxReclusterer {
    fn recluster(
        &self,
        entries: &[EmbeddingLogEntry],
        budget: RefineBudget,
    ) -> Result<Vec<usize>, DomainError> {
        budget.check()?;
        let frames = frames(entries);
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        if frames.len() > VBX_MAX_FRAMES {
            return Err(DomainError::DiarizationFailed(format!(
                "VBx: {} frames exceed the {VBX_MAX_FRAMES} cap",
                frames.len()
            )));
        }
        if frames.iter().any(|f| f.embedding.len() != self.plda.dim()) {
            return Err(DomainError::DiarizationFailed(
                "VBx: embedding dimension does not match the PLDA model".into(),
            ));
        }
        let y: Vec<Vec<f32>> = frames
            .iter()
            .map(|f| self.plda.transform(f.embedding))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&[f32]> = y.iter().map(Vec::as_slice).collect();
        let cut = match self.config.init_offset {
            Some(offset) => AhcCut::Calibrated {
                offset,
                fallback: self.init_threshold,
            },
            None => AhcCut::Similarity(self.init_threshold),
        };
        let init = strided_ahc(&refs, cut, VBX_INIT_MAX_REPRESENTATIVES, budget)?;
        let (init, s) = keep_largest(init, &refs, self.config.max_speakers.max(1));

        let t_len = frames.len();
        let on = self.config.init_smoothing.exp();
        let z = on + (s - 1) as f64;
        let mut gamma = vec![1.0 / z; t_len * s];
        for (t, &label) in init.iter().enumerate() {
            gamma[t * s + label] = on / z;
        }
        let dim = self.config.lda_dim.clamp(1, self.plda.lda_dim());
        let mut x: Vec<f64> = Vec::with_capacity(t_len * dim);
        for v in &y {
            x.extend(self.plda.project(v, dim)?);
        }
        drop(refs);
        drop(y);
        let phi: Vec<f64> = self.plda.psi()[..dim]
            .iter()
            .map(|p| f64::from(*p))
            .collect();
        let run_starts: Vec<usize> = (1..t_len)
            .filter(|&t| frames[t].epoch != frames[t - 1].epoch)
            .collect();
        let out = vbx(
            x,
            &phi,
            gamma,
            &run_starts,
            self.config.params,
            budget.deadline,
        )?;
        budget.check()?;

        // Each entry takes the speaker with the most posterior mass over
        // its frames.
        let mut votes = vec![0.0_f64; entries.len() * s];
        for (t, f) in frames.iter().enumerate() {
            for k in 0..s {
                votes[f.entry * s + k] += out.gamma[t * s + k];
            }
        }
        let raw: Vec<usize> = votes
            .chunks_exact(s)
            .map(|v| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(k, _)| k)
            })
            .collect();
        Ok(compact(&raw))
    }
}

/// Keep the `max` largest clusters of `labels`; members of the others
/// move to the nearest kept centroid. Returns compact labels and their
/// count.
fn keep_largest(labels: Vec<usize>, vectors: &[&[f32]], max: usize) -> (Vec<usize>, usize) {
    let k = labels.iter().max().map_or(0, |m| m + 1);
    if k <= max {
        return (labels, k.max(1));
    }
    let mut sizes: Vec<(usize, usize)> = (0..k).map(|c| (c, 0)).collect();
    for &l in &labels {
        sizes[l].1 += 1;
    }
    // Largest first; the earlier cluster wins ties.
    sizes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let kept: Vec<usize> = sizes[..max].iter().map(|(c, _)| *c).collect();
    let dim = vectors[0].len();
    let mut centroids = vec![vec![0.0_f32; dim]; max];
    for (v, &l) in vectors.iter().zip(&labels) {
        if let Some(i) = kept.iter().position(|&c| c == l) {
            for (c, x) in centroids[i].iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
    }
    for c in &mut centroids {
        l2_normalize(c);
    }
    let out = labels
        .iter()
        .zip(vectors)
        .map(|(&l, v)| {
            kept.iter()
                .position(|&c| c == l)
                .unwrap_or_else(|| nearest(&centroids, v))
        })
        .collect();
    (out, max)
}

/// Relabel to `0..k` in order of first appearance.
fn compact(labels: &[usize]) -> Vec<usize> {
    let mut map: Vec<(usize, usize)> = Vec::new();
    labels
        .iter()
        .map(|&l| match map.iter().find(|(from, _)| *from == l) {
            Some(&(_, to)) => to,
            None => {
                map.push((l, map.len()));
                map.len() - 1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Reference values from the official VBx code (`VBx.py`,
    // `diarization_lib.py` at commit 57466e6) on numpy 2.5.4 / scipy
    // 1.18.1: X scaled by 3, labels [0,0,0,1,1,0,1,2,0,0] one-hot
    // smoothed by softmax(5·q), Fa 0.5, Fb 2, loopP 0.8, 5 iterations.
    const X: [f64; 30] = [
        3.6, 0.9, -0.3, 3.0, 0.3, 0.6, 4.2, -0.6, 0.0, -2.7, 3.3, 0.9, -3.3, 2.7, -0.6, 3.3, 0.6,
        0.3, -3.0, 3.9, 0.0, -2.4, 3.0, 1.2, 3.9, 0.0, -0.9, 2.7, 1.2, 0.6,
    ];
    const PHI: [f64; 3] = [4.0, 2.0, 0.5];
    const ON: f64 = 0.986_703_291_042_268_2;
    const OFF: f64 = 0.006_648_354_478_866_005;
    const LABELS: [usize; 10] = [0, 0, 0, 1, 1, 0, 1, 2, 0, 0];
    const GAMMA: [f64; 30] = [
        0.999_988_497_640_047_7,
        1.150_230_726_556_124_4e-5,
        5.269_328_980_422_77e-11,
        0.999_995_613_688_155_6,
        4.386_258_289_239_202e-6,
        5.356_731_832_137_808_4e-11,
        0.999_993_114_477_300_3,
        6.885_446_012_973_353e-6,
        7.669_623_846_498_413e-11,
        5.826_448_178_283_477_4e-5,
        0.999_941_735_450_108_6,
        6.811_688_564_859_516e-11,
        2.257_645_355_707_023e-5,
        0.999_977_423_468_127_3,
        7.832_299_706_404_999e-11,
        0.981_536_385_381_304_4,
        0.018_463_612_866_313_32,
        1.752_391_383_415_490_3e-9,
        1.594_878_232_492_767_6e-5,
        0.999_984_051_187_230_1,
        3.045_187_021_277_548e-11,
        1.653_109_001_696_535_8e-4,
        0.999_834_688_974_142_5,
        1.256_906_091_181_256e-10,
        0.999_974_671_664_162_8,
        2.532_820_045_736_943e-5,
        1.353_821_541_229_269_7e-10,
        0.999_833_543_406_462_2,
        1.664_560_472_286_538e-4,
        5.463_106_186_942_346e-10,
    ];
    const PI: [f64; 3] = [
        0.609_683_651_927_628_9,
        0.390_316_347_964_586_4,
        1.077_848_144_178_132_2e-10,
    ];
    const ELBO: [f64; 5] = [
        -44.522_581_812_946_39,
        -41.218_825_450_670_59,
        -41.041_055_279_068_74,
        -41.040_161_369_939_56,
        -41.040_153_728_831_015,
    ];
    // forward_backward on fixed lls, tr = 0.9 I + 0.1 pi, ip = pi.
    const FB_PI: [f64; 3] = [0.5, 0.3, 0.2];
    const FB_P: [f64; 12] = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.3, 0.3, 0.4, 0.6, 0.3, 0.1];
    const FB_GAMMA: [f64; 12] = [
        0.588_237_959_830_076,
        0.396_057_612_892_983_14,
        0.015_704_427_276_939_846,
        0.517_055_654_165_448_9,
        0.468_537_368_424_715,
        0.014_406_977_409_835_103,
        0.549_718_115_830_646,
        0.430_367_611_844_635_57,
        0.019_914_272_324_717_774,
        0.585_385_456_747_517_3,
        0.395_968_991_659_264_97,
        0.018_645_551_593_217_857,
    ];
    const FB_TLL: f64 = -4.507_441_003_812_418;
    const GMM_THR: f64 = 0.395_185_693_592_410_8;

    fn far() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
        }
    }

    fn init() -> Vec<f64> {
        LABELS
            .iter()
            .flat_map(|&l| (0..3).map(move |k| if k == l { ON } else { OFF }))
            .collect()
    }

    fn params() -> VbxParams {
        VbxParams {
            fa: 0.5,
            fb: 2.0,
            loop_p: 0.8,
            max_iters: 5,
            epsilon: 1e-6,
        }
    }

    #[test]
    fn vbx_matches_the_reference_implementation() {
        let out = vbx(X.to_vec(), &PHI, init(), &[], params(), far()).unwrap();
        close(&out.gamma, &GAMMA, 1e-9);
        close(&out.pi, &PI, 1e-9);
        close(&out.elbo, &ELBO, 1e-9);
    }

    #[test]
    fn forward_backward_matches_the_reference_implementation() {
        let lls: Vec<f64> = FB_P.iter().map(|p| p.ln()).collect();
        let fb = forward_backward(&lls, 3, &FB_PI, 0.9, &[true, false, false, false]);
        close(&fb.gamma, &FB_GAMMA, 1e-12);
        assert!((fb.tll - FB_TLL).abs() < 1e-12);
    }

    #[test]
    fn a_run_start_makes_the_runs_independent() {
        let lls: Vec<f64> = FB_P.iter().map(|p| p.ln()).collect();
        let joint = forward_backward(&lls, 3, &FB_PI, 0.9, &[true, false, true, false]);
        let first = forward_backward(&lls[..6], 3, &FB_PI, 0.9, &[true, false]);
        let second = forward_backward(&lls[6..], 3, &FB_PI, 0.9, &[true, false]);
        let separate: Vec<f64> = first.gamma.iter().chain(&second.gamma).copied().collect();
        close(&joint.gamma, &separate, 1e-12);
        assert!((joint.tll - (first.tll + second.tll)).abs() < 1e-12);
    }

    #[test]
    fn two_gmm_threshold_matches_the_reference_implementation() {
        let mut s: Vec<f32> = (0..40)
            .map(|i| (-0.2 + 0.5 * f64::from(i) / 39.0) as f32)
            .collect();
        s.extend((0..20).map(|i| (0.5 + 0.4 * f64::from(i) / 19.0) as f32));
        let thr = two_gmm_threshold(&s).unwrap();
        // The reference ran on f64 scores; these are rounded to f32.
        assert!((thr - GMM_THR).abs() < 1e-5, "{thr}");
        assert_eq!(two_gmm_threshold(&[0.5]), None);
        assert_eq!(two_gmm_threshold(&[0.5, 0.5, 0.5]), None);
    }

    #[test]
    fn an_expired_deadline_is_an_error() {
        let past = Instant::now() - Duration::from_millis(1);
        let r = vbx(X.to_vec(), &PHI, init(), &[], params(), past);
        assert!(matches!(r, Err(DomainError::DiarizationFailed(_))));
    }

    fn log_entry(i: u32, epoch: u16, axis: usize, dim: usize) -> EmbeddingLogEntry {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        EmbeddingLogEntry::new(
            vec![echo_domain::SpanRef {
                chunk_index: i,
                span_index: 0,
            }],
            i * 3_000..(i + 1) * 3_000,
            epoch,
            None,
            echo_domain::SpeakerId::new(),
            v.into_boxed_slice(),
        )
    }

    fn reclusterer(dim: usize) -> VbxReclusterer {
        VbxReclusterer {
            plda: Arc::new(crate::plda::tests::identity(dim, 25.0)),
            config: VbxConfig {
                params: VbxParams {
                    fa: 1.0,
                    fb: 1.0,
                    loop_p: 0.5,
                    max_iters: 10,
                    epsilon: 1e-6,
                },
                init_offset: None,
                ..VbxConfig::default()
            },
            init_threshold: 0.5,
        }
    }

    fn budget() -> RefineBudget {
        RefineBudget { deadline: far() }
    }

    #[test]
    fn reclusterer_separates_two_voices_across_epochs() {
        let entries: Vec<_> = (0..8)
            .map(|i| log_entry(i, u16::from(i >= 4), (i % 2) as usize, 4))
            .collect();
        let labels = reclusterer(4).recluster(&entries, budget()).unwrap();
        assert_eq!(labels, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn dense_windows_outvote_the_span_embedding() {
        let mut entries: Vec<_> = (0..6)
            .map(|i| log_entry(i, 0, (i % 2) as usize, 4))
            .collect();
        // Entry 4's span embedding looks like voice 0, but its three
        // windows all sound like voice 1.
        let w = |axis: usize| {
            let mut v = vec![0.0_f32; 4];
            v[axis] = 1.0;
            v.into_boxed_slice()
        };
        entries[4].windows = vec![w(1), w(1), w(1)];
        let labels = reclusterer(4).recluster(&entries, budget()).unwrap();
        assert_eq!(labels[4], labels[1]);
        assert_ne!(labels[4], labels[0]);
    }

    #[test]
    fn a_dimension_mismatch_is_an_error() {
        let entries: Vec<_> = (0..4).map(|i| log_entry(i, 0, 0, 3)).collect();
        assert!(reclusterer(4).recluster(&entries, budget()).is_err());
    }

    #[test]
    fn keep_largest_moves_small_clusters_to_the_nearest_kept_one() {
        let v = |a: f32| vec![a.cos(), a.sin()];
        let vectors = [v(0.0), v(0.1), v(1.5), v(1.6), v(0.2)];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let (labels, k) = keep_largest(vec![0, 0, 1, 1, 2], &refs, 2);
        assert_eq!(k, 2);
        assert_eq!(labels, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn non_finite_inputs_and_bad_parameters_are_errors() {
        let bad = |x: Vec<f64>, phi: &[f64], gamma: Vec<f64>, p: VbxParams| {
            vbx(x, phi, gamma, &[], p, far()).is_err()
        };
        let mut x = X.to_vec();
        x[4] = f64::NAN;
        assert!(bad(x, &PHI, init(), params()), "NaN frame");
        let mut g = init();
        g[2] = f64::INFINITY;
        assert!(bad(X.to_vec(), &PHI, g, params()), "infinite gamma");
        assert!(
            bad(X.to_vec(), &[4.0, -2.0, 0.5], init(), params()),
            "phi < 0"
        );
        assert!(
            bad(X.to_vec(), &[4.0, f64::NAN, 0.5], init(), params()),
            "NaN phi"
        );
        for (name, p) in [
            (
                "fb = 0",
                VbxParams {
                    fb: 0.0,
                    ..params()
                },
            ),
            (
                "fb < 0",
                VbxParams {
                    fb: -1.0,
                    ..params()
                },
            ),
            (
                "fa NaN",
                VbxParams {
                    fa: f64::NAN,
                    ..params()
                },
            ),
            (
                "fa inf",
                VbxParams {
                    fa: f64::INFINITY,
                    ..params()
                },
            ),
            (
                "loopP > 1",
                VbxParams {
                    loop_p: 1.5,
                    ..params()
                },
            ),
            (
                "loopP < 0",
                VbxParams {
                    loop_p: -0.1,
                    ..params()
                },
            ),
            (
                "loopP NaN",
                VbxParams {
                    loop_p: f64::NAN,
                    ..params()
                },
            ),
        ] {
            assert!(bad(X.to_vec(), &PHI, init(), p), "{name}");
        }
    }

    #[test]
    fn run_starts_out_of_range_or_not_increasing_are_errors() {
        let run = |starts: &[usize]| vbx(X.to_vec(), &PHI, init(), starts, params(), far());
        assert!(run(&[3, 7]).is_ok());
        assert!(run(&[10]).is_err(), "past the last frame");
        assert!(run(&[0]).is_err(), "frame 0 always starts a run");
        assert!(run(&[7, 3]).is_err(), "decreasing");
        assert!(run(&[3, 3]).is_err(), "repeated");
    }

    #[test]
    fn a_run_start_makes_vbx_treat_identical_runs_identically() {
        // The sequence repeated twice. As two independent runs, both
        // copies see the same speaker models and are both entered from
        // the priors, so their posteriors match frame for frame. Joined
        // as one run, the second copy's first frame depends on the
        // first copy's last frame and the posteriors differ.
        let x: Vec<f64> = X.iter().chain(&X).copied().collect();
        let gamma: Vec<f64> = init().into_iter().chain(init()).collect();
        let split = vbx(x.clone(), &PHI, gamma.clone(), &[10], params(), far()).unwrap();
        let (a, b) = split.gamma.split_at(30);
        close(a, b, 1e-12);
        let joined = vbx(x, &PHI, gamma, &[], params(), far()).unwrap();
        let (a, b) = joined.gamma.split_at(30);
        assert!(
            a.iter().zip(b).any(|(p, q)| (p - q).abs() > 1e-6),
            "without a run start the copies are coupled"
        );
    }

    #[test]
    fn mismatched_shapes_are_errors_not_panics() {
        assert!(vbx(X[..29].to_vec(), &PHI, init(), &[], params(), far()).is_err());
        assert!(vbx(X.to_vec(), &PHI, vec![1.0; 7], &[], params(), far()).is_err());
        assert!(vbx(Vec::new(), &PHI, Vec::new(), &[], params(), far()).is_err());
    }
}
