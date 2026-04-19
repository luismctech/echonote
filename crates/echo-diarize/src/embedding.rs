//! Speaker embedding port and helpers.
//!
//! An embedder converts a chunk of mono 16 kHz audio into a fixed-size
//! float vector that lives in a metric space where same-speaker chunks
//! are close (cosine similarity ≈ 1) and different-speaker chunks are
//! far apart. The actual neural model (3D-Speaker ERes2Net, CAM++,
//! WeSpeaker, …) plugs in behind [`SpeakerEmbedder`].
//!
//! The trait is intentionally small and synchronous: ONNX inference
//! is CPU-bound and the call sites already run inside spawned tasks,
//! so async at the embedder boundary would add ceremony without
//! parallelism.

use echo_domain::{DomainError, Sample};

/// Pure function from audio chunk to fixed-size voice embedding.
///
/// Implementations must:
///
/// - Be deterministic for a given input (no hidden RNG state). The
///   downstream cluster relies on stable embeddings to converge.
/// - Return a vector whose L2 norm is non-zero. Adapters should
///   L2-normalise the output so callers can use plain cosine
///   similarity without re-normalising.
/// - Accept arbitrary chunk lengths and internally pad / truncate to
///   the model's window. Returning `Ok(None)` is preferred over
///   raising errors when the chunk is below the minimum window
///   length the model needs to produce a meaningful embedding.
pub trait SpeakerEmbedder: Send + Sync {
    /// Sample rate the model was trained on. Mixing rates degrades
    /// the embedding silently — resample upstream.
    fn sample_rate_hz(&self) -> u32;

    /// Embedding dimensionality. Used by the cluster to pre-allocate.
    fn dim(&self) -> usize;

    /// Compute one embedding for the chunk. Returns `Ok(None)` when
    /// the chunk is too short or otherwise unusable; the diarizer
    /// will skip it instead of polluting a cluster with garbage.
    fn embed(&mut self, samples: &[Sample]) -> Result<Option<Vec<f32>>, DomainError>;
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` when either vector has zero L2 norm — the cluster
/// should treat that as "no match" and either skip or seed a new
/// centroid depending on policy.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine: vectors must match in length");

    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// L2-normalise a vector in place so callers can rely on
/// [`cosine_similarity`] equating to plain dot product. No-op for
/// zero vectors (the embedder should never emit one, but we don't
/// want to panic if it slips through).
pub fn l2_normalize(v: &mut [f32]) {
    let mut n = 0.0_f32;
    for x in v.iter() {
        n += x * x;
    }
    if n <= f32::EPSILON {
        return;
    }
    let inv = 1.0 / n.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = [0.5_f32, -0.3, 0.8];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-6, "expected ~1.0, got {s}");
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_opposite_vectors_is_minus_one() {
        let a = [0.7_f32, -0.4];
        let b = [-0.7_f32, 0.4];
        let s = cosine_similarity(&a, &b);
        assert!((s + 1.0).abs() < 1e-6, "expected ~-1.0, got {s}");
    }

    #[test]
    fn cosine_handles_zero_vector_without_panicking() {
        let a = [0.0_f32, 0.0];
        let b = [1.0_f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_is_noop_on_zero_vector() {
        let mut v = vec![0.0_f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
