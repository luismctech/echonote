//! PLDA back-end for the VBx re-segmentation ([`crate::vbx`]).
//!
//! VBx does not cluster raw speaker embeddings. Following the
//! BUTSpeechFIT recipe, each embedding `x` is first moved to an
//! LDA space and length-normalised,
//!
//! `y = l2norm(LDAᵀ · l2norm(x - mean1) - mean2)`,
//!
//! and then projected with a two-covariance PLDA model fitted on the same
//! space: `f = T · (y - mu)`, where `T` whitens the within-speaker
//! covariance and diagonalises the across-speaker covariance, whose
//! diagonal is `psi` (sorted in decreasing order). VBx then sees
//! within-speaker noise of identity covariance and speakers drawn from
//! `N(0, diag(psi))`.
//!
//! The model must be fitted on embeddings of the same embedder
//! (`scripts/fit-plda.py`). The multilingual CAM++ model ships inside the
//! binary ([`Plda::campplus_multilingual`]); see `assets/README.md` for
//! its training data and licence.
//!
//! # File format
//!
//! Little-endian: the 8-byte magic `ECHOPLDA`, `u32` version (1), `u32`
//! embedding dimension `d`, `u32` LDA dimension `k`, then `f32` arrays:
//! `mean1[d]`, `lda[d*k]` (row-major, `d` rows), `mean2[k]`, `mu[k]`,
//! `tr[k*k]` (row-major, row `i` is the `i`-th output dimension) and
//! `psi[k]`.

use std::sync::{Arc, OnceLock};

use echo_domain::DomainError;

/// Magic bytes at the start of a PLDA file.
pub const PLDA_MAGIC: &[u8; 8] = b"ECHOPLDA";

/// The only file version this reader understands.
const PLDA_VERSION: u32 = 1;

/// PLDA model fitted on the multilingual CAM++ (192-dim) embeddings.
static CAMPP_MULTI_BYTES: &[u8] = include_bytes!("../assets/plda_campplus_multi.plda");

/// LDA transform plus two-covariance PLDA model (see the module docs).
#[derive(Clone, PartialEq)]
pub struct Plda {
    dim: usize,
    lda_dim: usize,
    mean1: Vec<f32>,
    lda: Vec<f32>,
    mean2: Vec<f32>,
    mu: Vec<f32>,
    tr: Vec<f32>,
    psi: Vec<f32>,
}

impl std::fmt::Debug for Plda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plda")
            .field("dim", &self.dim)
            .field("lda_dim", &self.lda_dim)
            .finish_non_exhaustive()
    }
}

fn invalid(msg: impl Into<String>) -> DomainError {
    DomainError::DiarizationFailed(format!("invalid PLDA model: {}", msg.into()))
}

/// Little-endian reader over the model bytes.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], DomainError> {
        if self.bytes.len() < n {
            return Err(invalid("truncated file"));
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, DomainError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, DomainError> {
        let len = n.checked_mul(4).ok_or_else(|| invalid("size overflow"))?;
        let v: Vec<f32> = self
            .take(len)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if v.iter().all(|x| x.is_finite()) {
            Ok(v)
        } else {
            Err(invalid("non-finite value"))
        }
    }
}

impl Plda {
    /// Parse a model in the format of the module docs.
    ///
    /// # Errors
    ///
    /// [`DomainError::DiarizationFailed`] for a wrong magic or version, a
    /// truncated or oversized file, zero or inconsistent dimensions, a
    /// non-finite value or a negative `psi`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DomainError> {
        let mut r = Reader { bytes };
        if r.take(8)? != PLDA_MAGIC {
            return Err(invalid("bad magic"));
        }
        if r.u32()? != PLDA_VERSION {
            return Err(invalid("unsupported version"));
        }
        let dim = r.u32()? as usize;
        let lda_dim = r.u32()? as usize;
        if dim == 0 || lda_dim == 0 || lda_dim > dim || dim > 4_096 {
            return Err(invalid(format!("dimensions {dim} -> {lda_dim}")));
        }
        let model = Self {
            dim,
            lda_dim,
            mean1: r.f32s(dim)?,
            lda: r.f32s(dim * lda_dim)?,
            mean2: r.f32s(lda_dim)?,
            mu: r.f32s(lda_dim)?,
            tr: r.f32s(lda_dim * lda_dim)?,
            psi: r.f32s(lda_dim)?,
        };
        if !r.bytes.is_empty() {
            return Err(invalid("trailing bytes"));
        }
        if model.psi.iter().any(|&p| p < 0.0) {
            return Err(invalid("negative across-class variance"));
        }
        Ok(model)
    }

    /// The model fitted on the multilingual CAM++ embeddings, parsed once.
    ///
    /// # Panics
    ///
    /// Never in a released build: the asset is checked by the unit tests.
    #[must_use]
    pub fn campplus_multilingual() -> Arc<Self> {
        static MODEL: OnceLock<Arc<Plda>> = OnceLock::new();
        Arc::clone(MODEL.get_or_init(|| {
            Arc::new(Self::from_bytes(CAMPP_MULTI_BYTES).expect("bundled PLDA asset is valid"))
        }))
    }

    /// Embedding dimension the model expects.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimension of the LDA / PLDA space.
    #[must_use]
    pub fn lda_dim(&self) -> usize {
        self.lda_dim
    }

    /// Across-speaker variances in the PLDA space, in decreasing order.
    #[must_use]
    pub fn psi(&self) -> &[f32] {
        &self.psi
    }

    /// Embedding moved to the length-normalised LDA space
    /// (`y` in the module docs). `x` must have [`Self::dim`] values.
    #[must_use]
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let mut centred: Vec<f64> = x
            .iter()
            .zip(&self.mean1)
            .map(|(a, m)| f64::from(*a) - f64::from(*m))
            .collect();
        normalise(&mut centred);
        let mut y = vec![0.0_f64; self.lda_dim];
        for (i, c) in centred.iter().enumerate() {
            let row = &self.lda[i * self.lda_dim..(i + 1) * self.lda_dim];
            for (acc, w) in y.iter_mut().zip(row) {
                *acc += c * f64::from(*w);
            }
        }
        for (acc, m) in y.iter_mut().zip(&self.mean2) {
            *acc -= f64::from(*m);
        }
        normalise(&mut y);
        y.into_iter().map(|v| v as f32).collect()
    }

    /// First `out_dim` PLDA coordinates of a transformed vector `y`
    /// (`f` in the module docs), `out_dim <= lda_dim`.
    #[must_use]
    pub fn project(&self, y: &[f32], out_dim: usize) -> Vec<f64> {
        debug_assert_eq!(y.len(), self.lda_dim);
        let k = self.lda_dim;
        let centred: Vec<f64> = y
            .iter()
            .zip(&self.mu)
            .map(|(a, m)| f64::from(*a) - f64::from(*m))
            .collect();
        (0..out_dim.min(k))
            .map(|i| {
                self.tr[i * k..(i + 1) * k]
                    .iter()
                    .zip(&centred)
                    .map(|(t, c)| f64::from(*t) * c)
                    .sum()
            })
            .collect()
    }
}

fn normalise(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > f64::EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Serialise a model (test helper; the fitting script writes the
    /// real files).
    pub(crate) fn encode(
        dim: usize,
        lda_dim: usize,
        arrays: [&[f32]; 6], // mean1, lda, mean2, mu, tr, psi
    ) -> Vec<u8> {
        let mut out = PLDA_MAGIC.to_vec();
        for v in [PLDA_VERSION, dim as u32, lda_dim as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for a in arrays {
            for x in a {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Model whose LDA is the identity and whose PLDA transform scales
    /// the unit vectors by 10 (a within-speaker spread of 0.1), with every
    /// across-class variance equal to `psi`.
    pub(crate) fn identity(dim: usize, psi: f32) -> Plda {
        let mut eye = vec![0.0_f32; dim * dim];
        let mut scaled = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            eye[i * dim + i] = 1.0;
            scaled[i * dim + i] = 10.0;
        }
        let zeros = vec![0.0_f32; dim];
        let psis = vec![psi; dim];
        Plda::from_bytes(&encode(
            dim,
            dim,
            [&zeros, &eye, &zeros, &zeros, &scaled, &psis],
        ))
        .unwrap()
    }

    /// 3 -> 2 model with hand-picked values; reference outputs computed
    /// with numpy using the VBx formulas (`vbhmm.py`).
    fn small() -> Plda {
        let bytes = encode(
            3,
            2,
            [
                &[0.1, -0.2, 0.3],
                &[1.0, 0.5, -0.5, 1.0, 0.25, 0.75],
                &[0.05, -0.1],
                &[0.02, 0.01],
                &[2.0, 0.5, -0.3, 1.5],
                &[3.0, 0.5],
            ],
        );
        Plda::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn transform_and_project_match_the_vbx_formulas() {
        let p = small();
        let y = p.transform(&[0.4, 0.3, -0.2]);
        // numpy: l2_norm(lda.T @ l2_norm(x - mean1) - mean2)
        let expected_y = [-0.306_802_24_f32, 0.951_773_3];
        for (a, b) in y.iter().zip(expected_y) {
            assert!((a - b).abs() < 1e-5, "{y:?}");
        }
        // numpy: ((y - mu) @ tr.T)[:2]
        let f = p.project(&y, 2);
        let expected_f = [-0.182_717_84_f64, 1.510_700_6];
        for (a, b) in f.iter().zip(expected_f) {
            assert!((a - b).abs() < 1e-5, "{f:?}");
        }
        assert_eq!(p.project(&y, 1).len(), 1);
    }

    #[test]
    fn malformed_files_are_rejected() {
        let good = encode(1, 1, [&[0.0], &[1.0], &[0.0], &[0.0], &[1.0], &[1.0]]);
        assert!(Plda::from_bytes(&good).is_ok());
        assert!(
            Plda::from_bytes(&good[..good.len() - 1]).is_err(),
            "truncated"
        );
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(Plda::from_bytes(&trailing).is_err(), "trailing");
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(Plda::from_bytes(&magic).is_err(), "magic");
        let nan = encode(1, 1, [&[f32::NAN], &[1.0], &[0.0], &[0.0], &[1.0], &[1.0]]);
        assert!(Plda::from_bytes(&nan).is_err(), "nan");
        let neg = encode(1, 1, [&[0.0], &[1.0], &[0.0], &[0.0], &[1.0], &[-1.0]]);
        assert!(Plda::from_bytes(&neg).is_err(), "negative psi");
        let wide = encode(
            1,
            2,
            [
                &[0.0],
                &[1.0, 1.0],
                &[0.0; 2],
                &[0.0; 2],
                &[1.0; 4],
                &[1.0; 2],
            ],
        );
        assert!(Plda::from_bytes(&wide).is_err(), "lda_dim > dim");
    }

    #[test]
    fn bundled_campplus_model_loads_with_sorted_psi() {
        let p = Plda::campplus_multilingual();
        assert_eq!(p.dim(), 192);
        assert!(p.lda_dim() <= 192);
        assert!(p.psi().windows(2).all(|w| w[0] >= w[1]));
        let y = p.transform(&[0.1; 192]);
        let norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
