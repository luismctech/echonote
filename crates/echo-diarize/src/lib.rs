//! # echo-diarize
//!
//! Speaker diarization for EchoNote: turn a stream of voiced audio
//! chunks into stable speaker identities. The crate splits into:
//!
//! - [`embedding`]: the [`SpeakerEmbedder`] trait that ONNX adapters
//!   (3D-Speaker ERes2Net, CAM++, …) implement, plus a couple of
//!   shared linear-algebra helpers (`cosine_similarity`, `l2_normalize`).
//! - [`cluster`]: [`OnlineCluster`], a threshold-based incremental
//!   clusterer that keeps one normalised running-mean centroid per
//!   speaker.
//! - [`online_diarizer`]: [`OnlineDiarizer`], the [`echo_domain::Diarizer`]
//!   adapter that wires an embedder to a cluster.
//!
//! Two ONNX speaker embedder adapters are provided:
//!
//! - [`eres2net`]: 3D-Speaker ERes2Net (English VoxCeleb, ~26 MB).
//! - [`cam_plus_plus`]: 3D-Speaker CAM++ (multilingual, ~28 MB, lower
//!   EER, recommended for Spanish-primary meetings).

#![warn(rust_2018_idioms, clippy::all)]

pub mod cam_plus_plus;
pub mod cluster;
pub mod embedding;
pub mod eres2net;
pub mod offline_refine;
pub mod online_diarizer;
pub mod pyannote_segmenter;

pub use cam_plus_plus::{
    CamPlusPlusConfig, CamPlusPlusEmbedder, CAMPP_EMBED_DIM, CAMPP_FBANK_DIM, CAMPP_MIN_SAMPLES,
    CAMPP_SAMPLE_RATE, CAMPP_TARGET_FRAMES,
};
pub use cluster::{OnlineCluster, OnlineClusterConfig};
pub use embedding::{cosine_similarity, l2_normalize, SpeakerEmbedder};
pub use eres2net::{
    Eres2NetConfig, Eres2NetEmbedder, ERES2NET_EMBED_DIM, ERES2NET_FBANK_DIM, ERES2NET_MIN_SAMPLES,
    ERES2NET_SAMPLE_RATE, ERES2NET_TARGET_FRAMES,
};
pub use offline_refine::{refine_speakers, OfflineRefineConfig};
pub use online_diarizer::OnlineDiarizer;
pub use pyannote_segmenter::{
    PyannoteSegmenter, PyannoteSegmenterConfig, PYANNOTE_CHUNK_SAMPLES, PYANNOTE_HOP_SAMPLES,
    PYANNOTE_MAX_LOCAL_SPEAKERS, PYANNOTE_SAMPLE_RATE,
};
