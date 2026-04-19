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
//! Sprint 1 day 5 ships the port + clustering + a deterministic stub
//! embedder for tests; the real ERes2Net adapter (and its model
//! download) lands in day 6, behind the same `SpeakerEmbedder` trait.

#![warn(rust_2018_idioms, clippy::all)]

pub mod cluster;
pub mod embedding;
pub mod online_diarizer;

pub use cluster::{OnlineCluster, OnlineClusterConfig};
pub use embedding::{cosine_similarity, l2_normalize, SpeakerEmbedder};
pub use online_diarizer::OnlineDiarizer;
