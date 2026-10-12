//! Per-embedder diarization presets shared by the desktop shell and the
//! offline benchmarks.
//!
//! The embedder family is chosen from the model file name and decides
//! both the adapter to load and the online clustering thresholds, so a
//! benchmark run scores exactly the configuration the app would build.
//!
//! The values mirror `build_diarizer` in `src-tauri/src/commands/mod.rs`
//! (CAM++ 0.85 with online merging disabled, ERes2Net 0.65). That
//! function still inlines them; switching it to
//! [`EmbedderFamily::cluster_config`] removes the duplication.

use std::path::Path;

use echo_domain::DomainError;

use crate::cam_plus_plus::CamPlusPlusEmbedder;
use crate::cluster::OnlineClusterConfig;
use crate::embedding::SpeakerEmbedder;
use crate::eres2net::Eres2NetEmbedder;

/// Speaker embedder families the app knows how to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderFamily {
    /// 3D-Speaker CAM++ (EN VoxCeleb 512-dim or multilingual 192-dim).
    CamPlusPlus,
    /// 3D-Speaker ERes2Net (EN VoxCeleb, 192-dim). Fallback family.
    Eres2Net,
}

impl EmbedderFamily {
    /// Family implied by a model file name: names containing
    /// `camplusplus` or `campplus` are CAM++, everything else ERes2Net.
    #[must_use]
    pub fn from_file_name(file_name: &str) -> Self {
        if file_name.contains("camplusplus") || file_name.contains("campplus") {
            Self::CamPlusPlus
        } else {
            Self::Eres2Net
        }
    }

    /// Family implied by the file name component of `path`.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        Self::from_file_name(path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
    }

    /// Default online-cluster similarity threshold for this family.
    ///
    /// CAM++ cosines run hot (cross-speaker 0.70-0.90 on same-channel
    /// audio), hence 0.85; ERes2Net is calibrated at 0.65.
    #[must_use]
    pub fn default_similarity_threshold(self) -> f32 {
        match self {
            Self::CamPlusPlus => 0.85,
            Self::Eres2Net => 0.65,
        }
    }

    /// Online-cluster configuration the app uses for this family,
    /// optionally overriding the similarity threshold.
    ///
    /// CAM++ disables online centroid merging (`merge_threshold = 1.0`):
    /// its running centroids drift together and would cascade-merge; the
    /// offline refinement pass does the merging instead.
    #[must_use]
    pub fn cluster_config(self, similarity_threshold: Option<f32>) -> OnlineClusterConfig {
        let similarity_threshold =
            similarity_threshold.unwrap_or_else(|| self.default_similarity_threshold());
        match self {
            Self::CamPlusPlus => OnlineClusterConfig {
                similarity_threshold,
                merge_threshold: 1.0,
                ..OnlineClusterConfig::default()
            },
            Self::Eres2Net => OnlineClusterConfig {
                similarity_threshold,
                ..OnlineClusterConfig::default()
            },
        }
    }

    /// Load the embedder adapter for this family from `path`.
    ///
    /// # Errors
    ///
    /// Propagates the adapter's load error (missing file, bad graph).
    pub fn load(self, path: &Path) -> Result<Box<dyn SpeakerEmbedder>, DomainError> {
        Ok(match self {
            Self::CamPlusPlus => Box::new(CamPlusPlusEmbedder::new(path)?),
            Self::Eres2Net => Box::new(Eres2NetEmbedder::new(path)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_follows_the_file_name() {
        for name in [
            "campplus_zh_en_common_advanced.onnx",
            "campplus_en_voxceleb.onnx",
            "my_camplusplus.onnx",
        ] {
            assert_eq!(
                EmbedderFamily::from_file_name(name),
                EmbedderFamily::CamPlusPlus
            );
        }
        assert_eq!(
            EmbedderFamily::from_file_name("eres2net_en_voxceleb.onnx"),
            EmbedderFamily::Eres2Net
        );
        assert_eq!(
            EmbedderFamily::from_path(Path::new("/m/embedder/campplus_en_voxceleb.onnx")),
            EmbedderFamily::CamPlusPlus
        );
    }

    #[test]
    fn camplusplus_preset_matches_the_app() {
        let cfg = EmbedderFamily::CamPlusPlus.cluster_config(None);
        assert_eq!(cfg.similarity_threshold, 0.85);
        assert_eq!(cfg.merge_threshold, 1.0);
        let defaults = OnlineClusterConfig::default();
        assert_eq!(cfg.max_speakers, defaults.max_speakers);
        assert_eq!(cfg.max_centroid_history, defaults.max_centroid_history);
        assert_eq!(cfg.sticky_bias, defaults.sticky_bias);
    }

    #[test]
    fn eres2net_preset_matches_the_app() {
        let cfg = EmbedderFamily::Eres2Net.cluster_config(None);
        assert_eq!(cfg.similarity_threshold, 0.65);
        assert_eq!(
            cfg.merge_threshold,
            OnlineClusterConfig::default().merge_threshold
        );
    }

    #[test]
    fn threshold_override_only_changes_similarity() {
        let cfg = EmbedderFamily::CamPlusPlus.cluster_config(Some(0.7));
        assert_eq!(cfg.similarity_threshold, 0.7);
        assert_eq!(cfg.merge_threshold, 1.0);
        assert_eq!(
            EmbedderFamily::Eres2Net.default_similarity_threshold(),
            0.65
        );
    }
}
