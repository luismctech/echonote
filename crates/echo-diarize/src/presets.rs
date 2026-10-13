//! Per-embedder diarization presets shared by the desktop shell and the
//! offline benchmarks.
//!
//! The embedder family decides the adapter to load, the online
//! clustering thresholds and the offline refinement threshold. It is
//! keyed by the model catalog id ([`EmbedderFamily::from_model_id`]);
//! the file-name heuristic ([`EmbedderFamily::from_file_name`]) is only
//! the fallback for user-supplied model files the catalog does not know.
//! `build_diarizer` in `src-tauri` and `echo-proto bench der` both build
//! their diarizer from here, so a benchmark run scores exactly the
//! configuration the app builds.

use std::path::Path;

use echo_domain::DomainError;

use crate::cam_plus_plus::CamPlusPlusEmbedder;
use crate::cluster::OnlineClusterConfig;
use crate::embedding::SpeakerEmbedder;
use crate::eres2net::Eres2NetEmbedder;
use crate::offline_refine::OfflineRefineConfig;

/// End-of-session refinement the diarizer runs (ADR 0001, D6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OfflineRefineMode {
    /// No offline pass; the embedding log is not kept.
    Off,
    /// Fold over-split online speakers together (legacy
    /// `SpeakersMerged` path). The default until a re-segmenting
    /// refine beats it on the held-out DER sets.
    #[default]
    MergeOnly,
    /// Re-cluster every logged span and move segments between speakers
    /// (`SpeakersReassigned` path).
    Resegment,
}

impl OfflineRefineMode {
    /// Parse a CLI or environment value: `off`, `merge` / `merge-only`,
    /// `reseg` / `resegment` (case-insensitive). `None` otherwise.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "merge" | "merge-only" | "merge_only" => Some(Self::MergeOnly),
            "reseg" | "resegment" => Some(Self::Resegment),
            _ => None,
        }
    }
}

/// Speaker embedder families the app knows how to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderFamily {
    /// 3D-Speaker CAM++ trained on zh + en ("common advanced", 192-dim).
    /// The app's recommended embedder.
    CamPlusPlusMultilingual,
    /// 3D-Speaker CAM++ trained on EN VoxCeleb (512-dim).
    CamPlusPlus,
    /// 3D-Speaker ERes2Net (EN VoxCeleb, 192-dim). Fallback family.
    Eres2Net,
}

impl EmbedderFamily {
    /// Every family, in catalog order.
    pub const ALL: [Self; 3] = [
        Self::CamPlusPlusMultilingual,
        Self::CamPlusPlus,
        Self::Eres2Net,
    ];

    /// Model catalog id of this family's model (the `embedder-*` ids of
    /// the desktop model catalog).
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::CamPlusPlusMultilingual => "embedder-campplus-multi",
            Self::CamPlusPlus => "embedder-camplusplus",
            Self::Eres2Net => "embedder-eres2net",
        }
    }

    /// Family of the catalog model `model_id`, or `None` for an id no
    /// family claims. This is the primary dispatch: a catalog model
    /// always loads the adapter and preset it was calibrated with,
    /// whatever its file is called.
    #[must_use]
    pub fn from_model_id(model_id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.model_id() == model_id)
    }

    /// Fallback family for a model file outside the catalog, guessed
    /// from its name: CAM++ names (`camplusplus` or `campplus`) that
    /// also contain `zh_en` or `multi` are the multilingual CAM++, other
    /// CAM++ names the EN CAM++, everything else ERes2Net. Prefer
    /// [`Self::from_model_id`] whenever the catalog id is known.
    #[must_use]
    pub fn from_file_name(file_name: &str) -> Self {
        let cam = file_name.contains("camplusplus") || file_name.contains("campplus");
        if cam && (file_name.contains("zh_en") || file_name.contains("multi")) {
            Self::CamPlusPlusMultilingual
        } else if cam {
            Self::CamPlusPlus
        } else {
            Self::Eres2Net
        }
    }

    /// [`Self::from_file_name`] applied to the file name of `path`.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        Self::from_file_name(path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
    }

    /// Default online-cluster similarity threshold for this family.
    ///
    /// Multilingual CAM++: 0.45, calibrated on held-out data with
    /// `echo-proto bench der` (T6b, `docs/spikes/2026-10-der-baseline.md`):
    /// DER is flat for 0.45-0.50 and rises sharply below 0.40 as speakers
    /// merge. The EN CAM++ (0.85) and ERes2Net (0.65) values predate the
    /// DER harness and are not calibrated with it.
    #[must_use]
    pub fn default_similarity_threshold(self) -> f32 {
        match self {
            Self::CamPlusPlusMultilingual => 0.45,
            Self::CamPlusPlus => 0.85,
            Self::Eres2Net => 0.65,
        }
    }

    /// Online-cluster configuration the app uses for this family,
    /// optionally overriding the similarity threshold.
    ///
    /// CAM++ disables online centroid merging (`merge_threshold = 1.0`):
    /// its running centroids drift together and would cascade-merge; the
    /// offline refinement pass does the merging instead. The multilingual
    /// CAM++ caps at 8 speakers: the default cap of 4 cost 3-6 DER points
    /// on the calibration data, where recordings have up to 6 speakers.
    #[must_use]
    pub fn cluster_config(self, similarity_threshold: Option<f32>) -> OnlineClusterConfig {
        let similarity_threshold =
            similarity_threshold.unwrap_or_else(|| self.default_similarity_threshold());
        match self {
            Self::CamPlusPlusMultilingual => OnlineClusterConfig {
                similarity_threshold,
                merge_threshold: 1.0,
                max_speakers: 8,
                ..OnlineClusterConfig::default()
            },
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

    /// End-of-session offline refinement configuration for this family.
    ///
    /// Multilingual CAM++: average-linkage merge threshold 0.50 (T6b).
    /// At the default 0.85 the pass never merged; 0.40-0.50 merges the
    /// online over-splits and is the flattest region of the calibration
    /// grid. Other families keep [`OfflineRefineConfig::default`].
    #[must_use]
    pub fn refine_config(self) -> OfflineRefineConfig {
        match self {
            Self::CamPlusPlusMultilingual => OfflineRefineConfig {
                merge_threshold: 0.50,
                ..OfflineRefineConfig::default()
            },
            Self::CamPlusPlus | Self::Eres2Net => OfflineRefineConfig::default(),
        }
    }

    /// End-of-session refinement mode for this family. `MergeOnly` for
    /// every family until re-segmentation is shown to lower DER on the
    /// held-out sets (ADR 0001, D6).
    #[must_use]
    pub fn refine_mode(self) -> OfflineRefineMode {
        OfflineRefineMode::MergeOnly
    }

    /// Load the embedder adapter for this family from `path`.
    ///
    /// # Errors
    ///
    /// Propagates the adapter's load error (missing file, bad graph).
    pub fn load(self, path: &Path) -> Result<Box<dyn SpeakerEmbedder>, DomainError> {
        Ok(match self {
            Self::CamPlusPlusMultilingual | Self::CamPlusPlus => {
                Box::new(CamPlusPlusEmbedder::new(path)?)
            }
            Self::Eres2Net => Box::new(Eres2NetEmbedder::new(path)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_model_ids_dispatch_to_their_family() {
        assert_eq!(
            EmbedderFamily::from_model_id("embedder-campplus-multi"),
            Some(EmbedderFamily::CamPlusPlusMultilingual)
        );
        assert_eq!(
            EmbedderFamily::from_model_id("embedder-camplusplus"),
            Some(EmbedderFamily::CamPlusPlus)
        );
        assert_eq!(
            EmbedderFamily::from_model_id("embedder-eres2net"),
            Some(EmbedderFamily::Eres2Net)
        );
        for unknown in ["", "embedder-", "embedder-wespeaker", "asr-small"] {
            assert_eq!(EmbedderFamily::from_model_id(unknown), None, "{unknown}");
        }
    }

    #[test]
    fn every_family_round_trips_through_its_model_id() {
        for family in EmbedderFamily::ALL {
            assert_eq!(
                EmbedderFamily::from_model_id(family.model_id()),
                Some(family)
            );
        }
    }

    #[test]
    fn family_follows_the_file_name() {
        assert_eq!(
            EmbedderFamily::from_file_name("campplus_zh_en_common_advanced.onnx"),
            EmbedderFamily::CamPlusPlusMultilingual
        );
        for name in ["campplus_en_voxceleb.onnx", "my_camplusplus.onnx"] {
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
    fn multilingual_campplus_uses_the_held_out_calibration() {
        // T6b: calibrated on MLS es dev + VoxConverse dev (held out).
        let family = EmbedderFamily::CamPlusPlusMultilingual;
        let cfg = family.cluster_config(None);
        assert_eq!(cfg.similarity_threshold, 0.45);
        assert_eq!(cfg.max_speakers, 8);
        assert_eq!(cfg.merge_threshold, 1.0);
        let defaults = OnlineClusterConfig::default();
        assert_eq!(cfg.max_centroid_history, defaults.max_centroid_history);
        assert_eq!(cfg.sticky_bias, defaults.sticky_bias);
        assert_eq!(family.refine_config().merge_threshold, 0.50);
    }

    #[test]
    fn en_campplus_keeps_its_uncalibrated_preset() {
        let cfg = EmbedderFamily::CamPlusPlus.cluster_config(None);
        assert_eq!(cfg.similarity_threshold, 0.85);
        assert_eq!(cfg.merge_threshold, 1.0);
        let defaults = OnlineClusterConfig::default();
        assert_eq!(cfg.max_speakers, defaults.max_speakers);
        assert_eq!(cfg.max_centroid_history, defaults.max_centroid_history);
        assert_eq!(cfg.sticky_bias, defaults.sticky_bias);
        assert_eq!(
            EmbedderFamily::CamPlusPlus.refine_config().merge_threshold,
            OfflineRefineConfig::default().merge_threshold
        );
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
    fn refine_config_is_part_of_the_preset() {
        for family in [
            EmbedderFamily::CamPlusPlusMultilingual,
            EmbedderFamily::CamPlusPlus,
            EmbedderFamily::Eres2Net,
        ] {
            let refine = family.refine_config();
            assert!((-1.0..=1.0).contains(&refine.merge_threshold));
            assert_eq!(
                refine.min_embeddings,
                OfflineRefineConfig::default().min_embeddings
            );
        }
    }

    #[test]
    fn every_family_defaults_to_merge_only_refine() {
        for family in EmbedderFamily::ALL {
            assert_eq!(
                family.refine_mode(),
                OfflineRefineMode::MergeOnly,
                "{family:?}"
            );
        }
        assert_eq!(OfflineRefineMode::default(), OfflineRefineMode::MergeOnly);
    }

    #[test]
    fn refine_mode_parses_cli_and_env_spellings() {
        for (text, mode) in [
            ("off", OfflineRefineMode::Off),
            ("merge", OfflineRefineMode::MergeOnly),
            ("merge-only", OfflineRefineMode::MergeOnly),
            ("RESEG", OfflineRefineMode::Resegment),
            ("resegment", OfflineRefineMode::Resegment),
        ] {
            assert_eq!(OfflineRefineMode::parse(text), Some(mode), "{text}");
        }
        assert_eq!(OfflineRefineMode::parse("vbx"), None);
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
