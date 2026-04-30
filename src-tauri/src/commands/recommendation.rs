//! Model recommendation engine based on hardware profile.
//!
//! Scores each model against the detected hardware and returns
//! the recommended model for both ASR and LLM tasks.

use serde::Serialize;

use super::hardware::{get_hardware_profile, HardwareProfile};

/// A single model recommendation.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelPick {
    /// Catalog model id (e.g. `"asr-large-v3-turbo"`).
    pub model_id: String,
    /// Short reason explaining the pick.
    pub reason: String,
    /// Confidence level: "high", "medium", "low".
    pub confidence: String,
}

/// Full recommendation result.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelRecommendation {
    /// Recommended ASR model.
    pub asr: ModelPick,
    /// Recommended LLM model (None if RAM is too low for any LLM).
    pub llm: Option<ModelPick>,
    /// Optional warning for the user (e.g. "RAM is below minimum").
    pub warning: Option<String>,
    /// The hardware profile used to produce this recommendation.
    pub hardware: HardwareProfile,
}

const GB: u64 = 1_073_741_824;

/// Analyse system hardware and recommend the optimal models.
#[tauri::command]
#[specta::specta]
pub fn get_model_recommendation() -> ModelRecommendation {
    let hw = get_hardware_profile();
    let ram_gb = hw.total_ram_bytes / GB;

    let (asr, llm, warning) = recommend(ram_gb, &hw);

    ModelRecommendation {
        asr,
        llm,
        warning,
        hardware: hw,
    }
}

fn recommend(ram_gb: u64, hw: &HardwareProfile) -> (ModelPick, Option<ModelPick>, Option<String>) {
    match ram_gb {
        0..=5 => (
            ModelPick {
                model_id: "asr-base".into(),
                reason: "Very limited RAM — only the smallest model fits safely".into(),
                confidence: "high".into(),
            },
            None,
            Some(
                "System RAM is very low. LLM features (summaries, chat) are not recommended."
                    .into(),
            ),
        ),
        6..=9 => (
            ModelPick {
                model_id: "asr-large-v3-turbo-q5".into(),
                reason: "Good quality-to-size ratio for 8 GB systems".into(),
                confidence: "high".into(),
            },
            Some(ModelPick {
                model_id: "llm-qwen3-8b".into(),
                reason: "Fits within available memory with quantization".into(),
                confidence: "medium".into(),
            }),
            Some("With 8 GB RAM, avoid running ASR and LLM simultaneously.".into()),
        ),
        10..=19 => {
            let asr = if hw.unified_memory && hw.cpu_cores >= 8 {
                ModelPick {
                    model_id: "asr-large-v3-turbo".into(),
                    reason: "Unified memory + multi-core allows full-precision turbo model".into(),
                    confidence: "high".into(),
                }
            } else {
                ModelPick {
                    model_id: "asr-large-v3-turbo-q5".into(),
                    reason: "Quantized turbo model balances quality and memory usage".into(),
                    confidence: "high".into(),
                }
            };
            (
                asr,
                Some(ModelPick {
                    model_id: "llm-qwen3-8b".into(),
                    reason: "Fits comfortably alongside ASR in 16 GB".into(),
                    confidence: "high".into(),
                }),
                None,
            )
        }
        _ => (
            ModelPick {
                model_id: "asr-large-v3-turbo".into(),
                reason: "Best quality model — plenty of RAM available".into(),
                confidence: "high".into(),
            },
            Some(ModelPick {
                model_id: "llm-qwen3-14b".into(),
                reason: "Largest available model fits well with 24+ GB RAM".into(),
                confidence: "high".into(),
            }),
            None,
        ),
    }
}
