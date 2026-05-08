//! Model recommendation engine based on hardware profile.
//!
//! Scores each model against the detected hardware and returns
//! the recommended model for both ASR and LLM tasks.
//!
//! ## Decision logic
//!
//! The engine uses four hardware signals ranked by importance:
//! 1. Total RAM — primary gate for which models can run at all.
//! 2. Apple Silicon (`cpu_brand` contains "Apple M" + `unified_memory`) —
//!    unlocks Metal-accelerated full-precision models at lower RAM thresholds.
//! 3. Metal working set (`gpu_max_working_set`) — how much the GPU can
//!    actively hold; used to size LLM recommendations on Apple Silicon.
//! 4. Available RAM — if the system is already under pressure, add a warning.
//!
//! ## Memory budgets used
//!
//! | Model                  | ~RAM at inference |
//! |------------------------|-------------------|
//! | asr-base               |  160 MB           |
//! | asr-small              |  500 MB           |
//! | asr-large-v3-turbo-q5  |  620 MB           |
//! | asr-large-v3-turbo     |  1.7 GB           |
//! | llm-qwen3-4b Q4_K_M   |  3.0 GB           |
//! | llm-qwen3-8b Q4_K_M   |  5.5 GB           |
//! | llm-qwen3-14b Q4_K_M  |  9.5 GB           |

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
    // Apple Silicon: unified memory pool shared by CPU + Metal GPU.
    // Parse the brand string so that other vendors with unified_memory=true
    // (AMD APUs, etc.) are not incorrectly treated as Apple Silicon.
    let is_apple_silicon = hw.unified_memory && hw.cpu_brand.contains("Apple M");

    // Metal working set in GiB — the actual capacity available to the GPU.
    // M1 Pro 16 GB → ~10.7 GB; M1 8 GB → ~5.3 GB; M2 Max 32 GB → ~21.3 GB.
    let metal_gb = hw.gpu_max_working_set / GB;

    // Available RAM signal: warns when the OS is already under pressure.
    // Only meaningful on non-unified platforms (macOS compresses aggressively
    // so available RAM fluctuates a lot even when the system is healthy).
    let available_gb = hw.available_ram_bytes / GB;
    let memory_pressure = !is_apple_silicon && available_gb < 3;

    match ram_gb {
        // ── ≤ 5 GB: transcription only ───────────────────────────────────────
        0..=5 => (
            ModelPick {
                model_id: "asr-base".into(),
                reason: "Very limited RAM — only the smallest Whisper model (~148 MB) fits safely"
                    .into(),
                confidence: "high".into(),
            },
            None,
            Some(
                "System RAM is too low for LLM features (summaries, chat). \
                 Transcription only."
                    .into(),
            ),
        ),

        // ── 6-9 GB (e.g. 8 GB MacBook M1/M2, 8 GB Windows laptop) ──────────
        6..=9 => {
            let (asr_reason, llm_reason) = if is_apple_silicon {
                (
                    "Quantized turbo (~620 MB) runs via Metal on Apple Silicon 8 GB".into(),
                    "2.5 GB on disk; with macOS memory compression fits alongside Whisper".into(),
                )
            } else {
                (
                    "Good quality-to-size ratio for 8 GB systems".into(),
                    "Only 2.5 GB on disk (~3 GB RAM) — leaves headroom for ASR and OS".into(),
                )
            };
            let warn = if is_apple_silicon {
                Some(
                    "Running ASR + LLM simultaneously uses ~4 GB of your 8 GB. \
                     Close other apps for best performance."
                        .into(),
                )
            } else {
                Some(
                    "With 8 GB RAM the 4B LLM is the safest choice — \
                     it leaves room for ASR and OS."
                        .into(),
                )
            };
            (
                ModelPick {
                    model_id: "asr-large-v3-turbo-q5".into(),
                    reason: asr_reason,
                    confidence: "high".into(),
                },
                Some(ModelPick {
                    model_id: "llm-qwen3-4b".into(),
                    reason: llm_reason,
                    confidence: "high".into(),
                }),
                warn,
            )
        }

        // ── 10-15 GB (e.g. 12 GB MacBook Air M2, 12-15 GB Windows laptop) ───
        10..=15 => {
            if is_apple_silicon {
                // Full turbo is fine (1.7 GB); 4B LLM (3 GB) + ASR (1.7 GB) + OS (4 GB)
                // = ~9 GB → comfortable. Recommending 8B here risks pressure (5.5 + 1.7 + 4 = 11.2).
                (
                    ModelPick {
                        model_id: "asr-large-v3-turbo".into(),
                        reason: "Full-precision turbo runs via Metal on Apple Silicon".into(),
                        confidence: "high".into(),
                    },
                    Some(ModelPick {
                        model_id: "llm-qwen3-4b".into(),
                        reason:
                            "3 GB runtime footprint is safe alongside ASR on 12 GB unified memory"
                                .into(),
                        confidence: "high".into(),
                    }),
                    None,
                )
            } else {
                // x86 12-15 GB: use Q5 ASR (faster on CPU-only inference)
                (
                    ModelPick {
                        model_id: "asr-large-v3-turbo-q5".into(),
                        reason: "Quantized turbo is faster on CPU inference and uses only ~620 MB"
                            .into(),
                        confidence: "high".into(),
                    },
                    Some(ModelPick {
                        model_id: "llm-qwen3-4b".into(),
                        reason: "2.5 GB on disk — safe alongside ASR with OS overhead on 12 GB"
                            .into(),
                        confidence: "high".into(),
                    }),
                    if memory_pressure {
                        Some(
                            "System memory appears to be under pressure. \
                             Close other apps before recording."
                                .into(),
                        )
                    } else {
                        None
                    },
                )
            }
        }

        // ── 16-23 GB (e.g. M1 Pro/M2 Pro 16 GB, typical 16 GB laptop) ──────
        16..=23 => {
            if is_apple_silicon {
                // M1 Pro/M2 Pro 16 GB: Metal working set ~10.7 GB.
                // ASR turbo (1.7 GB) + 8B LLM (5.5 GB) + OS (5 GB) ≈ 12 GB → comfortable.
                // 14B LLM (9.5 GB) + ASR (1.7 GB) + OS (5 GB) ≈ 16 GB → too tight.
                let asr_reason = if metal_gb > 0 {
                    format!(
                        "Full-precision turbo — Metal can use up to {metal_gb} GB of your \
                         unified memory pool"
                    )
                } else {
                    "Full-precision turbo runs via Metal on Apple Silicon 16 GB".into()
                };
                (
                    ModelPick {
                        model_id: "asr-large-v3-turbo".into(),
                        reason: asr_reason,
                        confidence: "high".into(),
                    },
                    Some(ModelPick {
                        model_id: "llm-qwen3-8b".into(),
                        reason:
                            "~5.5 GB runtime fits comfortably alongside ASR in 16 GB unified memory"
                                .into(),
                        confidence: "high".into(),
                    }),
                    None,
                )
            } else {
                // x86 16 GB: no Metal acceleration, CPU inference is the bottleneck.
                // Full turbo on CPU is slow; Q5 is a better UX default.
                // Exception: high core count (≥12) makes full turbo viable.
                let asr = if hw.cpu_cores >= 12 {
                    ModelPick {
                        model_id: "asr-large-v3-turbo".into(),
                        reason: "High core count offsets CPU-only inference — full turbo is viable"
                            .into(),
                        confidence: "medium".into(),
                    }
                } else {
                    ModelPick {
                        model_id: "asr-large-v3-turbo-q5".into(),
                        reason:
                            "Quantized turbo is faster on CPU-only inference with similar accuracy"
                                .into(),
                        confidence: "high".into(),
                    }
                };
                (
                    asr,
                    Some(ModelPick {
                        model_id: "llm-qwen3-8b".into(),
                        reason: "Fits comfortably alongside ASR on 16 GB RAM".into(),
                        confidence: "high".into(),
                    }),
                    if memory_pressure {
                        Some(
                            "System memory is under pressure. \
                             Consider closing other applications before recording."
                                .into(),
                        )
                    } else {
                        None
                    },
                )
            }
        }

        // ── 24+ GB (e.g. M1 Max/M2 Max/M3 Pro 24-96 GB, 32 GB workstation) ─
        _ => {
            if is_apple_silicon {
                // ASR turbo (1.7 GB) + 14B LLM (9.5 GB) + OS (5 GB) ≈ 16 GB → fits in 24+ GB.
                let llm_reason = if metal_gb >= 16 {
                    format!(
                        "14B model (~9.5 GB runtime) fits well — Metal working set is \
                         {metal_gb} GB on your device"
                    )
                } else {
                    "14B model (~9.5 GB runtime) fits alongside ASR with room to spare on 24+ GB"
                        .into()
                };
                (
                    ModelPick {
                        model_id: "asr-large-v3-turbo".into(),
                        reason:
                            "Best accuracy — large unified memory pool enables fast Metal inference"
                                .into(),
                        confidence: "high".into(),
                    },
                    Some(ModelPick {
                        model_id: "llm-qwen3-14b".into(),
                        reason: llm_reason,
                        confidence: "high".into(),
                    }),
                    None,
                )
            } else {
                // High-end Windows/Linux workstation with dedicated GPU or lots of RAM.
                (
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
                )
            }
        }
    }
}
