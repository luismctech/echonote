//! Fixtures shared by the gated real-model acceptance tests: a synthetic
//! Spanish meeting with one planted fact, persisted in an in-memory store.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use echo_domain::{
    AudioFormat, CreateMeeting, MeetingId, MeetingStore, Segment, SegmentId, Speaker,
};
use echo_storage::SqliteMeetingStore;

/// Number the planted fact must surface as in the summary.
pub const PLANTED_NUMBER: &str = "437";
pub const PLANTED_SENTENCE: &str =
    "Quiero dejar claro que el presupuesto final del proyecto Atlas es de 437 mil euros.";

pub fn model_from_env() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("ECHO_LLM_MODEL").ok()?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("ECHO_LLM_MODEL points to {path:?} but it does not exist");
        None
    }
}

/// Deterministic pseudo-random index so the dialogue is varied but stable.
pub fn pick(seed: &mut u64, len: usize) -> usize {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 33) as usize) % len
}

/// Builds `count` varied Spanish meeting utterances rotating over four
/// speakers; the planted fact replaces the one at `plant_at`.
pub fn synthetic_dialogue(count: usize, plant_at: Option<usize>) -> Vec<String> {
    const TOPICS: [&str; 10] = [
        "la migración de la base de datos",
        "el módulo de facturación",
        "la campaña de captación de clientes",
        "el plan de formación del equipo",
        "la integración con el proveedor de pagos",
        "el rediseño de la aplicación móvil",
        "la auditoría de seguridad",
        "el calendario de lanzamientos",
        "la atención al cliente en horario nocturno",
        "el informe trimestral para la dirección",
    ];
    const OPENERS: [&str; 8] = [
        "Sobre {t}, creo que deberíamos revisar los plazos antes de comprometernos.",
        "Respecto a {t}, el equipo ha avanzado más de lo previsto esta semana.",
        "Me preocupa {t} porque todavía hay dependencias sin resolver.",
        "Hemos hablado con el cliente acerca de {t} y quiere ver una demostración pronto.",
        "En cuanto a {t}, propongo que Marta prepare un borrador para el jueves.",
        "Tengo dudas sobre {t}; necesitamos más datos antes de decidir nada.",
        "Para {t} haría falta reforzar el equipo con una persona más durante un mes.",
        "Veo bien {t}, aunque convendría documentar mejor los riesgos asociados.",
    ];
    const FOLLOW_UPS: [&str; 8] = [
        "Estoy de acuerdo, pero habría que validarlo con el área legal.",
        "Entonces lo dejamos anotado y lo retomamos en la próxima reunión.",
        "Puedo encargarme yo de hacer el seguimiento durante las dos próximas semanas.",
        "No me parece realista con los recursos que tenemos ahora mismo.",
        "Perfecto, así evitamos retrasos con el resto de entregables.",
        "Deberíamos medir el impacto antes de ampliar el alcance.",
        "Si nadie se opone, avanzamos con esa propuesta y revisamos el viernes.",
        "Eso nos obligaría a mover la fecha de entrega, ¿lo ven razonable?",
    ];

    let mut seed = 0x5EED_u64;
    (0..count)
        .map(|i| {
            if plant_at == Some(i) {
                return PLANTED_SENTENCE.to_string();
            }
            let topic = TOPICS[pick(&mut seed, TOPICS.len())];
            let opener = OPENERS[pick(&mut seed, OPENERS.len())].replace("{t}", topic);
            let follow = FOLLOW_UPS[pick(&mut seed, FOLLOW_UPS.len())];
            // Alternate single sentences and two-sentence turns.
            if i % 3 == 0 {
                opener
            } else {
                format!("{opener} {follow}")
            }
        })
        .collect()
}

/// Persists `utterances` as a meeting and returns its id plus the store.
pub async fn seed_meeting(utterances: &[String]) -> (MeetingId, Arc<SqliteMeetingStore>) {
    let store = Arc::new(SqliteMeetingStore::open_in_memory().await.unwrap());
    let id = MeetingId::new();
    store
        .create(CreateMeeting {
            id,
            title: "Reunión de seguimiento".into(),
            input_format: AudioFormat::WHISPER,
        })
        .await
        .unwrap();

    let speakers: Vec<Speaker> = ["Ana", "Luis", "Marta", "Pablo"]
        .iter()
        .enumerate()
        .map(|(slot, name)| Speaker::anonymous(slot as u32).renamed(*name))
        .collect();
    for speaker in &speakers {
        store.upsert_speaker(id, speaker).await.unwrap();
    }

    let mut clock_ms = 0_u32;
    let segments: Vec<Segment> = utterances
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let start_ms = clock_ms;
            clock_ms += 4_000 + (text.len() as u32) * 60;
            Segment {
                id: SegmentId::new(),
                start_ms,
                end_ms: clock_ms,
                text: text.clone(),
                speaker_id: Some(speakers[i % speakers.len()].id),
                confidence: None,
            }
        })
        .collect();
    store.append_segments(id, &segments).await.unwrap();
    (id, store)
}
