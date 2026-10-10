//! Transcript rendering shared by the summary and chat use cases.
//!
//! Both use cases feed the meeting transcript to the model as one
//! line per non-empty segment. This module owns that line format and
//! the token-budgeted chunking the summary uses when a transcript has
//! to be split (map-reduce). The legacy head + tail truncation
//! ([`render_truncated`]) remains only as the chat path until
//! retrieval replaces it.
//!
//! ## Chunking rule
//!
//! Chunks are cut **only on segment boundaries**. A segment is the
//! smallest unit the model can attribute (and, in chat, cite), so a
//! segment is never split mid-line and never dropped: when one line
//! alone exceeds the budget it becomes a chunk of its own and the
//! caller decides what to do with the overflow.

use std::fmt::Write;

use echo_domain::{Meeting, Segment, Speaker};

/// Prefix of the citation marker the chat model is told to copy.
/// Picked to be unambiguous (no overlap with English/Spanish prose)
/// and short (cheap on tokens).
pub(crate) const CITE_OPEN: &str = "[seg:";

/// Maximum characters of transcript text the chat feeds to the model
/// through [`render_truncated`]. Legacy, chat-only cap: the summary no
/// longer truncates and is budgeted in tokens instead.
pub(crate) const MAX_TRANSCRIPT_CHARS: usize = 6_000;

/// Line format of a rendered transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flavour {
    /// `Label: text` — what the summariser reads.
    Summary,
    /// `[seg:UUID] (m:ss) Label: text` — what the chat reads; the id is
    /// the citation target the model must copy verbatim.
    Chat,
}

/// Render one line per non-empty segment, in transcript order.
///
/// Segments whose text is blank after trimming are skipped. Speakers
/// resolve to their user label, else `Speaker N`; segments without a
/// known speaker read `Speaker`.
pub(crate) fn render_lines(meeting: &Meeting, flavour: Flavour) -> Vec<String> {
    meeting
        .segments
        .iter()
        .filter_map(|seg| {
            let text = seg.text.trim();
            if text.is_empty() {
                return None;
            }
            let label = seg
                .speaker_id
                .and_then(|id| meeting.speakers.iter().find(|s| s.id == id))
                .map(speaker_display_name)
                .unwrap_or_else(|| "Speaker".to_string());
            let mut line = String::new();
            if flavour == Flavour::Chat {
                push_citation_prefix(&mut line, seg);
            }
            line.push_str(&label);
            line.push_str(": ");
            line.push_str(text);
            Some(line)
        })
        .collect()
}

/// Group `lines` into chunks of at most `budget` tokens, cutting only
/// between lines. Each chunk is its lines joined with `\n`, so joining
/// the chunks with `\n` reproduces the full transcript.
///
/// `count` measures one piece of text in tokens; each joining newline
/// is charged one extra token so the estimate never undercounts. A
/// line that alone exceeds `budget` is emitted as its own chunk.
pub(crate) fn chunk_lines(
    lines: &[String],
    budget: usize,
    count: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut used = 0usize;
    for line in lines {
        let cost = count(line);
        if !current.is_empty() && used + 1 + cost > budget {
            chunks.push(std::mem::take(&mut current));
            used = 0;
        }
        if current.is_empty() {
            used = cost;
        } else {
            current.push('\n');
            used += 1 + cost;
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Legacy chat path: render the transcript as one block, keeping the head (4/6) and the
/// tail (2/6) of [`MAX_TRANSCRIPT_CHARS`] with an explicit elision
/// marker when it is longer. Surviving citation markers stay intact, so
/// a citation the model emits still points at a real segment id.
pub(crate) fn render_truncated(meeting: &Meeting, flavour: Flavour) -> String {
    let buf = render_lines(meeting, flavour).join("\n");
    if buf.chars().count() <= MAX_TRANSCRIPT_CHARS {
        return buf;
    }

    let head_chars = MAX_TRANSCRIPT_CHARS * 4 / 6;
    let tail_chars = MAX_TRANSCRIPT_CHARS - head_chars;
    let head: String = buf.chars().take(head_chars).collect();
    let tail: String = buf
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n[… (transcript truncated for length) …]\n{tail}")
}

fn push_citation_prefix(buf: &mut String, seg: &Segment) {
    buf.push_str(CITE_OPEN);
    buf.push_str(&seg.id.0.to_string());
    buf.push_str("] ");
    // Timestamp in mm:ss so the model can reference when things were said.
    let secs = seg.start_ms / 1000;
    let mm = secs / 60;
    let ss = secs % 60;
    write!(buf, "({mm}:{ss:02}) ").expect("write to String");
}

fn speaker_display_name(s: &Speaker) -> String {
    s.label
        .clone()
        .unwrap_or_else(|| format!("Speaker {}", s.slot + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    use echo_domain::{AudioFormat, MeetingId, MeetingSummary, SegmentId};
    use pretty_assertions::assert_eq;
    use time::OffsetDateTime;

    fn meeting_with(texts: &[&str]) -> Meeting {
        let speaker = Speaker::anonymous(0).renamed("Ana");
        let segments = texts
            .iter()
            .enumerate()
            .map(|(i, text)| Segment {
                id: SegmentId::new(),
                start_ms: (i as u32) * 61_000,
                end_ms: (i as u32) * 61_000 + 1_000,
                text: (*text).to_string(),
                speaker_id: Some(speaker.id),
                confidence: None,
            })
            .collect();
        Meeting {
            summary: MeetingSummary {
                id: MeetingId::new(),
                title: "T".into(),
                started_at: OffsetDateTime::now_utc(),
                ended_at: None,
                duration_ms: 0,
                language: None,
                segment_count: 0,
            },
            input_format: AudioFormat::WHISPER,
            segments,
            speakers: vec![speaker],
            notes: vec![],
        }
    }

    /// One token per character: makes budgets in tests exact.
    fn per_char(text: &str) -> usize {
        text.chars().count()
    }

    fn lines(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn summary_flavour_renders_label_and_text_and_skips_empty_segments() {
        let meeting = meeting_with(&["Hola", "   ", "Adiós"]);
        assert_eq!(
            render_lines(&meeting, Flavour::Summary),
            vec!["Ana: Hola", "Ana: Adiós"]
        );
    }

    #[test]
    fn chat_flavour_keeps_segment_markers_and_timestamps() {
        let meeting = meeting_with(&["Hola", "Adiós"]);
        let rendered = render_lines(&meeting, Flavour::Chat);
        assert_eq!(rendered.len(), 2);
        assert_eq!(
            rendered[1],
            format!("[seg:{}] (1:01) Ana: Adiós", meeting.segments[1].id.0)
        );
        assert!(
            rendered[0].starts_with(&format!("{CITE_OPEN}{}] (0:00) ", meeting.segments[0].id.0))
        );
    }

    #[test]
    fn unknown_speaker_reads_speaker() {
        let mut meeting = meeting_with(&["Hola"]);
        meeting.segments[0].speaker_id = None;
        assert_eq!(
            render_lines(&meeting, Flavour::Summary),
            vec!["Speaker: Hola"]
        );
    }

    #[test]
    fn chunker_returns_one_chunk_when_everything_fits() {
        let input = lines(&["aaa", "bbb", "ccc"]);
        // 3 + 1 + 3 + 1 + 3 = 11 tokens.
        assert_eq!(chunk_lines(&input, 11, &per_char), vec!["aaa\nbbb\nccc"]);
    }

    #[test]
    fn chunker_splits_on_segment_boundaries_within_budget() {
        let input = lines(&["aaa", "bbb", "ccc", "ddd"]);
        // Two lines cost 3 + 1 + 3 = 7 tokens; a third would reach 11.
        assert_eq!(
            chunk_lines(&input, 8, &per_char),
            vec!["aaa\nbbb", "ccc\nddd"]
        );
    }

    #[test]
    fn chunker_emits_an_oversized_segment_as_its_own_chunk() {
        let input = lines(&["aa", "bbbbbbbbbbbb", "cc"]);
        assert_eq!(
            chunk_lines(&input, 5, &per_char),
            vec!["aa", "bbbbbbbbbbbb", "cc"]
        );
    }

    #[test]
    fn chunker_returns_no_chunks_for_empty_input() {
        assert!(chunk_lines(&[], 10, &per_char).is_empty());
    }

    #[test]
    fn chunker_preserves_order_and_drops_nothing() {
        let input: Vec<String> = (0..40)
            .map(|i| format!("line-{i}-{}", "x".repeat(i % 7)))
            .collect();
        let chunks = chunk_lines(&input, 25, &per_char);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.join("\n"), input.join("\n"));
        for chunk in &chunks {
            let single_line = !chunk.contains('\n');
            assert!(per_char(chunk) <= 25 || single_line, "over budget: {chunk}");
        }
    }

    #[test]
    fn truncation_keeps_short_transcripts_untouched() {
        let meeting = meeting_with(&["Hola", "Adiós"]);
        assert_eq!(
            render_truncated(&meeting, Flavour::Summary),
            "Ana: Hola\nAna: Adiós"
        );
    }

    #[test]
    fn truncation_keeps_head_and_tail_with_a_marker() {
        let long = "x".repeat(MAX_TRANSCRIPT_CHARS);
        let meeting = meeting_with(&[&long, "TAIL"]);
        let rendered = render_truncated(&meeting, Flavour::Chat);
        assert!(rendered.contains("transcript truncated"), "{rendered}");
        assert!(rendered.starts_with(CITE_OPEN), "{rendered}");
        assert!(rendered.ends_with("TAIL"), "{rendered}");
    }
}
