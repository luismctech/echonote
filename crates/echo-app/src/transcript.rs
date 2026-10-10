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
//! segment is never dropped. A segment that alone exceeds the budget is
//! the one exception to "never split": it is cut into pieces that each
//! fit (sentences, then words, then characters), every piece repeating
//! the speaker label so attribution survives.

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
/// is charged one extra token so the estimate never undercounts.
///
/// A line that alone exceeds `budget` is first split into pieces that
/// each fit (see [`split_oversized`]), so no emitted line is over budget
/// unless `budget` is too small for even one character after the label.
pub(crate) fn chunk_lines(
    lines: &[String],
    budget: usize,
    count: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut used = 0usize;
    for line in lines {
        let pieces = if count(line) > budget {
            split_oversized(line, budget, count)
        } else {
            vec![line.clone()]
        };
        for piece in &pieces {
            let cost = count(piece);
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
            current.push_str(piece);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Split one line that exceeds `budget` into lines that each fit.
///
/// The text up to and including the first `": "` (the speaker label,
/// plus the citation marker in the chat flavour) is repeated on every
/// piece, so each piece keeps its attribution and, in chat, its citable
/// segment id. The remaining body is cut at the coarsest boundary that
/// fits: sentence ends (`.`, `?`, `!`, `…` followed by whitespace), then
/// whitespace, then characters. Only the whitespace at a cut is
/// dropped; the cut never falls inside a UTF-8 character.
fn split_oversized(line: &str, budget: usize, count: &dyn Fn(&str) -> usize) -> Vec<String> {
    let (prefix, body) = line
        .split_once(": ")
        .map_or(("", line), |(label, body)| (&line[..label.len() + 2], body));
    let fits = |body: &str| count(&format!("{prefix}{body}")) <= budget;
    split_body(body, &fits, Boundary::Sentence)
        .into_iter()
        .map(|piece| format!("{prefix}{piece}"))
        .collect()
}

/// Boundary a body is cut at, from coarsest to finest.
#[derive(Clone, Copy)]
enum Boundary {
    Sentence,
    Whitespace,
    Char,
}

/// Greedy packing of `body` into pieces accepted by `fits`: atoms of the
/// given `boundary` are appended while they fit; an atom that does not
/// fit on its own is split again at the next finer boundary.
fn split_body(body: &str, fits: &dyn Fn(&str) -> bool, boundary: Boundary) -> Vec<String> {
    if fits(body) {
        return vec![body.to_string()];
    }
    let atoms = match boundary {
        Boundary::Sentence => sentence_atoms(body),
        Boundary::Whitespace => body.split_inclusive(char::is_whitespace).collect(),
        Boundary::Char => return split_by_chars(body, fits),
    };
    let finer = match boundary {
        Boundary::Sentence => Boundary::Whitespace,
        _ => Boundary::Char,
    };

    let mut pieces = Vec::new();
    let mut current = String::new();
    for atom in atoms {
        let candidate = format!("{current}{atom}");
        if fits(candidate.trim_end()) {
            current = candidate;
            continue;
        }
        if !current.trim().is_empty() {
            pieces.push(current.trim().to_string());
        }
        current.clear();
        if fits(atom.trim_end()) {
            current.push_str(atom);
        } else {
            pieces.extend(split_body(atom.trim(), fits, finer));
        }
    }
    if !current.trim().is_empty() {
        pieces.push(current.trim().to_string());
    }
    pieces
}

/// Cut `body` into the longest prefixes accepted by `fits`, always on a
/// char boundary and always advancing by at least one character.
fn split_by_chars(body: &str, fits: &dyn Fn(&str) -> bool) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest = body;
    while !rest.is_empty() {
        let ends: Vec<usize> = rest.char_indices().map(|(i, c)| i + c.len_utf8()).collect();
        // Binary search for the longest prefix that fits; the count of a
        // prefix grows with its length, and one char is always taken.
        let (mut lo, mut hi) = (1usize, ends.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if fits(&rest[..ends[mid - 1]]) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let (piece, tail) = rest.split_at(ends[lo - 1]);
        pieces.push(piece.to_string());
        rest = tail;
    }
    pieces
}

/// Split after every sentence terminator (`.`, `?`, `!`, `…`) that is
/// followed by whitespace; the whitespace stays with the sentence it
/// ends. A text without terminators is a single atom.
fn sentence_atoms(text: &str) -> Vec<&str> {
    let mut atoms = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((_, c)) = chars.next() {
        if !matches!(c, '.' | '?' | '!' | '…') {
            continue;
        }
        if let Some(&(j, next)) = chars.peek().filter(|(_, n)| n.is_whitespace()) {
            let end = j + next.len_utf8();
            chars.next();
            atoms.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        atoms.push(&text[start..]);
    }
    atoms
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

    /// Every line of every chunk, in order.
    fn flat_lines(chunks: &[String]) -> Vec<String> {
        chunks
            .iter()
            .flat_map(|c| c.split('\n').map(str::to_string))
            .collect()
    }

    fn without_whitespace(text: &str) -> String {
        text.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn oversized_segment_is_split_at_sentence_boundaries_keeping_the_label() {
        let input = lines(&["Ana: Uno dos. Tres cuatro. Cinco seis."]);
        let chunks = chunk_lines(&input, 25, &per_char);
        assert_eq!(
            flat_lines(&chunks),
            vec!["Ana: Uno dos.", "Ana: Tres cuatro.", "Ana: Cinco seis."]
        );
        assert!(chunks.iter().all(|c| per_char(c) <= 25));
    }

    #[test]
    fn oversized_sentence_falls_back_to_whitespace() {
        let input = lines(&["Ana: aaa bbb ccc ddd eee"]);
        let chunks = chunk_lines(&input, 14, &per_char);
        assert_eq!(
            flat_lines(&chunks),
            vec!["Ana: aaa bbb", "Ana: ccc ddd", "Ana: eee"]
        );
    }

    #[test]
    fn oversized_word_falls_back_to_char_boundaries_without_splitting_utf8() {
        let body = "ñ".repeat(23);
        let input = lines(&[&format!("Ana: {body}")]);
        let chunks = chunk_lines(&input, 10, &per_char);
        let pieces = flat_lines(&chunks);
        assert!(pieces.len() > 1);
        assert!(pieces
            .iter()
            .all(|p| p.starts_with("Ana: ") && per_char(p) <= 10));
        let rejoined: String = pieces.iter().map(|p| &p["Ana: ".len()..]).collect();
        assert_eq!(rejoined, body);
    }

    #[test]
    fn splitting_an_oversized_segment_drops_nothing_and_keeps_the_neighbours() {
        let long = "Ana: Primera frase larga, con comas. ¿Segunda frase? ¡Tercera! \
                    Cuarta sin puntuación final alguna que se pueda cortar";
        let input = lines(&["Beto: antes", long, "Beto: después"]);
        let chunks = chunk_lines(&input, 30, &per_char);
        let pieces = flat_lines(&chunks);
        assert_eq!(pieces.first().map(String::as_str), Some("Beto: antes"));
        assert_eq!(pieces.last().map(String::as_str), Some("Beto: después"));
        let middle = &pieces[1..pieces.len() - 1];
        assert!(middle.iter().all(|p| p.starts_with("Ana: ")));
        assert!(pieces.iter().all(|p| per_char(p) <= 30));
        let rejoined: String = middle.iter().map(|p| &p["Ana: ".len()..]).collect();
        assert_eq!(
            without_whitespace(&rejoined),
            without_whitespace(&long["Ana: ".len()..])
        );
    }

    #[test]
    fn chat_pieces_repeat_the_citation_prefix() {
        let prefix = "[seg:abc] (0:05) Ana: ";
        let input = lines(&[&format!("{prefix}Uno dos tres. Cuatro cinco seis.")]);
        let chunks = chunk_lines(&input, 40, &per_char);
        let pieces = flat_lines(&chunks);
        assert_eq!(pieces.len(), 2);
        assert!(pieces.iter().all(|p| p.starts_with(prefix)), "{pieces:?}");
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
