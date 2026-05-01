//! Render a [`Meeting`] (with optional [`Summary`]) into Markdown or plain text.
//!
//! These functions are pure: they take domain entities and return strings.
//! The Tauri shell layer handles path validation, file I/O, and IPC.

use echo_domain::{Meeting, Note, SpeakerId, Summary, SummaryContent};

/// Supported export formats.
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// GitHub-flavoured Markdown with headings and checkboxes.
    Markdown,
    /// Plain UTF-8 text.
    Txt,
}

/// Render a meeting (with optional summary) into the requested format.
pub fn render(meeting: &Meeting, summary: Option<&Summary>, format: ExportFormat) -> String {
    match format {
        ExportFormat::Markdown => render_markdown(meeting, summary),
        ExportFormat::Txt => render_plain_text(meeting, summary),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn format_ts(ms: u32) -> String {
    let total_s = ms / 1000;
    let m = total_s / 60;
    let s = total_s % 60;
    format!("{m:02}:{s:02}")
}

fn speaker_label(meeting: &Meeting, speaker_id: Option<&SpeakerId>) -> Option<String> {
    let sid = speaker_id?;
    meeting.speakers.iter().find(|sp| &sp.id == sid).map(|sp| {
        sp.label
            .clone()
            .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
    })
}

// ---------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------

fn render_markdown(meeting: &Meeting, summary: Option<&Summary>) -> String {
    let m = &meeting.summary;
    let date = &m.started_at;
    let dur_s = m.duration_ms / 1000;
    let dur = if dur_s < 60 {
        format!("{dur_s}s")
    } else {
        format!("{}m {:02}s", dur_s / 60, dur_s % 60)
    };

    let mut out = format!("# {}\n\n", m.title);
    out.push_str(&format!(
        "**Date:** {}  \n**Duration:** {}  \n**Language:** {}  \n**Segments:** {}\n\n",
        date,
        dur,
        m.language.as_deref().unwrap_or("?"),
        m.segment_count,
    ));

    if !meeting.speakers.is_empty() {
        out.push_str("**Participants:** ");
        let names: Vec<String> = meeting
            .speakers
            .iter()
            .map(|sp| {
                sp.label
                    .clone()
                    .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
            })
            .collect();
        out.push_str(&names.join(", "));
        out.push_str("\n\n");
    }

    out.push_str("---\n\n");

    if let Some(s) = summary {
        out.push_str("## Summary\n\n");
        out.push_str(&render_summary_body_md(s));
        out.push_str("---\n\n");
    }

    out.push_str("## Transcript\n\n");
    for seg in &meeting.segments {
        let ts = format_ts(seg.start_ms);
        let label = speaker_label(meeting, seg.speaker_id.as_ref());
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match label {
            Some(name) => out.push_str(&format!("**[{ts}] {name}:** {text}\n\n")),
            None => out.push_str(&format!("**[{ts}]** {text}\n\n")),
        }
    }

    if !meeting.notes.is_empty() {
        out.push_str("---\n\n## Notes\n\n");
        render_notes_md(&mut out, &meeting.notes);
    }

    out
}

fn render_summary_body_md(summary: &Summary) -> String {
    let mut out = String::new();

    match &summary.content {
        SummaryContent::General {
            summary: s,
            key_points,
            decisions,
            action_items,
            open_questions,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if !key_points.is_empty() {
                out.push_str("### Key points\n\n");
                for p in key_points {
                    out.push_str(&format!("- {p}\n"));
                }
                out.push('\n');
            }
            if !decisions.is_empty() {
                out.push_str("### Decisions\n\n");
                for d in decisions {
                    out.push_str(&format!("- {d}\n"));
                }
                out.push('\n');
            }
            if !action_items.is_empty() {
                out.push_str("### Action items\n\n");
                for ai in action_items {
                    let mut line = format!("- [ ] {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" — *{o}*"));
                    }
                    if let Some(d) = &ai.due {
                        line.push_str(&format!(" (due: {d})"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            if !open_questions.is_empty() {
                out.push_str("### Open questions\n\n");
                for q in open_questions {
                    out.push_str(&format!("- {q}\n"));
                }
                out.push('\n');
            }
        }
        SummaryContent::OneOnOne {
            summary: s,
            wins,
            blockers,
            growth_feedback,
            next_steps,
            follow_up_topics,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Wins", wins);
            md_list(&mut out, "### Blockers", blockers);
            md_list(&mut out, "### Growth feedback", growth_feedback);
            if !next_steps.is_empty() {
                out.push_str("### Next steps\n\n");
                for ai in next_steps {
                    let mut line = format!("- [ ] {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" — *{o}*"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            md_list(&mut out, "### Follow-up topics", follow_up_topics);
        }
        SummaryContent::SprintReview {
            summary: s,
            completed_items,
            carry_over,
            risks,
            next_sprint_priorities,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Completed", completed_items);
            md_list(&mut out, "### Carry-over", carry_over);
            md_list(&mut out, "### Risks", risks);
            md_list(
                &mut out,
                "### Next sprint priorities",
                next_sprint_priorities,
            );
        }
        SummaryContent::Interview {
            summary: s,
            quotes,
            themes,
            pain_points,
            opportunities,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if !quotes.is_empty() {
                out.push_str("### Quotes\n\n");
                for q in quotes {
                    out.push_str(&format!(
                        "> \"{}\"\n> — {}{}\n\n",
                        q.quote,
                        q.speaker,
                        q.context
                            .as_deref()
                            .map(|c| format!(" ({c})"))
                            .unwrap_or_default()
                    ));
                }
            }
            md_list(&mut out, "### Themes", themes);
            md_list(&mut out, "### Pain points", pain_points);
            md_list(&mut out, "### Opportunities", opportunities);
        }
        SummaryContent::SalesCall {
            summary: s,
            customer_context,
            pain_points,
            interest_signals,
            objections,
            next_steps,
            deal_stage_indicator,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            if let Some(ctx) = customer_context {
                out.push_str(&format!("**Customer context:** {ctx}\n\n"));
            }
            md_list(&mut out, "### Pain points", pain_points);
            md_list(&mut out, "### Interest signals", interest_signals);
            md_list(&mut out, "### Objections", objections);
            if !next_steps.is_empty() {
                out.push_str("### Next steps\n\n");
                for ai in next_steps {
                    out.push_str(&format!("- [ ] {}\n", ai.task));
                }
                out.push('\n');
            }
            if let Some(stage) = deal_stage_indicator {
                out.push_str(&format!("**Deal stage:** {stage}\n\n"));
            }
        }
        SummaryContent::Lecture {
            summary: s,
            concepts_covered,
            definitions,
            examples,
            homework_or_next,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            md_list(&mut out, "### Concepts covered", concepts_covered);
            if !definitions.is_empty() {
                out.push_str("### Definitions\n\n");
                for d in definitions {
                    out.push_str(&format!("- **{}**: {}\n", d.term, d.definition));
                }
                out.push('\n');
            }
            md_list(&mut out, "### Examples", examples);
            md_list(&mut out, "### Homework / next", homework_or_next);
        }
        SummaryContent::FreeText { text } => {
            out.push_str(text);
            out.push_str("\n\n");
        }
        _ => {}
    }
    out
}

fn md_list(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(heading);
    out.push_str("\n\n");
    for item in items {
        out.push_str(&format!("- {item}\n"));
    }
    out.push('\n');
}

fn render_notes_md(out: &mut String, notes: &[Note]) {
    for note in notes {
        let ts = format_ts(note.timestamp_ms);
        out.push_str(&format!("- **[{ts}]** {}\n", note.text));
    }
    out.push('\n');
}

// ---------------------------------------------------------------------------
// Plain text
// ---------------------------------------------------------------------------

fn render_plain_text(meeting: &Meeting, summary: Option<&Summary>) -> String {
    let m = &meeting.summary;
    let dur_s = m.duration_ms / 1000;
    let dur = if dur_s < 60 {
        format!("{dur_s}s")
    } else {
        format!("{}m {:02}s", dur_s / 60, dur_s % 60)
    };

    let mut out = format!("{}\n{}\n\n", m.title, "=".repeat(m.title.len()));
    out.push_str(&format!(
        "Date:     {}\nDuration: {}\nLanguage: {}\nSegments: {}\n",
        m.started_at,
        dur,
        m.language.as_deref().unwrap_or("?"),
        m.segment_count,
    ));

    if !meeting.speakers.is_empty() {
        let names: Vec<String> = meeting
            .speakers
            .iter()
            .map(|sp| {
                sp.label
                    .clone()
                    .unwrap_or_else(|| format!("Speaker {}", sp.slot + 1))
            })
            .collect();
        out.push_str(&format!("Participants: {}\n", names.join(", ")));
    }

    if let Some(s) = summary {
        out.push_str("\n\nSUMMARY\n-------\n\n");
        out.push_str(&render_summary_body_txt(s));
    }

    out.push_str("\n\nTRANSCRIPT\n----------\n\n");
    for seg in &meeting.segments {
        let ts = format_ts(seg.start_ms);
        let label = speaker_label(meeting, seg.speaker_id.as_ref());
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match label {
            Some(name) => out.push_str(&format!("[{ts}] {name}: {text}\n")),
            None => out.push_str(&format!("[{ts}] {text}\n")),
        }
    }

    if !meeting.notes.is_empty() {
        out.push_str("\n\nNOTES\n-----\n\n");
        for note in &meeting.notes {
            let ts = format_ts(note.timestamp_ms);
            out.push_str(&format!("[{ts}] {}\n", note.text));
        }
    }

    out
}

fn render_summary_body_txt(summary: &Summary) -> String {
    let mut out = String::new();

    match &summary.content {
        SummaryContent::General {
            summary: s,
            key_points,
            decisions,
            action_items,
            open_questions,
        } => {
            out.push_str(s);
            out.push_str("\n\n");
            txt_list(&mut out, "KEY POINTS", key_points);
            txt_list(&mut out, "DECISIONS", decisions);
            if !action_items.is_empty() {
                out.push_str("ACTION ITEMS\n");
                for ai in action_items {
                    let mut line = format!("  - {}", ai.task);
                    if let Some(o) = &ai.owner {
                        line.push_str(&format!(" ({o})"));
                    }
                    if let Some(d) = &ai.due {
                        line.push_str(&format!(" [due: {d}]"));
                    }
                    out.push_str(&line);
                    out.push('\n');
                }
                out.push('\n');
            }
            txt_list(&mut out, "OPEN QUESTIONS", open_questions);
        }
        _ => {
            let md = render_summary_body_md(summary);
            for line in md.lines() {
                let stripped = line
                    .trim_start_matches('#')
                    .trim_start_matches("**")
                    .trim_end_matches("**")
                    .trim_start_matches("- [ ] ")
                    .trim_start_matches("- ")
                    .trim_start_matches("> ");
                out.push_str(stripped.trim());
                out.push('\n');
            }
        }
    }
    out
}

fn txt_list(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(heading);
    out.push('\n');
    for item in items {
        out.push_str(&format!("  - {item}\n"));
    }
    out.push('\n');
}
