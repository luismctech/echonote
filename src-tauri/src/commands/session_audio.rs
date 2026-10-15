//! Optional per-session audio files.
//!
//! A session started with `saveAudio` writes its audio (16 kHz mono,
//! 16-bit PCM WAV, see `StreamingPipeline::with_audio_sink`) to a
//! partial file in the recordings directory, next to the meeting
//! database. The meeting id only exists once the recorder handled
//! `Started`, so the partial file is renamed to `<meeting_id>.wav` as
//! soon as the meeting exists (the pipeline keeps writing through its
//! open handle), or removed at the end when no meeting was recorded.
//! Partial files left by a crash are swept at startup. Deleting the
//! meeting deletes its file.

use std::io;
use std::path::{Path, PathBuf};

use echo_app::FrameSink;
use echo_audio::{WavSink, WriteOptions};
use echo_domain::{AudioFormat, AudioFrame, DomainError, MeetingId};

/// Sub-directory of the database directory that holds session audio.
pub(crate) const RECORDINGS_DIR: &str = "recordings";

/// Final audio file of a meeting.
pub(crate) fn meeting_audio_path(dir: &Path, meeting: MeetingId) -> PathBuf {
    dir.join(format!("{meeting}.wav"))
}

/// Fresh partial file name for a session that has no meeting id yet.
pub(crate) fn partial_audio_path(dir: &Path) -> PathBuf {
    // The counter keeps two sessions apart even when the clock reads
    // the same for both.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    dir.join(format!(
        "session-{}-{}.partial.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// Create the recordings directory if needed and open a 16 kHz mono
/// WAV sink at `path`.
pub(crate) fn create_sink(path: &Path) -> Result<Box<dyn FrameSink>, DomainError> {
    let io_error = |e: &dyn std::fmt::Display| {
        DomainError::Storage(format!("session audio at {}: {e}", path.display()))
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io_error(&e))?;
    }
    let sink = WavSink::create(path, AudioFormat::WHISPER, WriteOptions::default())
        .map_err(|e| io_error(&e))?;
    Ok(Box::new(WavFrameSink { inner: Some(sink) }))
}

/// [`WavSink`] behind the [`FrameSink`] port.
struct WavFrameSink {
    inner: Option<WavSink>,
}

impl FrameSink for WavFrameSink {
    fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
        let sink = self
            .inner
            .as_mut()
            .ok_or_else(|| DomainError::Storage("session audio already finished".into()))?;
        sink.write_frame(frame)
            .map_err(|e| DomainError::Storage(format!("session audio: {e}")))
    }

    fn finish(&mut self) -> Result<(), DomainError> {
        match self.inner.take() {
            Some(sink) => sink
                .finalize()
                .map(drop)
                .map_err(|e| DomainError::Storage(format!("session audio: {e}"))),
            None => Ok(()),
        }
    }
}

/// Move the finished partial file to the meeting's audio path, or
/// remove it when the session recorded no meeting. Returns the final
/// path when the audio was kept.
pub(crate) fn finalize(
    dir: &Path,
    partial: &Path,
    meeting: Option<MeetingId>,
) -> io::Result<Option<PathBuf>> {
    match meeting {
        Some(id) => {
            let path = meeting_audio_path(dir, id);
            std::fs::rename(partial, &path)?;
            Ok(Some(path))
        }
        None => {
            remove_if_present(partial)?;
            Ok(None)
        }
    }
}

/// Remove a meeting's audio file; a meeting without one is fine.
pub(crate) fn delete_meeting_audio(dir: &Path, meeting: MeetingId) -> io::Result<()> {
    remove_if_present(&meeting_audio_path(dir, meeting))
}

/// Remove `*.partial.wav` files a previous run left behind (crash,
/// forced quit before the meeting id was known). Call at startup, when
/// no session is writing. Returns how many files were removed.
pub(crate) fn sweep_partial_files(dir: &Path) -> io::Result<usize> {
    let entries = match std::fs::read_dir(dir) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        other => other?,
    };
    let mut removed = 0;
    for entry in entries {
        let path = entry?.path();
        let is_partial = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".partial.wav"));
        if is_partial {
            remove_if_present(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "echonote-audio-{name}-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ))
    }

    fn frame(samples: Vec<f32>) -> AudioFrame {
        AudioFrame {
            samples,
            format: AudioFormat::WHISPER,
            captured_at_ns: 0,
        }
    }

    #[test]
    fn sink_writes_a_16k_mono_16bit_wav_in_a_new_directory() {
        let dir = temp_dir("sink").join(RECORDINGS_DIR);
        let path = partial_audio_path(&dir);
        let mut sink = create_sink(&path).unwrap();
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        sink.accept(&frame(vec![-0.5; 1_600])).unwrap();
        sink.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(
            bytes.len(),
            44 + 3_200 * 2,
            "16-bit samples after a 44-byte header"
        );
        let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        assert_eq!(u16_at(22), 1, "mono");
        assert_eq!(u32_at(24), 16_000, "16 kHz");
        assert_eq!(u16_at(34), 16, "16-bit");
    }

    #[test]
    fn sink_rejects_audio_that_is_not_16k_mono() {
        let dir = temp_dir("format");
        let mut sink = create_sink(&partial_audio_path(&dir)).unwrap();
        let stereo = AudioFrame {
            samples: vec![0.0; 4],
            format: AudioFormat {
                sample_rate_hz: 48_000,
                channels: 2,
            },
            captured_at_ns: 0,
        };
        assert!(sink.accept(&stereo).is_err());
    }

    #[test]
    fn finalize_moves_the_partial_file_to_the_meeting_path() {
        let dir = temp_dir("keep");
        std::fs::create_dir_all(&dir).unwrap();
        let partial = partial_audio_path(&dir);
        std::fs::write(&partial, b"wav").unwrap();
        let meeting = MeetingId::new();

        let kept = finalize(&dir, &partial, Some(meeting)).unwrap();
        assert_eq!(kept, Some(meeting_audio_path(&dir, meeting)));
        assert!(!partial.exists());
        assert_eq!(
            std::fs::read(meeting_audio_path(&dir, meeting)).unwrap(),
            b"wav"
        );
    }

    #[test]
    fn finalize_removes_the_partial_file_without_a_meeting() {
        let dir = temp_dir("orphan");
        std::fs::create_dir_all(&dir).unwrap();
        let partial = partial_audio_path(&dir);
        std::fs::write(&partial, b"wav").unwrap();

        assert_eq!(finalize(&dir, &partial, None).unwrap(), None);
        assert!(!partial.exists());
    }

    #[test]
    fn deleting_a_meeting_audio_removes_it_and_tolerates_a_missing_file() {
        let dir = temp_dir("delete");
        std::fs::create_dir_all(&dir).unwrap();
        let meeting = MeetingId::new();
        std::fs::write(meeting_audio_path(&dir, meeting), b"wav").unwrap();

        delete_meeting_audio(&dir, meeting).unwrap();
        assert!(!meeting_audio_path(&dir, meeting).exists());
        delete_meeting_audio(&dir, meeting).unwrap();
        delete_meeting_audio(&temp_dir("never-created"), meeting).unwrap();
    }

    #[test]
    fn audio_moved_to_the_meeting_path_while_open_is_completed_there() {
        // The drain renames the file as soon as the meeting exists, while
        // the pipeline is still writing through the open handle.
        let dir = temp_dir("early");
        let partial = partial_audio_path(&dir);
        let mut sink = create_sink(&partial).unwrap();
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        let meeting = MeetingId::new();
        let path = finalize(&dir, &partial, Some(meeting)).unwrap().unwrap();
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        sink.finish().unwrap();

        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), 44 + 3_200 * 2);
        assert!(!partial.exists());
    }

    #[test]
    fn sweep_removes_leftover_partial_files_only() {
        let dir = temp_dir("sweep");
        std::fs::create_dir_all(&dir).unwrap();
        let partial = partial_audio_path(&dir);
        std::fs::write(&partial, b"wav").unwrap();
        let kept = meeting_audio_path(&dir, MeetingId::new());
        std::fs::write(&kept, b"wav").unwrap();

        assert_eq!(sweep_partial_files(&dir).unwrap(), 1);
        assert!(!partial.exists());
        assert!(kept.exists());
        assert_eq!(sweep_partial_files(&temp_dir("absent")).unwrap(), 0);
    }

    #[test]
    fn partial_paths_are_unique_and_never_look_like_meeting_audio() {
        let dir = PathBuf::from("/r");
        let a = partial_audio_path(&dir);
        let b = partial_audio_path(&dir);
        assert_ne!(a, b);
        assert!(a.to_string_lossy().ends_with(".partial.wav"));
    }
}
