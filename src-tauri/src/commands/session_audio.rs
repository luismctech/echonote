//! Optional per-session audio files.
//!
//! A session started with `saveAudio` keeps its audio (16 kHz mono,
//! 16-bit PCM WAV, see `StreamingPipeline::with_audio_sink`) as
//! `recordings/<meeting_id>.wav`, next to the meeting database.
//!
//! The meeting id only exists once the recorder handled `Started`, a
//! moment after the pipeline starts. Until the drain task publishes the
//! final path through an [`AudioTarget`], the sink holds the audio in
//! memory, so no file is ever renamed or removed while open (which
//! Windows forbids) and a session that never gets a meeting leaves no
//! file at all. The header is brought up to date after every chunk, so
//! the file stays readable if the process dies. Deleting the meeting
//! deletes its file.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use echo_app::FrameSink;
use echo_audio::{WavSink, WriteOptions};
use echo_domain::{AudioFormat, AudioFrame, DomainError, MeetingId};

/// Sub-directory of the database directory that holds session audio.
pub(crate) const RECORDINGS_DIR: &str = "recordings";

/// Final path of a session's audio, set once by the drain task when
/// the meeting exists.
pub(crate) type AudioTarget = Arc<OnceLock<PathBuf>>;

/// Audio held in memory while the target is unknown: 5 minutes at
/// 16 kHz (about 19 MB). The meeting normally exists within
/// milliseconds; past this the session is not saved.
const MAX_PENDING_SAMPLES: usize = 16_000 * 300;

/// Audio file of a meeting.
pub(crate) fn meeting_audio_path(dir: &Path, meeting: MeetingId) -> PathBuf {
    dir.join(format!("{meeting}.wav"))
}

/// A sink that writes 16 kHz mono WAV to `target` once it is set.
pub(crate) fn deferred_sink(target: AudioTarget) -> Box<dyn FrameSink> {
    Box::new(DeferredWavSink::new(target, MAX_PENDING_SAMPLES))
}

/// Remove a meeting's audio file; a meeting without one is fine.
pub(crate) fn delete_meeting_audio(dir: &Path, meeting: MeetingId) -> io::Result<()> {
    match std::fs::remove_file(meeting_audio_path(dir, meeting)) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// [`WavSink`] behind the [`FrameSink`] port, opened once its target
/// path is known.
struct DeferredWavSink {
    target: AudioTarget,
    /// Audio received before the target was set.
    pending: Vec<f32>,
    max_pending: usize,
    sink: Option<WavSink>,
}

impl DeferredWavSink {
    fn new(target: AudioTarget, max_pending: usize) -> Self {
        Self {
            target,
            pending: Vec::new(),
            max_pending,
            sink: None,
        }
    }

    /// The open file, created (with the pending audio) once the target
    /// is known; `None` while it is not.
    fn open(&mut self) -> Result<Option<&mut WavSink>, DomainError> {
        if self.sink.is_none() {
            let Some(path) = self.target.get() else {
                return Ok(None);
            };
            let mut sink = WavSink::create(path, AudioFormat::WHISPER, WriteOptions::default())
                .map_err(storage_error)?;
            let pending = std::mem::take(&mut self.pending);
            sink.write_frame(&whisper_frame(pending))
                .map_err(storage_error)?;
            self.sink = Some(sink);
        }
        Ok(self.sink.as_mut())
    }
}

impl FrameSink for DeferredWavSink {
    fn accept(&mut self, frame: &AudioFrame) -> Result<(), DomainError> {
        if frame.format != AudioFormat::WHISPER {
            return Err(DomainError::Storage(format!(
                "session audio must be 16 kHz mono, got {:?}",
                frame.format
            )));
        }
        let max_pending = self.max_pending;
        let pending = self.pending.len();
        match self.open()? {
            Some(sink) => {
                sink.write_frame(frame).map_err(storage_error)?;
                // Keep the header current so a killed process leaves a
                // readable file.
                sink.flush().map_err(storage_error)
            }
            None if pending + frame.samples.len() > max_pending => {
                self.pending = Vec::new();
                Err(DomainError::Storage(
                    "no meeting to save the session audio to".into(),
                ))
            }
            None => {
                self.pending.extend_from_slice(&frame.samples);
                Ok(())
            }
        }
    }

    fn finish(&mut self) -> Result<(), DomainError> {
        self.open()?;
        // Without a meeting the pending audio is dropped: no file.
        self.pending = Vec::new();
        match self.sink.take() {
            Some(sink) => sink.finalize().map(drop).map_err(storage_error),
            None => Ok(()),
        }
    }
}

fn whisper_frame(samples: Vec<f32>) -> AudioFrame {
    AudioFrame {
        samples,
        format: AudioFormat::WHISPER,
        captured_at_ns: 0,
    }
}

fn storage_error(e: impl std::fmt::Display) -> DomainError {
    DomainError::Storage(format!("session audio: {e}"))
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

    /// Format fields and the sample count the header declares.
    fn header(path: &Path) -> (u16, u32, u16, u32) {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        assert_eq!(
            bytes.len() as u32,
            44 + u32_at(40),
            "data matches the header"
        );
        (u16_at(22), u32_at(24), u16_at(34), u32_at(40) / 2)
    }

    fn files_in(dir: &Path) -> usize {
        std::fs::read_dir(dir).map_or(0, |d| d.count())
    }

    #[test]
    fn audio_before_the_target_is_kept_and_written_first() {
        let dir = temp_dir("deferred").join(RECORDINGS_DIR);
        let target = AudioTarget::default();
        let mut sink = deferred_sink(Arc::clone(&target));
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        assert_eq!(files_in(&dir), 0, "no file before the meeting exists");

        let path = meeting_audio_path(&dir, MeetingId::new());
        target.set(path.clone()).unwrap();
        sink.accept(&frame(vec![-0.5; 800])).unwrap();
        sink.finish().unwrap();

        assert_eq!(
            header(&path),
            (1, 16_000, 16, 2_400),
            "mono, 16 kHz, 16-bit"
        );
        let bytes = std::fs::read(&path).unwrap();
        let first = i16::from_le_bytes([bytes[44], bytes[45]]);
        let last = i16::from_le_bytes([bytes[bytes.len() - 2], bytes[bytes.len() - 1]]);
        assert!(first > 0 && last < 0, "buffered audio comes first");
    }

    #[test]
    fn a_target_set_before_finish_still_gets_the_buffered_audio() {
        let dir = temp_dir("late");
        let target = AudioTarget::default();
        let mut sink = deferred_sink(Arc::clone(&target));
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        let path = meeting_audio_path(&dir, MeetingId::new());
        target.set(path.clone()).unwrap();
        sink.finish().unwrap();
        assert_eq!(header(&path).3, 1_600);
    }

    #[test]
    fn a_session_without_a_meeting_leaves_no_file() {
        let dir = temp_dir("orphan");
        let mut sink = deferred_sink(AudioTarget::default());
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        sink.finish().unwrap();
        assert_eq!(files_in(&dir), 0);
    }

    #[test]
    fn too_much_audio_without_a_target_stops_saving() {
        let mut sink = DeferredWavSink::new(AudioTarget::default(), 2_000);
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        assert!(sink.accept(&frame(vec![0.5; 1_600])).is_err());
    }

    #[test]
    fn the_file_is_readable_without_finish() {
        // A killed process never finishes the sink.
        let dir = temp_dir("killed");
        let path = meeting_audio_path(&dir, MeetingId::new());
        let target = AudioTarget::default();
        target.set(path.clone()).unwrap();
        let mut sink = deferred_sink(target);
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        sink.accept(&frame(vec![0.5; 1_600])).unwrap();
        std::mem::forget(sink);
        assert_eq!(header(&path).3, 3_200);
    }

    #[test]
    fn sink_rejects_audio_that_is_not_16k_mono() {
        let mut sink = deferred_sink(AudioTarget::default());
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
}
