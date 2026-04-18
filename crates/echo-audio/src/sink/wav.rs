//! WAV file sink.
//!
//! Writes 16-bit PCM RIFF/WAV by default — the universally compatible
//! format for inspection in QuickTime, Audacity and ffprobe. The writer
//! converts the `f32` samples produced by [`echo_domain::AudioFrame`]
//! into `i16` with simple clamping.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavSpec, WavWriter};

use echo_domain::{AudioFormat, AudioFrame};

/// Errors specific to the WAV sink. We keep them concrete here and let
/// the application layer wrap them into a [`echo_domain::DomainError`]
/// when crossing a port boundary.
#[derive(Debug, thiserror::Error)]
pub enum WavError {
    /// Underlying I/O failed.
    #[error("wav io: {0}")]
    Io(#[from] std::io::Error),
    /// Hound reported a write/encoding error.
    #[error("wav writer: {0}")]
    Encoder(#[from] hound::Error),
    /// A frame whose format diverged from the writer's spec was pushed.
    #[error("frame format mismatch: writer expects {expected:?}, frame is {actual:?}")]
    FormatMismatch {
        expected: AudioFormat,
        actual: AudioFormat,
    },
    /// `finalize` was already called.
    #[error("wav sink already finalized")]
    AlreadyFinalized,
}

/// Bit depth and sample format used to encode the WAV file. We default
/// to 16-bit PCM because Whisper and downstream tooling all accept it.
#[derive(Debug, Clone, Copy)]
pub struct WriteOptions {
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        }
    }
}

/// Append-only writer that turns a sequence of [`AudioFrame`]s into a
/// WAV file. Drop the value or call [`Self::finalize`] to flush headers.
pub struct WavSink {
    path: PathBuf,
    spec: AudioFormat,
    writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
    samples_written: u64,
}

impl WavSink {
    /// Create a new sink at `path` for the given format.
    pub fn create(
        path: impl Into<PathBuf>,
        spec: AudioFormat,
        options: WriteOptions,
    ) -> Result<Self, WavError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let wav_spec = WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate_hz,
            bits_per_sample: options.bits_per_sample,
            sample_format: options.sample_format,
        };
        let writer = WavWriter::create(&path, wav_spec)?;
        Ok(Self {
            path,
            spec,
            writer: Some(writer),
            samples_written: 0,
        })
    }

    /// Path the file is being written to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of PCM samples (per channel × channel count) written so far.
    #[must_use]
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }

    /// Push a captured frame. The format must match the one declared at
    /// [`Self::create`] time.
    pub fn write_frame(&mut self, frame: &AudioFrame) -> Result<(), WavError> {
        if frame.format != self.spec {
            return Err(WavError::FormatMismatch {
                expected: self.spec,
                actual: frame.format,
            });
        }
        let writer = self.writer.as_mut().ok_or(WavError::AlreadyFinalized)?;
        for sample in &frame.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            // Map [-1.0, 1.0] -> [i16::MIN+1, i16::MAX] symmetric.
            let scaled = (clamped * f32::from(i16::MAX)) as i16;
            writer.write_sample(scaled)?;
        }
        self.samples_written += frame.samples.len() as u64;
        Ok(())
    }

    /// Flush the WAV header and close the file.
    pub fn finalize(mut self) -> Result<PathBuf, WavError> {
        if let Some(w) = self.writer.take() {
            w.finalize()?;
        }
        Ok(self.path.clone())
    }
}

impl Drop for WavSink {
    fn drop(&mut self) {
        if let Some(w) = self.writer.take() {
            // Best-effort flush: errors at drop time would be eaten by
            // the runtime, so log and move on.
            if let Err(e) = w.finalize() {
                tracing::warn!(path = %self.path.display(), error = %e, "wav sink drop: finalize failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    fn frame(samples: Vec<f32>, fmt: AudioFormat) -> AudioFrame {
        AudioFrame {
            samples,
            format: fmt,
            captured_at_ns: 0,
        }
    }

    #[test]
    fn writes_a_valid_wav_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.wav");
        let fmt = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 1,
        };

        let mut sink = WavSink::create(&path, fmt, WriteOptions::default()).unwrap();
        // 0.5 s of a 440 Hz tone at -6 dBFS.
        let n = 8_000;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 16_000.0;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        sink.write_frame(&frame(samples, fmt)).unwrap();
        assert_eq!(sink.samples_written(), n as u64);
        let written_path = sink.finalize().unwrap();

        let reader = hound::WavReader::open(&written_path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(reader.len(), n as u32);
    }

    #[test]
    fn rejects_frame_with_wrong_format() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.wav");
        let fmt = AudioFormat::WHISPER;
        let mut sink = WavSink::create(&path, fmt, WriteOptions::default()).unwrap();
        let other = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let err = sink.write_frame(&frame(vec![0.0; 4], other)).unwrap_err();
        assert!(matches!(err, WavError::FormatMismatch { .. }));
    }
}
