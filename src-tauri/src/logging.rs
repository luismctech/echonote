//! Logging for the desktop shell: the console, plus a file in the OS
//! app log directory so a session can be diagnosed after the fact.
//!
//! The subscriber is installed first thing in [`init`], before Tauri
//! can resolve the log directory; the file is attached later with
//! [`attach_file`], from the app's `setup`. Lines logged before that
//! only reach the console. Log lines never carry transcript text or
//! audio.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Name of the log file inside the app log directory.
pub const LOG_FILE_NAME: &str = "echonote.log";

/// Size above which the log file is rotated at startup.
const ROTATE_AT_BYTES: u64 = 10 * 1024 * 1024;

/// The open log file, once [`attach_file`] has run.
static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Install the global subscriber: an env filter (`ECHO_LOG`, then
/// `RUST_LOG`, default `info`), a console layer and a file layer that
/// writes nothing until [`attach_file`] runs. Also bridges `log`
/// records into `tracing`. A second call is a no-op.
pub fn init() {
    let filter = EnvFilter::try_from_env("ECHO_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false).compact())
        .with(
            fmt::layer()
                .with_target(false)
                .with_ansi(false)
                .with_writer(|| FileWriter),
        )
        .try_init();
}

/// Start writing logs to [`LOG_FILE_NAME`] in `dir` (created if
/// missing), after rotating a file larger than 10 MB. Returns the log
/// file path.
pub fn attach_file(dir: &Path) -> io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(LOG_FILE_NAME);
    rotate_if_larger(&path, ROTATE_AT_BYTES)?;
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    *LOG_FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(file);
    Ok(path)
}

/// Move `path` to `<path>.1` when it is larger than `max_bytes`,
/// replacing any previous `.1`, so at most one old file is kept.
/// Returns whether it rotated. A missing file is not an error.
fn rotate_if_larger(path: &Path, max_bytes: u64) -> io::Result<bool> {
    let len = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    if len <= max_bytes {
        return Ok(false);
    }
    let mut backup = path.as_os_str().to_owned();
    backup.push(".1");
    let backup = PathBuf::from(backup);
    // Removed first: `rename` onto an existing file is not equally
    // supported on every platform.
    match std::fs::remove_file(&backup) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    std::fs::rename(path, &backup)?;
    Ok(true)
}

/// Writer handed to the file layer for each event; drops the output
/// while no file is attached.
struct FileWriter;

impl Write for FileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut slot = LOG_FILE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(file) = slot.as_mut() {
            // One locked `write_all` per event keeps lines whole when
            // several threads log at once.
            file.write_all(buf)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut slot = LOG_FILE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.as_mut().map_or(Ok(()), File::flush)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique temporary directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "echonote-logging-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn small_or_missing_log_is_not_rotated() {
        let dir = TempDir::new("small");
        let log = dir.0.join(LOG_FILE_NAME);
        assert!(!rotate_if_larger(&log, 10).unwrap(), "missing file");
        std::fs::write(&log, b"0123456789").unwrap();
        assert!(!rotate_if_larger(&log, 10).unwrap(), "at the threshold");
        assert_eq!(std::fs::read(&log).unwrap(), b"0123456789");
        assert!(!dir.0.join("echonote.log.1").exists());
    }

    #[test]
    fn large_log_moves_to_dot_one_replacing_the_previous_one() {
        let dir = TempDir::new("large");
        let log = dir.0.join(LOG_FILE_NAME);
        let backup = dir.0.join("echonote.log.1");
        std::fs::write(&backup, b"oldest").unwrap();
        std::fs::write(&log, b"0123456789A").unwrap();
        assert!(rotate_if_larger(&log, 10).unwrap());
        assert!(!log.exists());
        assert_eq!(std::fs::read(&backup).unwrap(), b"0123456789A");
    }

    #[test]
    fn attached_file_receives_log_output_in_a_created_directory() {
        let dir = TempDir::new("attach");
        let logs = dir.0.join("nested").join("logs");
        let path = attach_file(&logs).unwrap();
        assert_eq!(path, logs.join(LOG_FILE_NAME));
        FileWriter.write_all(b"line\n").unwrap();
        FileWriter.flush().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"line\n");
        // Detach so the directory can be removed on every platform.
        *LOG_FILE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}
