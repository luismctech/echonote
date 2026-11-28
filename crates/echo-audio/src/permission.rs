//! The "System Audio Recording" permission, asked up front.
//!
//! On macOS 14.2+ system audio comes from a Core Audio process tap, which
//! needs this permission (see [`crate::capture::macos_tap`]). Asking for
//! it during onboarding means the first recording does not have to fall
//! back to ScreenCaptureKit (and show a second, Screen Recording prompt)
//! while macOS waits for the answer.

/// Outcome of [`request_system_audio_permission`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemAudioPermission {
    /// The permission is granted; system audio uses the Core Audio tap.
    Granted,
    /// The user declined (now or earlier). System audio falls back to
    /// ScreenCaptureKit until the permission is enabled in System
    /// Settings › Privacy & Security.
    Denied,
    /// Nothing to ask: another platform, macOS older than 14.2, or the
    /// permission cannot be read. System audio uses its other adapter.
    NotNeeded,
}

/// Asks the OS for the permission when it applies and has not been
/// granted, and resolves once the user answers. Never prompts twice in
/// one process; a permission declined earlier resolves to
/// [`SystemAudioPermission::Denied`] without a prompt.
pub async fn request_system_audio_permission() -> SystemAudioPermission {
    #[cfg(target_os = "macos")]
    {
        crate::capture::macos_tap::request_permission().await
    }
    #[cfg(not(target_os = "macos"))]
    {
        SystemAudioPermission::NotNeeded
    }
}

// macOS decisions are tested next to the TCC code in `macos_tap`.
#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn other_platforms_need_no_permission() {
        assert_eq!(
            request_system_audio_permission().await,
            SystemAudioPermission::NotNeeded
        );
    }
}
