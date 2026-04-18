// Emit TAURI_* cfg flags and platform assets consumed by the tauri crate.
fn main() {
    tauri_build::build();

    // Best-effort short git hash. Fails silently when .git is absent
    // (e.g. when vendored in a source tarball) and defaults to "unknown".
    let hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=ECHO_GIT_HASH={hash}");
    println!("cargo:rerun-if-changed=../.git/HEAD");
}
