# EchoNote — Release & Distribution Plan

## Overview

EchoNote is a Tauri 2.x desktop app. Distribution uses GitHub Releases as the
primary channel, with the Tauri updater plugin for in-app auto-updates.

---

## 1. Release Workflow

### Trigger
```bash
# Bump version in Cargo.toml, package.json, tauri.conf.json
# Then tag and push:
git tag v0.2.0
git push origin v0.2.0
```

### What Happens
1. The `release.yml` workflow triggers on `v*.*.*` tags
2. Builds run in parallel across 4 matrix entries:
   - **macOS ARM64** (Apple Silicon) → `.dmg`, `.app.tar.gz`
   - **macOS x64** (Intel) → `.dmg`, `.app.tar.gz`
   - **Linux x64** → `.deb`, `.AppImage`
   - **Windows x64** → `.msi`, `.exe` (NSIS)
3. Each build uploads artifacts to a **draft** GitHub Release
4. `latest.json` is auto-generated for the Tauri updater
5. You review the draft and publish it manually

### Artifacts Per Platform

| Platform | Installer | Updater Bundle | Signature |
|----------|-----------|---------------|-----------|
| macOS ARM | `.dmg` | `.app.tar.gz` | `.app.tar.gz.sig` |
| macOS x64 | `.dmg` | `.app.tar.gz` | `.app.tar.gz.sig` |
| Linux | `.deb`, `.AppImage` | `.AppImage` | `.AppImage.sig` |
| Windows | `.msi`, `.exe` | `.exe` | `.exe.sig` |

---

## 2. Secrets Setup (Required Before First Release)

### 2a. Tauri Updater Signing Key (Required)

```bash
# Install Tauri CLI if not already
pnpm tauri signer generate -w ~/.tauri/echonote.key

# This outputs:
#   Private key saved to ~/.tauri/echonote.key
#   Public key: dW50cnVzdGVkIGNvbW1lbnQ...
```

Add to GitHub repo secrets:
- `TAURI_SIGNING_PRIVATE_KEY` → contents of `~/.tauri/echonote.key`
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` → the password you chose (or empty)

Then update `tauri.conf.json`:
```json
"updater": {
  "pubkey": "<paste the public key here>"
}
```

### 2b. Apple Code Signing (Optional but Recommended for macOS)

Without code signing, macOS users see:
> "EchoNote.app is damaged and can't be opened."

**Requires an Apple Developer account ($99/year).**

| Secret | Description |
|--------|-------------|
| `APPLE_CERTIFICATE` | Base64-encoded `.p12` Developer ID certificate |
| `APPLE_CERTIFICATE_PASSWORD` | Password for the `.p12` file |
| `APPLE_SIGNING_IDENTITY` | e.g. `"Developer ID Application: Your Name (TEAMID)"` |
| `APPLE_API_ISSUER` | App Store Connect API Issuer ID |
| `APPLE_API_KEY` | App Store Connect API Key ID |
| `APPLE_API_KEY_CONTENT` | Contents of the `.p8` private key file |

**Workaround without Apple Developer account:**
Users can run: `xattr -cr /Applications/EchoNote.app` to bypass Gatekeeper.
Document this in the README under Installation.

### 2c. Windows Code Signing (Optional)

Without signing, Windows shows SmartScreen warnings for new apps.
Options:
- **Azure Code Signing** ($9/year through Azure Trusted Signing)
- **Traditional EV cert** (more expensive, ~$200-400/year)
- Or: users click "More info → Run anyway" on SmartScreen

---

## 3. Distribution Channels

### Phase 1 — GitHub Releases (Day 1)

**Primary channel.** All installers uploaded automatically.

Users install by:
1. Go to https://github.com/luismctech/echonote/releases/latest
2. Download the installer for their platform
3. Run the installer

**Auto-update:** Once installed, the Tauri updater checks `latest.json` on
each app launch and prompts users to update.

### Phase 2 — Homebrew Tap (macOS)

Create a tap repository: `luismctech/homebrew-tap`

```ruby
# Formula/echonote.rb
cask "echonote" do
  version "0.2.0"
  sha256 "..."

  url "https://github.com/luismctech/echonote/releases/download/v#{version}/EchoNote_#{version}_aarch64.dmg"
  name "EchoNote"
  desc "Private, local-first meeting transcription"
  homepage "https://github.com/luismctech/echonote"

  livecheck do
    url :url
    strategy :github_latest
  end

  app "EchoNote.app"
end
```

Users install:
```bash
brew tap luismctech/tap
brew install --cask echonote
```

**Automate:** Add a step to `release.yml` that opens a PR to the tap repo
with the updated version and SHA.

### Phase 3 — Winget (Windows)

Submit a manifest to [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs):

```yaml
PackageIdentifier: luismctech.EchoNote
PackageVersion: 0.2.0
InstallerType: msi
Installers:
  - Architecture: x64
    InstallerUrl: https://github.com/luismctech/echonote/releases/download/v0.2.0/EchoNote_0.2.0_x64-setup.msi
    InstallerSha256: ...
```

Users install:
```powershell
winget install luismctech.EchoNote
```

**Automate:** Use [vedantmgoyal9/winget-releaser](https://github.com/vedantmgoyal9/winget-releaser)
GitHub Action to auto-submit on release.

### Phase 4 — Flathub / AUR (Linux)

**Flathub** (broad reach):
- Create a `app.echonote.desktop.yml` Flatpak manifest
- Submit PR to [flathub/flathub](https://github.com/flathub/flathub)
- Users: `flatpak install flathub app.echonote.desktop`

**AUR** (Arch Linux):
- Create `PKGBUILD` that downloads the `.AppImage` or builds from source
- Users: `yay -S echonote-bin`

### Phase 5 — Website (Landing Page)

A simple landing page at `echonote.app` or `albertomcruz.github.io/echonote`:
- Hero with app screenshot
- Platform download buttons (auto-detect OS)
- Features list
- Privacy messaging (all on-device)
- Links to GitHub

---

## 4. Version Bump Checklist

Before tagging a release:

- [ ] Update `version` in `Cargo.toml` (workspace)
- [ ] Update `version` in `package.json`
- [ ] Update `version` in `src-tauri/tauri.conf.json`
- [ ] Update CHANGELOG.md
- [ ] Ensure CI is green on `develop`
- [ ] Merge `develop` → `main` via PR
- [ ] Tag `main`: `git tag v0.X.0 && git push origin v0.X.0`
- [ ] Review the draft release on GitHub, edit notes, publish

**Future automation:** Use `cargo release` or a custom script to sync versions
across all three files and create the tag.

---

## 5. Recommended Rollout Order

| Priority | Action | Effort |
|----------|--------|--------|
| **P0** | Generate updater signing key + add secrets | 10 min |
| **P0** | Test release workflow with `v0.1.0` tag | 30 min |
| **P1** | Document install instructions in README | 30 min |
| **P2** | Create Homebrew tap | 1 hour |
| **P2** | Submit to Winget | 1 hour |
| **P3** | Landing page | 2-4 hours |
| **P3** | Flathub / AUR packages | 2-4 hours |
| **P3** | Apple Developer account + code signing | 1-2 hours |

---

## 6. Architecture Diagram

```
                    git tag v0.2.0
                         │
                         ▼
               ┌─────────────────┐
               │  release.yml    │
               │  (GitHub Actions)│
               └────────┬────────┘
                         │
          ┌──────────────┼──────────────┐──────────────┐
          ▼              ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ macOS ARM│  │ macOS x64│  │ Linux x64│  │ Win x64  │
    │ .dmg     │  │ .dmg     │  │ .deb     │  │ .msi     │
    │ .app.gz  │  │ .app.gz  │  │ .AppImage│  │ .exe     │
    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │              │              │              │
         └──────────────┴──────┬───────┴──────────────┘
                               ▼
                    ┌───────────────────┐
                    │  GitHub Release   │
                    │  (draft → publish)│
                    │  + latest.json    │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Direct download   Homebrew     Winget
        from Releases     `brew install` `winget install`
              │
              ▼
        Tauri Updater (in-app auto-update)
```
