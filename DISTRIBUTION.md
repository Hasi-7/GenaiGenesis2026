# Desktop Distribution

## Current Packaging Targets

The desktop client is packaged with `electron-builder` from `frontend/package.json`.

- Windows: `nsis`
- macOS: `dmg`
- Linux: `AppImage` and `deb`

Artifacts are emitted to `frontend/release/`.

## Local Build Commands

```bash
cd frontend
npm ci
npm run dist
```

## Release Automation

GitHub Actions release packaging lives in `.github/workflows/desktop-release.yml`.

- manual trigger: `workflow_dispatch`
- tagged releases: `v*`

The workflow builds platform-specific installers on native runners and uploads them as artifacts.

## Shipping Model

The desktop app is a thin client.

- It captures local camera/microphone.
- It streams media to the remote analysis backend over TCP `9000`.
- It does not bundle the Python backend.

## Runtime Configuration

The Electron main process persists a local runtime config in the user data directory as `client-config.json`.

Stored fields:

- `mediaHost`
- `mediaPort`
- `apiBaseUrl`
- `deviceId`
- `deviceName`

This lets shipped builds connect to a remote server without rebuilding the app.

## Signing / Hardening To Add Next

- macOS signing + notarization
- Windows code signing
- auto-update channel using GitHub Releases or a private release feed
