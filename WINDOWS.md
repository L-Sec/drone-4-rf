# Windows installation

The supported Windows release is a per-user installer. It does not require
administrator rights and does not place writable RF data under the application
directory.

## Install and run

1. Run `Drone-4-RF-<version>-Windows-x64-Setup.exe`.
2. Keep the default per-user installation location unless you have a reason to
   change it.
3. Launch **Drone 4-RF** from the Start menu.

Configuration, baselines, event databases, and captures are stored under
`%LOCALAPPDATA%\Drone4RF`. The uninstaller deliberately retains that directory
to prevent accidental loss of observations. Remove it manually only when you
intend to erase those files.

Live HackRF operation still requires the device to use the WinUSB driver. If the
radio is not detected, use Zadig to bind the HackRF to WinUSB.

## Build the installer

Building requires:

- the repository's Python 3.9 environment with PyInstaller and GUI dependencies;
- PothosSDR with Python 3.9 bindings and the HackRF support module; and
- Inno Setup 6.

From PowerShell at the repository root:

```powershell
.\scripts\build-windows-installer.ps1
```

The build validates the packaged application before creating the Setup EXE and
its `.sha256` checksum under `dist\windows`. Generated build content and
installers are excluded from Git.
