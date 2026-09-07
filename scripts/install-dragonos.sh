#!/usr/bin/env bash
#
# install-dragonos.sh - native install of drone4rf on DragonOS / Debian / Ubuntu
# for real-time capture + survey on the end user's machine.
#
# Why a script and not just "pip install .": SoapySDR's Python bindings are
# installed at OS level (apt), NOT from PyPI. A plain `python -m venv` cannot
# see them, so live HackRF use fails with "SoapySDR Python bindings not found".
# This script creates the venv with --system-site-packages so the apt-installed
# bindings are visible, and verifies the HackRF actually enumerates.
#
# DragonOS already ships SoapySDR + SoapyHackRF + hackrf, so the apt step is
# usually a no-op there; it is included for plain Debian/Ubuntu.
#
# Usage:
#   ./scripts/install-dragonos.sh              # venv install (recommended)
#   ./scripts/install-dragonos.sh --system     # install into system python3 (no venv)
#   ./scripts/install-dragonos.sh --no-apt      # skip the apt step (deps already present)
#   ./scripts/install-dragonos.sh --dev         # include the [dev] extra (pytest)

set -euo pipefail

VENV_DIR=".venv"
USE_VENV=1
RUN_APT=1
PIP_EXTRA=""

for arg in "$@"; do
  case "$arg" in
    --system) USE_VENV=0 ;;
    --no-apt) RUN_APT=0 ;;
    --dev)    PIP_EXTRA=".[dev]" ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Resolve repo root (this script lives in scripts/) and work from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n'  "$*" >&2; }

if [[ "$(uname -s)" != "Linux" ]]; then
  warn "This installer targets Linux (DragonOS/Debian/Ubuntu)."
  exit 1
fi

# --- 1. OS-level SDR stack (SoapySDR bindings + HackRF driver + tools) --------
if [[ "$RUN_APT" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    say "Installing SDR packages (SoapySDR bindings, HackRF module + tools)…"
    sudo apt-get update
    sudo apt-get install -y \
      python3-soapysdr soapysdr-tools soapysdr-module-hackrf hackrf \
      python3-venv python3-pip
  else
    warn "apt-get not found; skipping OS package step. Ensure SoapySDR + HackRF are present."
  fi
else
  say "Skipping apt step (--no-apt)."
fi

# --- 2. Python environment ----------------------------------------------------
if [[ "$USE_VENV" -eq 1 ]]; then
  # --system-site-packages is the crucial flag: it lets the venv see the
  # apt-installed python3-soapysdr, which is NOT pip-installable.
  say "Creating venv at $VENV_DIR (with --system-site-packages so SoapySDR is visible)…"
  python3 -m venv --system-site-packages "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY="$VENV_DIR/bin/python"
else
  say "Installing into system python3 (no venv)…"
  PY="python3"
fi

say "Installing Drone 4-RF (${PIP_EXTRA:-.})…"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "${PIP_EXTRA:-.}"

# --- 3. Non-root HackRF access (udev / plugdev) -------------------------------
if ! groups "$USER" | tr ' ' '\n' | grep -qx plugdev; then
  warn "User '$USER' is not in the 'plugdev' group; HackRF may need sudo."
  warn "  sudo usermod -aG plugdev $USER   # then log out and back in"
fi

# --- 4. Verify ----------------------------------------------------------------
say "Verifying SoapySDR sees the device…"
if command -v SoapySDRUtil >/dev/null 2>&1; then
  SoapySDRUtil --find || warn "SoapySDRUtil found no device (is the HackRF plugged in?)"
fi

say "Verifying Drone 4-RF can import SoapySDR and enumerate devices…"
"$PY" -m drone4rf devices || warn "drone4rf devices reported no hardware (simulator still works)."

cat <<EOF

$(say "Done.")
Run the real-time survey dashboard:

  $( [[ "$USE_VENV" -eq 1 ]] && echo "source $VENV_DIR/bin/activate" )
  drone4rf web            # launch dashboard, then pick the source
                            # (hackrf / simulated / file) and press Start
                            # in the browser

Then open the printed http://localhost:8731/ URL. To sanity-check without
hardware first: drone4rf scan --source simulated
EOF
