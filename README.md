# Drone 4-RF

Drone 4-RF is distributed as the `drone-4-rf` Python project. Import the
package as `drone4rf` and run the command-line interface with `drone4rf`.

> ### 🚧 Work in progress
> This is an active experiment, not a finished product. The pipeline runs
> end to end and is covered by 167 tests, but **every threshold, weight
> and signature in it was tuned against a single site, one antenna and
> one HackRF One.** Expect false positives - a busy 2.4 GHz band full of
> Wi-Fi and Bluetooth will produce "possible drone activity" until the
> analytics are calibrated more broadly. Nothing here is validated for
> safety-, security- or life-critical use.
>
> **If you run this, please consider sharing your calibration data** -
> labelled feature datasets from other sites are the single thing most
> likely to make the detection signatures actually generalise. The export
> contains no IQ, no message content and no absolute frequencies, and
> takes one command. See [CONTRIBUTING.md](CONTRIBUTING.md).

Passive SDR scanner that watches configurable frequency ranges, learns the
local RF background, and flags anomalous activity whose behavior is
consistent with drone command/control, video links, or motor/ESC
electromagnetic interference. Primary hardware: **HackRF One** (via
SoapySDR, so Airspy / RTL-SDR / LimeSDR / USRP / SDRplay can follow).

> ## ⚠️ Legal notice
> **This software is receive-only.** It contains no transmit path and
> cannot jam, spoof, or interfere with any radio system. **You are
> responsible for complying with all local radio, aviation, surveillance,
> and privacy laws** in your jurisdiction. Passive reception itself is
> regulated in some countries. Detections are *probabilistic indicators*,
> never proof - this tool will not and cannot output "confirmed drone."

## What it does (Stages 2–6 - complete)

**Optional ML (Stage 6):** a CPU-only scikit-learn subsystem, disabled
until *you* train it on *your* labeled observations - no fabricated
labels, ever. Mark false positives in the GUI and label authorized
drone-test assessments (`drone4rf ml label --event-id N --label
drone`), then:

```sh
python -m drone4rf ml build-dataset --site mysite    # features + labels from the DB
python -m drone4rf ml train                          # calibrated RF + CV metrics
python -m drone4rf ml train-anomaly                  # site novelty (no labels needed)
# review the printed metrics, then set ml.enabled: true in your config
```

The classifier's calibrated P(drone) becomes one more fusion term;
uncertain classifications are **rejected and contribute nothing**;
models refuse to load across feature-schema changes; and
`ml synth-test` exercises the whole pipeline on fabricated data that is
loudly marked as such.

### Browser dashboard (recommended)

No extra dependencies - the server is stdlib-only and the UI is one
static page:

```sh
python -m drone4rf web
```

Opens `http://localhost:8731` automatically: live spectrum with the
adaptive threshold overlaid, scrolling waterfall, active tracks,
color-coded assessments with evidence/penalty chips, and an event
reviewer for structured crowd-survey labels that also feed the ML classifier.
Start, Stop, and Calibrate are in the toolbar.

Use **Band focus** to narrow the receiver like an SDR tuning control. Search a
configured band name (for example `ism_2400`), or type any center frequency in
MHz, then click **Find** and **Focus scan**. If a scan is already running it is
stopped and restarted at the requested center with a fresh adaptive baseline;
this avoids applying a baseline learned at one frequency to another. **Visible
span MHz** zooms the spectrum and waterfall together without throwing away the
rest of the samples, and **Full span** returns to the receiver sample-rate view.
Bands wider than the instantaneous sample rate remain available in Sweep mode.

### Real-time refining survey

The **Refining** tab turns each streamed assessment and stored event into a
structured survey record. Pick `confirmed drone`, `false positive`, or
`unsure`; then identify the drone model or the controlled false-positive device
class, add optional notes and a contributor handle, and save. Labels overwrite
cleanly when a second review corrects the first one, while the original RF event
and behavioral feature payload remain unchanged.

The dashboard's **Export labels** button downloads the current JSON survey set.
**Export RF signatures** downloads the complete captured signature corpus,
including unlabeled observations, RF measurements, detector/assessment type,
behavioral features, evidence and penalties, quality flags, schema version, and
any attached survey verdict. The same exports are available without the
browser:

```sh
drone4rf ml export-labels --db data/events.db --format json --out labels.json
drone4rf ml export-labels --db data/events.db --format csv  --out labels.csv
drone4rf ml export-signatures --db data/events.db --format json --out signatures.json
drone4rf ml export-signatures --db data/events.db --format csv  --out signatures.csv
# Add --labeled-only to export-signatures when only reviewed records are wanted.
```

Existing v0.7 databases upgrade automatically on first open; see
[MIGRATION.md](MIGRATION.md). The export contains the stored event features,
structured verdict, model/device class, contributor, and UTC label time.

### Workshop recording and replay

The dashboard can preserve a live or simulated survey for a classroom where no
RF source or SDR will be available. Start a scan, enter an optional session
name, and click **Record** in the Workshop session toolbar. The recorder keeps:

- timestamped, peak-preserving spectrum and threshold frames (the source for
  reconstructing the scrolling waterfall);
- every detector firing and its behavioral RF feature dictionary;
- track promotions/closures, fused assessments, and labels submitted while the
  recording is active;
- a decimated I/Q waveform preview for live display and offline replay.

Recordings are saved under `data/sessions/` as versioned, gzip-compressed
`.dwsession` files. Select one in the same toolbar, choose a speed from 0.5x to
10x, and click **Replay**. The spectrum, waterfall, tracks, assessment cards,
and counters follow the original timing through the dashboard's normal stream,
with no radio attached. Replayed assessment cards are read-only so their old
event IDs cannot modify unrelated rows in the current event database.

The compact replay stream is always recorded. Three synchronized, opt-in
sidecars are available for a **single-band** scan:

- **CF32 (Inspectrum):** little-endian interleaved float32 I/Q. Open with
  `inspectrum -r <sample-rate> recording.cf32`; center frequency and sample rate
  are also written to the IQ metadata JSON.
- **IQ/WAV:** stereo PCM16 with I in the left channel and Q in the right. WAV
  headers carry the RF sample rate, and long captures are split before the RIFF
  4 GiB limit.
- **Detector CSV:** synchronized spectrum peak/mean/median/noise values,
  threshold averages, detector features, track behavior, fusion evidence, and
  explanations.

Raw IQ is deliberately opt-in and unavailable in sweep mode because joining
different center frequencies into one sample stream would be misleading. At
10 MS/s, CF32 writes about 80 MB/s (4.8 GB/minute) and PCM16 IQ/WAV about
40 MB/s; the dashboard shows the estimated combined rate before recording and
reports writer drops afterward. None of these formats contain demodulated
payloads, but raw IQ and spectra can reveal frequency use and activity timing,
so review files before distributing them.

The server binds **loopback only** by default. `--host 0.0.0.0` exposes
it to your LAN (it warns when you do) - anyone who can reach the port
can then start scans and read your event database, so only do that on a
trusted network. It validates the `Host` header (blocking DNS-rebinding)
and requires a per-run token on every state-changing request, so other
sites open in your browser cannot drive your radio.

### Qt desktop interface (alternative)

`pip install .[gui]` then

```sh
python -m drone4rf gui
```

Live spectrum with the adaptive detection threshold overlaid, rolling
waterfall, active signal tracks, color-coded assessments with their full
explanations, and an event browser where detections can be **marked as
false positives** (stored in the database for future classifier
training). Start/Stop, scan-vs-sweep mode, environment selection,
frozen-baseline toggle, and a Calibrate button (learn, then press Stop
to save) - plus receiver-health warnings in the status bar.

**Drone-oriented analytics (Stage 4):** on top of anomaly tracking, the
scanner now extracts behavioral signatures - sub-millisecond burst
timing, frequency-hopping patterns, motor/ESC-style harmonic combs,
multi-band correlation - rejects lookalikes (Wi-Fi, Bluetooth, fixed
infrastructure), and fuses everything into **explainable, confidence-
ranked assessments**:

Example output, produced by the built-in simulator so you can reproduce
it yourself with `python -m drone4rf scan --source sim --duration 10`:

```
[ASSESSMENT] PROBABLE DRONE ACTIVITY
  2.4370 GHz  span 7.00 MHz  confidence 0.71  (hopper, 163 observations)
  9 distinct narrowband channels between 2433.5 and 2440.5 MHz within 3 s;
  mean per-channel duty 0.44; channel-spacing regularity 0.74; peak SNR 44 dB
  across 163 channel detections; hop pattern partially matches Bluetooth/BLE
  behavior; harmonic comb: 4 peaks spaced ~452 kHz around 2438.17 MHz -
  consistent with motor/ESC switching noise, but equally with: switching power
  supply, LED driver, VFD / industrial motor controller. Confidence reduced by:
  Bluetooth-like hopping. Passive RF indicators only - not a confirmed drone.
```

Note what the engine does even on its own synthetic hopper: it flags the
Bluetooth resemblance, lists competing explanations for the harmonic
comb, and applies a penalty - then still refuses to call it confirmed.

Categories are capped at *high-confidence drone-related activity* -
"confirmed" is not in the vocabulary. Entities with too few
observations, EMI-only evidence, or receiver clipping are hard-capped at
*possible* regardless of score. All weights and thresholds live in the
`analytics:` config section.

## Multiband sweeping (Stage 3)

Cycles the HackRF through a
configurable band plan (2.4 GHz ISM and 5.8 GHz video by default;
sub-GHz presets included but disabled - allocations are regional),
with priority/revisit scheduling, adaptive re-scanning of frequencies
showing activity, per-band baselines saved as **named environments**,
operator-defined **exclusion ranges** for known emitters, and optional
short **triggered IQ captures** around persistent-track promotions
(off by default). Tracks are window-aware: a 2.4 GHz signal is not
"lost" while the radio is looking at 5.8 GHz.

```sh
# Learn per-band baselines for your site, saved under a name you choose
python -m drone4rf sweep --source hackrf --calibrate --environment mysite --duration 120

# Monitor the whole band plan against the frozen environment
python -m drone4rf sweep --source hackrf --frozen-baseline --environment mysite

# No hardware? Watch the sweep work against simulated multi-band emitters
python -m drone4rf sweep --source sim --duration 15
```

## Single-band mode (Stage 2)

- Streams IQ from a HackRF One, a **simulated source** (synthetic tones,
  bursts, frequency hoppers, EMI combs), or a **recorded IQ file**
  (cf32 or hackrf_transfer cs8).
- Welch PSD with DC-offset removal and HackRF DC-spike masking.
- Adaptive, poisoning-resistant per-bin background baseline (save/load,
  freeze).
- Energy detector (vs. baseline) and OS-CFAR detector.
- Frequency-clustered persistence tracking (repeated bursts vs.
  continuous links).
- SQLite event log with explanations and data-quality flags.
- Receiver-overload (clipping) and dropped-chunk warnings.
- Full pytest suite that runs with **no SDR hardware**.

See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture and the
Stage 5–6 roadmap (GUI visualization, optional ML).

## Installation

### DragonOS / Linux native install (recommended)

Drone 4-RF is designed to run natively on DragonOS for live HackRF survey use.
After cloning the repository, run:

```sh
cd drone-4-rf
./scripts/install-dragonos.sh --no-apt
```

DragonOS normally includes SoapySDR, SoapyHackRF, and the HackRF tools already.
The installer creates a Python virtual environment that can access those system
packages, installs Drone 4-RF, checks non-root device access, and verifies SDR
discovery. On Debian or Ubuntu, omit `--no-apt` so the installer installs the
required OS packages:

```sh
./scripts/install-dragonos.sh
```

Then activate the environment and launch the local dashboard:

```sh
source .venv/bin/activate
drone4rf web
```

Open the printed `http://localhost:8731/` address. The server binds to loopback
by default and is not exposed to the network unless you explicitly change its
host setting.

### Installing SoapySDR (only needed for live HackRF use)

SoapySDR's Python bindings are **not on PyPI**; install them at OS level:

- **Windows**: install [PothosSDR](https://downloads.myriadrf.org/builds/PothosSDR/)
  (`winget install Pothosware.PothosSDR`) - it bundles SoapySDR,
  SoapyHackRF, and Python bindings. **The bindings are built for one
  specific CPython version** (Python 3.9 in the 2021.07.25 build), so
  live HackRF use must run under that Python:

  ```powershell
  py -3.9 -m venv .venv39
  .venv39\Scripts\pip install -e .
  .venv39\Scripts\python -m drone4rf devices
  ```

  drone4rf auto-detects PothosSDR at `C:\Program Files\PothosSDR`
  (override with the `POTHOSSDR_ROOT` env var) and configures the DLL
  search path itself - no manual `PATH`/`PYTHONPATH` setup needed.
  If the HackRF doesn't enumerate, use [Zadig](https://zadig.akeo.ie/)
  (`winget install akeo.ie.Zadig`) to bind it to the WinUSB driver.
- **Debian/Ubuntu**: `sudo apt install python3-soapysdr soapysdr-module-hackrf hackrf`
- Verify: `SoapySDRUtil --find` should list the HackRF, and
  `python -m drone4rf devices` should print it.

  > **venv gotcha:** `python3-soapysdr` installs into the *system* Python, so a
  > plain `python -m venv .venv` cannot see it and live HackRF use fails with
  > "SoapySDR Python bindings not found". Create the venv with
  > `python3 -m venv --system-site-packages .venv` (or install into system
  > Python), so the apt-installed bindings are visible.

### Development or simulator-only install

Python 3.9 or newer is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

This installs NumPy, SciPy, PyYAML, and pytest-enough for the simulator, file
playback, and the complete test suite without SDR hardware.

### Windows installer

Windows users can install the desktop application with the per-user Setup EXE.
See [WINDOWS.md](WINDOWS.md) for installation, data-location, driver, and build
details. DragonOS remains the recommended platform for live survey use.

## Usage

```sh
# List detected SDR devices
python -m drone4rf devices

# Demo with the simulated source (no hardware): a CW tone, a bursty
# emitter, and a frequency hopper over synthetic noise
python -m drone4rf scan --source sim --duration 20

# Replay a recorded capture (hackrf_transfer -r capture.cs8 ...)
python -m drone4rf scan --source file --iq-file capture.cs8 --iq-format cs8

# Live HackRF scan of the configured band (default 2.437 GHz, 10 MS/s)
python -m drone4rf scan --source hackrf

# Calibrate: learn the local background for 60 s, then save the baseline
python -m drone4rf calibrate --source hackrf --duration 60

# Scan using the saved (frozen) baseline
python -m drone4rf scan --source hackrf --frozen-baseline
```

## Configuration

Every parameter is documented inline in
[config/default.yaml](config/default.yaml). Configuration is resolved in
this order, so a fresh clone works with no flags:

1. an explicit `--config <path>`;
2. `config/local.yaml` - **your** settings; git-ignored, because gain
   values, exclusion ranges and band plans describe your site;
3. `config/default.yaml` - the repo defaults;
4. built-in defaults (no sweep bands: band plans are regional, so they
   must be a deliberate choice rather than an assumption).

To tune it for your machine and site:

```sh
cp config/local.example.yaml config/local.yaml   # Windows: copy
```

and edit your copy - it is picked up automatically from then on. The
template starts from a reduced FFT overlap, which roughly halves CPU
cost per chunk; raise it again if your machine keeps up.

All output stays local: events land in `data/events.db` (SQLite,
inspect with any browser), baselines in `data/`, and the whole `data/`
directory is git-ignored - it describes the radio environment of
wherever you recorded it.

## Interpreting output

Stage 2 reports **RF anomalies**, not drones. A line like

```
[TRACK persistent] 2.4372 GHz  bw 1.1 MHz  SNR 21 dB  hits 12  duty 0.43
  energy+cfar agree; intermittent (burst-like) occupancy
```

means "something not in your learned background is transmitting here
repeatedly." Drone-specific scoring (hopping, multi-band correlation,
motor EMI, background rejection) arrives in Stage 4 and will still speak
in calibrated confidence bands, never certainty.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `SoapySDR module not found` | Bindings not installed, or interpreter version doesn't match the PothosSDR build (needs Python 3.9) - see install section. Sim & file sources work on any Python. |
| `LoadLibrary() failed` for `*Support.dll` | PothosSDR's DLLs not resolvable - run via drone4rf (it wires the DLL path automatically); if you import SoapySDR yourself, call `os.add_dll_directory` on PothosSDR's `bin` first. |
| Steady `queue drops` during scan | CPU can't keep up with the sample rate: lower `dsp.overlap` to 0.25, reduce `dsp.fft_size`, or lower `device.sample_rate` to 8 MS/s. Drop-oldest keeps results correct but reduces time coverage. |
| No devices found (Windows) | Wrong USB driver - run Zadig, select the HackRF, install WinUSB. |
| `O` overflow counts climbing | USB can't keep up: use a rear USB2/3 port, no hubs, lower `sample_rate`. |
| Clipping warnings | Front end overloaded: lower `vga_gain_db`, then `lna_gain_db`; keep `amp_enabled: false`. |
| Everything is a detection | Baseline not learned yet - run `calibrate` first, or let the scan warm up; check antenna isn't next to your own router. |
| Detections vanish after minutes | A persistent emitter was absorbed into the baseline - freeze the baseline after calibration (`--frozen-baseline`). |
| Weak sensitivity below 8 MS/s | HackRF baseband filters degrade; stay at ≥ 8 MS/s. |

## Contributing calibration data

The detectors work; the *calibration* is the weak link, because it comes
from one environment. If you have run drone4rf anywhere - with or
without a drone - a labelled feature dataset would genuinely help:

```sh
# review events in the dashboard (legacy CLI labels remain supported)
python -m drone4rf ml build-dataset --site <nickname> --out mydata.npz
python -m drone4rf ml export-labels --format json --out refined-labels.json
python -m drone4rf ml export-signatures --format json --out rf-signatures.json
```

The NPZ training dataset holds only abstract behavioural scores and your
labels: no IQ, no demodulated content, no absolute frequencies, no timestamps.
The survey JSON/CSV export deliberately includes stored event metadata for
research provenance, so review it before sharing. Negative data ("no drone was
present here") is just as useful as positive data.

Full details, what *not* to send, and the privacy/legal ground rules:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Security & privacy

- Receive-only; no TX code exists in this repository.
- No demodulation or payload decoding - only power-spectrum statistics
  and timing features are computed or stored.
- Workshop recordings always store compressed processed spectra, waveform
  previews, detections, tracks, assessments, and labels. Continuous CF32 and
  IQ/WAV sidecars are explicit opt-ins, single-band only, and may be very large.
- Short triggered IQ capture remains separately configurable and off by
  default.
- The event database is local SQLite; nothing is transmitted anywhere.

## Development

```sh
pytest            # full suite, no hardware needed
```

Package layout, threading model, and DSP details: [docs/DESIGN.md](docs/DESIGN.md).
