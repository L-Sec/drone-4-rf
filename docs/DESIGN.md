# Drone 4-RF - Engineering Design

Passive SDR scanner for detecting probable drone RF and electromagnetic
signatures. Primary hardware: HackRF One. **Receive-only by design: the
codebase contains no transmit path and never calls any TX API.**

---

## 1. System architecture

Six layers, each an independent package with typed interfaces:

```
+--------------------------------------------------------------+
|  UI layer          CLI (Stage 2)  ->  GUI/dashboard (Stage 5) |
+--------------------------------------------------------------+
|  Fusion layer      track manager, confidence fusion,          |
|                    explanation builder            (Stage 2/4) |
+--------------------------------------------------------------+
|  Detection layer   energy, OS-CFAR (Stage 2); burst, hopper,  |
|                    EMI-comb, multiband corr.       (Stage 4)  |
+--------------------------------------------------------------+
|  Modeling layer    adaptive per-bin baseline, environments    |
+--------------------------------------------------------------+
|  DSP layer         DC removal, windowed Welch PSD, dBFS,      |
|                    artifact masking, noise-floor estimation   |
+--------------------------------------------------------------+
|  Source layer      SoapySDR (HackRF), simulated IQ, IQ files  |
+--------------------------------------------------------------+
         |                    |
     SQLite event store   structured logging
```

### Module map (Stage 2 implementation)

| Module | Responsibility |
|---|---|
| `drone4rf.config` | Validated dataclass configuration from YAML |
| `drone4rf.sdr.base` | `SDRSource` ABC, `IQChunk`, `SourceHealth` |
| `drone4rf.sdr.hackrf` | SoapySDR-backed HackRF RX source |
| `drone4rf.sdr.simulated` | Deterministic synthetic-emitter source |
| `drone4rf.sdr.file_source` | Playback of cf32 / cs8 (hackrf_transfer) files |
| `drone4rf.dsp.preprocess` | DC removal, Welch PSD, dBFS, DC-spike mask, clip check |
| `drone4rf.dsp.noise_floor` | Robust (median/MAD) noise-floor estimation |
| `drone4rf.baseline` | Adaptive per-bin background model, save/load |
| `drone4rf.detectors.*` | Energy and OS-CFAR detectors -> `Detection` |
| `drone4rf.tracking` | Frequency-clustered persistence tracks |
| `drone4rf.events` | SQLite event store |
| `drone4rf.pipeline` | Threads, bounded queue, health stats, orchestration |
| `drone4rf.cli` | `devices` / `scan` / `calibrate` commands |

## 2. Data flow

```
SDR / sim / file
   |  IQChunk (complex64, center, fs, timestamp, seq, drop count)
   v
bounded Queue (drop-oldest, drops counted)          [acquisition thread]
   |
   v
DC offset removal -> clip check -> Welch PSD (Hann, 50% overlap)
   -> dBFS -> DC-spike mask                          [processing thread]
   |
   +--> BaselineModel.update()  (skipped when frozen)
   +--> noise-floor estimate (median/MAD)
   |
   v
Detectors (energy vs. baseline threshold, OS-CFAR)
   |  list[Detection]
   v
SignalTracker (cluster by frequency, count hits/misses)
   |  TrackEvent on promotion (persistent) and close (with burst stats)
   v
EventStore (SQLite)  +  UI callback (CLI printout)
```

## 3. Threading model

Two threads plus the main thread; communication only through a bounded
`queue.Queue` and thread-safe stats:

- **Acquisition thread** - blocks on the SDR stream, wraps buffers into
  `IQChunk`s, `put_nowait` into the queue. On `queue.Full` it drops the
  *oldest* chunk (fresh data beats stale data for live monitoring) and
  increments a drop counter. Never does DSP.
- **Processing thread** - pops chunks, runs the DSP/detect/track/store
  chain. All numerical work is NumPy-vectorized so the GIL is released
  for the bulk of the time.
- **Main thread** - CLI status display and Ctrl+C handling. Shutdown is a
  single `threading.Event`; the pipeline joins both threads and closes
  the SDR stream in a `finally` block.

No multiprocessing in Stage 2: one HackRF channel at 10-20 MS/s with a
4096-point Welch PSD is comfortably real-time on a modern laptop core.
The queue bound (default 8 chunks) caps memory regardless of load.

## 4. DSP pipeline detail

1. **DC offset**: subtract per-chunk complex mean (removes most of the
   HackRF DC term before the FFT).
2. **Welch PSD**: Hann window, 50 % overlap, `fft_size` 4096. Power is
   normalized by the window's coherent gain so a full-scale tone reads
   ~0 dBFS; all measurements are *relative* dBFS (HackRF is uncalibrated).
3. **fftshift** so bin 0 = lowest frequency; frequency axis =
   `center + fftshift(fftfreq(N, 1/fs))`.
4. **DC-spike mask**: replace the center ±`dc_mask_bins` bins with the
   median of their neighbors - the residual LO/DC artifact must never
   reach the detectors.
5. **Clip check**: fraction of samples with |I| or |Q| > 0.95 full scale;
   above the configured fraction the chunk is flagged and the event
   records a quality warning (detections during clipping are suspect).

## 5. Background modeling

Per-bin asymmetric exponentially weighted model (robust in spirit to a
median/MAD tracker but O(N) per update and streaming-friendly):

- `level_db[bin]` rises with small `alpha_up` (0.02) and falls with large
  `alpha_down` (0.2). A new persistent emitter therefore takes minutes,
  not seconds, to be absorbed - and bins currently *above* threshold get
  an additional `hot_bin_damping` factor (0.1×), the anti-poisoning
  guard.
- `spread_db[bin]` is an EW mean absolute deviation.
- Threshold: `level + max(k_mad * spread, min_delta_db)`.
- Freeze flag stops all adaptation (post-calibration operation).
- Save/load as `.npz` with center/fs/fft-size metadata; a baseline is
  refused at load if it does not match the current tuning - this is the
  seed of Stage 3's named-environment store.

## 6. Detection fusion (initial formula)

Stage 2 ships per-detector scores plus persistence tracking; Stage 4
combines them. The fusion design, fixed now so detectors emit compatible
scores in [0, 1]:

```
raw = Σ_i w_i · s_i          (evidence terms)
    - Σ_j p_j · b_j          (background-similarity / quality penalties)

confidence = 1 / (1 + exp(-(raw - bias)))     # logistic squash
```

Initial weights (to be re-fit against labeled captures in Stage 4):

| term | weight |
|---|---|
| anomaly-above-baseline score | 1.0 |
| burst-pattern score | 1.5 |
| frequency-hopping score | 2.0 |
| motor/ESC EMI score | 1.0 (supporting evidence only) |
| multi-band correlation score | 2.0 |
| known-drone fingerprint score | 2.5 |
| persistence (scan cycles seen, saturating) | 1.0 |
| **penalties** | |
| Wi-Fi/BT background similarity | −1.5 |
| receiver overload / clipping present | −2.0 |
| single observation only | −1.0 |

Category mapping: `<0.2` background, `<0.4` unclassified RF activity,
`<0.6` possible, `<0.8` probable, `≥0.8` high-confidence drone-related
activity. **"Confirmed drone" is never emitted** - no passive RF-only
method independently validates identity. Every event carries a
human-readable explanation assembled from the contributing terms.

## 7. Configuration schema

See `config/default.yaml` - every key is commented there and validated in
`drone4rf/config.py` (typed dataclasses; hard errors with the offending
key path). HackRF gain values are checked against the hardware's legal
steps (LNA 0–40/8, VGA 0–62/2) and sample rate against 2–20 MS/s.

## 8. Database schema (SQLite, WAL mode)

```sql
CREATE TABLE events (
  id               INTEGER PRIMARY KEY,
  ts_utc           TEXT NOT NULL,       -- ISO 8601
  ts_local         TEXT NOT NULL,
  source           TEXT NOT NULL,       -- hackrf | simulated | file
  detector         TEXT NOT NULL,
  kind             TEXT NOT NULL,       -- detection | track_persistent | track_closed
  center_hz        REAL NOT NULL,
  bandwidth_hz     REAL,
  peak_db          REAL,
  avg_db           REAL,
  snr_db           REAL,
  duration_s       REAL,
  score            REAL,
  confidence_label TEXT,
  explanation      TEXT,
  features_json    TEXT,                -- extracted feature vector
  quality_json     TEXT,                -- clipping, drops, warnings
  config_version   TEXT
);
```

Stage 3 adds `environments` (named baselines) and `iq_captures` (short
triggered recordings, disabled by default); Stage 6 adds `model_version`.

## 9. Safety, legality, privacy

- **No transmit path.** The source layer only ever opens RX streams; no
  TX API is imported anywhere. The tool cannot jam, spoof, or inject.
- **No demodulation of content.** Only power-spectral statistics and
  timing features are computed. Protocol-family recognition (Stage 4) is
  a classifier over spectral shape/timing, never payload decoding.
- IQ recording is **off by default**; when enabled (Stage 3) it is
  limited to short triggered captures with a retention policy.
- A legal notice (radio/aviation/surveillance/privacy compliance is the
  operator's responsibility) prints on every start and heads the README.
- Uncertainty is first-class: capped confidence vocabulary, mandatory
  explanations, and explicit quality warnings on degraded data.

## 10. HackRF One limitations and mitigations

| Limitation | Mitigation |
|---|---|
| Half-duplex, single tuner | Sweep scheduler (Stage 3) revisits bands; no pretense of simultaneous coverage |
| 8-bit ADC, ~48 dB usable dynamic range | Conservative default gains, clip detection + event quality flags |
| DC spike / LO leakage | Per-chunk DC removal + center-bin masking before detection |
| Image artifacts | Baseline absorbs stationary images; artifact tests in suite |
| USB throughput / dropped buffers | Overflow counting via SoapySDR return codes, drop-oldest queue, drop-rate warnings |
| Frequency-dependent sensitivity | Relative dBFS + per-band baselines; optional correction curves later |
| Filters degrade < 8 MS/s | Config validator warns; default 10 MS/s |

## 11. Expected bottlenecks

1. **FFT throughput** - Welch @ 10 MS/s / 4096 pt / 50 % overlap ≈ 5 k
   FFTs/s: fine in NumPy; PyFFTW is an optional drop-in if profiling
   demands (`fast` extra).
2. **Retune latency (Stage 3)** - HackRF retune ≈ 5–50 ms + settling;
   dwell times must dominate retunes; the scheduler will batch
   nearby steps.
3. **SQLite write bursts** - WAL mode + one insert per event (events are
   rare relative to chunks); no per-chunk writes.
4. **Queue backpressure** - drop-oldest keeps latency bounded; drop rate
   is surfaced as a health metric.

## 11b. Stage 3 additions (implemented)

- **SweepScheduler** (`drone4rf/scheduler.py`): bands are divided into
  steps of `sample_rate * usable_fraction`; selection is oldest-due
  first with priority tie-break (starvation-free); `revisit_s` controls
  cadence; adaptive rescanning pulls steps with recent detections
  forward (`hot_revisit_s` for `hot_visits` rounds).
- **Sweep acquisition**: the acquisition thread retunes, discards
  `settle_chunks` (LO transient), dwells `dwell_s` worth of chunks, then
  reschedules. Sources gained `retune()`; the simulator gained an
  absolute-frequency mode so multi-band scenarios are testable.
- **BaselineBank / named environments**: one BaselineModel per sweep
  step, saved as a directory of .npz files plus manifest
  (`data/environments/<name>/`); `sweep --calibrate --environment X`
  writes it, `--frozen-baseline` restores and freezes it (newly visited
  steps inherit the frozen state).
- **Window-aware tracking**: tracks only accrue hits/misses while the
  receiver is tuned to a window containing them - a 2.4 GHz track is
  not closed for "misses" recorded while observing 5.8 GHz.
- **Exclusion ranges**: operator-declared known emitters; detections
  centered inside them are suppressed before tracking.
- **Triggered IQ capture**: on track promotion, the promoting chunk plus
  the next `chunks_per_event-1` same-center chunks are written as .cf32
  with JSON sidecars; disabled by default, capped by `max_files`.

## 11c. Stage 4 additions (implemented)

The `drone4rf/analytics/` package implements §6's fusion design:

- **Sub-chunk temporal profiling** (`temporal.py`): the Welch FFT pass
  now also yields a spectrogram (~0.1-0.3 ms rows); every detection gets
  duty cycle, burst count, and burst-period regularity, which flow into
  track history.
- **Hop analysis** (`hopping.py`): narrowband detections are clustered
  into channels over a sliding window; scoring rewards channel count,
  LOW per-channel duty (the discriminator vs. independent continuous
  emitters, which are gated down), and spacing regularity.
- **EMI comb analysis** (`emi.py`): evenly spaced peak runs are flagged
  as motor/ESC-consistent, always with competing explanations attached,
  and only ever attach as supporting evidence to the single nearest
  entity (a room's switching supply must not inflate a whole band).
- **Background rejection** (`background.py`): Wi-Fi similarity
  (bandwidth + channel-grid position + bursty duty), Bluetooth
  similarity (channels confined to 2402-2480 MHz; IRREGULAR spacing is
  treated as BT-like since drone FHSS grids are regular), and
  fixed-infrastructure detection (old, near-continuous tracks) become
  confidence penalties.
- **Fusion** (`fusion.py`): weighted log-odds per §6, five capped
  categories, hard caps (fewer than 3 observations, clipping present,
  or EMI-only evidence → at most "possible"), and assembled plain-text
  explanations naming every evidence and penalty term. Multi-band
  correlation links strong entities separated by ≥200 MHz.
- **Emission hysteresis** (`engine.py`): escalations announce
  immediately; lateral flapping and de-escalations respect a 30 s
  cooldown per entity.

Known behaviour in a congested 2.4 GHz band: dense Bluetooth/BLE hopping
combined with a nearby switching-supply harmonic comb can still reach
"possible", and occasionally "probable" - with the Bluetooth penalty and
the competing-EMI explanations shown in the output. That is the designed
response to genuinely ambiguous evidence rather than a bug, but it does
mean the default weights are not yet trustworthy in busy environments.
The escalation path is site-specific exclusion ranges, then Stage 6
classifiers trained on labelled data from multiple sites.

## 11d. Stage 5 additions (implemented)

Optional PyQt6/pyqtgraph GUI (`pip install .[gui]`, `drone4rf gui`):

- **Threading contract**: `gui/controller.py` is Qt-free and bridges the
  pipeline's worker-thread callbacks (spectrum frames, track events,
  assessments) into thread-safe containers; Qt timers on the main thread
  poll them (~30 fps spectrum, 2 Hz status). Pipeline threads never
  touch Qt objects, and the UI never blocks DSP.
- **Views**: live spectrum with adaptive-threshold overlay, rolling
  waterfall (restarts per sweep retune), active-tracks table,
  color-coded assessment panel with full explanations, and an event
  browser over the SQLite store with one-click **false-positive
  marking** (persisted in the new `user_feedback` column - the future
  Stage 6 label source).
- **Controls**: source (hackrf/sim), mode (scan/sweep), environment
  name, frozen-baseline toggle, Start/Stop, and Calibrate (learn then
  save on Stop). Status bar carries tuned frequency, chunk/detection
  counters, drop/overflow counts, and a clipping warning.
- The pipeline gained a generic `on_spectrum(SpectrumFrame)` callback -
  any future dashboard (web UI, headless recorder) can consume the same
  stream without touching GUI code.

**Browser dashboard** (`drone4rf/web/`, `drone4rf web`): the session
controller was promoted out of the Qt package to `drone4rf/control.py`
and made multi-consumer - the latest spectrum frame carries a sequence
number and assessments/track events accumulate in cursor-addressed logs,
so several viewers (browser tabs, the Qt window) can read the same
session without consuming each other's data.

- Transport: stdlib `ThreadingHTTPServer` + Server-Sent Events. No
  third-party dependency, no build step, one static HTML file. SSE over
  WebSockets because the data flow is one-way and SSE self-reconnects;
  control actions are ordinary JSON POSTs.
- Spectra are **peak-decimated** to ~1024 points before transport (~6 KB
  per frame at ~12 fps): a mean would smooth away exactly the
  narrowband signals the tool exists to find.
- Local-service hardening, because this process controls radio hardware:
  loopback bind by default with a warning otherwise, `Host` header
  validation (blocks DNS rebinding), a per-run token required on all
  state-changing requests, JSON-only POST bodies, no CORS headers, and
  environment names validated against a strict charset so a network
  request cannot traverse out of the data directory.

## 11e. Stage 6 additions (implemented)

`drone4rf/ml/` - optional, CPU-only (scikit-learn), disabled by
default, with strict separation of detection / feature extraction /
classification / calibration:

- **Feature schema** (`features.py`): a versioned vector derived from
  the fusion engine's own deterministic scores, so assessment rows in
  the events DB double as training examples and runtime featurization is
  training featurization by construction. `ml*` terms are excluded from
  features - a live model can never see its own output.
- **Dataset building** (`dataset.py`): labels come exclusively from
  operator feedback (`user_feedback`: false_positive → 0, drone → 1;
  everything else stays unlabeled). Provenance records source DB, exact
  event ids, counts, and versions. No labels are ever fabricated.
- **Models** (`models.py`): Random Forest (class_weight='balanced') with
  sigmoid probability calibration when each class has ≥15 examples,
  stratified-CV precision/recall/F1 stored in the bundle,
  **site-independent GroupKFold metrics** when ≥2 sites contributed,
  and **unknown-class rejection** at inference (below the calibrated
  confidence threshold the model contributes nothing). Isolation Forest
  gives per-site behavioral novelty from unlabeled data. Bundles carry
  feature-schema version (mismatch refuses to load), sklearn version,
  training date, metrics, and provenance.
- **Fusion integration**: accepted P(drone) ≥ 0.5 becomes evidence
  `ml`, P(drone) < 0.5 becomes penalty `ml_background`, novelty > 0.5
  becomes small evidence `ml_anomaly`; abstentions are noted in the
  explanation. Enabling ML without a valid model fails fast with an
  actionable error.
- **Synthetic mode** (`synth.py`, `drone4rf ml synth-test`): fabricated
  feature data exercises the full train/calibrate/evaluate/infer path;
  output is loudly marked SYNTHETIC-PIPELINE-TEST-ONLY in provenance.
- **CLI**: `ml label / build-dataset / train / train-anomaly /
  evaluate / synth-test`.

## 12. Phased implementation plan

| Stage | Content | Status |
|---|---|---|
| 1 | This design document | ✅ |
| 2 | MVP: HackRF/sim/file sources, PSD, baseline, energy + OS-CFAR, persistence tracks, CLI, SQLite, tests | ✅ (this repo) |
| 3 | Sweep scheduler, scan profiles, adaptive rescan, exclusion ranges, named-environment baselines, triggered IQ capture | ✅ (this repo) |
| 4 | Burst / hop / multi-band / motor-EMI detectors, background rejection, fusion engine per §6 | ✅ (this repo) |
| 5 | PyQtGraph GUI: spectrum, waterfall, timeline, tracks, explanations | ✅ (this repo) |
| 6 | Optional ML (Isolation Forest anomaly first, calibrated RF classifier second), synthetic-data mode for pipeline testing only | ✅ (this repo) |

## 13. Major technical risks

1. **False positives from Wi-Fi/BT** - the 2.4/5 GHz bands are saturated;
   until Stage 4's rejection layer, outputs are capped at "unclassified
   RF activity" semantics: the CLI labels detections as *anomalies*, not
   drones.
2. **Baseline poisoning** - a drone hovering during calibration becomes
   background. Mitigated by asymmetric/hot-bin damping, freeze mode, and
   the calibration workflow's "review persistent emitters" step.
3. **HackRF dynamic range** - strong nearby emitters (own Wi-Fi router)
   can compress the front end and mask weak drone links. Mitigation:
   overload detection, gain guidance, and honest quality flags.
4. **EMI ambiguity** - motor/ESC signatures resemble many appliances;
   treated strictly as supporting evidence with competing explanations.
5. **Windows driver friction** - SoapySDR/WinUSB setup is the #1 support
   issue; README dedicates a section, and the sim/file sources make the
   whole pipeline testable with no hardware at all.
