"""Command-line interface: devices / scan / calibrate.

Examples:
    python -m drone4rf devices
    python -m drone4rf scan --source sim --duration 20
    python -m drone4rf scan --source hackrf --config config/default.yaml
    python -m drone4rf scan --source file --iq-file cap.cs8 --iq-format cs8
    python -m drone4rf calibrate --source hackrf --duration 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from drone4rf import LEGAL_NOTICE, __version__
from drone4rf.analytics import DroneAssessment
from drone4rf.baseline import BaselineBank, BaselineModel
from drone4rf.config import AppConfig, ConfigError, load_config
from drone4rf.detectors.base import Detection
from drone4rf.events import EventStore
from drone4rf.pipeline import ScannerPipeline
from drone4rf.scheduler import SweepScheduler
from drone4rf.sdr.base import SDRSource
from drone4rf.sdr.file_source import FileSource
from drone4rf.sdr.simulated import (
    SimulatedSource,
    demo_scenario,
    sweep_demo_scenario,
)
from drone4rf.tracking import TrackEvent

log = logging.getLogger(__name__)


def resolve_config_path(explicit: str | None) -> str | None:
    """Pick the configuration file to load.

    Precedence: an explicit --config, then your own config/local.yaml,
    then the repo's config/default.yaml, then the built-in dataclass
    defaults. Auto-discovery is what makes a fresh clone work without
    flags - the built-in defaults deliberately ship no sweep bands
    (band plans are regional and must be a deliberate choice), so
    'sweep' would otherwise fail on a clean checkout.
    """
    if explicit:
        return explicit
    for candidate in (Path("config/local.yaml"), Path("config/default.yaml")):
        if candidate.is_file():
            print(f"using configuration: {candidate}", file=sys.stderr)
            return str(candidate)
    return None


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_source(
    args: argparse.Namespace, cfg: AppConfig, sweep: bool = False
) -> SDRSource:
    if args.source == "sim":
        return SimulatedSource(
            scenario=sweep_demo_scenario() if sweep else demo_scenario(),
            center_hz=cfg.device.center_freq_hz,
            sample_rate=cfg.device.sample_rate,
            duration_s=args.duration,
        )
    if args.source == "file":
        if not args.iq_file:
            raise SystemExit("--iq-file is required with --source file")
        return FileSource(
            path=args.iq_file,
            iq_format=args.iq_format,
            center_hz=cfg.device.center_freq_hz,
            sample_rate=cfg.device.sample_rate,
        )
    # Deferred import: SoapySDR is optional.
    from drone4rf.sdr.hackrf import HackRFSource

    return HackRFSource(cfg.device)


def _fmt_freq(hz: float) -> str:
    return f"{hz / 1e9:.4f} GHz" if hz >= 1e9 else f"{hz / 1e6:.3f} MHz"


def _print_detection(det: Detection) -> None:
    print(
        f"  [{det.detector}] {_fmt_freq(det.center_hz)}  "
        f"bw {det.bandwidth_hz / 1e3:.0f} kHz  SNR {det.snr_db:.1f} dB  "
        f"score {det.score:.2f}"
        + (f"  flags={','.join(det.flags)}" if det.flags else "")
    )


def _print_track_event(ev: TrackEvent) -> None:
    t = ev.track
    label = "TRACK persistent" if ev.kind == "track_persistent" else "TRACK closed"
    print(
        f"\n[{label}] {_fmt_freq(t.center_hz)}  bw {t.bandwidth_hz / 1e6:.2f} MHz  "
        f"SNR {t.max_snr_db:.1f} dB  hits {t.hits}  occupancy {t.occupancy:.2f}"
    )
    print(f"  {ev.explanation}\n")


def _print_assessment(a: DroneAssessment) -> None:
    label = a.category.replace("_", " ").upper()
    print(
        f"\n{'=' * 70}\n"
        f"[ASSESSMENT] {label}\n"
        f"  {_fmt_freq(a.center_hz)}  span {a.freq_span_hz / 1e6:.2f} MHz  "
        f"confidence {a.confidence:.2f}  ({a.entity}, {a.observations} observations)\n"
        f"  {a.explanation}\n"
        f"{'=' * 70}\n"
    )


def _status_loop(pipeline: ScannerPipeline, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        s = pipeline.snapshot()
        tuned = (
            f"tuned {_fmt_freq(s.current_center_hz)}  "
            if s.current_center_hz == s.current_center_hz  # not NaN
            else ""
        )
        sweep_part = f"steps {s.steps_visited}  " if pipeline.scheduler else ""
        print(
            f"status: {tuned}{sweep_part}chunks {s.chunks_processed}  "
            f"noise {s.noise_floor_db:.1f} dBFS  detections {s.detections}  "
            f"drops {s.queue_drops}  overflows {s.source_overflows}"
            + ("  ⚠ CLIPPING" if s.clipped_chunks else ""),
            file=sys.stderr,
        )


def cmd_devices(_: argparse.Namespace) -> int:
    from drone4rf.sdr.hackrf import list_devices

    devices = list_devices()
    if not devices:
        print(
            "No SDR devices found (or SoapySDR bindings missing).\n"
            "Simulated and file sources remain available."
        )
        return 1
    for i, dev in enumerate(devices):
        print(f"[{i}] " + ", ".join(f"{k}={v}" for k, v in dev.items()))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    cfg = load_config(resolve_config_path(args.config))
    _setup_logging(cfg.storage.log_level)
    source = _build_source(args, cfg)
    store = EventStore(cfg.storage.database_path)

    baseline: BaselineModel | None = None
    if args.frozen_baseline:
        path = Path(cfg.baseline.path)
        if not path.exists():
            raise SystemExit(
                f"--frozen-baseline requested but no baseline at {path}; "
                "run 'calibrate' first"
            )
        baseline = BaselineModel.load(
            path,
            cfg.baseline,
            cfg.device.center_freq_hz,
            cfg.device.sample_rate,
            cfg.dsp.fft_size,
        )
        baseline.frozen = True
        print(f"loaded frozen baseline from {path} ({baseline.frames_seen} frames)")

    pipeline = ScannerPipeline(
        cfg,
        source,
        store=store,
        baseline=baseline,
        on_detection=_print_detection if args.verbose_detections else None,
        on_track_event=_print_track_event,
        on_assessment=_print_assessment,
    )

    print(
        f"scanning {_fmt_freq(cfg.device.center_freq_hz)} "
        f"@ {cfg.device.sample_rate / 1e6:.1f} MS/s  source={source.name}  "
        f"(Ctrl+C to stop)"
    )
    status_stop = threading.Event()
    status = threading.Thread(
        target=_status_loop, args=(pipeline, status_stop), daemon=True
    )
    try:
        with source:
            status.start()
            stats = pipeline.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\ninterrupted - shutting down cleanly", file=sys.stderr)
        pipeline.stop()
        stats = pipeline.snapshot()
    finally:
        status_stop.set()
        store.close()

    print(
        f"\ndone: {stats.chunks_processed} chunks, {stats.detections} detections, "
        f"{stats.track_events} track events, {stats.queue_drops} queue drops, "
        f"{stats.source_overflows} USB overflows"
    )
    for w in stats.warnings:
        print(f"⚠ {w}")
    print(f"events stored in {cfg.storage.database_path}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = load_config(resolve_config_path(args.config))
    _setup_logging(cfg.storage.log_level)
    if not cfg.sweep.enabled_bands():
        raise SystemExit(
            "no enabled bands in the sweep plan; enable at least one in "
            "the config's sweep.bands section"
        )
    if args.source == "file":
        raise SystemExit("file source cannot retune; use 'scan' for recordings")
    if args.calibrate and args.frozen_baseline:
        raise SystemExit("--calibrate and --frozen-baseline are mutually exclusive")
    source = _build_source(args, cfg, sweep=True)
    scheduler = SweepScheduler(
        cfg.sweep, cfg.device.sample_rate, cfg.dsp.chunk_samples
    )
    env_dir = Path(cfg.sweep.environments_dir) / args.environment

    bank: BaselineBank
    if args.frozen_baseline:
        if not (env_dir / "manifest.json").exists():
            raise SystemExit(
                f"--frozen-baseline requested but no environment at {env_dir}; "
                "run 'sweep --calibrate' first"
            )
        bank = BaselineBank.load(env_dir, cfg.baseline, cfg.dsp.fft_size)
        bank.freeze_all()
        print(f"loaded frozen environment '{args.environment}' from {env_dir}")
    else:
        bank = BaselineBank(cfg.baseline, cfg.dsp.fft_size)

    store = EventStore(cfg.storage.database_path) if not args.calibrate else None
    pipeline = ScannerPipeline(
        cfg,
        source,
        store=store,
        bank=bank,
        scheduler=scheduler,
        on_detection=_print_detection if args.verbose_detections else None,
        on_track_event=_print_track_event if not args.calibrate else None,
        on_assessment=_print_assessment if not args.calibrate else None,
    )

    bands = ", ".join(
        f"{b.name} ({b.start_hz / 1e6:.0f}-{b.stop_hz / 1e6:.0f} MHz)"
        for b in cfg.sweep.enabled_bands()
    )
    n_steps = len(scheduler.steps)
    mode = "calibrating" if args.calibrate else "sweeping"
    print(f"{mode} {n_steps} steps across: {bands}  source={source.name}")
    status_stop = threading.Event()
    status = threading.Thread(
        target=_status_loop, args=(pipeline, status_stop), daemon=True
    )
    try:
        with source:
            status.start()
            stats = pipeline.run(duration_s=args.duration)
    except KeyboardInterrupt:
        print("\ninterrupted - shutting down cleanly", file=sys.stderr)
        pipeline.stop()
        stats = pipeline.snapshot()
    finally:
        status_stop.set()
        if store is not None:
            store.close()

    if args.calibrate:
        if not bank.any_ready():
            print("calibration too short: no step baseline converged; nothing saved")
            return 1
        bank.save(env_dir, environment=args.environment)
        ready = sum(1 for m in bank.models.values() if m.ready)
        print(
            f"environment '{args.environment}' saved to {env_dir} "
            f"({ready}/{len(bank.models)} step baselines converged)"
        )
        if ready < n_steps:
            print(
                f"⚠ only {ready} of {n_steps} sweep steps converged - "
                "calibrate longer for full coverage"
            )
    print(
        f"\ndone: {stats.steps_visited} step visits, {stats.chunks_processed} chunks, "
        f"{stats.detections} detections, {stats.track_events} track events, "
        f"{stats.captures_written} captures, {stats.queue_drops} queue drops"
    )
    for w in stats.warnings:
        print(f"⚠ {w}")
    return 0


def cmd_ml(args: argparse.Namespace) -> int:
    """ML subsystem management: dataset building, training, evaluation."""
    from drone4rf.ml import dataset as ds
    from drone4rf.ml import models as ml_models
    from drone4rf.labels import export_csv, export_json

    _setup_logging("INFO")
    try:
        if args.ml_command == "label":
            store = EventStore(args.db)
            try:
                store.set_feedback(args.event_id, args.label)
            finally:
                store.close()
            print(f"event {args.event_id} labeled '{args.label}'")
            return 0

        if args.ml_command == "export-labels":
            store = EventStore(args.db)
            try:
                records = store.export_labels()
            finally:
                store.close()
            content = (
                export_csv(records)
                if args.format == "csv"
                else export_json(records)
            )
            if args.out == "-":
                sys.stdout.write(content)
            else:
                destination = Path(args.out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
                print(f"exported {len(records)} labels to {destination}")
            return 0

        if args.ml_command == "export-signatures":
            store = EventStore(args.db)
            try:
                records = store.export_signatures(
                    labeled_only=args.labeled_only
                )
            finally:
                store.close()
            content = (
                export_csv(records)
                if args.format == "csv"
                else export_json(records)
            )
            if args.out == "-":
                sys.stdout.write(content)
            else:
                destination = Path(args.out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
                print(f"exported {len(records)} RF signatures to {destination}")
            return 0

        if args.ml_command == "build-dataset":
            X, y, sites, meta = ds.build_dataset(args.db, site=args.site)
            ds.save_dataset(args.out, X, y, sites, meta)
            print(
                f"dataset: {meta.n_total} assessments from {args.db}\n"
                f"  labeled drone: {meta.n_drone}   labeled not-drone: "
                f"{meta.n_not_drone}   unlabeled: {meta.n_unlabeled}\n"
                f"saved to {args.out}"
            )
            if meta.n_drone == 0:
                print(
                    "⚠ no drone-labeled examples yet - run an authorized test "
                    "flight and label its assessments with 'drone4rf ml label'"
                )
            return 0

        if args.ml_command == "train":
            X, y, sites, meta = ds.load_dataset(args.dataset)
            bundle = ml_models.train_classifier(X, y, sites=sites, meta=meta)
            ml_models.save_model(bundle, args.out)
            _print_metrics(bundle)
            return 0

        if args.ml_command == "train-anomaly":
            X, y, sites, meta = ds.load_dataset(args.dataset)
            bundle = ml_models.train_anomaly(X, meta=meta)
            ml_models.save_model(bundle, args.out)
            print(f"anomaly model trained on {len(X)} vectors -> {args.out}")
            return 0

        if args.ml_command == "evaluate":
            X, y, sites, meta = ds.load_dataset(args.dataset)
            bundle = ml_models.load_model(args.model, expected_kind="classifier")
            scorer = ml_models.MLScorer(bundle, None, args.min_confidence)
            labeled = y != ds.LABEL_UNLABELED
            correct = rejected = 0
            for fv, label in zip(X[labeled], y[labeled]):
                r = scorer.score(fv)
                if r.rejected:
                    rejected += 1
                elif (r.p_drone >= 0.5) == (label == ds.LABEL_DRONE):
                    correct += 1
            n = int(labeled.sum())
            print(
                f"{n} labeled vectors: {correct} correct, {rejected} rejected "
                f"as unknown, {n - correct - rejected} wrong"
            )
            _print_metrics(bundle)
            return 0

        if args.ml_command == "synth-test":
            from drone4rf.ml.synth import SYNTHETIC_MARKER, synthetic_dataset

            print(
                "=== SYNTHETIC PIPELINE TEST ===\n"
                "Fabricated data validates the ML plumbing ONLY. The resulting\n"
                "model says nothing about real drones and must not be deployed."
            )
            X, y, sites, meta = synthetic_dataset()
            bundle = ml_models.train_classifier(X, y, sites=sites, meta=meta)
            _print_metrics(bundle)
            scorer = ml_models.MLScorer(bundle, None, 0.6)
            r = scorer.score(X[0])
            print(f"inference roundtrip: P(drone)={r.p_drone}, rejected={r.rejected}")
            if args.out:
                ml_models.save_model(bundle, args.out)
                print(f"⚠ synthetic model saved to {args.out} "
                      f"(provenance: {SYNTHETIC_MARKER})")
            return 0
    except ml_models.MLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise SystemExit(f"unknown ml command {args.ml_command!r}")


def _print_metrics(bundle: dict) -> None:
    m = bundle["metrics"]
    prov = bundle["provenance"]
    print(
        f"classifier trained {bundle['trained_utc'][:19]} "
        f"(calibrated: {bundle['calibrated']}, sklearn {bundle['sklearn_version']})\n"
        f"  training data: drone={prov['n_drone']} not_drone={prov['n_not_drone']} "
        f"sites={prov['sites']}\n"
        f"  {m['cv_splits']}-fold CV - drone: P {m['precision_drone']:.2f} "
        f"R {m['recall_drone']:.2f} F1 {m['f1_drone']:.2f} | not-drone: "
        f"P {m['precision_not_drone']:.2f} R {m['recall_not_drone']:.2f} "
        f"F1 {m['f1_not_drone']:.2f}"
    )
    if m.get("site_independent_f1_drone") is not None:
        print(
            f"  site-independent F1 - drone {m['site_independent_f1_drone']:.2f}, "
            f"not-drone {m['site_independent_f1_not_drone']:.2f}"
        )
    else:
        print("  ⚠ single-site data: site-independent performance unmeasured")


def cmd_web(args: argparse.Namespace) -> int:
    cfg = load_config(resolve_config_path(args.config))
    _setup_logging(cfg.storage.log_level)
    from drone4rf.web import run_server

    return run_server(
        cfg, host=args.host, port=args.port, open_browser=not args.no_browser
    )


def cmd_gui(args: argparse.Namespace) -> int:
    cfg = load_config(resolve_config_path(args.config))
    _setup_logging(cfg.storage.log_level)
    try:
        from drone4rf.gui.app import run_gui
    except ImportError as exc:
        raise SystemExit(
            f"GUI dependencies missing ({exc}); install with: pip install .[gui]"
        ) from exc
    return run_gui(cfg)


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(resolve_config_path(args.config))
    _setup_logging(cfg.storage.log_level)
    source = _build_source(args, cfg)
    pipeline = ScannerPipeline(cfg, source, store=None, update_baseline=True)

    print(
        f"calibrating background at {_fmt_freq(cfg.device.center_freq_hz)} for "
        f"{args.duration:.0f} s - keep the environment 'normal' (no drone flying!)"
    )
    try:
        with source:
            stats = pipeline.run(duration_s=args.duration)
    except KeyboardInterrupt:
        pipeline.stop()
        stats = pipeline.snapshot()

    if not pipeline.baseline.ready:
        print("calibration too short: baseline not converged; nothing saved")
        return 1
    pipeline.baseline.save(cfg.baseline.path)
    print(
        f"baseline saved to {cfg.baseline.path} "
        f"({pipeline.baseline.frames_seen} frames, noise floor "
        f"{stats.noise_floor_db:.1f} dBFS)"
    )
    if stats.warnings:
        print("⚠ calibration quality warnings - consider re-running:")
        for w in stats.warnings:
            print(f"  - {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone4rf",
        description="Passive SDR scanner for probable drone RF/EMI activity.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list SoapySDR devices").set_defaults(
        func=cmd_devices
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML config path")
    common.add_argument(
        "--source", choices=("hackrf", "sim", "file"), default="hackrf"
    )
    common.add_argument("--iq-file", default=None, help="IQ file for --source file")
    common.add_argument("--iq-format", choices=("cf32", "cs8"), default="cf32")
    common.add_argument(
        "--duration", type=float, default=None, help="seconds to run (default: forever)"
    )

    scan = sub.add_parser("scan", parents=[common], help="monitor the configured band")
    scan.add_argument(
        "--frozen-baseline",
        action="store_true",
        help="load the saved baseline and freeze adaptation",
    )
    scan.add_argument(
        "--verbose-detections",
        action="store_true",
        help="print every raw per-frame detection (noisy)",
    )
    scan.set_defaults(func=cmd_scan)

    cal = sub.add_parser(
        "calibrate", parents=[common], help="learn and save the local RF background"
    )
    cal.set_defaults(func=cmd_calibrate, duration=60.0)

    sweep = sub.add_parser(
        "sweep", parents=[common], help="cycle through the configured band plan"
    )
    sweep.add_argument(
        "--environment",
        default="default",
        help="named baseline environment to use/save (default: 'default')",
    )
    sweep.add_argument(
        "--calibrate",
        action="store_true",
        help="learn per-band baselines and save them as the environment",
    )
    sweep.add_argument(
        "--frozen-baseline",
        action="store_true",
        help="load the saved environment and freeze adaptation",
    )
    sweep.add_argument(
        "--verbose-detections",
        action="store_true",
        help="print every raw per-frame detection (noisy)",
    )
    sweep.set_defaults(func=cmd_sweep)

    web = sub.add_parser(
        "web", help="launch the browser dashboard (no extra dependencies)"
    )
    web.add_argument("--config", default=None, help="YAML config path")
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; anything other than loopback exposes control of "
        "your radio to the network (default: 127.0.0.1)",
    )
    web.add_argument("--port", type=int, default=8731)
    web.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    web.set_defaults(func=cmd_web)

    gui = sub.add_parser("gui", help="launch the Qt desktop interface")
    gui.add_argument("--config", default=None, help="YAML config path")
    gui.set_defaults(func=cmd_gui)

    ml = sub.add_parser("ml", help="optional machine-learning subsystem")
    mlsub = ml.add_subparsers(dest="ml_command", required=True)

    m_label = mlsub.add_parser("label", help="label an event for training")
    m_label.add_argument("--db", default="data/events.db")
    m_label.add_argument("--event-id", type=int, required=True)
    m_label.add_argument(
        "--label", required=True,
        choices=("drone", "not_drone", "false_positive", "true_positive"),
    )

    m_ds = mlsub.add_parser("build-dataset", help="extract features+labels from the DB")
    m_ds.add_argument("--db", default="data/events.db")
    m_ds.add_argument("--out", default="data/ml/dataset.npz")
    m_ds.add_argument("--site", default="unspecified-site",
                      help="site name for site-independent evaluation")

    m_export = mlsub.add_parser(
        "export-labels", help="export structured crowd-survey labels"
    )
    m_export.add_argument("--db", default="data/events.db")
    m_export.add_argument(
        "--out", default="-", help="output path, or - for stdout (default)"
    )
    m_export.add_argument("--format", choices=("json", "csv"), default="json")

    m_sig = mlsub.add_parser(
        "export-signatures", help="export captured RF signatures"
    )
    m_sig.add_argument("--db", default="data/events.db")
    m_sig.add_argument(
        "--out", default="-", help="output path, or - for stdout (default)"
    )
    m_sig.add_argument("--format", choices=("json", "csv"), default="json")
    m_sig.add_argument(
        "--labeled-only",
        action="store_true",
        help="export only signatures that have a survey verdict",
    )

    m_tr = mlsub.add_parser("train", help="train the calibrated classifier")
    m_tr.add_argument("--dataset", default="data/ml/dataset.npz")
    m_tr.add_argument("--out", default="data/ml/model.joblib")

    m_an = mlsub.add_parser("train-anomaly", help="train the Isolation Forest")
    m_an.add_argument("--dataset", default="data/ml/dataset.npz")
    m_an.add_argument("--out", default="data/ml/anomaly.joblib")

    m_ev = mlsub.add_parser("evaluate", help="evaluate a model on a dataset")
    m_ev.add_argument("--model", default="data/ml/model.joblib")
    m_ev.add_argument("--dataset", default="data/ml/dataset.npz")
    m_ev.add_argument("--min-confidence", type=float, default=0.6)

    m_sy = mlsub.add_parser(
        "synth-test", help="pipeline test on fabricated data (never deployable)"
    )
    m_sy.add_argument("--out", default=None,
                      help="optionally save the synthetic model here")
    ml.set_defaults(func=cmd_ml)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may use legacy code pages (cp1252) that cannot
    # encode characters like '⚠'; degrade gracefully instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    print(LEGAL_NOTICE + "\n", file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "calibrate" and args.duration is None:
        args.duration = 60.0
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
