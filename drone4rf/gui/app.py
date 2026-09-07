"""Main window: spectrum, waterfall, tracks, assessments, event browser.

Threading rule: everything in this module runs on the Qt main thread.
Pipeline data arrives exclusively through PipelineController's
thread-safe containers, polled by QTimers.
"""

from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from drone4rf import LEGAL_NOTICE, __version__
from drone4rf.analytics import DroneAssessment
from drone4rf.config import AppConfig
from drone4rf.control import PipelineController
from drone4rf.events import EventStore

WATERFALL_ROWS = 200
LEVELS = (-110.0, -30.0)  # dBFS display range

_CATEGORY_COLORS = {
    "background_activity": "#7f8c8d",
    "unclassified_rf_activity": "#f0f0f0",
    "possible_drone_activity": "#f39c12",
    "probable_drone_activity": "#e67e22",
    "high_confidence_drone_activity": "#e74c3c",
}


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.controller = PipelineController(cfg)
        self._event_store = EventStore(cfg.storage.database_path)
        self._wf_buffer: np.ndarray | None = None
        self._wf_center: float | None = None
        # Cursors into the controller's multi-consumer logs.
        self._frame_seq_seen = -1
        self._assess_cursor = 0

        self.setWindowTitle(f"Drone 4-RF {__version__} - passive RF scanner")
        self.resize(1280, 800)
        self._build_toolbar()
        self._build_body()
        self.statusBar().showMessage(LEGAL_NOTICE.replace("\n", " "))

        self._fast_timer = QTimer(self)
        self._fast_timer.timeout.connect(self._poll_spectrum)
        self._fast_timer.start(33)  # ~30 fps
        self._slow_timer = QTimer(self)
        self._slow_timer.timeout.connect(self._poll_slow)
        self._slow_timer.start(500)

    # -- construction -------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["hackrf", "sim"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["sweep", "scan"])
        self.env_edit = QLineEdit("default")
        self.env_edit.setMaximumWidth(120)
        self.frozen_check = QCheckBox("frozen baseline")
        self.frozen_check.setChecked(True)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start_stop)
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setToolTip(
            "Learn the RF background; press Stop to save the environment"
        )
        self.calibrate_btn.clicked.connect(self._on_calibrate)

        for w in (
            QLabel("Source:"), self.source_combo,
            QLabel("Mode:"), self.mode_combo,
            QLabel("Environment:"), self.env_edit,
            self.frozen_check, self.start_btn, self.calibrate_btn,
        ):
            lay.addWidget(w)
        lay.addStretch(1)
        self.health_label = QLabel("idle")
        lay.addWidget(self.health_label)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(bar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        self._splitter = splitter
        self.setCentralWidget(container)

    def _build_body(self) -> None:
        # Left: spectrum over waterfall.
        left = QSplitter(Qt.Orientation.Vertical)
        self.spec_plot = pg.PlotWidget(title="Spectrum")
        self.spec_plot.setLabel("bottom", "Frequency", units="MHz")
        self.spec_plot.setLabel("left", "Power", units="dBFS")
        self.spec_plot.setYRange(*LEVELS)
        self.psd_curve = self.spec_plot.plot(pen=pg.mkPen("#00d0ff", width=1))
        self.thr_curve = self.spec_plot.plot(
            pen=pg.mkPen("#ffcc00", width=1, style=Qt.PenStyle.DashLine)
        )
        left.addWidget(self.spec_plot)

        wf_widget = pg.PlotWidget(title="Waterfall")
        wf_widget.setLabel("bottom", "Frequency", units="MHz")
        wf_widget.hideAxis("left")
        self.wf_item = pg.ImageItem(axisOrder="row-major")
        cmap = pg.colormap.get("inferno")
        self.wf_item.setLookupTable(cmap.getLookupTable(nPts=256))
        wf_widget.addItem(self.wf_item)
        self._wf_plot = wf_widget
        left.addWidget(wf_widget)
        self._splitter.addWidget(left)

        # Right: tabs.
        tabs = QTabWidget()
        self.assess_text = QTextEdit()
        self.assess_text.setReadOnly(True)
        tabs.addTab(self.assess_text, "Assessments")

        self.tracks_table = QTableWidget(0, 6)
        self.tracks_table.setHorizontalHeaderLabels(
            ["MHz", "BW kHz", "SNR dB", "hits", "occupancy", "detectors"]
        )
        tabs.addTab(self.tracks_table, "Tracks")

        events_widget = QWidget()
        ev_lay = QVBoxLayout(events_widget)
        self.events_table = QTableWidget(0, 6)
        self.events_table.setHorizontalHeaderLabels(
            ["id", "time (UTC)", "kind", "MHz", "label", "feedback"]
        )
        btns = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_events)
        fp_btn = QPushButton("Mark selected as false positive")
        fp_btn.clicked.connect(self._mark_false_positive)
        btns.addWidget(refresh_btn)
        btns.addWidget(fp_btn)
        btns.addStretch(1)
        ev_lay.addLayout(btns)
        ev_lay.addWidget(self.events_table)
        tabs.addTab(events_widget, "Events")

        self._splitter.addWidget(tabs)
        self._splitter.setSizes([800, 480])

    # -- control handlers ----------------------------------------------------

    def _on_start_stop(self) -> None:
        if self.controller.running:
            self.controller.stop()
            self.start_btn.setText("Start")
            self.calibrate_btn.setEnabled(True)
            return
        self._start(calibrate=False)

    def _on_calibrate(self) -> None:
        if self.controller.running:
            return
        self.frozen_check.setChecked(False)
        self._start(calibrate=True)

    def _start(self, calibrate: bool) -> None:
        try:
            self.controller.start(
                source_name=self.source_combo.currentText(),
                mode=self.mode_combo.currentText(),
                environment=self.env_edit.text().strip() or "default",
                frozen=self.frozen_check.isChecked() and not calibrate,
                calibrate=calibrate,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Drone 4-RF", str(exc))
            return
        self.start_btn.setText("Stop")
        self.calibrate_btn.setEnabled(False)
        self._wf_buffer = None  # restart the waterfall

    # -- polling --------------------------------------------------------------

    def _poll_spectrum(self) -> None:
        item = self.controller.peek_spectrum()
        if item is None:
            return
        seq, frame = item
        if seq == self._frame_seq_seen:
            return  # nothing new since the last repaint
        self._frame_seq_seen = seq
        mhz = frame.freqs_hz / 1e6
        self.psd_curve.setData(mhz, frame.psd_db)
        if frame.threshold_db is not None:
            self.thr_curve.setData(mhz, frame.threshold_db)
        else:
            self.thr_curve.clear()

        if self._wf_center != frame.center_hz or self._wf_buffer is None:
            # Retuned (sweep step change): restart the waterfall history.
            self._wf_center = frame.center_hz
            self._wf_buffer = np.full(
                (WATERFALL_ROWS, len(frame.psd_db)), LEVELS[0], dtype=np.float32
            )
            self.spec_plot.setTitle(
                f"Spectrum - {frame.center_hz / 1e9:.4f} GHz"
            )
            # setImage must precede setRect (older pyqtgraph requires an
            # image before the display rect can be assigned).
            self.wf_item.setImage(self._wf_buffer, autoLevels=False, levels=LEVELS)
            self.wf_item.setRect(
                mhz[0], 0.0, mhz[-1] - mhz[0], float(WATERFALL_ROWS)
            )
        self._wf_buffer = np.roll(self._wf_buffer, 1, axis=0)
        self._wf_buffer[0] = frame.psd_db.astype(np.float32)
        self.wf_item.setImage(
            self._wf_buffer, autoLevels=False, levels=LEVELS
        )

    def _poll_slow(self) -> None:
        for seq, a in self.controller.assessments_since(self._assess_cursor):
            self._append_assessment(a)
            self._assess_cursor = seq
        stats = self.controller.snapshot()
        if stats is not None and self.controller.running:
            tuned = (
                f"{stats.current_center_hz / 1e9:.4f} GHz"
                if stats.current_center_hz == stats.current_center_hz
                else "-"
            )
            clip = "  ⚠ CLIPPING" if stats.clipped_chunks else ""
            self.health_label.setText(
                f"{self.controller.status_message}  |  tuned {tuned}  "
                f"chunks {stats.chunks_processed}  det {stats.detections}  "
                f"drops {stats.queue_drops}  ovf {stats.source_overflows}{clip}"
            )
        else:
            self.health_label.setText(self.controller.status_message)
            if not self.controller.running and self.start_btn.text() == "Stop":
                self.start_btn.setText("Start")
                self.calibrate_btn.setEnabled(True)
        self._refresh_tracks()

    def _append_assessment(self, a: DroneAssessment) -> None:
        color = _CATEGORY_COLORS.get(a.category, "#f0f0f0")
        label = a.category.replace("_", " ").upper()
        self.assess_text.append(
            f'<div style="color:{color}"><b>[{label}]</b> '
            f"{a.center_hz / 1e9:.4f} GHz - confidence {a.confidence:.2f} "
            f"({a.entity}, {a.observations} obs)<br/>{a.explanation}</div><br/>"
        )

    def _refresh_tracks(self) -> None:
        tracks = self.controller.active_tracks()
        self.tracks_table.setRowCount(len(tracks))
        for row, tr in enumerate(
            sorted(tracks, key=lambda t: -t.max_snr_db)
        ):
            cells = (
                f"{tr.center_hz / 1e6:.3f}",
                f"{tr.bandwidth_hz / 1e3:.0f}",
                f"{tr.max_snr_db:.1f}",
                str(tr.hits),
                f"{tr.occupancy:.2f}",
                "+".join(sorted(tr.detectors)),
            )
            for col, text in enumerate(cells):
                self.tracks_table.setItem(row, col, QTableWidgetItem(text))

    def _refresh_events(self) -> None:
        rows = self._event_store.recent(limit=200)
        self.events_table.setRowCount(len(rows))
        for r, ev in enumerate(rows):
            cells = (
                str(ev["id"]),
                (ev["ts_utc"] or "")[:19],
                ev["kind"],
                f"{ev['center_hz'] / 1e6:.3f}",
                ev["confidence_label"] or "",
                ev["user_feedback"] or "",
            )
            for c, text in enumerate(cells):
                self.events_table.setItem(r, c, QTableWidgetItem(text))

    def _mark_false_positive(self) -> None:
        row = self.events_table.currentRow()
        if row < 0:
            return
        item = self.events_table.item(row, 0)
        if item is None:
            return
        event_id = int(item.text())
        self._event_store.set_feedback(event_id, "false_positive")
        self._refresh_events()

    # -- shutdown --------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.controller.stop()
        self._event_store.close()
        super().closeEvent(event)


def run_gui(cfg: AppConfig) -> int:
    app = QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=False, background="k", foreground="w")
    window = MainWindow(cfg)
    window.show()
    return app.exec()
