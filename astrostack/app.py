"""AstroStack GUI entrypoint.

Uso:
  python -m astrostack.app
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "astrostack"

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .arduino import ArduinoController
from .mount import MountModel
from .plate_solve_service import PlateSolveResult, PlateSolveSettings, solve_from_stack
from .preprocessing import StretchConfig, stretch_to_u8
from .simulations import StarFieldSimulator
from .stacking import StackingConfig, StackingEngine
from .star_detection import StarDetectionConfig, detect_stars as detect_stars_candidates
from .tracking import TrackingConfig, TrackingEngine

try:
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
    from astropy.time import Time
    import astropy.units as u

    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

SITE_LAT_DEG = -33.4489
SITE_LON_DEG = -70.6693
SITE_ELEV_M = 570.0


@dataclass
class AppState:
    running: bool = False
    last_frame: Optional[np.ndarray] = None
    last_stack: Optional[np.ndarray] = None
    last_plate_result: Optional[PlateSolveResult] = None
    last_detected_stars: Optional[list[tuple[float, float, float]]] = None


class AstroStackApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AstroStack")
        self.resize(1120, 760)
        self.setMinimumSize(980, 680)

        self.state = AppState()
        self.msg_queue: queue.Queue[str] = queue.Queue()
        self.solve_queue: queue.Queue[tuple[Optional[PlateSolveResult], Optional[str]]] = queue.Queue()
        self.status_queue: queue.Queue[dict[str, str]] = queue.Queue()
        self.preview_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.stacker = StackingEngine(StackingConfig())
        self.tracker = TrackingEngine(TrackingConfig())
        self.simulator = StarFieldSimulator()
        self.arduino = ArduinoController()
        self.mount = MountModel()
        self.worker: Optional[threading.Thread] = None
        self.solve_worker: Optional[threading.Thread] = None
        self.goto_worker: Optional[threading.Thread] = None
        self.manual_move_worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self._build_ui()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(200)
        self.poll_timer.timeout.connect(self._poll_messages)
        self.poll_timer.start()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QtWidgets.QFrame()
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(4, 4, 4, 4)

        title = QtWidgets.QLabel("AstroStack")
        title_font = QtGui.QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QtWidgets.QLabel("Tracking · Stacking · Plate Solving · GoTo")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        layout.addWidget(header)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.tab_overview = QtWidgets.QWidget()
        self.tab_tracking = QtWidgets.QWidget()
        self.tab_stacking = QtWidgets.QWidget()
        self.tab_solve = QtWidgets.QWidget()
        self.tab_goto = QtWidgets.QWidget()
        self.tab_logs = QtWidgets.QWidget()

        self.tabs.addTab(self.tab_overview, "Overview")
        self.tabs.addTab(self.tab_tracking, "Tracking")
        self.tabs.addTab(self.tab_stacking, "Stacking")
        self.tabs.addTab(self.tab_solve, "Plate Solve")
        self.tabs.addTab(self.tab_goto, "GoTo")
        self.tabs.addTab(self.tab_logs, "Logs")

        self._build_overview_tab()
        self._build_tracking_tab()
        self._build_stacking_tab()
        self._build_plate_tab()
        self._build_goto_tab()
        self._build_logs_tab()

        self.setCentralWidget(container)

    def _build_overview_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_overview)
        layout.setContentsMargins(20, 20, 20, 20)

        status_box = QtWidgets.QGroupBox("Session status")
        status_layout = QtWidgets.QGridLayout(status_box)

        self.capture_status_label = QtWidgets.QLabel("Idle")
        self.stack_status_label = QtWidgets.QLabel("No stack yet")
        self.track_status_label = QtWidgets.QLabel("No tracking yet")
        self.solve_status_label = QtWidgets.QLabel("No plate solve yet")

        labels = [
            ("Capture", self.capture_status_label),
            ("Tracking", self.track_status_label),
            ("Stacking", self.stack_status_label),
            ("Plate Solve", self.solve_status_label),
        ]

        for row, (label_text, value_label) in enumerate(labels):
            label = QtWidgets.QLabel(label_text)
            label_font = label.font()
            label_font.setBold(True)
            label.setFont(label_font)
            status_layout.addWidget(label, row, 0)
            status_layout.addWidget(value_label, row, 1)

        layout.addWidget(status_box)

        controls = QtWidgets.QGroupBox("Capture control")
        controls_layout = QtWidgets.QVBoxLayout(controls)

        source_layout = QtWidgets.QHBoxLayout()
        source_layout.addWidget(QtWidgets.QLabel("Source"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Simulation"])
        source_layout.addWidget(self.source_combo)
        source_layout.addStretch()
        controls_layout.addLayout(source_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start capture")
        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)
        self.btn_reset = QtWidgets.QPushButton("Reset stack")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self.reset_stack)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_reset)
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)

        layout.addWidget(controls)
        layout.addStretch()

    def _build_tracking_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_tracking)
        layout.setContentsMargins(20, 20, 20, 20)

        heading = QtWidgets.QLabel("Tracking configuration")
        heading_font = heading.font()
        heading_font.setPointSize(14)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        config_box = QtWidgets.QGroupBox("Phase correlation preprocessing")
        config_layout = QtWidgets.QGridLayout(config_box)

        self.tracking_inputs = {
            "sigma_hp": QtWidgets.QLineEdit(str(self.tracker.config.sigma_hp)),
            "sigma_smooth": QtWidgets.QLineEdit(str(self.tracker.config.sigma_smooth)),
            "bright_percentile": QtWidgets.QLineEdit(str(self.tracker.config.bright_percentile)),
            "resp_min": QtWidgets.QLineEdit(str(self.tracker.config.resp_min)),
            "max_shift": QtWidgets.QLineEdit(str(self.tracker.config.max_shift_per_frame_px)),
            "bg_alpha": QtWidgets.QLineEdit(str(self.tracker.config.bg_ema_alpha)),
        }

        self._add_form_row(config_layout, "σ high-pass", self.tracking_inputs["sigma_hp"], 0)
        self._add_form_row(config_layout, "σ smooth", self.tracking_inputs["sigma_smooth"], 1)
        self._add_form_row(config_layout, "Bright %", self.tracking_inputs["bright_percentile"], 2)
        self._add_form_row(config_layout, "Resp min", self.tracking_inputs["resp_min"], 3)
        self._add_form_row(config_layout, "Max shift/frame", self.tracking_inputs["max_shift"], 4)
        self._add_form_row(config_layout, "BG EMA α", self.tracking_inputs["bg_alpha"], 5)

        self.subtract_bg_checkbox = QtWidgets.QCheckBox("Subtract background EMA")
        self.subtract_bg_checkbox.setChecked(self.tracker.config.subtract_bg_ema)
        config_layout.addWidget(self.subtract_bg_checkbox, 0, 2, 1, 1)

        layout.addWidget(config_box)

        arduino_box = QtWidgets.QGroupBox("Arduino mount")
        arduino_layout = QtWidgets.QGridLayout(arduino_box)

        self.arduino_port_input = QtWidgets.QLineEdit(self.arduino.port)
        self.arduino_baud_input = QtWidgets.QLineEdit(str(self.arduino.baud))
        self.arduino_connect_button = QtWidgets.QPushButton("Connect")
        self.arduino_connect_button.clicked.connect(self.toggle_arduino_connection)
        self.arduino_enable_checkbox = QtWidgets.QCheckBox("Enable mount")
        self.arduino_enable_checkbox.stateChanged.connect(self.toggle_arduino_enable)
        self.arduino_stop_button = QtWidgets.QPushButton("Stop")
        self.arduino_stop_button.clicked.connect(self.stop_arduino)
        self.arduino_status_label = QtWidgets.QLabel("Not connected")

        arduino_layout.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        arduino_layout.addWidget(self.arduino_port_input, 0, 1)
        arduino_layout.addWidget(QtWidgets.QLabel("Baud"), 0, 2)
        arduino_layout.addWidget(self.arduino_baud_input, 0, 3)
        arduino_layout.addWidget(self.arduino_connect_button, 0, 4)
        arduino_layout.addWidget(self.arduino_enable_checkbox, 1, 0, 1, 2)
        arduino_layout.addWidget(self.arduino_stop_button, 1, 2, 1, 1)
        arduino_layout.addWidget(self.arduino_status_label, 1, 3, 1, 2)

        layout.addWidget(arduino_box)

        apply_button = QtWidgets.QPushButton("Apply tracking settings")
        apply_button.clicked.connect(self.apply_tracking)
        layout.addWidget(apply_button, alignment=QtCore.Qt.AlignLeft)

        reset_button = QtWidgets.QPushButton("Reset keyframe")
        reset_button.clicked.connect(self.reset_tracking_keyframe)
        layout.addWidget(reset_button, alignment=QtCore.Qt.AlignLeft)

        status_box = QtWidgets.QGroupBox("Tracking telemetry")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.track_detail_label = QtWidgets.QLabel("Waiting for capture…")
        status_layout.addWidget(self.track_detail_label)
        layout.addWidget(status_box)
        layout.addStretch()

    def _build_stacking_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_stacking)
        layout.setContentsMargins(20, 20, 20, 20)

        heading = QtWidgets.QLabel("Stacking configuration")
        heading_font = heading.font()
        heading_font.setPointSize(14)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        config_box = QtWidgets.QGroupBox("Noise-aware stacking")
        config_layout = QtWidgets.QGridLayout(config_box)

        self.stacking_inputs = {
            "sigma_bg": QtWidgets.QLineEdit(str(self.stacker.config.sigma_bg)),
            "sigma_floor": QtWidgets.QLineEdit(str(self.stacker.config.sigma_floor_p)),
            "z_clip": QtWidgets.QLineEdit(str(self.stacker.config.z_clip)),
            "peak_p": QtWidgets.QLineEdit(str(self.stacker.config.peak_p)),
            "peak_blur": QtWidgets.QLineEdit(str(self.stacker.config.peak_blur)),
            "resp_min": QtWidgets.QLineEdit(str(self.stacker.config.resp_min)),
            "max_rad": QtWidgets.QLineEdit(str(self.stacker.config.max_rad)),
            "hot_z": QtWidgets.QLineEdit(str(self.stacker.config.hot_z)),
            "hot_max": QtWidgets.QLineEdit(str(self.stacker.config.hot_max)),
        }

        self._add_form_row(config_layout, "σ background", self.stacking_inputs["sigma_bg"], 0)
        self._add_form_row(config_layout, "Sigma floor %", self.stacking_inputs["sigma_floor"], 1)
        self._add_form_row(config_layout, "Z clip", self.stacking_inputs["z_clip"], 2)
        self._add_form_row(config_layout, "Peak %", self.stacking_inputs["peak_p"], 3)
        self._add_form_row(config_layout, "Peak blur", self.stacking_inputs["peak_blur"], 4)
        self._add_form_row(config_layout, "Resp min", self.stacking_inputs["resp_min"], 5)
        self._add_form_row(config_layout, "Max radius", self.stacking_inputs["max_rad"], 6)
        self._add_form_row(config_layout, "Hot-pixel Z", self.stacking_inputs["hot_z"], 7)
        self._add_form_row(config_layout, "Hot-pixel max", self.stacking_inputs["hot_max"], 8)

        layout.addWidget(config_box)

        apply_button = QtWidgets.QPushButton("Apply stacking settings")
        apply_button.clicked.connect(self.apply_stacking)
        layout.addWidget(apply_button, alignment=QtCore.Qt.AlignLeft)

        status_box = QtWidgets.QGroupBox("Stack status")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.stack_detail_label = QtWidgets.QLabel("Waiting for capture…")
        status_layout.addWidget(self.stack_detail_label)
        layout.addWidget(status_box)

        preview_box = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.preview_label = QtWidgets.QLabel("No preview yet.")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setScaledContents(False)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_box)

        save_button = QtWidgets.QPushButton("Save stacked image")
        save_button.clicked.connect(self.save_stack)
        layout.addWidget(save_button, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch()
        layout.setStretch(4, 1)

    def _build_plate_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_solve)
        layout.setContentsMargins(20, 20, 20, 20)

        heading = QtWidgets.QLabel("Plate solving")
        heading_font = heading.font()
        heading_font.setPointSize(14)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        settings_box = QtWidgets.QGroupBox("Solve settings")
        settings_layout = QtWidgets.QGridLayout(settings_box)

        self.solve_inputs = {
            "target": QtWidgets.QLineEdit("M42"),
            "radius_deg": QtWidgets.QLineEdit("1.0"),
            "gmax": QtWidgets.QLineEdit("12.0"),
            "pixel_um": QtWidgets.QLineEdit("2.9"),
            "focal_mm": QtWidgets.QLineEdit("900.0"),
            "max_gaia": QtWidgets.QLineEdit("8000"),
        }

        self._add_form_row(settings_layout, "Target (name/coord)", self.solve_inputs["target"], 0, width=28)
        self._add_form_row(settings_layout, "Search radius (deg)", self.solve_inputs["radius_deg"], 1)
        self._add_form_row(settings_layout, "G max", self.solve_inputs["gmax"], 2)
        self._add_form_row(settings_layout, "Pixel size (µm)", self.solve_inputs["pixel_um"], 3)
        self._add_form_row(settings_layout, "Focal length (mm)", self.solve_inputs["focal_mm"], 4)
        self._add_form_row(settings_layout, "Max Gaia sources", self.solve_inputs["max_gaia"], 5)

        layout.addWidget(settings_box)

        detect_row = QtWidgets.QHBoxLayout()
        self.detect_button = QtWidgets.QPushButton("Detect stars")
        self.detect_button.clicked.connect(self.detect_stars)
        detect_row.addWidget(self.detect_button)
        detect_row.addStretch()
        layout.addLayout(detect_row)

        detect_box = QtWidgets.QGroupBox("Detection summary")
        detect_layout = QtWidgets.QVBoxLayout(detect_box)
        self.detect_detail_label = QtWidgets.QLabel("No detections yet.")
        detect_layout.addWidget(self.detect_detail_label)
        layout.addWidget(detect_box)

        preview_box = QtWidgets.QGroupBox("Detection preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.detect_preview_label = QtWidgets.QLabel("No detection preview.")
        self.detect_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detect_preview_label.setMinimumSize(480, 360)
        self.detect_preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        preview_layout.addWidget(self.detect_preview_label)
        layout.addWidget(preview_box)

        self.solve_button = QtWidgets.QPushButton("Solve from current stack")
        self.solve_button.clicked.connect(self.start_plate_solve)
        layout.addWidget(self.solve_button, alignment=QtCore.Qt.AlignLeft)

        self.accept_button = QtWidgets.QPushButton("Accept solution for GoTo")
        self.accept_button.clicked.connect(self.accept_plate_solution)
        layout.addWidget(self.accept_button, alignment=QtCore.Qt.AlignLeft)

        status_box = QtWidgets.QGroupBox("Solution summary")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.solve_detail_label = QtWidgets.QLabel("No solution yet.")
        status_layout.addWidget(self.solve_detail_label)
        layout.addWidget(status_box)
        layout.addStretch()

    def _build_goto_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_goto)
        layout.setContentsMargins(20, 20, 20, 20)

        heading = QtWidgets.QLabel("GoTo helper")
        heading_font = heading.font()
        heading_font.setPointSize(14)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        goto_box = QtWidgets.QGroupBox("Target offsets")
        goto_layout = QtWidgets.QGridLayout(goto_box)

        self.goto_inputs = {
            "target_ra": QtWidgets.QLineEdit("83.8221"),
            "target_dec": QtWidgets.QLineEdit("-5.3911"),
        }

        self._add_form_row(goto_layout, "Target RA (deg)", self.goto_inputs["target_ra"], 0)
        self._add_form_row(goto_layout, "Target Dec (deg)", self.goto_inputs["target_dec"], 1)

        layout.addWidget(goto_box)

        compute_button = QtWidgets.QPushButton("Compute offset")
        compute_button.clicked.connect(self.compute_goto_offset)
        layout.addWidget(compute_button, alignment=QtCore.Qt.AlignLeft)

        mount_box = QtWidgets.QGroupBox("Mount calibration")
        mount_layout = QtWidgets.QGridLayout(mount_box)
        self.microstep_az_input = QtWidgets.QLineEdit(str(self.mount.state.ms_az))
        self.microstep_alt_input = QtWidgets.QLineEdit(str(self.mount.state.ms_alt))
        self.sign_az_checkbox = QtWidgets.QCheckBox("Invert AZ sign")
        self.sign_alt_checkbox = QtWidgets.QCheckBox("Invert ALT sign")
        mount_layout.addWidget(QtWidgets.QLabel("AZ microsteps"), 0, 0)
        mount_layout.addWidget(self.microstep_az_input, 0, 1)
        mount_layout.addWidget(QtWidgets.QLabel("ALT microsteps"), 1, 0)
        mount_layout.addWidget(self.microstep_alt_input, 1, 1)
        mount_layout.addWidget(self.sign_az_checkbox, 0, 2)
        mount_layout.addWidget(self.sign_alt_checkbox, 1, 2)
        self.apply_mount_button = QtWidgets.QPushButton("Apply mount settings")
        self.apply_mount_button.clicked.connect(self.apply_mount_settings)
        mount_layout.addWidget(self.apply_mount_button, 2, 0, 1, 3)
        layout.addWidget(mount_box)

        objects_box = QtWidgets.QGroupBox("Objects now (Alt/Az)")
        objects_layout = QtWidgets.QGridLayout(objects_box)
        self.refresh_objects_button = QtWidgets.QPushButton("Refresh objects")
        self.refresh_objects_button.clicked.connect(self.refresh_objects)
        self.objects_combo = QtWidgets.QComboBox()
        self.objects_combo.addItems(["(none)"])
        self.goto_delay_az_input = QtWidgets.QLineEdit("220")
        self.goto_delay_alt_input = QtWidgets.QLineEdit("220")
        self.goto_objects_button = QtWidgets.QPushButton("GoTo target")
        self.goto_objects_button.clicked.connect(self.goto_selected_object)
        objects_layout.addWidget(self.refresh_objects_button, 0, 0)
        objects_layout.addWidget(self.objects_combo, 0, 1, 1, 3)
        objects_layout.addWidget(QtWidgets.QLabel("AZ delay (us)"), 1, 0)
        objects_layout.addWidget(self.goto_delay_az_input, 1, 1)
        objects_layout.addWidget(QtWidgets.QLabel("ALT delay (us)"), 1, 2)
        objects_layout.addWidget(self.goto_delay_alt_input, 1, 3)
        objects_layout.addWidget(self.goto_objects_button, 2, 0, 1, 4)
        layout.addWidget(objects_box)

        manual_box = QtWidgets.QGroupBox("Manual mount movement")
        manual_layout = QtWidgets.QGridLayout(manual_box)
        self.manual_steps_az_input = QtWidgets.QLineEdit("200")
        self.manual_delay_az_input = QtWidgets.QLineEdit("1500")
        self.manual_steps_alt_input = QtWidgets.QLineEdit("200")
        self.manual_delay_alt_input = QtWidgets.QLineEdit("1500")
        self.manual_az_left_button = QtWidgets.QPushButton("AZ ◀")
        self.manual_az_right_button = QtWidgets.QPushButton("AZ ▶")
        self.manual_alt_up_button = QtWidgets.QPushButton("ALT ▲")
        self.manual_alt_down_button = QtWidgets.QPushButton("ALT ▼")
        self.manual_az_left_button.clicked.connect(lambda: self._start_manual_move("A", "REV"))
        self.manual_az_right_button.clicked.connect(lambda: self._start_manual_move("A", "FWD"))
        self.manual_alt_up_button.clicked.connect(lambda: self._start_manual_move("B", "FWD"))
        self.manual_alt_down_button.clicked.connect(lambda: self._start_manual_move("B", "REV"))

        manual_layout.addWidget(QtWidgets.QLabel("AZ steps"), 0, 0)
        manual_layout.addWidget(self.manual_steps_az_input, 0, 1)
        manual_layout.addWidget(QtWidgets.QLabel("AZ delay (us)"), 0, 2)
        manual_layout.addWidget(self.manual_delay_az_input, 0, 3)
        manual_layout.addWidget(self.manual_az_left_button, 1, 0)
        manual_layout.addWidget(self.manual_az_right_button, 1, 1)

        manual_layout.addWidget(QtWidgets.QLabel("ALT steps"), 2, 0)
        manual_layout.addWidget(self.manual_steps_alt_input, 2, 1)
        manual_layout.addWidget(QtWidgets.QLabel("ALT delay (us)"), 2, 2)
        manual_layout.addWidget(self.manual_delay_alt_input, 2, 3)
        manual_layout.addWidget(self.manual_alt_up_button, 3, 0)
        manual_layout.addWidget(self.manual_alt_down_button, 3, 1)
        layout.addWidget(manual_box)

        status_box = QtWidgets.QGroupBox("GoTo result")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.goto_detail_label = QtWidgets.QLabel("Need a plate solution to compute offsets.")
        status_layout.addWidget(self.goto_detail_label)
        layout.addWidget(status_box)
        layout.addStretch()

    def _build_logs_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.tab_logs)
        layout.setContentsMargins(10, 10, 10, 10)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def _add_form_row(
        self,
        layout: QtWidgets.QGridLayout,
        label: str,
        widget: QtWidgets.QLineEdit,
        row: int,
        width: int = 12,
    ) -> None:
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        widget.setMaximumWidth(width * 10)
        layout.addWidget(widget, row, 1)

    def _safe_float(self, value: str, default: float) -> float:
        try:
            return float(value)
        except ValueError:
            return default

    def _safe_int(self, value: str, default: int) -> int:
        try:
            return int(float(value))
        except ValueError:
            return default

    def _poll_messages(self) -> None:
        while True:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.append(msg)

        while True:
            try:
                result, error = self.solve_queue.get_nowait()
            except queue.Empty:
                break
            self.solve_button.setEnabled(True)
            if error:
                self.solve_status_label.setText("Solve failed")
                self.solve_detail_label.setText(error)
                self.log(error)
                continue
            if result is not None:
                self.state.last_plate_result = result
                self._update_plate_status(result)

        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if "track_status" in status:
                self.track_status_label.setText(status["track_status"])
                self.track_detail_label.setText(status["track_detail"])
            if "stack_status" in status:
                self.stack_status_label.setText(status["stack_status"])
                self.stack_detail_label.setText(status["stack_detail"])

        while True:
            try:
                preview = self.preview_queue.get_nowait()
            except queue.Empty:
                break
            preview = np.ascontiguousarray(preview)
            if preview.ndim == 2:
                height, width = preview.shape
                image_format = QtGui.QImage.Format_Grayscale8
                bytes_per_line = width
            else:
                height, width, channels = preview.shape
                image_format = QtGui.QImage.Format_RGB888
                bytes_per_line = width * channels
            qimage = QtGui.QImage(preview.data, width, height, bytes_per_line, image_format)
            pixmap = QtGui.QPixmap.fromImage(qimage.copy())
            self.preview_label.setPixmap(pixmap)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.msg_queue.put(f"[{timestamp}] {message}")

    def detect_stars(self) -> None:
        if self.state.last_stack is None:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "No stacked image available yet.")
            return
        detect_cfg = StarDetectionConfig()
        preview_u8 = np.ascontiguousarray(stretch_to_u8(self.state.last_stack, StretchConfig()))
        stars = detect_stars_candidates(preview_u8, config=detect_cfg)
        self.state.last_detected_stars = [(float(x), float(y), float(score)) for (y, x, score) in stars]
        count = len(stars)
        self.detect_detail_label.setText(f"Detected {count} stars.")
        self.log(f"Detected {count} stars for plate solving.")

        overlay = QtGui.QImage(preview_u8.data, preview_u8.shape[1], preview_u8.shape[0], preview_u8.shape[1], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(overlay.copy())
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        for y, x, _score in stars:
            painter.drawEllipse(QtCore.QPointF(float(x), float(y)), 6.0, 6.0)
        painter.end()
        self.detect_preview_label.setPixmap(pixmap)

    def apply_tracking(self) -> None:
        cfg = TrackingConfig(
            sigma_hp=self._safe_float(self.tracking_inputs["sigma_hp"].text(), self.tracker.config.sigma_hp),
            sigma_smooth=self._safe_float(
                self.tracking_inputs["sigma_smooth"].text(), self.tracker.config.sigma_smooth
            ),
            bright_percentile=self._safe_float(
                self.tracking_inputs["bright_percentile"].text(), self.tracker.config.bright_percentile
            ),
            resp_min=self._safe_float(self.tracking_inputs["resp_min"].text(), self.tracker.config.resp_min),
            max_shift_per_frame_px=self._safe_float(
                self.tracking_inputs["max_shift"].text(), self.tracker.config.max_shift_per_frame_px
            ),
            bg_ema_alpha=self._safe_float(
                self.tracking_inputs["bg_alpha"].text(), self.tracker.config.bg_ema_alpha
            ),
            subtract_bg_ema=self.subtract_bg_checkbox.isChecked(),
        )
        self.tracker.config = cfg
        self.log("Tracking settings updated.")

    def reset_tracking_keyframe(self) -> None:
        self.tracker.reset()
        self.log("Tracking keyframe reset.")
        self.track_detail_label.setText("Tracking reset; awaiting new frames.")

    def toggle_arduino_connection(self) -> None:
        if self.arduino.is_connected:
            status = self.arduino.disconnect()
            self.arduino_connect_button.setText("Connect")
            self.arduino_enable_checkbox.setChecked(False)
        else:
            port = self.arduino_port_input.text().strip()
            baud = self._safe_int(self.arduino_baud_input.text(), self.arduino.baud)
            status = self.arduino.connect(port=port, baud=baud)
            if status.connected:
                self.arduino_connect_button.setText("Disconnect")
        self.arduino_status_label.setText(status.message)
        self.log(f"Arduino: {status.message}")

    def toggle_arduino_enable(self) -> None:
        if not self.arduino.is_connected:
            self.arduino_enable_checkbox.setChecked(False)
            self.arduino_status_label.setText("Connect first")
            return
        enabled = self.arduino_enable_checkbox.isChecked()
        reply = self.arduino.enable(enabled)
        state = "enabled" if enabled else "disabled"
        message = f"Mount {state} ({reply or 'no reply'})"
        self.arduino_status_label.setText(message)
        self.log(f"Arduino: {message}")

    def stop_arduino(self) -> None:
        if not self.arduino.is_connected:
            self.arduino_status_label.setText("Connect first")
            return
        reply = self.arduino.stop()
        message = f"Stop sent ({reply or 'no reply'})"
        self.arduino_status_label.setText(message)
        self.log(f"Arduino: {message}")

    def apply_stacking(self) -> None:
        cfg = StackingConfig(
            sigma_bg=self._safe_float(self.stacking_inputs["sigma_bg"].text(), self.stacker.config.sigma_bg),
            sigma_floor_p=self._safe_float(
                self.stacking_inputs["sigma_floor"].text(), self.stacker.config.sigma_floor_p
            ),
            z_clip=self._safe_float(self.stacking_inputs["z_clip"].text(), self.stacker.config.z_clip),
            peak_p=self._safe_float(self.stacking_inputs["peak_p"].text(), self.stacker.config.peak_p),
            peak_blur=self._safe_float(self.stacking_inputs["peak_blur"].text(), self.stacker.config.peak_blur),
            resp_min=self._safe_float(self.stacking_inputs["resp_min"].text(), self.stacker.config.resp_min),
            max_rad=self._safe_float(self.stacking_inputs["max_rad"].text(), self.stacker.config.max_rad),
            hot_z=self._safe_float(self.stacking_inputs["hot_z"].text(), self.stacker.config.hot_z),
            hot_max=self._safe_int(self.stacking_inputs["hot_max"].text(), self.stacker.config.hot_max),
        )
        self.stacker.config = cfg
        self.log("Stacking settings updated.")

    def start_capture(self) -> None:
        if self.state.running:
            return
        self.state.running = True
        self.stop_event.clear()
        self.stacker.start(height=self.simulator.config.height, width=self.simulator.config.width)
        self.tracker.reset()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.capture_status_label.setText("Running (simulation)")
        self.log("Capture started (simulation).")
        self.worker = threading.Thread(target=self._run_capture_loop, daemon=True)
        self.worker.start()

    def stop_capture(self) -> None:
        if not self.state.running:
            return
        self.stop_event.set()
        self.state.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.capture_status_label.setText("Stopped")
        self.log("Capture stopped.")

    def reset_stack(self) -> None:
        if self.simulator:
            self.stacker.start(height=self.simulator.config.height, width=self.simulator.config.width)
        self.state.last_stack = None
        self.stack_status_label.setText("Stack reset")
        self.stack_detail_label.setText("Stack reset.")
        self.log("Stack reset.")

    def _run_capture_loop(self) -> None:
        stretch_cfg = StretchConfig()
        last_update = time.time()
        for frame in self.simulator.frames():
            if self.stop_event.is_set():
                break
            self.state.last_frame = frame

            reg = self.tracker.preprocess(frame)
            now = time.time()
            dx, dy, resp = self.tracker.update(reg, now)

            used, stack_img = self.stacker.step(frame)
            self.state.last_stack = stack_img

            if now - last_update > 0.25:
                frames = self.stacker.state.frames_used
                self.status_queue.put(
                    {
                        "track_status": f"dx={dx:+.2f}px dy={dy:+.2f}px resp={resp:.3f}",
                        "track_detail": (
                            f"dx={dx:+.2f}px dy={dy:+.2f}px resp={resp:.3f} "
                            f"(EMA bg={'on' if self.tracker.config.subtract_bg_ema else 'off'})"
                        ),
                        "stack_status": f"{frames} frames used",
                        "stack_detail": (
                            f"Frames used: {frames} | last used: {used} | resp={self.stacker.state.last_resp:.3f}"
                        ),
                    }
                )

                if stack_img is not None:
                    preview = stretch_to_u8(stack_img, stretch_cfg)
                    self.preview_queue.put(preview.copy())
                    self.log(f"Preview refreshed ({preview.shape[1]}x{preview.shape[0]}).")
                last_update = now
            time.sleep(0.03)

        self.stop_capture()

    def save_stack(self) -> None:
        stack = self.state.last_stack
        if stack is None:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "No stack available yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save stacked image",
            "",
            "PNG (*.png);;TIFF (*.tif);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.stacker.save_png(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "AstroStack", f"Failed to save: {exc}")
            return
        self.log(f"Stack saved to {path}.")

    def _collect_plate_settings(self) -> PlateSolveSettings:
        return PlateSolveSettings(
            target=self.solve_inputs["target"].text().strip() or "M42",
            radius_deg=self._safe_float(self.solve_inputs["radius_deg"].text(), 1.0),
            gmax=self._safe_float(self.solve_inputs["gmax"].text(), 12.0),
            pixel_size_um=self._safe_float(self.solve_inputs["pixel_um"].text(), 2.9),
            focal_length_mm=self._safe_float(self.solve_inputs["focal_mm"].text(), 900.0),
            max_gaia_sources=self._safe_int(self.solve_inputs["max_gaia"].text(), 8000),
        )

    def start_plate_solve(self) -> None:
        if self.solve_worker and self.solve_worker.is_alive():
            self.log("Plate solving already running.")
            return
        if self.state.last_stack is None:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "No stacked image available yet.")
            return
        settings = self._collect_plate_settings()
        self.solve_button.setEnabled(False)
        self.solve_status_label.setText("Solving…")
        self.log("Plate solve started.")
        self.solve_worker = threading.Thread(
            target=self._run_plate_solve,
            args=(self.state.last_stack.copy(), settings),
            daemon=True,
        )
        self.solve_worker.start()

    def _run_plate_solve(self, stack: np.ndarray, settings: PlateSolveSettings) -> None:
        try:
            result = solve_from_stack(stack, settings=settings)
        except Exception as exc:
            self.solve_queue.put((None, f"Plate solving failed: {exc}"))
            return
        self.solve_queue.put((result, None))

    def _update_plate_status(self, result: PlateSolveResult) -> None:
        stars = len(result.stars)
        if not result.solution:
            self.solve_detail_label.setText(f"Detected {stars} stars. Not enough matches for a solution.")
            self.solve_status_label.setText("Solve incomplete")
            self.log("Plate solve incomplete (not enough stars or matches).")
            return
        metrics = result.solution.get("metrics", {})
        err = metrics.get("err_median", None)
        center = result.center_radec
        if center:
            center_text = f"Center RA {center[0]:.4f}° Dec {center[1]:.4f}°"
        else:
            center_text = "Center unavailable"
        err_text = f"err_med={err:.3f}\"" if err is not None else "err_med=--"
        self.solve_detail_label.setText(f"Stars: {stars} | {err_text} | {center_text}")
        self.solve_status_label.setText("Solved")
        self.log("Plate solve complete.")

    def accept_plate_solution(self) -> None:
        result = self.state.last_plate_result
        if result is None or result.center_radec is None:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "No plate solution available.")
            return
        if not ASTROPY_OK:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "Astropy is required for Alt/Az conversion.")
            return
        az_deg, alt_deg = radec_to_altaz(result.center_radec[0], result.center_radec[1])
        ms_az = self._safe_int(self.microstep_az_input.text(), self.mount.state.ms_az)
        ms_alt = self._safe_int(self.microstep_alt_input.text(), self.mount.state.ms_alt)
        self.mount.calibrate(
            az_center_deg=az_deg,
            alt_center_deg=alt_deg,
            az_micro_now=self.mount.state.az_micro,
            alt_micro_now=self.mount.state.alt_micro,
            ms_az=ms_az,
            ms_alt=ms_alt,
            c_cm_now=self.mount.state.c_cm_cal,
        )
        self.goto_detail_label.setText(
            f"GoTo calibrated at AZ={az_deg:.2f}° ALT={alt_deg:.2f}°"
        )
        self.log("GoTo calibration accepted from plate solve.")

    def compute_goto_offset(self) -> None:
        result = self.state.last_plate_result
        if result is None or result.center_radec is None:
            QtWidgets.QMessageBox.warning(self, "AstroStack", "No plate solution available.")
            return
        target_ra = self._safe_float(self.goto_inputs["target_ra"].text(), result.center_radec[0])
        target_dec = self._safe_float(self.goto_inputs["target_dec"].text(), result.center_radec[1])
        solved_ra, solved_dec = result.center_radec
        delta_ra = (target_ra - solved_ra + 540.0) % 360.0 - 180.0
        delta_dec = target_dec - solved_dec
        self.goto_detail_label.setText(
            f"ΔRA={delta_ra:+.3f}° ({delta_ra * 60:+.2f}′) | ΔDec={delta_dec:+.3f}° ({delta_dec * 60:+.2f}′)"
        )
        self.log("GoTo offset computed.")

    def apply_mount_settings(self) -> None:
        ms_az = self._safe_int(self.microstep_az_input.text(), self.mount.state.ms_az)
        ms_alt = self._safe_int(self.microstep_alt_input.text(), self.mount.state.ms_alt)
        self.mount.state.ms_az = ms_az
        self.mount.state.ms_alt = ms_alt
        self.mount.state.sign_az = -1.0 if self.sign_az_checkbox.isChecked() else 1.0
        self.mount.state.sign_alt = -1.0 if self.sign_alt_checkbox.isChecked() else 1.0
        if self.arduino.is_connected:
            reply = self.arduino.set_microsteps(ms_az, ms_alt)
            self.log(f"Arduino microsteps updated ({reply or 'no reply'}).")
        else:
            self.log("Mount settings updated (Arduino not connected).")

    def refresh_objects(self) -> None:
        if not ASTROPY_OK:
            self.log("Astropy not available: objects list disabled.")
            self.objects_combo.clear()
            self.objects_combo.addItems(["(none)"])
            return
        rows, error = now_planets_altaz()
        if error is not None:
            self.log(error)
            self.objects_combo.clear()
            self.objects_combo.addItems(["(none)"])
            return
        self.objects_combo.clear()
        if not rows:
            self.objects_combo.addItems(["(none)"])
            self.log("No objects above the horizon.")
            return
        self.objects_combo.addItems(
            ["(none)"]
            + [
                f"{row['name']} | alt={row['alt_deg']:.1f} az={row['az_deg']:.1f}"
                for row in rows
                if row["alt_deg"] > 0
            ]
        )
        self.log(f"Objects refreshed ({len(rows)} total).")

    def goto_selected_object(self) -> None:
        if self.goto_worker and self.goto_worker.is_alive():
            self.log("GoTo already running.")
            return
        target_text = self.objects_combo.currentText()
        if target_text == "(none)":
            self.log("Select a target before GoTo.")
            return
        if not ASTROPY_OK:
            self.log("Astropy not available: GoTo disabled.")
            return
        rows, error = now_planets_altaz()
        if error is not None:
            self.log(error)
            return
        name = target_text.split("|")[0].strip().lower()
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            self.log(f"Target '{name}' not found.")
            return
        rate_az = self._safe_int(self.goto_delay_az_input.text(), 220)
        rate_alt = self._safe_int(self.goto_delay_alt_input.text(), 220)
        self.goto_worker = threading.Thread(
            target=self._run_goto,
            args=(row["az_deg"], row["alt_deg"], rate_az, rate_alt),
            daemon=True,
        )
        self.goto_worker.start()

    def _run_goto(self, az_deg: float, alt_deg: float, rate_az_us: int, rate_alt_us: int) -> None:
        if not self.arduino.is_connected:
            self.log("GoTo failed: Arduino not connected.")
            return
        plan = self.mount.compute_goto(target_az_deg=az_deg, target_alt_deg=alt_deg)
        if plan is None:
            self.log("GoTo failed: mount not calibrated.")
            return
        d_micro_az, d_micro_alt, da, _c_target, _c_cur = plan
        delay_az = int(max(200, min(5000, rate_az_us)))
        delay_alt = int(max(200, min(5000, rate_alt_us)))
        if abs(d_micro_az) >= 1.0:
            axis = "A"
            direction = "FWD" if d_micro_az >= 0 else "REV"
            steps = int(abs(d_micro_az))
            self.arduino.rate(0.0, 0.0)
            self.arduino.move(axis, direction, steps, delay_az)
            self.mount.apply_move(np.sign(d_micro_az) * steps, 0.0)
        if abs(d_micro_alt) >= 1.0:
            axis = "B"
            direction = "FWD" if d_micro_alt >= 0 else "REV"
            steps = int(abs(d_micro_alt))
            self.arduino.rate(0.0, 0.0)
            self.arduino.move(axis, direction, steps, delay_alt)
            self.mount.apply_move(0.0, np.sign(d_micro_alt) * steps)
        self.log(
            f"GoTo completed: target AZ={az_deg:.2f}° ALT={alt_deg:.2f}° (ΔAZ={da:+.2f}°)."
        )

    def _start_manual_move(self, axis: str, direction: str) -> None:
        if self.manual_move_worker and self.manual_move_worker.is_alive():
            self.log("Manual move already running.")
            return
        if not self.arduino.is_connected:
            self.log("Manual move failed: Arduino not connected.")
            return
        if axis == "A":
            steps = self._safe_int(self.manual_steps_az_input.text(), 200)
            delay_us = self._safe_int(self.manual_delay_az_input.text(), 1500)
        else:
            steps = self._safe_int(self.manual_steps_alt_input.text(), 200)
            delay_us = self._safe_int(self.manual_delay_alt_input.text(), 1500)
        self.manual_move_worker = threading.Thread(
            target=self._run_manual_move,
            args=(axis, direction, steps, delay_us),
            daemon=True,
        )
        self.manual_move_worker.start()

    def _run_manual_move(self, axis: str, direction: str, steps: int, delay_us: int) -> None:
        if steps <= 0:
            self.log("Manual move skipped: steps must be positive.")
            return
        delay = int(max(200, min(5000, delay_us)))
        self.arduino.rate(0.0, 0.0)
        self.arduino.move(axis, direction, steps, delay)
        delta = float(steps if direction == "FWD" else -steps)
        if axis == "A":
            self.mount.apply_move(delta, 0.0)
        else:
            self.mount.apply_move(0.0, delta)
        self.log(f"Manual move {axis} {direction} steps={steps} delay={delay}us.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.state.running:
            self.stop_event.set()
        if self.arduino.is_connected:
            self.arduino.disconnect()
        event.accept()


def now_planets_altaz() -> tuple[list[dict[str, float | str]], Optional[str]]:
    if not ASTROPY_OK:
        return [], "Astropy not available: Alt/Az helpers disabled."
    loc = EarthLocation(lat=SITE_LAT_DEG * u.deg, lon=SITE_LON_DEG * u.deg, height=SITE_ELEV_M * u.m)
    t = Time.now()
    frame = AltAz(obstime=t, location=loc)
    names = ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    rows: list[dict[str, float | str]] = []
    for name in names:
        try:
            sc = get_body(name, t, loc).transform_to(frame)
        except Exception:
            continue
        rows.append(
            {
                "name": name,
                "alt_deg": float(sc.alt.deg),
                "az_deg": float(sc.az.deg),
                "dist_au": float(sc.distance.to(u.au).value),
            }
        )
    rows.sort(key=lambda r: float(r["alt_deg"]), reverse=True)
    return rows, None


def radec_to_altaz(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    if not ASTROPY_OK:
        raise RuntimeError("Astropy not available for RA/Dec -> AltAz conversion.")
    loc = EarthLocation(lat=SITE_LAT_DEG * u.deg, lon=SITE_LON_DEG * u.deg, height=SITE_ELEV_M * u.m)
    t = Time.now()
    frame = AltAz(obstime=t, location=loc)
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    aa = sc.transform_to(frame)
    return float(aa.az.deg), float(aa.alt.deg)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = AstroStackApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
