from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from .plate_solve_service import PlateSolveResult, PlateSolveSettings, solve_from_stack
from .preprocessing import StretchConfig, stretch_to_u8
from .simulations import StarFieldSimulator
from .stacking import StackingConfig, StackingEngine
from .tracking import TrackingConfig, TrackingEngine


@dataclass
class AppState:
    running: bool = False
    last_frame: Optional[np.ndarray] = None
    last_stack: Optional[np.ndarray] = None
    last_plate_result: Optional[PlateSolveResult] = None


class AstroStackApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AstroStack")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.style = ttk.Style(self.root)
        if "aqua" in self.style.theme_names():
            self.style.theme_use("aqua")
        elif "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.state = AppState()
        self.msg_queue: queue.Queue[str] = queue.Queue()
        self.solve_queue: queue.Queue[tuple[Optional[PlateSolveResult], Optional[str]]] = queue.Queue()
        self.status_queue: queue.Queue[dict[str, str]] = queue.Queue()

        self.stacker = StackingEngine(StackingConfig())
        self.tracker = TrackingEngine(TrackingConfig())
        self.simulator = StarFieldSimulator()
        self.worker: Optional[threading.Thread] = None
        self.solve_worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self._build_ui()
        self._poll_messages()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=16)
        header.pack(side=tk.TOP, fill=tk.X)

        title = ttk.Label(header, text="AstroStack", font=("SF Pro Display", 22, "bold"))
        subtitle = ttk.Label(header, text="Tracking · Stacking · Plate Solving · GoTo")
        title.pack(anchor="w")
        subtitle.pack(anchor="w")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_overview = ttk.Frame(notebook)
        self.tab_tracking = ttk.Frame(notebook)
        self.tab_stacking = ttk.Frame(notebook)
        self.tab_solve = ttk.Frame(notebook)
        self.tab_goto = ttk.Frame(notebook)
        self.tab_logs = ttk.Frame(notebook)

        notebook.add(self.tab_overview, text="Overview")
        notebook.add(self.tab_tracking, text="Tracking")
        notebook.add(self.tab_stacking, text="Stacking")
        notebook.add(self.tab_solve, text="Plate Solve")
        notebook.add(self.tab_goto, text="GoTo")
        notebook.add(self.tab_logs, text="Logs")

        self._build_overview_tab()
        self._build_tracking_tab()
        self._build_stacking_tab()
        self._build_plate_tab()
        self._build_goto_tab()
        self._build_logs_tab()

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.tab_overview, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.LabelFrame(frame, text="Session status", padding=12)
        status_frame.pack(fill=tk.X)

        self.capture_status_var = tk.StringVar(value="Idle")
        self.stack_status_var = tk.StringVar(value="No stack yet")
        self.track_status_var = tk.StringVar(value="No tracking yet")
        self.solve_status_var = tk.StringVar(value="No plate solve yet")

        for row, (label, var) in enumerate(
            [
                ("Capture", self.capture_status_var),
                ("Tracking", self.track_status_var),
                ("Stacking", self.stack_status_var),
                ("Plate Solve", self.solve_status_var),
            ]
        ):
            ttk.Label(status_frame, text=label, font=("SF Pro Text", 12, "bold")).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            ttk.Label(status_frame, textvariable=var).grid(row=row, column=1, sticky="w", pady=4)

        controls = ttk.LabelFrame(frame, text="Capture control", padding=12)
        controls.pack(fill=tk.X, pady=16)

        source_frame = ttk.Frame(controls)
        source_frame.pack(fill=tk.X)
        ttk.Label(source_frame, text="Source").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value="Simulation")
        ttk.OptionMenu(source_frame, self.source_var, "Simulation", "Simulation").pack(
            side=tk.LEFT, padx=8
        )

        btn_frame = ttk.Frame(controls)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        self.btn_start = ttk.Button(btn_frame, text="Start capture", command=self.start_capture)
        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.btn_reset = ttk.Button(btn_frame, text="Reset stack", command=self.reset_stack, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_reset.pack(side=tk.LEFT)

    def _build_tracking_tab(self) -> None:
        frame = ttk.Frame(self.tab_tracking, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Tracking configuration", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        config_frame = ttk.LabelFrame(frame, text="Phase correlation preprocessing", padding=12)
        config_frame.pack(fill=tk.X, pady=(10, 16))

        self.tracking_vars = {
            "sigma_hp": tk.StringVar(value=str(self.tracker.config.sigma_hp)),
            "sigma_smooth": tk.StringVar(value=str(self.tracker.config.sigma_smooth)),
            "bright_percentile": tk.StringVar(value=str(self.tracker.config.bright_percentile)),
            "resp_min": tk.StringVar(value=str(self.tracker.config.resp_min)),
            "max_shift": tk.StringVar(value=str(self.tracker.config.max_shift_per_frame_px)),
            "bg_alpha": tk.StringVar(value=str(self.tracker.config.bg_ema_alpha)),
            "subtract_bg": tk.BooleanVar(value=self.tracker.config.subtract_bg_ema),
        }

        self._add_form_row(config_frame, "σ high-pass", self.tracking_vars["sigma_hp"], 0)
        self._add_form_row(config_frame, "σ smooth", self.tracking_vars["sigma_smooth"], 1)
        self._add_form_row(config_frame, "Bright %", self.tracking_vars["bright_percentile"], 2)
        self._add_form_row(config_frame, "Resp min", self.tracking_vars["resp_min"], 3)
        self._add_form_row(config_frame, "Max shift/frame", self.tracking_vars["max_shift"], 4)
        self._add_form_row(config_frame, "BG EMA α", self.tracking_vars["bg_alpha"], 5)

        ttk.Checkbutton(
            config_frame,
            text="Subtract background EMA",
            variable=self.tracking_vars["subtract_bg"],
        ).grid(row=0, column=2, sticky="w", padx=10)

        ttk.Button(frame, text="Apply tracking settings", command=self.apply_tracking).pack(
            anchor="w", pady=(0, 16)
        )

        status_frame = ttk.LabelFrame(frame, text="Tracking telemetry", padding=12)
        status_frame.pack(fill=tk.X)
        self.track_detail_var = tk.StringVar(value="Waiting for capture…")
        ttk.Label(status_frame, textvariable=self.track_detail_var).pack(anchor="w")

    def _build_stacking_tab(self) -> None:
        frame = ttk.Frame(self.tab_stacking, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Stacking configuration", font=("SF Pro Text", 14, "bold")).pack(anchor="w")

        config_frame = ttk.LabelFrame(frame, text="Noise-aware stacking", padding=12)
        config_frame.pack(fill=tk.X, pady=(10, 16))

        self.stacking_vars = {
            "sigma_bg": tk.StringVar(value=str(self.stacker.config.sigma_bg)),
            "sigma_floor": tk.StringVar(value=str(self.stacker.config.sigma_floor_p)),
            "z_clip": tk.StringVar(value=str(self.stacker.config.z_clip)),
            "peak_p": tk.StringVar(value=str(self.stacker.config.peak_p)),
            "peak_blur": tk.StringVar(value=str(self.stacker.config.peak_blur)),
            "resp_min": tk.StringVar(value=str(self.stacker.config.resp_min)),
            "max_rad": tk.StringVar(value=str(self.stacker.config.max_rad)),
            "hot_z": tk.StringVar(value=str(self.stacker.config.hot_z)),
            "hot_max": tk.StringVar(value=str(self.stacker.config.hot_max)),
        }

        self._add_form_row(config_frame, "σ background", self.stacking_vars["sigma_bg"], 0)
        self._add_form_row(config_frame, "Sigma floor %", self.stacking_vars["sigma_floor"], 1)
        self._add_form_row(config_frame, "Z clip", self.stacking_vars["z_clip"], 2)
        self._add_form_row(config_frame, "Peak %", self.stacking_vars["peak_p"], 3)
        self._add_form_row(config_frame, "Peak blur", self.stacking_vars["peak_blur"], 4)
        self._add_form_row(config_frame, "Resp min", self.stacking_vars["resp_min"], 5)
        self._add_form_row(config_frame, "Max radius", self.stacking_vars["max_rad"], 6)
        self._add_form_row(config_frame, "Hot-pixel Z", self.stacking_vars["hot_z"], 7)
        self._add_form_row(config_frame, "Hot-pixel max", self.stacking_vars["hot_max"], 8)

        ttk.Button(frame, text="Apply stacking settings", command=self.apply_stacking).pack(
            anchor="w", pady=(0, 16)
        )

        status_frame = ttk.LabelFrame(frame, text="Stack status", padding=12)
        status_frame.pack(fill=tk.X)
        self.stack_detail_var = tk.StringVar(value="Waiting for capture…")
        ttk.Label(status_frame, textvariable=self.stack_detail_var).pack(anchor="w")

        ttk.Button(frame, text="Save stacked image", command=self.save_stack).pack(anchor="w", pady=(12, 0))

    def _build_plate_tab(self) -> None:
        frame = ttk.Frame(self.tab_solve, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Plate solving", font=("SF Pro Text", 14, "bold")).pack(anchor="w")

        settings_frame = ttk.LabelFrame(frame, text="Solve settings", padding=12)
        settings_frame.pack(fill=tk.X, pady=(10, 16))

        self.solve_vars = {
            "target": tk.StringVar(value="M42"),
            "radius_deg": tk.StringVar(value="1.0"),
            "gmax": tk.StringVar(value="12.0"),
            "pixel_um": tk.StringVar(value="2.9"),
            "focal_mm": tk.StringVar(value="400.0"),
            "max_gaia": tk.StringVar(value="8000"),
        }

        self._add_form_row(settings_frame, "Target (name/coord)", self.solve_vars["target"], 0, width=28)
        self._add_form_row(settings_frame, "Search radius (deg)", self.solve_vars["radius_deg"], 1)
        self._add_form_row(settings_frame, "G max", self.solve_vars["gmax"], 2)
        self._add_form_row(settings_frame, "Pixel size (µm)", self.solve_vars["pixel_um"], 3)
        self._add_form_row(settings_frame, "Focal length (mm)", self.solve_vars["focal_mm"], 4)
        self._add_form_row(settings_frame, "Max Gaia sources", self.solve_vars["max_gaia"], 5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 12))
        self.solve_button = ttk.Button(btn_frame, text="Solve from current stack", command=self.start_plate_solve)
        self.solve_button.pack(side=tk.LEFT)

        status_frame = ttk.LabelFrame(frame, text="Solution summary", padding=12)
        status_frame.pack(fill=tk.X)
        self.solve_detail_var = tk.StringVar(value="No solution yet.")
        ttk.Label(status_frame, textvariable=self.solve_detail_var).pack(anchor="w")

    def _build_goto_tab(self) -> None:
        frame = ttk.Frame(self.tab_goto, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="GoTo helper", font=("SF Pro Text", 14, "bold")).pack(anchor="w")

        goto_frame = ttk.LabelFrame(frame, text="Target offsets", padding=12)
        goto_frame.pack(fill=tk.X, pady=(10, 16))

        self.goto_vars = {
            "target_ra": tk.StringVar(value="83.8221"),
            "target_dec": tk.StringVar(value="-5.3911"),
        }

        self._add_form_row(goto_frame, "Target RA (deg)", self.goto_vars["target_ra"], 0)
        self._add_form_row(goto_frame, "Target Dec (deg)", self.goto_vars["target_dec"], 1)

        ttk.Button(frame, text="Compute offset", command=self.compute_goto_offset).pack(anchor="w")

        status_frame = ttk.LabelFrame(frame, text="GoTo result", padding=12)
        status_frame.pack(fill=tk.X, pady=(12, 0))
        self.goto_detail_var = tk.StringVar(value="Need a plate solution to compute offsets.")
        ttk.Label(status_frame, textvariable=self.goto_detail_var).pack(anchor="w")

    def _build_logs_tab(self) -> None:
        frame = ttk.Frame(self.tab_logs, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _add_form_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        row: int,
        width: int = 12,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=4)

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
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

        while True:
            try:
                result, error = self.solve_queue.get_nowait()
            except queue.Empty:
                break
            self.solve_button.configure(state=tk.NORMAL)
            if error:
                self.solve_status_var.set("Solve failed")
                self.solve_detail_var.set(error)
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
                self.track_status_var.set(status["track_status"])
                self.track_detail_var.set(status["track_detail"])
            if "stack_status" in status:
                self.stack_status_var.set(status["stack_status"])
                self.stack_detail_var.set(status["stack_detail"])

        self.root.after(200, self._poll_messages)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.msg_queue.put(f"[{timestamp}] {message}")

    def apply_tracking(self) -> None:
        cfg = TrackingConfig(
            sigma_hp=self._safe_float(self.tracking_vars["sigma_hp"].get(), self.tracker.config.sigma_hp),
            sigma_smooth=self._safe_float(self.tracking_vars["sigma_smooth"].get(), self.tracker.config.sigma_smooth),
            bright_percentile=self._safe_float(
                self.tracking_vars["bright_percentile"].get(), self.tracker.config.bright_percentile
            ),
            resp_min=self._safe_float(self.tracking_vars["resp_min"].get(), self.tracker.config.resp_min),
            max_shift_per_frame_px=self._safe_float(
                self.tracking_vars["max_shift"].get(), self.tracker.config.max_shift_per_frame_px
            ),
            bg_ema_alpha=self._safe_float(self.tracking_vars["bg_alpha"].get(), self.tracker.config.bg_ema_alpha),
            subtract_bg_ema=bool(self.tracking_vars["subtract_bg"].get()),
        )
        self.tracker.config = cfg
        self.log("Tracking settings updated.")

    def apply_stacking(self) -> None:
        cfg = StackingConfig(
            sigma_bg=self._safe_float(self.stacking_vars["sigma_bg"].get(), self.stacker.config.sigma_bg),
            sigma_floor_p=self._safe_float(self.stacking_vars["sigma_floor"].get(), self.stacker.config.sigma_floor_p),
            z_clip=self._safe_float(self.stacking_vars["z_clip"].get(), self.stacker.config.z_clip),
            peak_p=self._safe_float(self.stacking_vars["peak_p"].get(), self.stacker.config.peak_p),
            peak_blur=self._safe_float(self.stacking_vars["peak_blur"].get(), self.stacker.config.peak_blur),
            resp_min=self._safe_float(self.stacking_vars["resp_min"].get(), self.stacker.config.resp_min),
            max_rad=self._safe_float(self.stacking_vars["max_rad"].get(), self.stacker.config.max_rad),
            hot_z=self._safe_float(self.stacking_vars["hot_z"].get(), self.stacker.config.hot_z),
            hot_max=self._safe_int(self.stacking_vars["hot_max"].get(), self.stacker.config.hot_max),
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
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_reset.configure(state=tk.NORMAL)
        self.capture_status_var.set("Running (simulation)")
        self.log("Capture started (simulation).")
        self.worker = threading.Thread(target=self._run_capture_loop, daemon=True)
        self.worker.start()

    def stop_capture(self) -> None:
        if not self.state.running:
            return
        self.stop_event.set()
        self.state.running = False
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.capture_status_var.set("Stopped")
        self.log("Capture stopped.")

    def reset_stack(self) -> None:
        if self.simulator:
            self.stacker.start(height=self.simulator.config.height, width=self.simulator.config.width)
        self.state.last_stack = None
        self.stack_status_var.set("Stack reset")
        self.stack_detail_var.set("Stack reset.")
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
                    self.log(f"Preview refreshed ({preview.shape[1]}x{preview.shape[0]}).")
                last_update = now
            time.sleep(0.03)

        self.stop_capture()

    def save_stack(self) -> None:
        stack = self.state.last_stack
        if stack is None:
            messagebox.showwarning("AstroStack", "No stack available yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.stacker.save_png(path)
        except Exception as exc:
            messagebox.showerror("AstroStack", f"Failed to save: {exc}")
            return
        self.log(f"Stack saved to {path}.")

    def _collect_plate_settings(self) -> PlateSolveSettings:
        return PlateSolveSettings(
            target=self.solve_vars["target"].get().strip() or "M42",
            radius_deg=self._safe_float(self.solve_vars["radius_deg"].get(), 1.0),
            gmax=self._safe_float(self.solve_vars["gmax"].get(), 12.0),
            pixel_size_um=self._safe_float(self.solve_vars["pixel_um"].get(), 2.9),
            focal_length_mm=self._safe_float(self.solve_vars["focal_mm"].get(), 400.0),
            max_gaia_sources=self._safe_int(self.solve_vars["max_gaia"].get(), 8000),
        )

    def start_plate_solve(self) -> None:
        if self.solve_worker and self.solve_worker.is_alive():
            self.log("Plate solving already running.")
            return
        if self.state.last_stack is None:
            messagebox.showwarning("AstroStack", "No stacked image available yet.")
            return
        settings = self._collect_plate_settings()
        self.solve_button.configure(state=tk.DISABLED)
        self.solve_status_var.set("Solving…")
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
            self.solve_detail_var.set(f"Detected {stars} stars. Not enough matches for a solution.")
            self.solve_status_var.set("Solve incomplete")
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
        self.solve_detail_var.set(f"Stars: {stars} | {err_text} | {center_text}")
        self.solve_status_var.set("Solved")
        self.log("Plate solve complete.")

    def compute_goto_offset(self) -> None:
        result = self.state.last_plate_result
        if result is None or result.center_radec is None:
            messagebox.showwarning("AstroStack", "No plate solution available.")
            return
        target_ra = self._safe_float(self.goto_vars["target_ra"].get(), result.center_radec[0])
        target_dec = self._safe_float(self.goto_vars["target_dec"].get(), result.center_radec[1])
        solved_ra, solved_dec = result.center_radec
        delta_ra = (target_ra - solved_ra + 540.0) % 360.0 - 180.0
        delta_dec = target_dec - solved_dec
        self.goto_detail_var.set(
            f"ΔRA={delta_ra:+.3f}° ({delta_ra * 60:+.2f}′) | ΔDec={delta_dec:+.3f}° ({delta_dec * 60:+.2f}′)"
        )
        self.log("GoTo offset computed.")


def main() -> None:
    root = tk.Tk()
    app = AstroStackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
