from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk

import numpy as np

from .preprocessing import StretchConfig, stretch_to_u8
from .simulations import StarFieldSimulator
from .stacking import StackingConfig, StackingEngine
from .tracking import TrackingConfig, TrackingEngine


@dataclass
class DemoState:
    running: bool = False
    last_frame: Optional[np.ndarray] = None
    last_stack: Optional[np.ndarray] = None


class AstroStackApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AstroStack")
        self.root.geometry("980x720")

        self.style = ttk.Style(self.root)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.state = DemoState()
        self.msg_queue: queue.Queue[str] = queue.Queue()

        self.stacker = StackingEngine(StackingConfig())
        self.tracker = TrackingEngine(TrackingConfig())
        self.simulator = StarFieldSimulator()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self._build_ui()
        self._poll_messages()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=10)
        header.pack(side=tk.TOP, fill=tk.X)

        title = ttk.Label(header, text="AstroStack", font=("SF Pro Display", 20, "bold"))
        subtitle = ttk.Label(header, text="Tracking · Stacking · Plate Solving · GoTo")
        title.pack(anchor="w")
        subtitle.pack(anchor="w")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_tracking = ttk.Frame(notebook)
        self.tab_stacking = ttk.Frame(notebook)
        self.tab_solve = ttk.Frame(notebook)
        self.tab_goto = ttk.Frame(notebook)
        self.tab_logs = ttk.Frame(notebook)

        notebook.add(self.tab_tracking, text="Tracking")
        notebook.add(self.tab_stacking, text="Stacking")
        notebook.add(self.tab_solve, text="Plate Solve")
        notebook.add(self.tab_goto, text="GoTo")
        notebook.add(self.tab_logs, text="Logs")

        self._build_tracking_tab()
        self._build_stacking_tab()
        self._build_plate_tab()
        self._build_goto_tab()
        self._build_logs_tab()

    def _build_tracking_tab(self) -> None:
        frame = ttk.Frame(self.tab_tracking, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Tracking status", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        self.track_status = ttk.Label(frame, text="Idle")
        self.track_status.pack(anchor="w", pady=(6, 12))

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)
        self._add_labeled_entry(grid, "σ HP", "10.0", 0)
        self._add_labeled_entry(grid, "σ Smooth", "2.0", 1)
        self._add_labeled_entry(grid, "Bright %", "99.3", 2)
        self._add_labeled_entry(grid, "Resp Min", "0.06", 3)

        ttk.Separator(frame).pack(fill=tk.X, pady=18)

        ttk.Label(frame, text="Demo control", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(anchor="w", pady=(8, 0))
        self.btn_start = ttk.Button(btn_frame, text="Start Demo", command=self.start_demo)
        self.btn_stop = ttk.Button(btn_frame, text="Stop Demo", command=self.stop_demo, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop.pack(side=tk.LEFT)

    def _build_stacking_tab(self) -> None:
        frame = ttk.Frame(self.tab_stacking, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Stacking status", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        self.stack_status = ttk.Label(frame, text="No frames stacked")
        self.stack_status.pack(anchor="w", pady=(6, 12))

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)
        self._add_labeled_entry(grid, "Sigma BG", "35.0", 0)
        self._add_labeled_entry(grid, "Peak %", "99.75", 1)
        self._add_labeled_entry(grid, "Resp Min", "0.05", 2)
        self._add_labeled_entry(grid, "Max Rad", "400.0", 3)

    def _build_plate_tab(self) -> None:
        frame = ttk.Frame(self.tab_solve, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Plate solving", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Integra plate solving desde plate_solve_pipeline.py usando la última imagen apilada.",
            wraplength=720,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Button(frame, text="Solve (placeholder)", command=self._not_implemented).pack(
            anchor="w", pady=(12, 0)
        )

    def _build_goto_tab(self) -> None:
        frame = ttk.Frame(self.tab_goto, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="GoTo", font=("SF Pro Text", 14, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Conecta la calibración de la montura y objetivos a partir de la solución de placas.",
            wraplength=720,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Button(frame, text="GoTo (placeholder)", command=self._not_implemented).pack(
            anchor="w", pady=(12, 0)
        )

    def _build_logs_tab(self) -> None:
        frame = ttk.Frame(self.tab_logs, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, default: str, column: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=8, pady=4, sticky="w")
        ttk.Label(frame, text=label).pack(anchor="w")
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default)
        entry.pack(anchor="w")

    def _not_implemented(self) -> None:
        self.log("Funcionalidad en construcción: conectaremos plate solving y GoTo aquí.")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.msg_queue.put(f"[{timestamp}] {message}")

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
        self.root.after(200, self._poll_messages)

    def start_demo(self) -> None:
        if self.state.running:
            return
        self.state.running = True
        self.stop_event.clear()
        self.stacker.start(height=self.simulator.config.height, width=self.simulator.config.width)
        self.tracker.reset()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.log("Demo iniciado (simulación de estrellas).")
        self.worker = threading.Thread(target=self._run_demo_loop, daemon=True)
        self.worker.start()

    def stop_demo(self) -> None:
        if not self.state.running:
            return
        self.stop_event.set()
        self.state.running = False
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.log("Demo detenido.")

    def _run_demo_loop(self) -> None:
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
                self.track_status.configure(
                    text=f"dx={dx:+.2f}px dy={dy:+.2f}px resp={resp:.3f}"
                )
                frames = self.stacker.state.frames_used
                self.stack_status.configure(
                    text=f"Frames used: {frames} | last used: {used} | resp={self.stacker.state.last_resp:.3f}"
                )

                if stack_img is not None:
                    preview = stretch_to_u8(stack_img, stretch_cfg)
                    self.log(f"Preview actualizado ({preview.shape[1]}x{preview.shape[0]}).")
                last_update = now
            time.sleep(0.03)

        self.stop_demo()


def main() -> None:
    root = tk.Tk()
    app = AstroStackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
