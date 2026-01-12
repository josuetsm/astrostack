from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArduinoStatus:
    connected: bool
    message: str


class ArduinoController:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, timeout: float = 1.0) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial = None
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._serial is not None

    def connect(self, port: Optional[str] = None, baud: Optional[int] = None, timeout: Optional[float] = None) -> ArduinoStatus:
        if port is not None:
            self.port = port
        if baud is not None:
            self.baud = int(baud)
        if timeout is not None:
            self.timeout = float(timeout)

        if self._serial is not None:
            return ArduinoStatus(True, f"Already connected on {self.port}")

        import serial

        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(1.8)
            pong = self._send_cmd("PING") or "???"
            return ArduinoStatus(True, f"Connected on {self.port} (PING={pong})")
        except Exception as exc:
            self._serial = None
            return ArduinoStatus(False, f"Connection failed: {exc}")

    def disconnect(self) -> ArduinoStatus:
        if self._serial is None:
            return ArduinoStatus(False, "Not connected")
        try:
            self._serial.close()
        finally:
            self._serial = None
        return ArduinoStatus(False, "Disconnected")

    def enable(self, on: bool) -> str:
        return self._send_cmd(f"ENABLE {1 if on else 0}")

    def rate(self, v_az: float, v_alt: float) -> str:
        return self._send_cmd(f"RATE {float(v_az):.3f} {float(v_alt):.3f}")

    def stop(self) -> str:
        return self._send_cmd("STOP")

    def move(self, axis: str, direction: str, steps: int, delay_us: int, timeout_s: float = 3.0) -> str:
        return self._send_cmd(f"MOVE {axis} {direction} {int(steps)} {int(delay_us)}", timeout_s=timeout_s)

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        return self._send_cmd(f"MS {int(az_div)} {int(alt_div)}")

    def _send_cmd(self, cmd: str, timeout_s: float = 1.0) -> str:
        if self._serial is None:
            return ""
        cmd = cmd.strip()
        if not cmd:
            return ""
        with self._lock:
            try:
                self._serial.reset_input_buffer()
            except Exception:
                pass
            self._serial.write((cmd + "\n").encode("ascii", errors="ignore"))
            self._serial.flush()
            t0 = time.time()
            while True:
                try:
                    line = self._serial.readline().decode(errors="ignore").strip()
                except Exception:
                    line = ""
                if line:
                    return line
                if (time.time() - t0) > timeout_s:
                    return ""
