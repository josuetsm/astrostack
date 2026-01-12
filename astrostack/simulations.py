from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class StarFieldConfig:
    width: int = 800
    height: int = 600
    star_count: int = 120
    noise_sigma: float = 6.0
    drift_px_per_frame: Tuple[float, float] = (0.4, -0.25)
    seed: int = 7


class StarFieldSimulator:
    def __init__(self, config: StarFieldConfig | None = None) -> None:
        self.config = config or StarFieldConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.stars = self._init_stars()
        self.offset = np.array([0.0, 0.0], dtype=np.float32)

    def _init_stars(self) -> np.ndarray:
        xs = self.rng.uniform(0, self.config.width, size=self.config.star_count)
        ys = self.rng.uniform(0, self.config.height, size=self.config.star_count)
        mags = self.rng.uniform(0.6, 1.0, size=self.config.star_count)
        return np.stack([xs, ys, mags], axis=1)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = np.zeros((self.config.height, self.config.width), dtype=np.float32)
            for x, y, m in self.stars:
                xi = int(np.clip(x + self.offset[0], 0, self.config.width - 1))
                yi = int(np.clip(y + self.offset[1], 0, self.config.height - 1))
                frame[yi, xi] += 2000.0 * m
            noise = self.rng.normal(0.0, self.config.noise_sigma, frame.shape)
            frame = np.clip(frame + noise, 0, 65535).astype(np.uint16)
            self.offset += np.array(self.config.drift_px_per_frame, dtype=np.float32)
            yield frame
