import math
from collections import deque

import numpy as np

from birdgame import HORIZON
from birdgame.stats.fewvar import FEWVar
from birdgame.trackers.trackerbase import TrackerBase


class EMWAConstants:
    FADE_FACTOR = 0.0001
    ROLLING_WINDOW = 20


class EMWAVarTracker(TrackerBase):
    """
    EWMA variance mixture tracker with feature-driven mean extrapolation and
    uncertainty scaling (velocity, acceleration, rolling volatility, direction
    reversals, falcon proximity).
    """

    def __init__(self, horizon=HORIZON):
        super().__init__(horizon)
        fade = EMWAConstants.FADE_FACTOR
        self.current_x = None
        self.ewa_dx_core = FEWVar(fading_factor=fade)
        self.ewa_dx_tail = FEWVar(fading_factor=fade)
        self.ewa_velocity = FEWVar(fading_factor=fade)
        self.ewa_accel = FEWVar(fading_factor=fade)

        self.weights = [0.92, 0.08]
        self.winsor_mult = 2.5
        self.tail_update_mult = 2.5
        self.scale_core = 0.97
        self.scale_tail = 1.10
        self.velocity_blend = 0.4
        self.vol_mult_floor = 0.85
        self.vol_mult_cap = 1.25
        self.direction_change_vol_mult = 1.08
        self.falcon_proximity_threshold = 0.05
        self.falcon_proximity_vol_mult = 1.10

        self.prev_dx = None
        self.last_direction_change = False
        self.falcon_offset = None
        self.x_window = deque(maxlen=EMWAConstants.ROLLING_WINDOW)
        self.dx_window = deque(maxlen=EMWAConstants.ROLLING_WINDOW)

    def _rolling_stats(self):
        if not self.x_window:
            return None
        xs = np.asarray(self.x_window, dtype=float)
        stats = {
            "rolling_mean_x": float(xs.mean()),
            "rolling_std_x": float(xs.std()) if len(xs) > 1 else 0.0,
            "rolling_range_x": float(xs.max() - xs.min()) if len(xs) > 1 else 0.0,
        }
        if self.dx_window:
            dxs = np.asarray(self.dx_window, dtype=float)
            stats["rolling_mean_dx"] = float(dxs.mean())
            stats["rolling_std_dx"] = float(dxs.std()) if len(dxs) > 1 else 0.0
            stats["rolling_mean_abs_dx"] = float(np.mean(np.abs(dxs)))
        return stats

    def _volatility_multiplier(self, rolling):
        if rolling is None or self.count <= 0:
            return 1.0

        baseline = max(math.sqrt(self.ewa_dx_core.get()), 1e-6)
        rolling_std_dx = rolling.get("rolling_std_dx", 0.0)
        vol_mult = rolling_std_dx / baseline if baseline > 0 else 1.0
        vol_mult = float(np.clip(vol_mult, self.vol_mult_floor, self.vol_mult_cap))

        if self.last_direction_change:
            vol_mult = min(vol_mult * self.direction_change_vol_mult, self.vol_mult_cap)

        if self.falcon_offset is not None:
            if abs(self.falcon_offset) < self.falcon_proximity_threshold:
                vol_mult = min(vol_mult * self.falcon_proximity_vol_mult, self.vol_mult_cap)

        return vol_mult

    def _extrapolated_mean(self):
        if self.current_x is None:
            return None
        velocity = self.ewa_velocity.get_mean() if self.ewa_velocity.ewa is not None else 0.0
        return self.current_x + self.velocity_blend * velocity

    def tick(self, payload, performance_metrics=None):
        x = payload["dove_location"]
        t = payload["time"]
        self.add_to_quarantine(t, x)
        self.current_x = x
        self.x_window.append(x)

        if "falcon_location" in payload:
            self.falcon_offset = payload["falcon_location"] - x
        else:
            self.falcon_offset = None

        prev_x = self.pop_from_quarantine(t)

        if prev_x is not None:
            x_change = x - prev_x
            self.dx_window.append(x_change)

            threshold = self.winsor_mult * math.sqrt(
                self.ewa_dx_core.get() if self.count > 0 else 1.0
            )
            winsorized_x_change = (
                np.clip(x_change, -threshold, threshold) if threshold > 0 else x_change
            )

            self.ewa_dx_core.update(winsorized_x_change)
            self.ewa_dx_tail.update(self.tail_update_mult * x_change)
            self.ewa_velocity.update(x_change)

            if self.prev_dx is not None:
                self.ewa_accel.update(x_change - self.prev_dx)
                if x_change != 0 and self.prev_dx != 0:
                    self.last_direction_change = np.sign(x_change) != np.sign(self.prev_dx)
                else:
                    self.last_direction_change = False
            else:
                self.last_direction_change = False

            self.prev_dx = x_change
            self.count += 1

    def predict(self):
        x_mean = self._extrapolated_mean()
        if x_mean is None:
            x_mean = self.current_x

        vol_mult = self._volatility_multiplier(self._rolling_stats())
        scales = [self.scale_core, self.scale_tail]
        components = []

        for i, ewa_dx in enumerate([self.ewa_dx_core, self.ewa_dx_tail]):
            try:
                x_std = math.sqrt(ewa_dx.get())
            except Exception:
                x_std = 1.0

            x_std = max(x_std * scales[i] * vol_mult, 1e-6)

            components.append({
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": x_mean, "scale": x_std},
                },
                "weight": self.weights[i],
            })

        return {"type": "mixture", "components": components}
