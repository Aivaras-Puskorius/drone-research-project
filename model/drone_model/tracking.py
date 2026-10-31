"""One-target tracker fed with the single most confident detection of each frame.

Each frame, the most confident detection above the confidence threshold goes to a constant-velocity Kalman
filter in pixel coordinates. A track starts after `hits_required` consecutive detections that fit the
prediction (a 99% chi-square gate and at most 200 px), and coasts on its prediction for up to 0.35 s without
one. The prediction also moves with the camera, by the whole-frame background shift between f-1 and f. The
displayed box is the Kalman centre with the size of the last detection used.

replay_window's `association` picks the detection a frame feeds the tracker: 'top1' (the default, as in every
experiment through round O) takes the most confident one; 'gated' gives an established track the most confident
detection inside its gate, so a stronger detection elsewhere (a car, say) does not end the track, and falls back to
the most confident one when no established track exists or none fits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

UNINITIALIZED, TENTATIVE, TRACKING, PREDICTED, LOST = 'UNINITIALIZED', 'TENTATIVE', 'TRACKING', 'PREDICTED', 'LOST'
ASSOCIATIONS = ('top1', 'gated')


@dataclass(frozen=True)
class Step:
    state: str
    point: tuple[float, float] | None
    used: bool


class Tracker:
    """Call step() once per frame with strictly increasing timestamps in seconds."""

    def __init__(self, hits_required: int = 3, max_coast_seconds: float = 0.35, max_association_distance_px: float = 200.0,
                 gate_squared: float = 9.21034, measurement_std_px: float = 5.0, initial_velocity_std_px_s: float = 500.0,
                 acceleration_std_px_s2: float = 200.0):
        if hits_required < 2:
            raise ValueError('a track needs at least two consecutive hits')
        self.hits_required, self.max_coast = hits_required, max_coast_seconds
        self.max_distance, self.gate = max_association_distance_px, gate_squared
        self.measurement_std, self.velocity_std, self.acceleration_std = (
            measurement_std_px, initial_velocity_std_px_s, acceleration_std_px_s2)
        self.state = UNINITIALIZED
        self._x = self._p = None
        self._last_time = self._last_measurement = None
        self._origin = UNINITIALIZED
        self._hits = 0

    def _reset(self, state: str) -> None:
        self.state, self._x, self._p, self._hits = state, None, None, 0

    def _predict(self, dt: float, shift: np.ndarray) -> None:
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        noise_map = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        x = transition @ self._x
        x[:2] += shift
        p = transition @ self._p @ transition.T + self.acceleration_std ** 2 * (noise_map @ noise_map.T)
        self._x, self._p = x, (p + p.T) / 2

    def _start(self, point: np.ndarray) -> None:
        self._x = np.array([point[0], point[1], 0.0, 0.0])
        self._p = np.diag([self.measurement_std ** 2] * 2 + [self.velocity_std ** 2] * 2)

    def _update(self, point: np.ndarray) -> None:
        observation = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        noise = np.eye(2) * self.measurement_std ** 2
        gain = np.linalg.solve(self._p[:2, :2] + noise, self._p[:2, :]).T
        self._x += gain @ (point - self._x[:2])
        residual = np.eye(4) - gain @ observation
        self._p = residual @ self._p @ residual.T + gain @ noise @ gain.T
        self._p = (self._p + self._p.T) / 2

    def _fits(self, point: np.ndarray) -> bool:
        innovation = point - self._x[:2]
        covariance = self._p[:2, :2] + np.eye(2) * self.measurement_std ** 2
        distance = float(np.linalg.norm(innovation))
        mahalanobis = float(innovation @ np.linalg.solve(covariance, innovation))
        return distance <= self.max_distance and mahalanobis <= self.gate

    def established_fits(self, timestamp_s: float, points: list[tuple[float, float]], background_shift) -> list[bool] | None:
        """For an established track (TRACKING or PREDICTED, within its coast time), whether step() at this timestamp
        would take each point; None without one. The tracker is not changed."""
        if self.state not in (TRACKING, PREDICTED) or float(timestamp_s) - self._last_measurement > self.max_coast + 1e-12:
            return None
        saved = self._x, self._p
        try:
            self._predict(float(timestamp_s) - self._last_time, np.asarray(background_shift, dtype=np.float64))
            return [self._fits(np.asarray(point, dtype=np.float64)) for point in points]
        finally:
            self._x, self._p = saved  # _predict builds new arrays, so the saved ones are untouched

    def step(self, timestamp_s: float, detection: tuple[float, float] | None, background_shift) -> Step:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or (self._last_time is not None and timestamp <= self._last_time):
            raise ValueError('timestamps must be finite and strictly increasing')
        shift = np.asarray(background_shift, dtype=np.float64)
        if shift.shape != (2,) or not np.isfinite(shift).all():
            raise ValueError('background_shift must be two finite pixel offsets')
        point = None if detection is None else np.array(detection, dtype=np.float64)
        if point is not None and not np.isfinite(point).all():
            raise ValueError('detection must be finite')
        # A tentative track whose last hit is too old falls back; an established one is lost.
        if self.state == TENTATIVE and timestamp - self._last_measurement > self.max_coast:
            self._reset(self._origin)
            self._last_measurement = None
        if self.state in (TRACKING, PREDICTED) and timestamp - self._last_measurement > self.max_coast + 1e-12:
            self._reset(LOST)
        if self._x is not None:
            self._predict(timestamp - self._last_time, shift)
        self._last_time = timestamp
        used = False
        if self.state in (TRACKING, PREDICTED, TENTATIVE):
            if point is not None and self._fits(point):
                self._update(point)
                self._last_measurement, used = timestamp, True
                if self.state == TENTATIVE:
                    self._hits += 1
                    if self._hits >= self.hits_required:
                        self.state = TRACKING
                else:
                    self.state = TRACKING
            elif self.state == TENTATIVE:
                self._reset(self._origin)
                self._last_measurement = None
            elif timestamp - self._last_measurement <= self.max_coast + 1e-12:
                self.state = PREDICTED
            else:
                self._reset(LOST)
        # With no track, any detection starts a tentative one at one hit, in the same frame a tentative one failed.
        if self.state in (UNINITIALIZED, LOST) and point is not None:
            self._origin = self.state
            self._start(point)
            self._hits = 1
            self._last_measurement, used = timestamp, True
            self.state = TENTATIVE
        return Step(self.state, None if self._x is None else (float(self._x[0]), float(self._x[1])), used)


def select_candidate(proposals: list[dict], threshold: float, fits=None) -> dict | None:
    """The most confident proposal at or above the threshold; ties go to the smaller x, then y. With `fits` (a
    function from the passing proposals' points to a list of booleans, or None), only the fitting ones count when
    any fits."""
    passing = [p for p in proposals if all(math.isfinite(float(p[k])) for k in ('x', 'y', 'confidence'))
               and p['confidence'] >= threshold]
    if passing and fits is not None:
        inside = fits([(p['x'], p['y']) for p in passing])
        if inside and any(inside):
            passing = [p for p, ok in zip(passing, inside) if ok]
    return min(passing, key=lambda p: (-p['confidence'], p['x'], p['y'])) if passing else None


def replay_window(rows: list[dict], fps: float, width: int, height: int, *, threshold: float,
                  hits_required: int = 3, association: str = 'top1') -> tuple[list[dict], list[dict]]:
    """Track one window from its saved per-frame proposals.

    rows: one per consecutive frame with 'frame', 'proposals' and 'background_shift'. Returns per frame the
    tracked point and box (both None when no box is shown) and the acquisition events: each time a track is
    established, 'initial_acquisition' the first time and 'reacquisition' after that. `association` is 'top1' or
    'gated' (module docstring).
    """
    if association not in ASSOCIATIONS:
        raise ValueError(f'unknown association {association!r}')
    tracker = Tracker(hits_required=hits_required)
    size, frames, events, previous = (32., 32.), [], [], None
    for row in rows:
        if previous is not None and row['frame'] != previous + 1:
            raise ValueError('frames must be consecutive')
        previous = row['frame']
        fits = None
        if association == 'gated':
            def fits(points, row=row):
                return tracker.established_fits(row['frame'] / fps, points, row['background_shift'])
        chosen = select_candidate(row['proposals'], threshold, fits)
        before = tracker.state
        result = tracker.step(row['frame'] / fps, None if chosen is None else (chosen['x'], chosen['y']),
                              row['background_shift'])
        if result.used:
            x0, y0, x1, y1 = chosen['xyxy']
            size = (max(1., x1 - x0), max(1., y1 - y0))
        if result.state == TRACKING and before not in (TRACKING, PREDICTED):
            events.append({'frame': row['frame'], 'kind': 'reacquisition' if events else 'initial_acquisition'})
        point = box = None
        if result.state in (TRACKING, PREDICTED) and result.point is not None:
            x, y = result.point
            if 0 <= x < width and 0 <= y < height:
                point, box = [x, y], [x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2]
        frames.append({'frame': row['frame'], 'state': result.state, 'point': point, 'box': box,
                       'detection': None if chosen is None else [chosen['x'], chosen['y']]})
    return frames, events
