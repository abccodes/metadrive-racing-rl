"""Reference-line guidance features for fixed racing maps."""

from __future__ import annotations

from functools import lru_cache
from math import pi

import numpy as np
from metadrive.component.pg_space import Parameter

from .racing_maps import TRACK_SPECS


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _sample_centerline(specs, sample_spacing: float = 5.0) -> np.ndarray:
    """Approximate a map centerline from the block sequence."""
    pos = np.array([0.0, 0.0], dtype=np.float64)
    heading = 0.0
    points = [pos.copy()]

    for kind, params in specs:
        if kind == "straight":
            length = float(params[Parameter.length])
            distance = 0.0
            while distance + sample_spacing < length:
                distance += sample_spacing
                pos = pos + sample_spacing * np.array(
                    [np.cos(heading), np.sin(heading)], dtype=np.float64
                )
                points.append(pos.copy())
            remainder = length - distance
            if remainder > 1e-6:
                pos = pos + remainder * np.array(
                    [np.cos(heading), np.sin(heading)], dtype=np.float64
                )
                points.append(pos.copy())
            continue

        radius = float(params[Parameter.radius])
        angle_deg = float(params[Parameter.angle])
        turn_sign = -1.0 if int(params[Parameter.dir]) == 1 else 1.0
        normal = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
        center = pos + turn_sign * radius * normal
        start_vec = pos - center
        total_angle = np.deg2rad(angle_deg)
        arc_len = radius * total_angle
        arc_steps = max(int(np.ceil(arc_len / sample_spacing)), 1)

        for step in range(1, arc_steps + 1):
            delta = turn_sign * total_angle * (step / arc_steps)
            c = np.cos(delta)
            s = np.sin(delta)
            rot = np.array([[c, -s], [s, c]])
            pos = center + rot @ start_vec
            points.append(pos.copy())

        heading += turn_sign * total_angle

    return np.asarray(points, dtype=np.float32)


def _compute_headings(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0, append=points[-1:])
    if len(points) > 1:
        diffs[-1] = diffs[-2]
    return np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32)


def _compute_progress(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    progress = np.zeros(len(points), dtype=np.float32)
    if len(points) > 1:
        progress[1:] = np.cumsum(seg_lens)
    return progress


@lru_cache(maxsize=None)
def get_track_guidance(track_name: str, sample_spacing: float = 5.0):
    if track_name not in TRACK_SPECS:
        raise ValueError(f"Unknown track for guidance: {track_name}")
    points = _sample_centerline(TRACK_SPECS[track_name], sample_spacing=sample_spacing)
    headings = _compute_headings(points)
    progress = _compute_progress(points)
    return {
        "points": points,
        "headings": headings,
        "progress": progress,
        "length": float(progress[-1]) if len(progress) else 0.0,
    }


class TrackGuidance:
    def __init__(self, track_name: str, lookahead_steps=(10, 25, 45), sample_spacing: float = 5.0):
        self.track_name = track_name
        self.lookahead_steps = tuple(int(s) for s in lookahead_steps)
        self.sample_spacing = float(sample_spacing)
        self.data = get_track_guidance(track_name, sample_spacing=sample_spacing)
        self.points = self.data["points"]
        self.headings = self.data["headings"]
        self.progress = self.data["progress"]
        self.length = self.data["length"]

    @property
    def feature_dim(self) -> int:
        return 2 + 2 * len(self.lookahead_steps)

    def initial_state(self):
        return {"last_index": 0}

    def compute(self, position, heading: float, state: dict | None = None):
        if state is None:
            state = self.initial_state()

        pos = np.asarray(position, dtype=np.float32)
        start = max(0, int(state.get("last_index", 0)) - 15)
        end = min(len(self.points), start + 120)
        search_points = self.points[start:end]
        local_idx = int(np.argmin(np.sum((search_points - pos) ** 2, axis=1)))
        idx = start + local_idx

        ref = self.points[idx]
        tangent_heading = float(self.headings[idx])
        tangent = np.array([np.cos(tangent_heading), np.sin(tangent_heading)], dtype=np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        delta = pos - ref
        lateral_error = float(np.dot(delta, normal))
        heading_error = float(_wrap_angle(float(heading) - tangent_heading))

        features = [
            np.clip(lateral_error / 10.0, -1.0, 1.0),
            np.clip(heading_error / pi, -1.0, 1.0),
        ]

        for step in self.lookahead_steps:
            look_idx = min(idx + step, len(self.points) - 1)
            look_heading = float(self.headings[look_idx])
            rel_heading = _wrap_angle(look_heading - float(heading))
            curve_heading = _wrap_angle(look_heading - tangent_heading)
            features.append(np.clip(rel_heading / pi, -1.0, 1.0))
            features.append(np.clip(curve_heading / pi, -1.0, 1.0))

        updated_state = {"last_index": idx}
        metrics = {
            "index": idx,
            "progress": float(self.progress[idx]),
            "lateral_error": lateral_error,
            "heading_error": heading_error,
        }
        return np.asarray(features, dtype=np.float32), updated_state, metrics
