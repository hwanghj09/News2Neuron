"""Physical looming and hypothetical neural drive are deliberately separate."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stimulus:
    score: float
    variable: str
    level: str
    rate_hz: float
    l_over_v_ms: float


def encode(scores, config, variable="T", weights=None):
    for key in "NTA":
        if key not in scores or not np.isfinite(scores[key]) or not 0 <= scores[key] <= 1:
            raise ValueError("Normalized N/T/A must be finite and between 0 and 1")
    if variable == "composite":
        w = np.asarray(weights if weights is not None else config["composite_weights"], dtype=float)
        if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
            raise ValueError("Composite weights must be nonnegative and sum to one")
        score = float(w @ np.array([scores[k] for k in "NTA"]))
    elif variable in "NTA" and len(variable) == 1:
        score = float(scores[variable])
    else:
        raise ValueError("variable must be N, T, A or composite")
    thresholds = np.asarray(config["thresholds"], dtype=float)
    rates = np.asarray(config["rates_hz"], dtype=float)
    if thresholds.shape != (2,) or not 0 < thresholds[0] < thresholds[1] < 1 or rates.shape != (3,) or (rates < 0).any() or not np.isfinite(rates).all():
        raise ValueError("Invalid levels or rates")
    level = int(np.searchsorted(thresholds, score, side="right"))
    return Stimulus(score, variable, ["L", "M", "H"][level], float(rates[level]),
                    float(config["looming"]["l_over_v_ms"][level]))


def looming(stimulus: Stimulus, config, n_frames=121):
    """Constant-speed approach, constant initial/final angles, padded to equal duration.

    Speed changes imply different approach durations. Hold the initial disk first;
    align the final angle to presentation end. No collision or d<=0 is rendered.
    """
    c = config["looming"]
    radius = float(c["radius_m"])
    a0, a1 = np.deg2rad([c["initial_angle_deg"], c["final_angle_deg"]])
    if not (radius > 0 and 0 < a0 < a1 < np.pi and stimulus.l_over_v_ms > 0):
        raise ValueError("Invalid physical looming parameters")
    v = radius / (stimulus.l_over_v_ms / 1000)
    d0, d1 = radius / np.tan(a0 / 2), radius / np.tan(a1 / 2)
    duration_ms = (d0 - d1) / v * 1000
    # All levels share a window long enough for the slowest approach.
    common_ms = max(float(c["presentation_ms"]),
                    max(c["l_over_v_ms"]) * (1 / np.tan(a0 / 2) - 1 / np.tan(a1 / 2)))
    t = np.linspace(0, common_ms, n_frames)
    start = common_ms - duration_ms
    distance = np.maximum(d1, d0 - v * np.maximum(t - start, 0) / 1000)
    angle = 2 * np.arctan(radius / distance)
    return pd.DataFrame({"time_ms": t, "distance_m": distance, "angle_deg": np.rad2deg(angle),
                         "radius_m": radius, "speed_m_s": v, "l_over_v_ms": stimulus.l_over_v_ms,
                         "level": stimulus.level, "approach_start_ms": start,
                         "source": "physical_visual_stimulus_not_neural_calibration"})


def poisson_schedule(n_cells, rate_hz, sim, seed):
    """Continuous Poisson arrivals, quantized to dt; count multiplicity is retained."""
    if n_cells < 1 or rate_hz < 0 or not np.isfinite(rate_hz):
        raise ValueError("At least one input cell and a finite nonnegative rate required")
    rng = np.random.default_rng(seed)
    duration = sim["stimulus_ms"]
    n_steps = round((sim["baseline_ms"] + duration + sim["recovery_ms"]) / sim["dt_ms"])
    events = []
    for cell in range(n_cells):
        count = rng.poisson(rate_hz * duration / 1000)
        t = sim["baseline_ms"] + rng.uniform(0, duration, count)
        steps = np.minimum((t / sim["dt_ms"]).astype(int), n_steps - 1)
        events.extend((int(step), cell) for step in steps)
    return np.asarray(sorted(events), dtype=int).reshape(-1, 2)
