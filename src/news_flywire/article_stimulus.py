"""Continuous article-specific physical geometry and hypothetical neural drive."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def article_seed(news_id, seed):
    payload = f'{news_id}:{int(seed)}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def profile(score, config):
    """A constant-speed disk approach aligned to stimulus end; zero is static.

    80% of the expected input is tonic, 20% follows angular expansion. The
    complete rate profile integrates to score * maximum_mean_rate * duration.
    This is a defined transfer function, not measured visual-neuron tuning.
    """
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Normalized drive must be finite and in [0,1]')
    c, p = config['simulation'], config['physical']
    dt, duration = float(c['dt_ms']), float(c['stimulus_ms'])
    if dt <= 0 or duration <= 0 or not np.isclose(duration/dt, round(duration/dt)):
        raise ValueError('Positive duration must be divisible by dt')
    a0, a1 = np.deg2rad([p['initial_angle_deg'], p['final_angle_deg']])
    radius = float(p['radius_m'])
    slow, fast = float(p['slow_l_over_v_ms']), float(p['fast_l_over_v_ms'])
    mix, maximum = float(config['angular_modulation_weight']), float(config['maximum_mean_rate_hz'])
    proposal = float(config['proposal_rate_hz'])
    if not (radius > 0 and 0 < a0 < a1 < np.pi and 0 < fast <= slow and 0 <= mix <= 1 and maximum >= 0 and proposal > 0):
        raise ValueError('Invalid geometry or rate parameters')
    lv = slow+(fast-slow)*float(score)
    approach_ms = lv*(1/np.tan(a0/2)-1/np.tan(a1/2))
    if approach_ms > duration+1e-9:
        raise ValueError('Physical approach does not fit simulation stimulus duration')
    times = np.arange(round(duration/dt)+1)*dt
    start = duration-approach_ms
    d0, d1 = radius/np.tan(a0/2), radius/np.tan(a1/2)
    distances = np.maximum(d1, d0-radius/lv*np.maximum(times-start, 0))
    if score == 0:
        distances[:] = d0
    angles = 2*np.arctan(radius/distances)
    expansion = np.maximum(np.diff(angles), 0)
    shape = np.ones(len(times)-1)
    if score > 0:
        shape = (1-mix)+mix*expansion/expansion.mean()
    rates = maximum*float(score)*shape
    if rates.max(initial=0) > proposal+1e-9:
        raise ValueError('Poisson proposal rate does not bound the complete profile')
    frames = pd.DataFrame(dict(time_ms=times, angle_deg=np.rad2deg(angles),
        distance_m=distances, radius_m=radius, l_over_v_ms=lv,
        approach_start_ms=start, speed_m_s=0. if score == 0 else radius/lv*1000))
    return rates, frames


def schedules(news_id, seed, scores, n_cells, config):
    """Common Poisson proposals across R/T/N, exact reuse for circuit controls."""
    if n_cells < 1:
        raise ValueError('At least one input neuron is required')
    rates = {k: profile(v, config)[0] for k, v in scores.items()}
    sim, proposal = config['simulation'], float(config['proposal_rate_hz'])
    duration, dt = sim['stimulus_ms'], sim['dt_ms']
    if not np.isclose(sim['baseline_ms']/dt, round(sim['baseline_ms']/dt)):
        raise ValueError('Baseline duration must be divisible by dt')
    rng = np.random.default_rng(article_seed(news_id, seed))
    result = {k: [] for k in scores}
    for cell in range(n_cells):
        count = rng.poisson(proposal*duration/1000)
        time = rng.uniform(0, duration, count)
        indices = np.minimum((time/dt).astype(int), len(next(iter(rates.values())))-1)
        accept = rng.uniform(size=count)
        for key, values in rates.items():
            keep = accept < values[indices]/proposal
            steps = indices[keep]+round(sim['baseline_ms']/dt)
            result[key].extend((int(step), cell) for step in steps)
    return {k: np.asarray(sorted(v), dtype=np.int64).reshape(-1, 2) for k, v in result.items()}


def event_hash(events):
    return hashlib.sha256(np.asarray(events, dtype='<i8').reshape(-1, 2).tobytes()).hexdigest()
