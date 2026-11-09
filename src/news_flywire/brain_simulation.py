"""Reduced LIF adaptation; parameters from Shiu, explicit event schedules.

This is not a replacement claim for the unchanged upstream Brian2 model.
Subthreshold states are integrated analytically; delays are quantized to dt.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


@dataclass
class SimulationResult:
    spikes: pd.DataFrame
    effective_delay_ms: float
    requested_input_events: int


def simulate(circuit, input_ids, events, config, silenced=()):
    c = config
    dt = float(c["dt_ms"])
    total = sum(c[k] for k in ["baseline_ms", "stimulus_ms", "recovery_ms"])
    if dt <= 0 or c["tau_m_ms"] <= 0 or c["tau_syn_ms"] <= 0 or c["baseline_ms"] <= 0 or c["stimulus_ms"] <= 0 or c["recovery_ms"] <= 0:
        raise ValueError("Time constants and durations must be positive")
    for duration in [total, c["baseline_ms"], c["stimulus_ms"], c["recovery_ms"]]:
        if not np.isclose(duration / dt, round(duration / dt)):
            raise ValueError("Durations must be divisible by dt")
    if c["v_threshold_mv"] <= max(c["v_rest_mv"], c["v_reset_mv"]):
        raise ValueError("Spike threshold must exceed rest and reset")
    if c["delay_ms"] < 0 or c["refractory_ms"] < 0:
        raise ValueError("Delays and refractory period cannot be negative")
    if not set(input_ids).union(silenced).issubset(circuit.index):
        raise ValueError("Unknown input or silenced root IDs")
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("Input IDs must be unique")
    steps = round(total / dt)
    n = len(circuit.ids)
    events = np.asarray(events, dtype=int).reshape(-1, 2)
    if len(events) and ((events < 0).any() or (events[:, 0] >= steps).any() or (events[:, 1] >= len(input_ids)).any()):
        raise ValueError("Input event out of range")
    ei = np.array([circuit.index[v] for v in input_ids], dtype=int)
    blocked = {circuit.index[v] for v in silenced}
    pre = circuit.edges.pre.map(circuit.index).to_numpy(dtype=int)
    post = circuit.edges.post.map(circuit.index).to_numpy(dtype=int)
    weights = circuit.edges.signed_synapses.to_numpy(dtype=float) * c["weight_mv"]
    mask = ~np.isin(pre, list(blocked)) & ~np.isin(post, list(blocked))
    syn = csc_matrix((weights[mask], (post[mask], pre[mask])), shape=(n, n))
    # A minimum one-step transmission avoids instantaneous loops.
    delay_steps = max(1, round(c["delay_ms"] / dt))
    pending = np.zeros((delay_steps + 1, n))
    v = np.full(n, c["v_rest_mv"], dtype=float)
    g = np.zeros(n)
    refractory_until = np.zeros(n, dtype=int)
    ref_steps = np.full(n, int(np.ceil(c["refractory_ms"] / dt)), dtype=int)
    ref_steps[ei] = 0  # upstream Poisson-target convention
    tm, ts = c["tau_m_ms"], c["tau_syn_ms"]
    em, es = np.exp(-dt/tm), np.exp(-dt/ts)
    coupling = (dt/tm)*em if np.isclose(tm, ts) else ts/(ts-tm)*(es-em)
    schedule = {}
    for step, cell in events:
        schedule.setdefault(int(step), []).append(ei[cell])
    output_steps, output_cells = [], []
    for step in range(steps):
        slot = step % len(pending)
        active = step >= refractory_until
        # Like upstream (unless refractory), both v and g are held in refractory.
        v[active] = c["v_rest_mv"] + (v[active]-c["v_rest_mv"])*em + coupling*g[active]
        g[active] *= es
        spiking = np.flatnonzero((v > c["v_threshold_mv"]) & active)
        # Brian2 ordering: state update -> threshold -> synaptic events -> reset.
        # Inputs received in this step are first tested for spikes next step.
        g += pending[slot]
        pending[slot].fill(0)
        if step in schedule:
            np.add.at(v, schedule[step], c["input_jump_mv"])
        if len(spiking):
            output_steps.extend([step]*len(spiking))
            output_cells.extend(spiking.tolist())
            v[spiking] = c["v_reset_mv"]
            g[spiking] = 0
            refractory_until[spiking] = step + np.maximum(1, ref_steps[spiking])
            # Sparse column slicing transmits only from neurons that fired.
            pending[(step + delay_steps) % len(pending)] += np.asarray(syn[:, spiking].sum(axis=1)).ravel()
    spikes = pd.DataFrame({"time_ms": np.asarray(output_steps)*dt,
                           "root_id": pd.Series([circuit.ids[i] for i in output_cells], dtype="string")})
    return SimulationResult(spikes, delay_steps*dt, len(events))


def response_metrics(result, circuit, config):
    c = config
    spikes = result.spikes
    start, end = c["baseline_ms"], c["baseline_ms"] + c["stimulus_ms"]
    before = spikes.loc[spikes.time_ms < start].groupby("root_id").size().reindex(circuit.ids, fill_value=0) * 1000 / start
    during = spikes.loc[(spikes.time_ms >= start) & (spikes.time_ms < end)].groupby("root_id").size().reindex(circuit.ids, fill_value=0) * 1000 / c["stimulus_ms"]
    delta = during-before
    outputs = circuit.cells(role="output")
    if not outputs:
        raise ValueError("Output circuit must be specified before simulation")
    bin_ms = c["latency_bin_ms"]
    total = end + c["recovery_ms"]
    bins = np.arange(0, total+bin_ms, bin_ms)
    out = spikes[spikes.root_id.isin(outputs)]
    counts, _ = np.histogram(out.time_ms, bins=bins)
    rate = counts * 1000 / (np.diff(bins)*len(outputs))
    times = bins[:-1]
    base = float(before.loc[outputs].mean())
    stim = (times >= start) & (times < end)
    crossings = times[stim & (rate >= base + c["latency_threshold_hz"])]
    latency = float(crossings[0]-start) if len(crossings) else np.nan
    elevated = np.any(rate[stim] >= base+c["latency_threshold_hz"])
    recovery = np.nan
    after = np.flatnonzero(times >= end)
    if elevated:
        for i in after:
            # Sustained for the remaining window, with at least three bins.
            if len(rate)-i >= 3 and np.all(np.abs(rate[i:]-base) <= c["recovery_threshold_hz"]):
                recovery = float(times[i]-end)
                break
    noninput = circuit.neurons.loc[~circuit.neurons.role.eq("input"), "root_id"]
    metrics = {
        "baseline_rate_hz": base, "stimulus_rate_hz": float(during.loc[outputs].mean()),
        "delta_rate_hz": float(delta.loc[outputs].mean()), "latency_ms": latency,
        "auc_spikes_per_neuron": float(np.sum((rate[stim]-base)*np.diff(bins)[stim]/1000)),
        "recovery_ms": recovery,
        "recruited_neurons": int((delta.loc[noninput] >= c["recruitment_delta_hz"]).sum()),
        "recruitment_fraction": float((delta.loc[noninput] >= c["recruitment_delta_hz"]).mean()),
        "external_input_events": result.requested_input_events,
        "effective_delay_ms": result.effective_delay_ms,
    }
    cells = pd.DataFrame({"root_id":circuit.ids, "baseline_rate_hz":before.to_numpy(),
                          "stimulus_rate_hz":during.to_numpy(), "delta_rate_hz":delta.to_numpy()})
    curve = pd.DataFrame({"time_ms":times, "rate_hz":rate, "baseline_rate_hz":base})
    return metrics, cells, curve
