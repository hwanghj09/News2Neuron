from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

from .brain_simulation import simulate, response_metrics
from .flywire_graph import degree_matched_control
from .stimulus_encoder import encode, poisson_schedule
from .provenance import write_json, manifest

CONDITIONS = {
    "E0":"No input", "E1":"Low T", "E2":"Medium T", "E3":"High T",
    "E4":"High T, identical events on random cells", "E5":"High T, LPLC2 outgoing and incoming connections removed",
    "E6":"High T, LC4 outgoing and incoming connections removed", "E7":"High T, degree-matched random silencing",
    "E8":"High N / low T, N-only mapping", "E9a":"T=0.5, N=0.1", "E9b":"T=0.5, N=0.9",
}


def run_experiments(circuit, config, out_dir, seeds=None, news_scores=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.csv").exists():
        raise ValueError("Output already contains a run. Choose a fresh --out path to preserve provenance")
    n_seeds = seeds if seeds is not None else config["experiment"]["seeds"]
    if n_seeds < 1:
        raise ValueError("seeds must be positive")
    inputs, outputs = circuit.cells(role="input"), circuit.cells(role="output")
    lplc2, lc4 = circuit.cells(cell_type="LPLC2"), circuit.cells(cell_type="LC4")
    if not inputs or not outputs or not lplc2 or not lc4:
        raise ValueError("Required input/output types are missing")
    # Partial circuits may not contain enough independently annotated sensory cells.
    # Do not relabel arbitrary intermediates as sensory neurons.
    random_pool = sorted(set(circuit.ids)-set(inputs)-set(outputs))
    if len(random_pool) < len(inputs):
        raise ValueError("Insufficient disjoint random-input neurons")
    all_metrics, all_cells, all_curves, all_input, selections = [], [], [], [], []
    sim = config["simulation"]
    for k in range(n_seeds):
        seed = config["experiment"]["first_seed"] + k
        rng = np.random.default_rng(seed)
        random_inputs = rng.choice(random_pool, len(inputs), replace=False).tolist()
        silenced_control, mismatch = degree_matched_control(circuit, lplc2, set(inputs)|set(outputs), seed)
        for condition, description in CONDITIONS.items():
            scores = dict(config["experiment"]["high_scores"])
            variable = "T"
            if condition in ["E1", "E2"]:
                scores["T"] = {"E1":0.1, "E2":0.5}[condition]
            if condition == "E8":
                scores.update(N=0.9, T=0.1)
                variable = "N"
            if condition.startswith("E9"):
                scores.update(T=0.5, N=0.1 if condition == "E9a" else 0.9)
            stimulus = encode(scores, config, variable)
            rate = 0 if condition == "E0" else stimulus.rate_hz
            # Re-using the same seed gives E3/E4/E5/E6/E7 identical events, and E9a/E9b identical events.
            schedule = poisson_schedule(len(inputs), rate, sim, seed)
            active_inputs = random_inputs if condition == "E4" else inputs
            blocked = {"E5":lplc2, "E6":lc4, "E7":silenced_control}.get(condition, [])
            result = simulate(circuit, active_inputs, schedule, sim, blocked)
            metrics, cells, curve = response_metrics(result, circuit, sim)
            metadata = {"condition_id":condition, "description":description, "seed":seed,
                        "connectome_version":circuit.metadata["connectome_version"],
                        "data_source":"simulated_activity", "score_variable":variable,
                        "N":scores["N"], "T":scores["T"], "A":scores["A"], "rate_hz":rate,
                        "degree_matching_mean_log_distance":mismatch if condition == "E7" else np.nan,
                        "random_input_population":"non-input, non-output subcircuit neurons; not sensory-matched" if condition == "E4" else "target_inputs"}
            all_metrics.append({**metadata, **metrics})
            for frame, collector in [(cells, all_cells), (curve, all_curves)]:
                frame["condition_id"], frame["seed"] = condition, seed
                collector.append(frame)
            selections.append({"condition_id":condition, "seed":seed, "input_cell_ids":active_inputs,
                               "output_cell_ids":outputs, "silenced_cell_ids":blocked,
                               "stimulation_params":{**asdict(stimulus), "rate_hz":rate}})
            input_frame = pd.DataFrame({"step":schedule[:,0], "input_slot":schedule[:,1]})
            input_frame["time_ms"] = input_frame.step * sim["dt_ms"]
            input_frame["root_id"] = [active_inputs[i] for i in schedule[:,1]]
            input_frame["condition_id"], input_frame["seed"] = condition, seed
            all_input.append(input_frame)
            sp = result.spikes.copy()
            sp["condition_id"], sp["seed"] = condition, seed
            spike_dir = out / "spikes" / condition
            spike_dir.mkdir(parents=True, exist_ok=True)
            sp.to_parquet(spike_dir / f"seed{seed}.parquet", index=False)
            if k == 0:
                sp.to_parquet(out / f"spikes_{condition}_seed{seed}.parquet", index=False)
        print(f"Completed computational seed {k+1}/{n_seeds}", flush=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(all_cells, ignore_index=True).to_parquet(out / "cell_metrics.parquet", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(out / "output_curves.csv", index=False)
    pd.concat(all_input, ignore_index=True).to_parquet(out / "input_events.parquet", index=False)
    write_json(out / "cell_selections.json", selections)
    write_json(out / "resolved_config.json", config)
    write_json(out / "manifest.json", manifest(sorted(Path(__file__).parent.glob("*.py")), circuit=circuit.metadata, computational_repeats=n_seeds,
       engine="reduced_lif_analytic_v1", raw_spikes_saved="all conditions and seeds in spikes/{condition}/seed{seed}.parquet; first seed also saved at run root",
       inference_scope="Computational seed variability; not biological independent samples",
       controls={"E4":"exact external event-count matching; random cells not sensory-matched",
                 "E7":"same-size closest log in/out degree set; report residual mismatch"},
       limitations=["Not a calibrated visual-neuron encoder", "Not the full-brain Shiu implementation",
                    "No human physiological validation", "L/M/H rate mapping is a preregistered hypothesis, not a measured tuning curve"]))
    if news_scores is not None:
        # Link every article to a level-specific E1/E2/E3 per-seed response distribution.
        # Articles sharing a level share the same simulation, not independent evidence.
        articles = []
        for row in news_scores.to_dict("records"):
            encoded = encode(row, config)
            condition = {"L":"E1", "M":"E2", "H":"E3"}[encoded.level]
            for item in metrics[metrics.condition_id.eq(condition)].to_dict("records"):
                articles.append({**item, "news_id":row["news_id"], "event_group":row["event_group"],
                                 "N_ai":row["N"], "T_ai":row["T"], "A_ai":row["A"],
                                 "article_data_source":row["data_source"],
                                 "shared_level_simulation":True})
        pd.DataFrame(articles).to_csv(out / "article_responses.csv", index=False)
    return metrics
