from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def ordinal_alpha(labels, axis):
    """Krippendorff ordinal alpha from coincidence counts, including ties/missing units."""
    coincidence = np.zeros((5, 5))
    for _, group in labels.groupby("news_id"):
        values = group[axis].dropna().astype(int).to_numpy()
        if len(values) < 2:
            continue
        counts = np.bincount(values, minlength=5)
        coincidence += (np.outer(counts, counts)-np.diag(counts))/(len(values)-1)
    n = coincidence.sum()
    if n < 2:
        return None
    marginal = coincidence.sum(axis=0)
    mid = np.cumsum(marginal)-marginal/2
    distance = (mid[:,None]-mid[None,:])**2
    observed = (coincidence*distance).sum()/n
    expected = (np.outer(marginal,marginal)*distance).sum()/(n*(n-1))
    return float(1-observed/expected) if expected > 0 else None


def evaluate_ratings(scores, labels, splits, bootstrap=1000, seed=20261109):
    required = {"news_id", "rater_id", "N", "T", "A"}
    if not required.issubset(labels):
        raise ValueError("Human labels require news_id, rater_id, N, T, A")
    labels = labels.copy()
    for axis in "NTA":
        labels[axis] = pd.to_numeric(labels[axis], errors="raise")
        v = labels[axis].to_numpy(dtype=float)
        if not np.isfinite(v).all() or ((v < 0)|(v > 4)|(v != np.floor(v))).any():
            raise ValueError("Complete independent human ratings (integer 0..4) required; blank template is not data")
    if labels.duplicated(["news_id","rater_id"]).any() or labels.rater_id.fillna("").eq("").any():
        raise ValueError("One rating per article/rater and nonempty rater ID required")
    if (labels.groupby("news_id").rater_id.nunique() < 3).any():
        raise ValueError("At least 3 independent raters per article required")
    if not set(labels.news_id).issubset(set(scores.news_id)):
        raise ValueError("Labels contain unknown news IDs")
    if splits.news_id.duplicated().any() or set(scores.news_id) != set(splits.news_id):
        raise ValueError("Split manifest must cover every article exactly once")
    split_check = splits.groupby("event_group").split.nunique()
    if (split_check > 1).any():
        raise ValueError("Event leakage between development and test")
    source_events = scores.set_index("news_id").event_group
    if not splits.event_group.equals(splits.news_id.map(source_events)):
        raise ValueError("Split event groups differ from the article metadata")
    consensus = labels.groupby("news_id")[["N","T","A"]].median()/4
    consensus = consensus.rename(columns={k:f"{k}_human" for k in "NTA"})
    joined = scores.merge(consensus, left_on="news_id", right_index=True, validate="one_to_one")
    assignment = splits.set_index("news_id")["split"]
    joined["split"] = joined.news_id.map(assignment)
    test = joined[joined.split.eq("test")].copy()
    if test.empty:
        raise ValueError("No human-rated articles in fixed test set")
    rng = np.random.default_rng(seed)
    groups = sorted(test.event_group.unique())
    result = {"data_source":"human_vs_predictions", "n_test_articles":len(test),
              "n_test_events":len(groups), "bootstrap_unit":"event_group", "bootstrap_repeats":bootstrap,
              "ordinal_alpha":{axis:ordinal_alpha(labels,axis) for axis in "NTA"}, "metrics":{}}
    for axis in "NTA":
        truth, pred = test[f"{axis}_human"].to_numpy(), test[axis].to_numpy()
        rhos, maes = [], []
        for _ in range(bootstrap):
            sampled = rng.choice(groups,len(groups),replace=True)
            boot = pd.concat([test[test.event_group.eq(g)] for g in sampled],ignore_index=True)
            rho = spearman(boot[f"{axis}_human"],boot[axis])
            if rho is not None:
                rhos.append(rho)
            maes.append(float(np.mean(np.abs(boot[f"{axis}_human"]-boot[axis]))))
        result["metrics"][axis] = {"spearman_rho":spearman(truth,pred), "mae_0_to_1":float(np.mean(np.abs(truth-pred))),
            "spearman_ci95":np.quantile(rhos,[.025,.975]).tolist() if rhos else None,
            "mae_ci95":np.quantile(maes,[.025,.975]).tolist() if maes else None,
            "valid_rho_bootstraps":len(rhos)}
    subgroup_rows = []
    for field in ["category","source"]:
        if field not in test:
            continue
        for group, part in test.groupby(field):
            subgroup_rows.append({"group_field":field,"group":group,"n":len(part),
                                  "T_mae":float(np.mean(np.abs(part.T_human-part["T"]))),
                                  "T_spearman":spearman(part.T_human,part["T"])})
    return result, joined, pd.DataFrame(subgroup_rows)


def summarize_experiments(metrics):
    summary = metrics.groupby("condition_id").agg(
        seeds=("seed","count"), delta_mean_hz=("delta_rate_hz","mean"),
        delta_sd_hz=("delta_rate_hz","std"), latency_mean_ms=("latency_ms","mean"),
        recruitment_mean=("recruitment_fraction","mean"), auc_mean=("auc_spikes_per_neuron","mean"))
    pivot = metrics.pivot(index="seed", columns="condition_id", values="delta_rate_hz")
    contrasts = []
    for a,b in [("E3","E0"),("E3","E4"),("E3","E5"),("E3","E6"),("E3","E7"),("E7","E5"),("E9a","E9b")]:
        d = (pivot[a]-pivot[b]).dropna().to_numpy()
        rng=np.random.default_rng(20261109)
        estimates = rng.choice(d,(2000,len(d)),replace=True).mean(axis=1)
        lo,hi=np.quantile(estimates,[.025,.975])
        contrasts.append({"contrast":f"{a} minus {b}","mean_delta_hz":float(d.mean()),
                          "paired_seed_sd_hz":float(d.std(ddof=1)) if len(d)>1 else np.nan,
                          "computational_ci95_low":float(lo),"computational_ci95_high":float(hi),
                          "unit":"paired computational seed; not biological subjects"})
    return summary.reset_index(), pd.DataFrame(contrasts)


def compare_observations(observed, circuit, cells):
    """Compare measured response directions only at explicitly matched cell types.

    A calcium delta and a firing-rate delta have different units: no MAE or
    amplitude identity is asserted. Missing/unmatched types stay in the table.
    """
    required={"cell_type","response_delta","modality","source","alignment_level"}
    if not required.issubset(observed):
        raise ValueError(f"Observed data missing columns: {sorted(required-set(observed))}")
    if observed.source.fillna("").eq("").any() or observed.modality.fillna("").eq("").any():
        raise ValueError("Measured-data source and modality are required")
    observed=observed.copy()
    observed.response_delta=pd.to_numeric(observed.response_delta,errors="raise")
    if not np.isfinite(observed.response_delta).all():
        raise ValueError("Nonfinite measured response")
    typed=cells[cells.condition_id.eq("E3")].merge(circuit.neurons[["root_id","cell_type"]],on="root_id",validate="many_to_one")
    model=typed.groupby("cell_type").delta_rate_hz.mean()
    observed["model_delta_hz"]=observed.cell_type.map(model)
    observed["comparison_status"]=np.where(observed.model_delta_hz.isna(),"unmatched_cell_type",
        np.where(observed.alignment_level.ne("cell_type"),"unsupported_alignment_level","direction_only"))
    valid=observed.comparison_status.eq("direction_only")
    observed["direction_agreement"]=pd.Series(pd.NA,index=observed.index,dtype="boolean")
    observed.loc[valid,"direction_agreement"]=(np.sign(observed.loc[valid,"response_delta"])==np.sign(observed.loc[valid,"model_delta_hz"]))
    observed["limitation"]="Cell-type summary only; different animals and measurement modalities; no neuron identity or amplitude equivalence"
    return observed
