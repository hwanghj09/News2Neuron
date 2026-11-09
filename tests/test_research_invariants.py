import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from news_flywire.news_scoring import rule_score, event_split, import_llm_scores
from news_flywire.stimulus_encoder import encode, looming, poisson_schedule
from news_flywire.flywire_graph import Circuit, demo_circuit, degree_matched_control
from news_flywire.brain_simulation import simulate, response_metrics
from news_flywire.evaluation import ordinal_alpha, evaluate_ratings, compare_observations
from news_flywire.experiments import run_experiments

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT/"configs/stimulation_v1.yaml").read_text())


def test_sadness_and_immediate_threat_are_distinct():
    grief=rule_score("가족의 사별 경험을 담은 추모 기록")
    emergency=rule_score("태풍 경보로 즉시 대피")
    assert grief["N"] >= .75 and grief["T"] == 0
    assert emergency["T"] == 1
    assert grief["confidence"] is None


def test_event_split_never_leaks():
    news=pd.DataFrame({"news_id":list("abcdefgh"),"event_group":["x","x","y","y","z","z","q","q"]})
    split=event_split(news)
    assert split.groupby("event_group").split.nunique().max()==1
    pd.testing.assert_frame_equal(split,event_split(news))


@pytest.mark.parametrize("value",[-.01,1.01,float("nan"),float("inf")])
def test_encoder_rejects_unphysical_scores(config,value):
    with pytest.raises(ValueError):
        encode({"N":.5,"T":value,"A":.5},config)


def test_primary_encoder_is_invariant_to_N_and_A(config):
    assert encode({"N":0,"T":.5,"A":0},config)==encode({"N":1,"T":.5,"A":1},config)
    assert encode({"N":.5,"T":1/3,"A":.5},config).level=="M"
    assert encode({"N":.5,"T":2/3,"A":.5},config).level=="H"


def test_composite_weights_require_preregistration(config):
    with pytest.raises(ValueError):
        encode({"N":.5,"T":.5,"A":.5},config,"composite",[.5,.5,.5])


def test_physical_looming_geometry_and_controls(config):
    frames=[]
    for t in [.1,.5,.9]:
        s=encode({"N":.5,"T":t,"A":.5},config)
        f=looming(s,config)
        assert np.all(f.distance_m>0)
        assert np.all(np.diff(f.angle_deg)>=-1e-9)
        np.testing.assert_allclose(f.angle_deg, np.rad2deg(2*np.arctan(f.radius_m/f.distance_m)))
        assert f.angle_deg.iloc[0]==pytest.approx(2)
        assert f.angle_deg.iloc[-1]==pytest.approx(80)
        frames.append(f)
    assert len({f.time_ms.iloc[-1] for f in frames})==1
    assert frames[0].speed_m_s.iloc[0]<frames[2].speed_m_s.iloc[0]


def test_poisson_count_and_reproducibility(config):
    sim=config["simulation"]
    a=poisson_schedule(100,150,sim,17)
    np.testing.assert_array_equal(a,poisson_schedule(100,150,sim,17))
    expected=100*150*sim["stimulus_ms"]/1000
    assert abs(len(a)-expected)<6*np.sqrt(expected)
    assert a[:,0].min()*sim["dt_ms"]>=sim["baseline_ms"]
    assert a[:,0].max()*sim["dt_ms"]<sim["baseline_ms"]+sim["stimulus_ms"]
    assert poisson_schedule(2,0,sim,1).shape==(0,2)


def test_version_mixing_is_rejected():
    c=demo_circuit()
    c.neurons.loc[0,"version"]="FAFB_v630"
    with pytest.raises(ValueError,match="Mixed"):
        Circuit(c.neurons,c.edges,c.metadata)


def test_no_input_no_spikes(config):
    c=demo_circuit()
    result=simulate(c,c.cells(role="input"),np.empty((0,2),int),config["simulation"])
    assert result.spikes.empty
    metrics,_,_=response_metrics(result,c,config["simulation"])
    assert metrics["delta_rate_hz"]==0 and metrics["recruited_neurons"]==0
    assert np.isnan(metrics["latency_ms"]) and np.isnan(metrics["recovery_ms"])


def test_connection_removal_blocks_transmission(config):
    c=demo_circuit()
    inputs=c.cells(role="input")
    events=poisson_schedule(len(inputs),150,config["simulation"],42)
    intact=simulate(c,inputs,events,config["simulation"])
    blocked=simulate(c,inputs,events,config["simulation"],silenced=inputs)
    intact_m,_,_=response_metrics(intact,c,config["simulation"])
    blocked_m,_,_=response_metrics(blocked,c,config["simulation"])
    assert intact_m["delta_rate_hz"]>0
    assert blocked_m["delta_rate_hz"]==0
    assert len(blocked.spikes)>0  # disconnected input cells still receive external drive


def test_degree_control_is_disjoint_and_same_size():
    c=demo_circuit()
    targets=c.cells(cell_type="LPLC2")
    excluded=set(c.cells(role="input"))|set(c.cells(role="output"))
    selected,mismatch=degree_matched_control(c,targets,excluded,5)
    assert len(selected)==len(targets)==len(set(selected))
    assert not set(selected)&excluded
    assert mismatch>=0


def test_experiment_controls_and_article_linkage(config,tmp_path):
    config=copy.deepcopy(config)
    config["simulation"].update(baseline_ms=20,stimulus_ms=60,recovery_ms=20)
    c=demo_circuit()
    article=pd.DataFrame([dict(news_id="test_article",event_group="event",N=.9,T=.1,A=.5,data_source="synthetic")])
    metrics=run_experiments(c,config,tmp_path,seeds=2,news_scores=article)
    assert len(metrics)==22
    inp=pd.read_parquet(tmp_path/"input_events.parquet")
    for seed in metrics.seed.unique():
        ref=inp[(inp.seed==seed)&(inp.condition_id=="E3")][["step","input_slot"]].reset_index(drop=True)
        for condition in ["E4","E5","E6","E7"]:
            other=inp[(inp.seed==seed)&(inp.condition_id==condition)][["step","input_slot"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(ref,other)
        a=metrics[(metrics.seed==seed)&metrics.condition_id.eq("E9a")].delta_rate_hz.iloc[0]
        b=metrics[(metrics.seed==seed)&metrics.condition_id.eq("E9b")].delta_rate_hz.iloc[0]
        assert a==b
    article_results=pd.read_csv(tmp_path/"article_responses.csv")
    assert article_results.condition_id.eq("E1").all()
    assert article_results.shared_level_simulation.all()
    with pytest.raises(ValueError,match="already"):
        run_experiments(c,config,tmp_path,seeds=1)


@pytest.mark.parametrize("bad",[4.5,-1,True,float("nan")])
def test_llm_json_schema_catches_bad_scores(tmp_path,bad):
    p=tmp_path/"scores.jsonl"
    p.write_text(json.dumps({"news_id":"n","N":0,"T":bad,"A":0,"rationale":"explanation"}))
    with pytest.raises(ValueError):
        import_llm_scores(p,pd.DataFrame({"news_id":["n"]}),"model-v1")


def test_llm_json_import_normalizes_and_tracks_model(tmp_path):
    p=tmp_path/"scores.jsonl"
    p.write_text(json.dumps({"news_id":"n","N":4,"T":2,"A":1,"rationale":"explanation"}))
    scored=import_llm_scores(p,pd.DataFrame({"news_id":["n"]}),"model-v1")
    assert scored["T"].iloc[0]==.5 and scored.scorer.iloc[0]=="model-v1"


def test_ordinal_alpha_perfect_and_disagreement():
    labels=pd.DataFrame({"news_id":["a"]*3+["b"]*3+["c"]*3,"N":[0]*3+[2]*3+[4]*3})
    assert ordinal_alpha(labels,"N")==pytest.approx(1)
    labels.loc[1,"N"]=4
    assert ordinal_alpha(labels,"N")<1


def test_unrated_human_sheet_is_never_treated_as_labels():
    with pytest.raises(ValueError):
        evaluate_ratings(pd.DataFrame(),pd.DataFrame([dict(news_id="a",rater_id="r",N=np.nan,T=np.nan,A=np.nan)]),pd.DataFrame())


def test_heldout_metrics_and_tampered_group_manifest():
    # Explicit test fixture, not a human dataset supplied with the research results.
    scores=pd.DataFrame([dict(news_id=f"n{i}",event_group=f"e{i}",N=(i%5)/4,T=(i%5)/4,A=(i%5)/4,category="fixture") for i in range(8)])
    labels=pd.DataFrame([dict(news_id=f"n{i}",rater_id=f"r{r}",N=i%5,T=i%5,A=i%5) for i in range(8) for r in range(3)])
    splits=scores[["news_id","event_group"]].copy()
    splits["split"]=["development"]*4+["test"]*4
    result,joined,subgroups=evaluate_ratings(scores,labels,splits,bootstrap=20)
    assert result["n_test_articles"]==4
    assert result["metrics"]["T"]["spearman_rho"]==pytest.approx(1)
    assert result["metrics"]["T"]["mae_0_to_1"]==0
    assert result["ordinal_alpha"]["T"]==1
    assert len(joined)==8 and not subgroups.empty
    splits.loc[0,"event_group"]="tampered"
    with pytest.raises(ValueError,match="event groups"):
        evaluate_ratings(scores,labels,splits,bootstrap=2)


def test_measured_validation_preserves_unmatched_and_disagreement():
    c=demo_circuit()
    cells=pd.DataFrame({"root_id":c.cells(role="output"),"condition_id":"E3","delta_rate_hz":10})
    obs=pd.DataFrame({"cell_type":["GF","DNp06","missing"],"response_delta":[-1,2,3],
                      "modality":"calcium","source":"test_fixture_only","alignment_level":"cell_type"})
    comparison=compare_observations(obs,c,cells)
    assert comparison.comparison_status.tolist()==["direction_only","direction_only","unmatched_cell_type"]
    assert comparison.direction_agreement.iloc[0]==False
    assert comparison.direction_agreement.iloc[1]==True
    assert pd.isna(comparison.direction_agreement.iloc[2])
