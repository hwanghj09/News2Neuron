import numpy as np
import pytest

from news_flywire.risk_v6 import HybridRisk, packet
from news_flywire.risk_v6_evaluation import paired_evidence, score_metrics


def test_identical_predictions_never_establish_superiority():
    result=paired_evidence([0,20,80],[0,30,60],[0,30,60],['a','a','b'],resamples=100)
    assert result['MAE_reduction_95_percentile']==[0.,0.]
    assert result['within10_change']==0
    assert result['mcnemar_exact_one_sided_p']==1


def test_paired_stratification_preserves_publisher_mix():
    # Each publisher has one case, so within-publisher draws cannot change
    # the publisher mix; an unstratified bootstrap would give variable means.
    result=paired_evidence([0,100],[20,70],[0,100],['a','b'],resamples=100)
    assert result['MAE_reduction_95_percentile']==[25.,25.]
    assert result['within10_change_95_percentile']==[1.,1.]


def test_worse_model_produces_negative_evidence():
    result=paired_evidence([0,30,90],[0,30,90],[30,60,60],['a','a','b'],resamples=100)
    assert result['MAE_reduction_95_percentile'][1]<0
    assert result['within10_change']==-1
    assert result['mcnemar_exact_one_sided_p']==1


def test_zero_majority_cannot_hide_high_risk_collapse():
    result=score_metrics([0]*9+[80],[0]*10)
    assert result['within10_AI_agreement']==.9
    assert result['high_recall']==0
    assert result['nonzero_recall']==0
    assert result['error_gt30']==1


def test_evidence_rejects_missing_or_nonfinite_pairs():
    with pytest.raises(ValueError):paired_evidence([1],[1,2],[1],['a'])
    with pytest.raises(ValueError):paired_evidence([np.nan],[1],[1],['a'])


def test_full_body_audit_packets_stay_closed_before_freeze(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Freeze model'):
        packet('audit')


@pytest.mark.parametrize('positive',[False,True])
def test_hybrid_single_class_reference_handles_small_input(positive):
    x=np.random.default_rng(3).normal(size=(6,8))
    texts=np.array(['shared article context number '+str(i) for i in range(6)])
    y=np.full((6,7),4. if positive else 0.)
    model=HybridRisk(mode='conditional').fit(x,texts,y,np.arange(6))
    axes,risk=model.predict(x,texts)
    assert np.isfinite(axes).all() and np.isfinite(risk).all()
    assert ((risk>=0)&(risk<=100)).all()
    if not positive:assert (risk==0).all()
