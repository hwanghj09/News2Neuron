import numpy as np
import pytest

from news_flywire.risk_input_quality import incomplete_numeric_draft
from news_flywire.risk_v7 import KernelRisk, packet
from news_flywire.risk_v7_evaluation import paired_second_attempt
from news_flywire.risk_release import select_model


def test_unfilled_draft_never_invents_numbers():
    assert incomplete_numeric_draft('지난달 수출은 __% 감소했고 매출은 ___억 원이다.')
    assert not incomplete_numeric_draft('지난달 수출은 12% 감소했고 매출은 300억 원이다.')
    assert not incomplete_numeric_draft('문서에 __init__이라는 프로그램 이름이 나온다.')


def test_second_attempt_uses_wider_registered_interval():
    y=np.arange(20)*3
    r=paired_second_attempt(y,np.zeros(20),np.full(20,15),['a']*10+['b']*10,resamples=1000)
    assert r['primary_interval_level']==.975
    a,b=r['MAE_reduction_97_5_percentile'];c,d=r['MAE_reduction_95_percentile']
    assert a<=c<=d<=b


def test_new_audit_cannot_be_opened_before_freeze(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Freeze model'):packet()


@pytest.mark.parametrize('v6,v7,expected',[(False,False,'v5'),(True,False,'v6'),(False,True,'v7'),(True,True,'v7')])
def test_failed_candidates_never_replace_the_published_model(v6,v7,expected):
    evidence={'v6':{'preregistered_evidence_pass':v6},'v7':{'preregistered_evidence_pass':v7}}
    assert select_model(evidence)==expected


@pytest.mark.parametrize('positive',[False,True])
def test_new_kernel_handles_single_reference_class(positive):
    x=np.random.default_rng(7).normal(size=(8,6));y=np.full((8,7),4. if positive else 0.)
    texts=np.array(['shared article context '+str(i) for i in range(8)])
    m=KernelRisk(mode='gated').fit(x,texts,y,np.arange(8),np.ones(8,bool))
    p=m.predict(x,texts)
    assert np.isfinite(p).all() and ((p>=0)&(p<=100)).all()
    if not positive:assert (p==0).all()
