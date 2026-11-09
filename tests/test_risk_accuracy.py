import numpy as np
import pytest
from news_flywire.risk_assessment import danger_index
from news_flywire.risk_v4 import IndexModel,named_families,packet
from news_flywire.risk_v5 import ZeroAware


def test_nonlinear_index_is_not_index_of_marginal_means():
    states=np.array([[0,0,0,0],[4,4,4,4]])
    assert danger_index(states).mean()==50
    assert danger_index(states.mean(axis=0))<20


@pytest.mark.parametrize('positive',[False,True])
def test_zero_gate_handles_single_reference_class(positive):
    x=np.random.default_rng(4).normal(size=(12,8));y=np.full((12,7),4. if positive else 0.)
    model=ZeroAware(gate=True).fit(x,y,np.arange(len(x)))
    axes,risk=model.predict(x)
    assert np.isfinite(axes).all() and np.isfinite(risk).all()
    assert (risk>=0).all() and (risk<=100).all()
    if not positive:assert (risk==0).all()


def test_hurdle_candidate_without_positive_training_labels_does_not_crash():
    x=np.ones((5,4));model=IndexModel('hurdle').fit(x,np.zeros((5,7)))
    assert (model.predict(x)[1]==0).all()


def test_campaign_purge_is_conservative_across_changed_headlines():
    assert named_families('네팔 홍수 실종자 수색')&named_families('네팔 대홍수 이재민 지원')
    assert named_families('리야드 공항 공격')&named_families('후티 정부군 무력충돌')
    assert not named_families('이어폰 출시')


def test_fresh_audit_packet_cannot_open_before_model_freeze(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Freeze the final model'):
        packet('fresh_random',0,tmp_path/'experiment')
