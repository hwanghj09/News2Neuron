import numpy as np
import pytest

from news_flywire.risk_v8 import context_features
from news_flywire.risk_v10 import ContextCalibration


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_equal_raw_scores_can_preserve_current_harm_and_suppress_closed_harm(mode):
    ongoing = dict(title='치료 상황', body='환자가 현재 입원 치료를 받고 있다. 재활이 필요하고 아직 복귀하지 못했다.')
    closed = dict(title='치료 종료', body='치료가 모두 끝났다. 완전히 회복해 제한 없이 일상으로 복귀했다. 추가 피해는 없다.')
    articles = [ongoing, closed]*12
    context = context_features(articles)
    raw = np.full(len(articles), 10.)
    y = np.array([40., 0.]*12)
    model = ContextCalibration(C=1., mode=mode).fit(raw, context, y, np.arange(len(y)))
    p = model.predict(raw[:2], context[:2])
    assert p[0] >= 30 and p[1] <= 10
    assert ((p >= 0)&(p <= 100)).all()


@pytest.mark.parametrize('positive', [False, True])
def test_single_presence_class_is_supported(positive):
    model = ContextCalibration().fit(np.arange(8.), np.ones((8, 4)),
        np.full(8, 40. if positive else 0.), np.arange(8))
    p = model.predict(np.arange(8.), np.ones((8, 4)))
    assert np.allclose(p, 40. if positive else 0.)


@pytest.mark.parametrize('raw,context', [([np.nan], [[1]]), ([-1], [[1]]),
    ([101], [[1]]), ([1, 2], [[1]]), ([1], [[np.inf]])])
def test_invalid_context_or_score_is_rejected(raw, context):
    with pytest.raises(ValueError):
        ContextCalibration.inputs(raw, context)


def test_reference_axes_and_article_metadata_do_not_enter_context():
    a = dict(title='치료 상황', body='환자가 현재 입원 치료를 받고 있다.')
    b = dict(a, state='resolved', danger_score=0, case_id='pretend_safe', L=0)
    assert np.array_equal(context_features([a]), context_features([b]))
