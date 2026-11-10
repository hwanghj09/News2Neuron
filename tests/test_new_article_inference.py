import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_inference import ArticleFeatures, validate_article
from news_flywire.risk_v8 import context_features
from news_flywire.risk_review_updates import input_key, load_catalog, score_with_reviews


@pytest.mark.parametrize('title,body', [('', '본문'), ('제목', ''), ('제목', '   '),
    (None, '본문'), ('제목', '<article>기사 본문</article>'),
    ('제목', '지난달 매출 __% 감소, 영업이익 ___억 원이다.')])
def test_missing_or_unprocessed_input_is_rejected(title, body):
    with pytest.raises(ValueError):
        validate_article(title, body)


def test_synthetic_cases_are_marked_and_scenario_families_are_held_out():
    path = Path('data/risk/20261110_v8/synthetic_cases.jsonl')
    cases = [json.loads(l) for l in path.read_text().splitlines()]
    dev = {a['family'] for a in cases if a['role'] == 'development'}
    test = {a['family'] for a in cases if a['role'] == 'test'}
    assert len(cases) == 64 and len(dev) == len(test) == 16 and not dev & test
    assert all(a['origin'] == 'synthetic_authored_by_current_Codex' for a in cases)
    y = np.array([[a[k] for k in AXES] for a in cases])
    assert np.allclose(danger_index(y[:, 3:]), [a['danger_score'] for a in cases])


def test_reference_metadata_cannot_change_evidence_inputs():
    text = dict(title='전시 개막', body='과학관에서 새 전시를 연다. 관람객은 정상적으로 입장했다.')
    adversarial = dict(text, danger_score=100, S=4, L=4, I=4, X=4, state='imminent')
    assert np.array_equal(context_features([text]), context_features([adversarial]))


@pytest.fixture
def reviewed_article(tmp_path):
    article = dict(title='환자 재활 계속', body='환자는 수술 뒤 재활 치료를 계속하고 있다. 아직 업무에 복귀하지 못했다.')
    digest = hashlib.sha256(article['body'].encode()).hexdigest()
    axes = dict(zip(AXES, [2, 2, 1, 2, 4, 3, 1]))
    record = dict(title=article['title'], assessment_body_sha256=digest,
        input_sha256=input_key(article['title'], digest), case_id='review_1', **axes,
        state='ongoing', rationale='현재 재활과 복귀 제한이 명시된다.',
        danger_score=float(danger_index([2, 4, 3, 1])))
    path = tmp_path/'review.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False)+'\n')
    return article, record, path


def test_exact_review_is_reused_without_requiring_an_encoder(reviewed_article):
    article, reference, path = reviewed_article
    result = score_with_reviews([dict(article, danger_score=0)], path, '/missing/model')[0]
    assert result['prediction_kind'] == 'direct_Codex_session_judgment'
    assert result['danger_score'] == reference['danger_score']
    assert result['reference_case_id'] == 'review_1'


@pytest.mark.parametrize('change', ['title', 'body'])
def test_review_never_transfers_to_a_changed_article_or_spoofed_hash(reviewed_article, monkeypatch, change):
    article, reference, path = reviewed_article
    changed = dict(article, assessment_body_sha256=reference['assessment_body_sha256'],
                   case_id='review_1', danger_score=reference['danger_score'])
    changed[change] += ' 이후 완치해 제한 없이 복귀했다.'
    def pure(inputs, *args):
        return [dict(case_id='0', danger_score=0., prediction_kind='pure_model_prediction') for a in inputs]
    monkeypatch.setattr('news_flywire.risk_review_updates.score_articles', pure)
    result = score_with_reviews([changed], path)[0]
    assert result['prediction_kind'] == 'pure_model_prediction'
    assert result['danger_score'] == 0.


def test_corrupted_review_cannot_bypass_rubric_formula(reviewed_article):
    article, reference, path = reviewed_article
    reference['danger_score'] = 100
    path.write_text(json.dumps(reference)+'\n')
    with pytest.raises(ValueError, match='differs'):
        load_catalog(path)


@pytest.mark.skipif(not Path('external/multilingual-e5-small/tokenizer.json').exists(),
                    reason='Pinned optional encoder is not downloaded')
def test_new_article_features_are_batch_invariant_and_cover_late_body():
    extractor = ArticleFeatures()
    first = dict(title='지역 행사 안내', body='지역 행사가 정상적으로 열렸다. 시민들이 전시를 관람했다.')
    second = dict(title='구조 소식', body='구조 작업을 하고 있다. 작업자의 생사는 아직 확인되지 않았다.')
    a, _ = extractor.transform([first])
    b, _ = extractor.transform([first, second])
    assert np.array_equal(a[0], b[0])
    long = dict(title=first['title'], body=(first['body']+'\n')*50)
    changed = dict(long, body=long['body']+'\n그러나 실제 화재로 두 사람이 중태에 빠져 의식을 찾지 못하고 있다.')
    z, receipts = extractor.transform([long, changed])
    assert all(r['windows'] > 1 and r['all_body_tokens_processed'] for r in receipts)
    assert np.array_equal(z[0, :384], z[1, :384])
    assert not np.allclose(z[0, 384:1152], z[1, 384:1152])
    assert receipts[0]['body_sha256'] != receipts[1]['body_sha256']
