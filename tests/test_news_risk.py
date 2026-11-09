"""Scientific/provenance invariants for iterative AI risk assessment."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_flywire.risk_assessment import danger_index, read_judgments, risk_band, scaled_features


def test_severe_past_loss_is_not_ongoing_danger():
    assert danger_index([4, 0, 0, 4]) == 0
    assert danger_index([0, 4, 4, 4]) == 0
    assert danger_index([4, 4, 4, 4]) == 100
    assert danger_index([4, 4, 4, 1]) >= 80  # one endangered person still matters


def test_index_monotone_and_stable_without_imminence():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 4, (100, 4))
    original = danger_index(x)
    for axis in range(4):
        raised = x.copy()
        raised[:, axis] = np.minimum(4, raised[:, axis] + .5)
        assert (danger_index(raised) >= original - 1e-10).all()
    assert danger_index([4, 4, 1, 4]) > 50  # long-term severe harm not erased
    assert ((original >= 0) & (original <= 100)).all()


@pytest.mark.parametrize("x", [[4, -1, 4, 4], [4, 4.1, 4, 4], [4, np.nan, 4, 4], [1, 2, 3]])
def test_invalid_ordinal_inputs_fail(x):
    with pytest.raises(ValueError):
        danger_index(x)


def test_bands_have_explicit_non_quantile_boundaries():
    assert risk_band(np.array([0, 9.9, 10, 30, 60, 85, 100])).tolist() == ["minimal", "minimal", "low", "moderate", "high", "critical", "critical"]


def fixture_labels(tmp_path):
    news = pd.DataFrame([{"news_id": "n1", "body_sha256": "aaa", "split": "train"},
                         {"news_id": "n2", "body_sha256": "bbb", "split": "test"}])
    selected = pd.DataFrame([{"case_id": "c1", "news_id": "n1", "body_sha256": "aaa", "event_group": "g1", "role": "development", "round": 1},
                             {"case_id": "c2", "news_id": "n2", "body_sha256": "bbb", "event_group": "g2", "role": "audit", "round": 1}])
    decision = pd.DataFrame([{"case_id": "c1", "N": 4, "T": 0, "A": 3, "S": 4, "L": 0, "I": 0, "X": 1, "state": "resolved", "rationale": "이미 검거되어 기사에 추가 위해가 제시되지 않는다."},
                             {"case_id": "c2", "N": 4, "T": 4, "A": 4, "S": 4, "L": 4, "I": 4, "X": 2, "state": "imminent", "rationale": "화재가 계속되어 주민이 당장 대피해야 한다."}])
    selected.to_csv(tmp_path / "selected_cases.csv", index=False)
    decision.to_csv(tmp_path / "session_judgments.tsv", sep="\t", index=False)
    return news, selected, decision


def test_ai_decisions_are_hash_bound_and_temporally_separated(tmp_path):
    news, _, _ = fixture_labels(tmp_path)
    assert len(read_judgments(tmp_path, news)) == 2
    news.loc[0, "body_sha256"] = "changed"
    with pytest.raises(ValueError, match="body changed"):
        read_judgments(tmp_path, news)
    news.loc[0, "body_sha256"] = "aaa"
    news.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="Only temporal training"):
        read_judgments(tmp_path, news)


def test_claimed_resolution_cannot_hide_positive_remaining_risk(tmp_path):
    news, _, d = fixture_labels(tmp_path)
    d.loc[0, "L"] = 3
    d.to_csv(tmp_path / "session_judgments.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="No residual risk"):
        read_judgments(tmp_path, news)


def test_audit_event_groups_cannot_leak_into_development(tmp_path):
    news, selected, _ = fixture_labels(tmp_path)
    selected.loc[1, "event_group"] = "g1"
    selected.to_csv(tmp_path / "selected_cases.csv", index=False)
    with pytest.raises(ValueError, match="group overlap"):
        read_judgments(tmp_path, news)


def test_feature_scaling_does_not_mutate_cached_inputs():
    x = np.ones((2, 1168), dtype=np.float32)
    result = scaled_features(x)
    assert (x == 1).all()
    assert np.linalg.norm(result[:, 1152:]) < np.linalg.norm(result[:, :384])


def test_native_promotion_removal_keeps_both_article_sides():
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/correct_seoul_risk_inputs.py'
    spec=importlib.util.spec_from_file_location('input_repair',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload='''<div id="articleContent"><div class="viewContent"><p>교육청의 지원 계획이 확정됐다.</p><div class="sportLightPromotion"><div><p>전혀 다른 시의회 홍보 기사다.</p></div></div><p>교육감은 조직 개편 부결에 사과했다.</p></div></div>'''.encode()
    before,after,count=module.remove_promotion(payload)
    assert count==1 and '시의회' in before and '시의회' not in after
    assert '지원 계획이 확정됐다' in after and '부결에 사과했다' in after


def test_article_without_promotion_stays_identical():
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/correct_seoul_risk_inputs.py'
    spec=importlib.util.spec_from_file_location('input_repair',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before,after,count=module.remove_promotion('<div id="articleContent"><div class="viewContent"><p>현재 화재가 진행 중이다.</p><p>주민은 대피해야 한다.</p></div></div>'.encode())
    assert count==0 and before==after
