import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from news_flywire.article_stimulus import profile, schedules, event_hash
from news_flywire.article_bridge import encode_table


@pytest.fixture
def cfg():
    return yaml.safe_load(Path('configs/article_bridge_v1.yaml').read_text())


def test_geometry_and_rate_have_the_same_presentation_window(cfg):
    for score in [.01, .3, .7, 1.]:
        rates, frames = profile(score, cfg)
        assert frames.time_ms.iloc[-1] == cfg['simulation']['stimulus_ms']
        assert len(frames) == len(rates)+1
        assert frames.angle_deg.iloc[0] == pytest.approx(20)
        assert frames.angle_deg.iloc[-1] == pytest.approx(80)
        assert (np.diff(frames.angle_deg) >= -1e-10).all()
        assert (frames.distance_m > 0).all()
        np.testing.assert_allclose(frames.angle_deg,
            np.rad2deg(2*np.arctan(frames.radius_m/frames.distance_m)))
        assert rates.mean() == pytest.approx(score*150)
        assert rates.max() <= cfg['proposal_rate_hz']


def test_zero_score_means_no_drive_and_no_visual_expansion(cfg):
    rates, frames = profile(0., cfg)
    assert not rates.any()
    assert np.allclose(frames.angle_deg, 20)
    result = schedules('article', 3, {'R':0., 'N':0.}, 5, cfg)
    assert all(e.shape == (0, 2) for e in result.values())


def test_article_schedule_is_reproducible_and_not_a_level_lookup(cfg):
    a = schedules('article_A', 42, {'R':.5, 'N':.5}, 20, cfg)
    b = schedules('article_A', 42, {'R':.5, 'N':.5}, 20, cfg)
    c = schedules('article_B', 42, {'R':.5}, 20, cfg)
    assert np.array_equal(a['R'], a['N'])
    assert np.array_equal(a['R'], b['R'])
    assert not np.array_equal(a['R'], c['R'])
    assert event_hash(a['R']) != event_hash(c['R'])
    times = a['R'][:, 0]*cfg['simulation']['dt_ms']
    assert times.min() >= 100 and times.max() < 500


def test_time_varying_poisson_preserves_expected_count_and_temporal_shape(cfg):
    events = schedules('poisson_check', 21, {'R':.8}, 1000, cfg)['R']
    expected = 1000*120*.4
    assert abs(len(events)-expected) < 6*np.sqrt(expected)
    # The physical expansion is aligned to the end, so the last half has
    # more expected events than the uniform first half, not fewer.
    rate, _ = profile(.8, cfg)
    mid = len(rate)//2
    observed_late = (events[:, 0] >= 200+mid).sum()
    expected_late = rate[mid:].sum()*.5/1000*1000
    assert abs(observed_late-expected_late) < 6*np.sqrt(expected_late)


@pytest.mark.parametrize('score', [-1, 1.01, np.nan, np.inf])
def test_invalid_score_rejected(cfg, score):
    with pytest.raises(ValueError):
        profile(score, cfg)


def test_physical_approach_and_poisson_bounds_cannot_be_silently_stretched(cfg):
    short = copy.deepcopy(cfg); short['simulation']['stimulus_ms'] = 100
    with pytest.raises(ValueError, match='does not fit'):
        profile(.2, short)
    low = copy.deepcopy(cfg); low['proposal_rate_hz'] = 10
    with pytest.raises(ValueError, match='does not bound'):
        profile(1., low)


def test_quality_exclusion_cannot_receive_a_drive(cfg):
    data = pd.DataFrame([dict(news_id='real1', source='source', url='https://example.org/a',
        title='title', published_date='2026-11-09', assessment_body_sha256='digest',
        assessment_research_eligible=False, input_quality_flag='draft',
        score_source='direct_Codex_session_judgment', session_reviewed=True,
        decision_state='ongoing', N=.8, T=.9, A=.5, danger_score=90)])
    result = encode_table(data, cfg)
    assert not result.simulation_allowed.iloc[0]
    assert result[['R_mean_rate_hz','T_mean_rate_hz','N_mean_rate_hz']].isna().all().all()
