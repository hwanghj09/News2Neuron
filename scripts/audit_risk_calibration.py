"""Replay fixed v9 calibration and its fresh AI-reference audit."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_v4 import load_news, body_for
from news_flywire.risk_v8 import metrics
from news_flywire.risk_v9 import OUT, BASE, MedianCalibration

f = json.loads((OUT/'model_freeze.json').read_text())
for key, path in [('source_sha256', 'src/news_flywire/risk_v9.py'),
    ('base_sha256', BASE), ('model_sha256', OUT/'model.joblib'),
    ('selection_sha256', OUT/'selection.csv'), ('protocol_sha256', OUT/'protocol.json'),
    ('development_sha256', 'data/risk/20261110_v8/development_oof.csv')]:
    assert sha256(path) == f[key], path
reference = json.loads((OUT/'reference_freeze.json').read_text())
assert sha256(OUT/'judgments.tsv') == reference['judgments_sha256']
train = pd.read_csv('data/risk/20261110_v8/development_oof.csv')
selection = pd.read_csv(OUT/'selection.csv')
assert not set('real_'+str(i) for i in selection.family) & set(train.family)
assert selection.family.is_unique
model = joblib.load(OUT/'model.joblib')
weights = 1/train.groupby('family').family.transform('size').to_numpy()
refit = MedianCalibration().fit(train.v8_oof_score, train.reference_score, weights)
assert np.array_equal(model.calibration.x, refit.x)
assert np.array_equal(model.calibration.y, refit.y)
test = pd.read_csv(OUT/'test_results.csv')
assert test.case_id.tolist() == selection.case_id.tolist()
assert np.allclose(test.reference_score, danger_index(test[AXES].to_numpy()[:, 3:]))
news = load_news()
articles = [dict(title=r.title, body=body_for(news.iloc[r.row_index])) for r in selection.itertuples()]
x = np.load('data/raw/risk/20261116_v9/test_features.npy')
stored = json.loads((OUT/'test_metrics.json').read_text())
assert sha256('data/raw/risk/20261116_v9/test_features.npy') == stored['feature_sha256']
raw = joblib.load(BASE).predict_articles(x, articles)
p = model.predict_articles(x, articles)
assert np.allclose(raw, test.v8_score, atol=1e-8)
assert np.allclose(p, test.v9_score, atol=1e-8)
for version in ['v5', 'v8', 'v9']:
    r = metrics(test.reference_score, test[version+'_score'].to_numpy())
    for k, v in r.items():
        assert v is None if stored['results'][version][k] is None else np.isclose(v, stored['results'][version][k])
# Descriptive, source-stratified paired intervals. Same single AI reference;
# these intervals do not quantify uncertainty in the rubric judgments.
rng = np.random.default_rng(20261116)
draws = np.concatenate([rng.choice(indices, (10000, len(indices)), replace=True)
    for indices in test.groupby('source').indices.values()], axis=1)
intervals = {}
for version in ['v5', 'v8']:
    delta = test[version+'_error'].to_numpy()-test.v9_error.to_numpy()
    intervals[version] = dict(MAE_reduction=float(delta.mean()),
        conditional_95_percentile=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
write_json(OUT/'paired_intervals.json', dict(comparisons=intervals, resamples=10000,
    seed=20261116, scheme='resample within each of seven sources; 2 articles/source',
    limitation='Small descriptive sample and one AI rubric; not uncertainty in AI judgments or population accuracy proof.'))
write_json(OUT/'integrity_audit.json', dict(frozen_hashes_verified=True,
    original_v8_unchanged=True, calibration_reproduced_from_1043_OOF_predictions=True,
    fresh_families_disjoint=True, final_test=14,
    max_score_replay_difference=float(np.abs(p-test.v9_score).max()),
    metric_replay=True, corpus_default='v5 unchanged'))
print('Verified frozen model, calibration fit, 14 fresh predictions, references and metrics.')
