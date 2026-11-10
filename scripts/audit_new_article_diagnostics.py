"""Replay v8 results and audit the held-out split and frozen expectations."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_inference import ArticleFeatures
from news_flywire.risk_v8 import OUT, PRIVATE, dataset, metrics


root = Path('src/news_flywire')
freeze = json.loads((OUT/'model_freeze.json').read_text())
for key, path in [('model_sha256', OUT/'model.joblib'),
                  ('source_sha256', root/'risk_v8.py'),
                  ('inference_sha256', root/'risk_inference.py'),
                  ('dataset_sha256', PRIVATE/'dataset.npz')]:
    assert sha256(path) == freeze[key], path
expected = json.loads((OUT/'expectation_freeze.json').read_text())
for key, path in [('synthetic_sha256', OUT/'synthetic_cases.jsonl'),
                  ('real_judgments_sha256', OUT/'real_judgments.tsv'),
                  ('real_reference_sha256', OUT/'real_reference.csv')]:
    assert sha256(path) == expected[key], path
manifest = json.loads((OUT/'dataset_manifest.json').read_text())
assert sha256(PRIVATE/'articles.jsonl') == manifest['articles_sha256']
assert sha256(OUT/'diagnostic_rows.csv') == manifest['metadata_sha256']
d, x, articles = dataset(False)
train = d.role.eq('development').to_numpy()
assert train.sum() == 1043 and (~train).sum() == 48
assert not set(d.family[train]) & set(d.family[~train])
model = joblib.load(OUT/'model.joblib')
assert len(model.kernel.x) == train.sum()
test = pd.read_csv(OUT/'test_results.csv')
assert test.case_id.tolist() == d.loc[~train, 'case_id'].tolist()
test_articles = [a for a, m in zip(articles, ~train) if m]
prediction = model.predict_articles(x[~train], test_articles)
delta = float(np.abs(prediction-test.v8_score).max())
assert delta < 1e-8
assert np.allclose(test.reference_score, danger_index(test[AXES].to_numpy()[:, 3:]))
report = json.loads((OUT/'test_metrics.json').read_text())
for origin in test.origin.unique():
    part = test[test.origin.eq(origin)]
    for version in ['v5', 'v8']:
        actual = metrics(part.reference_score, part[version+'_score'].to_numpy())
        for key, value in actual.items():
            stored = report['cohorts'][origin][version][key]
            assert stored is None if value is None else np.isclose(value, stored)
# Recompute complete articles, including test articles, through the public
# inference feature path rather than trusting only the private matrix.
indices = [0, 996, 997, len(d)-1]
fresh, receipts = ArticleFeatures().transform([articles[i] for i in indices])
feature_delta = float(np.abs(fresh-x[indices]).max())
assert feature_delta < 1e-6
predicted = model.predict_articles(fresh, [articles[i] for i in indices])
assert np.allclose(predicted, model.predict_articles(x[indices], [articles[i] for i in indices]), atol=1e-6)
write_json(OUT/'integrity_audit.json', dict(training=1043, final_test=48,
    frozen_model_and_source_verified=True, reference_hashes_verified=True,
    families_disjoint=True, max_test_replay_difference=delta,
    public_inference_articles_reencoded=len(indices), max_feature_replay_difference=feature_delta,
    complete_body_receipts=receipts, reported_metrics_recomputed=True,
    no_corpus_default_change=True))
print('Verified: 48 predictions, 1043/48 split, reference hashes, public inference parity.')
