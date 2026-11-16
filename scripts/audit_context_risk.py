"""Audit the failed v10 candidate separately from exact-version AI review updates."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_v4 import load_news, body_for
from news_flywire.risk_v8 import metrics
from news_flywire.risk_v10 import OUT, PRIVATE, BASE, development, ContextCalibration
from news_flywire.risk_review_updates import load_review_catalogs, score_with_reviews
from news_flywire.risk_input_quality import incomplete_numeric_draft

f = json.loads((OUT/'model_freeze.json').read_text())
for key, path in [('source_sha256', 'src/news_flywire/risk_v10.py'),
    ('model_sha256', OUT/'model.joblib'), ('base_sha256', BASE),
    ('protocol_sha256', OUT/'protocol.json'), ('selection_sha256', OUT/'selection.csv'),
    ('development_sha256', OUT/'development_oof.csv')]:
    assert sha256(path) == f[key], path
reference = json.loads((OUT/'reference_freeze.json').read_text())
assert sha256(OUT/'judgments.tsv') == reference['judgments_sha256']
d, raw, context = development()
selection = pd.read_csv(OUT/'selection.csv')
assert not set('real_'+str(i) for i in selection.family) & set(d.family)
target = danger_index(d[AXES].to_numpy(float)[:, 3:])
params = json.loads((OUT/'training.json').read_text())['selected']
refit = ContextCalibration(**params).fit(raw, context, target, d.family.to_numpy())
model = joblib.load(OUT/'model.joblib')
assert np.allclose(refit.classifier.coef_, model.calibration.classifier.coef_)
assert np.array_equal(refit.curve.y, model.calibration.curve.y)
news = load_news()
articles = [dict(title=r.title, body=body_for(news.iloc[r.row_index])) for r in selection.itertuples()]
x = np.load(PRIVATE/'test_features.npy')
report = json.loads((OUT/'test_metrics.json').read_text())
assert sha256(PRIVATE/'test_features.npy') == report['feature_sha256']
test = pd.read_csv(OUT/'test_results.csv')
assert test.case_id.tolist() == selection.case_id.tolist()
assert np.allclose(test.reference_score, danger_index(test[AXES].to_numpy(float)[:, 3:]))
p = model.predict_articles(x, articles)
assert np.allclose(p, test.v10_score, atol=1e-8)
for cohort, part in test.groupby('cohort'):
    for version in ['v5', 'v8', 'v9', 'v10']:
        r = metrics(part.reference_score, part[version+'_score'].to_numpy())
        for k, v in r.items():
            stored = report['cohorts'][cohort][version][k]
            assert stored is None if v is None else np.isclose(v, stored)
catalog = load_review_catalogs()
assert len(catalog) == 1074
inputs = []; blocked = []
lookup = news.set_index('news_id')
for r in catalog.values():
    body = body_for(lookup.loc[r['news_id']])
    if incomplete_numeric_draft(body):
        blocked.append(r['news_id']); continue
    inputs.append(dict(case_id=r['case_id'], title=r['title'], body=body))
applied = score_with_reviews(inputs, model_path='/unused-for-exact-reviews')
assert len(applied) == 1073 and len(blocked) == 1
assert all(r['prediction_kind'] == 'direct_Codex_session_judgment' for r in applied)
exported = pd.read_parquet(PRIVATE/'current_export/news_risk_scores.parquet')
old = pd.read_parquet('data/risk/20261110_followup/news_risk_scores.parquet')
assert len(exported) == 8128 and exported.session_reviewed.sum() == 1074
assert np.array_equal(exported.model_danger_score, old.model_danger_score)
assert np.array_equal(exported.assessment_research_eligible, old.assessment_research_eligible)
changes = pd.read_csv(OUT/'applied_review_changes.csv')
untouched = ~old.news_id.isin(changes.news_id)
assert np.array_equal(exported.loc[untouched, 'danger_score'], old.loc[untouched, 'danger_score'])
assert (np.abs(exported.danger_score-old.danger_score) > 1e-8).sum() == 19
smoke = [json.loads(l) for l in (OUT/'current_cli_outputs.jsonl').read_text().splitlines()]
pure = [json.loads(l) for l in (OUT/'pure_cli_outputs.jsonl').read_text().splitlines()]
assert len(smoke) == 44 and sum(r['prediction_kind'] == 'direct_Codex_session_judgment' for r in smoke) == 42
assert all(r['prediction_kind'] == 'pure_model_prediction' for r in smoke if r['origin'].startswith('synthetic'))
assert len(pure) == 3 and all(r['prediction_kind'] == 'pure_model_prediction' for r in pure)
scores = test.set_index('case_id').v10_score
assert all(np.isclose(r['danger_score'], scores.loc[r['case_id']]) for r in pure)
write_json(OUT/'integrity_audit.json', dict(frozen_candidate_and_references_verified=True,
    development_refit=1105, fresh_test_replay=28, max_replay_difference=float(abs(p-test.v10_score).max()),
    exact_version_catalogs=1074, exact_review_routes_verified=1073, incomplete_drafts_blocked=1,
    new_review_patches=42, corpus_risk_indices_changed=19, untouched_model='v5',
    corpus_export=8128, default_CLI_real_direct=42, default_CLI_synthetic_model=2,
    pure_candidate_CLI=3, direct_review_application_is_not_accuracy_evidence=True))
print('Verified: 28 pure predictions; 1073 exact reviews; 42 new reviews, 19 corrected indices; v5 unchanged.')
