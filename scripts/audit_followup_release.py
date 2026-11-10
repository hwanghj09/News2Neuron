"""Check exact-version review routing without calling it accuracy evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_flywire.provenance import write_json, sha256
from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_input_quality import incomplete_numeric_draft
from news_flywire.risk_review_updates import OUT, score_with_reviews
from news_flywire.risk_v4 import load_news, body_for


old = pd.read_parquet('data/risk/20261110_release/news_risk_scores.parquet')
new = pd.read_parquet(OUT/'news_risk_scores.parquet')
csv = pd.read_csv(OUT/'news_risk_scores.csv')
direct = pd.read_csv(OUT/'decision_audit.csv')
fresh = pd.read_csv('data/risk/20261110_v8/real_reference.csv')
assert len(new) == 8128 and len(direct) == 1032 and new.session_reviewed.sum() == 1032
assert csv.news_id.tolist() == new.news_id.tolist()
assert np.allclose(csv.danger_score, new.danger_score)
assert old.news_id.tolist() == new.news_id.tolist()
unchanged = ~new.news_id.isin(fresh.news_id)
assert np.allclose(old.loc[unchanged, 'danger_score'], new.loc[unchanged, 'danger_score'])
assert np.allclose(old.model_danger_score, new.model_danger_score)
assert np.array_equal(old.assessment_research_eligible, new.assessment_research_eligible)
applied = new.set_index('news_id').loc[direct.news_id]
assert np.allclose(applied[AXES].to_numpy()*4, direct[AXES].to_numpy())
assert np.allclose(applied.danger_score, danger_index(direct[AXES].to_numpy()[:, 3:]))
news = load_news().set_index('news_id')
inputs = []
quarantined = []
for r in direct.itertuples():
    row = news.loc[r.news_id]
    body = body_for(row)
    if incomplete_numeric_draft(body):
        quarantined.append(r.news_id)
        continue
    inputs.append(dict(case_id=r.case_id, title=row.title, body=body))
results = score_with_reviews(inputs, model_path='/not-needed-for-exact-reviews')
assert len(results) == 1031 and len(quarantined) == 1
assert all(r['prediction_kind'] == 'direct_Codex_session_judgment' for r in results)
lookup = direct.set_index('case_id')
assert all(np.isclose(r['danger_score'], danger_index(lookup.loc[r['case_id'], ['S', 'L', 'I', 'X']].to_numpy(float))) for r in results)
smoke = [json.loads(l) for l in Path('data/risk/20261110_v8/inference_smoke_outputs.jsonl').read_text().splitlines()]
assert len(smoke) == 96
assert sum(r['prediction_kind'] == 'direct_Codex_session_judgment' for r in smoke) == 32
assert sum(r['prediction_kind'] == 'pure_model_prediction' for r in smoke) == 64
assert all(r['prediction_kind'] == 'pure_model_prediction' for r in smoke if r['origin'] == 'synthetic_authored_by_current_Codex')
write_json(OUT/'integrity_audit.json', dict(articles=8128, real_direct_reviews=1032,
    new_real_reviews=32, previous_scores_unchanged_except_new_reviews=True,
    model_scores_unchanged=True, csv_parquet_equal=True, research_eligibility_unchanged=True,
    exact_review_routes_verified=1031, incomplete_drafts_quarantined=quarantined,
    new_article_CLI_smoke=dict(inputs=96, real_direct=32, synthetic_pure_model=64),
    catalog_sha256=sha256(OUT/'review_catalog.jsonl'),
    verification_kind='application integrity, not held-out accuracy'))
print('Verified: 1031 exact reviews, 1 draft quarantined, 96 CLI inputs, unchanged v5 predictions.')
