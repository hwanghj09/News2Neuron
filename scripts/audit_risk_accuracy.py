"""Reproduce frozen predictions and validate direct-vs-generalized score provenance."""
from pathlib import Path
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from news_flywire.risk_assessment import AXES,danger_index
from news_flywire.risk_v4 import load_news,body_for
from news_flywire.provenance import sha256,write_json

out=Path('data/risk/20261109_v5');news=load_news();s=pd.read_parquet(out/'news_risk_scores.parquet');d=pd.read_csv(out/'decision_audit.csv')
assert len(s)==8128 and s.news_id.is_unique and s.news_id.tolist()==news.news_id.tolist()
assert s.assessment_body_sha256.tolist()==news.assessment_body_sha256.tolist()
assert np.isfinite(s[AXES+['danger_score','component_index','model_danger_score']].to_numpy()).all()
assert s[AXES].ge(0).all().all() and s[AXES].le(1).all().all() and s.danger_score.between(0,100).all()
assert int(s.session_reviewed.sum())==358
assert set(s.loc[s.session_reviewed,'news_id'])==set(d.news_id)
assert s.loc[~s.session_reviewed,'score_source'].eq('AI_supervised_v5_model_prediction').all()
direct=s[s.session_reviewed];assert np.allclose(danger_index(direct[['S','L','I','X']].to_numpy()*4),direct.danger_score)
freeze=json.loads((out/'pre_audit_freeze.json').read_text())
for path,digest in freeze['files'].items():assert sha256(path)==digest
x=np.load('data/raw/risk/20261109_v5/features.npy');model=joblib.load(out/'model.joblib');axes,pred=model.predict(x)
raw=pd.read_parquet(out/'predictions.parquet');assert np.allclose(pred,raw.danger_score) and np.allclose(pred,s.model_danger_score)
for j,a in enumerate(AXES):assert np.allclose(axes[:,j]/4,raw[a])
selection=pd.read_csv(out/'selection.csv');dev=pd.read_csv(out/'development.csv')
assert len(dev)==321 and not set(selection.news_id)&set(dev.news_id)
for r in news.itertuples():assert hashlib.sha256(body_for(r).encode()).hexdigest()==r.assessment_body_sha256
csv=pd.read_csv(out/'news_risk_scores.csv');assert np.allclose(csv.danger_score,s.danger_score)
assert max(len(r.title.split())+len(r.summary.split()) for r in s.itertuples())<=25
metrics=json.loads((out/'fresh_audit_metrics.json').read_text());err=pd.read_csv(out/'audit_errors.csv')
for record in metrics['results']:
    mask=err.model.eq(record['model'])
    cohort=record['cohort']
    if cohort.startswith('novel_'):mask &=err.included_in_primary
    if cohort not in ['all','novel_all']:mask &=err.cohort.eq(cohort.replace('novel_',''))
    assert mask.sum()==record['articles'] and np.isclose(err.loc[mask,'absolute_error'].mean(),record['MAE'])
write_json(out/'verification.json',{'status':'passed','articles':len(s),'body_hashes_checked':len(news),'direct_AI_judgments':358,'development_articles':321,
    'new_audit_articles':len(selection),'primary_novel_articles':metrics['primary_articles'],'all_model_predictions_reproduced':True,
    'pre_audit_hashes_unchanged':True,'fresh_labels_used_for_model_selection':False,'CSV_Parquet_equal':True,
    'limitation':'Numerical/provenance checks and same-AI agreement, not independent semantic truth.'})
print('Verified all 8128 scoring bodies and predictions; 358 actual AI decisions; fresh test kept out of training')
