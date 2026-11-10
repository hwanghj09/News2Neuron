"""Replay frozen models and check source, reference, split and release integrity."""
from pathlib import Path
import importlib.metadata
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, danger_index
from news_flywire.risk_v4 import load_news, body_for
from news_flywire import risk_v6, risk_v7
from news_flywire.risk_v6_evaluation import score_metrics

root=Path('data/risk');news=load_news();bodies=[body_for(r) for r in news.itertuples()]
assert len(news)==8128 and news.news_id.is_unique
checks={};portable={}
for v,module,classes in [('v6',risk_v6,['HybridRisk']),('v7',risk_v7,['Blend','ContextTree','KernelRisk'])]:
    out=root/f'20261109_{v}';freeze=json.loads((out/'pre_audit_freeze.json').read_text())
    for p,h in freeze['files'].items():assert sha256(p)==h,p
    assert sha256(module.__file__)==freeze.get('model_source_sha256',freeze.get('source_sha256'))
    # Original python -m training pickled classes under __main__. Resolve
    # only these known local classes, then export their identical fitted
    # objects under importable modules. Original frozen bytes stay unchanged.
    main=sys.modules['__main__']
    for name in classes:setattr(main,name,getattr(module,name))
    try:model=joblib.load(out/'model.joblib')
    finally:
        for name in classes:delattr(main,name)
    portable_path=out/'model_portable.joblib'
    joblib.dump(model,portable_path,compress=3)
    model=joblib.load(portable_path)
    portable[v]=dict(original_sha256=sha256(out/'model.joblib'),portable_sha256=sha256(portable_path),
        fitted_parameters_changed=False,retrained=False,reason='importable class paths for artifacts trained with python -m')
    x=np.load(f'data/raw/risk/20261109_{v}/features.npy') if v=='v7' else np.load('data/raw/risk/20261109_v5/features.npy')
    if v=='v7':assert sha256('data/raw/risk/20261109_v7/features.npy')==freeze['features_sha256']
    texts=np.array([(r.title+'\n')*(2 if v=='v7' else 3)+b for r,b in zip(news.itertuples(),bodies)])
    frozen=pd.read_parquet(out/'predictions.parquet');replayed=[]
    for start in range(0,len(news),128):
        if v=='v6':axes,risk=model.predict(x[start:start+128],texts[start:start+128])
        else:
            risk=model['risk'].predict(x[start:start+128],texts[start:start+128])
            axes=np.clip(model['heads'].predict(x[start:start+128]),0,4)
        replayed.extend(risk)
        assert np.allclose(axes/4,frozen[AXES].iloc[start:start+128],atol=1e-10)
    difference=float(np.max(np.abs(np.array(replayed)-frozen.danger_score.to_numpy())))
    assert difference<1e-8,(v,difference)
    selection=pd.read_csv(out/'audit_selection.csv');dev=pd.read_csv(out/'development.csv')
    families=pd.read_csv(out/'corpus_families.csv').set_index('news_id')
    assert not set(families.loc[dev.news_id].family)&set(selection.family)
    seen=[]
    for p in sorted(out.glob('packet_audit_*_receipt.json')):
        receipt=json.loads(p.read_text());private=Path(f'data/raw/risk/20261109_{v}')/(p.stem.removesuffix('_receipt')+'.json')
        assert sha256(private)==receipt['sha256']
        packet=json.loads(private.read_text());seen.extend(r['case_id'] for r in packet)
        for r in packet:
            assert hashlib.sha256(r['body'].encode()).hexdigest()==r['assessment_body_sha256']
            assert r['body']==bodies[news.index[news.news_id.eq(selection.set_index('case_id').loc[r['case_id']].news_id)][0]]
    assert len(seen)==len(set(seen))==len(selection)
    reference=json.loads((out/'reference_freeze.json').read_text())
    # v6 and v7 seal schemas differ; both committed before error tables.
    if v=='v6':
        for p,h in {**reference['files'],**reference['full_body_packets']}.items():assert sha256(p)==h,p
    else:
        assert sha256(out/'audit_judgments.tsv')==reference['labels_sha256']
        for p,h in reference['packet_receipts'].items():assert sha256(p)==h,p
    errors=pd.read_csv(out/'audit_errors.csv')
    metrics=json.loads((out/'fresh_audit_metrics.json').read_text())
    for item in metrics['results']:
        part=errors[errors.model.eq(item['model'])]
        if item['cohort']!='all':part=part[part.cohort.eq(item['cohort'])]
        recomputed=score_metrics(part.reference,part.prediction)
        for k,value in recomputed.items():
            assert value is None if item[k] is None else np.isclose(value,item[k]),(v,k)
    checks[v]=dict(replayed_articles=len(news),max_score_difference=difference,complete_packet_cases=len(seen),
        stored_body_hashes_verified=8128,dev_audit_families_disjoint=True,frozen_hashes_unchanged=True,reference_hashes_unchanged=True)
    print(v,checks[v],flush=True)

release=root/'20261110_release';csv=pd.read_csv(release/'news_risk_scores.csv');parquet=pd.read_parquet(release/'news_risk_scores.parquet')
assert csv.news_id.tolist()==parquet.news_id.tolist()
assert np.allclose(csv.danger_score,parquet.danger_score)
assert len(csv)==8128 and int(csv.session_reviewed.sum())==1000
direct=pd.read_csv(release/'decision_audit.csv');assert len(direct)==1000 and direct.news_id.is_unique
actual=parquet.set_index('news_id').loc[direct.news_id]
assert np.allclose(actual.danger_score,danger_index(direct[['S','L','I','X']].to_numpy(float)))
assert np.allclose(actual[AXES].to_numpy()*4,direct[AXES].to_numpy())
prior=pd.read_parquet(root/'20261109_v5'/'predictions.parquet').set_index('news_id')
remaining=parquet[~parquet.session_reviewed]
assert np.allclose(remaining.danger_score,prior.loc[remaining.news_id].danger_score)
assert parquet.summary.fillna('').str.split().str.len().max()<=25
assert int(parquet.input_quality_flag.ne('').sum())==1
assert not parquet.loc[parquet.input_quality_flag.ne(''),'assessment_research_eligible'].any()
write_json(release/'portable_model_manifest.json',portable)
write_json(release/'integrity_audit.json',dict(models=checks,articles=8128,direct_references=1000,
    final_csv_parquet_equal=True,all_direct_scores_recomputed=True,unreviewed_model='v5',unreviewed_articles=len(remaining),
    full_body_packet_read_receipts_verified=True,publisher_summaries_max_words=25,input_quality_exclusions=1,
    runtime={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib','pyarrow']},
    limitation='Checks replay stored judgments and fitted models, not reference truth or independent human accuracy.'))
print('Release integrity checks passed',flush=True)
