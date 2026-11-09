"""Verify score provenance and complete-token coverage; no semantic truth claim."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, danger_index, read_judgments, risk_band
from correct_seoul_risk_inputs import remove_promotion

p=argparse.ArgumentParser()
p.add_argument('--corpus',default='data/corpus/20261109_v2')
p.add_argument('--out',default='data/risk/20261109_v3')
p.add_argument('--raw',default='data/raw/corpus/20261109_v2')
p.add_argument('--cache',default='data/raw/risk/20261109_v3')
p.add_argument('--encoder',default='external/multilingual-e5-small')
a=p.parse_args()
out,corpus,cache,raw=map(Path,[a.out,a.corpus,a.cache,a.raw])
news=pd.read_parquet(corpus/'news.parquet')
scores=pd.read_parquet(out/'news_risk_scores.parquet')
assert len(scores)==len(news) and scores.news_id.is_unique
assert scores.news_id.tolist()==news.news_id.tolist()
assert scores.body_sha256.tolist()==news.body_sha256.tolist()
assert np.isfinite(scores[AXES+['danger_score','bootstrap_p10','bootstrap_p90']].to_numpy()).all()
assert scores[AXES].ge(0).all().all() and scores[AXES].le(1).all().all()
assert scores.danger_score.between(0,100).all()
assert np.allclose(danger_index(scores[['S','L','I','X']].to_numpy()*4),scores.danger_score)
assert (risk_band(scores.danger_score)==scores.danger_band).all()
assert scores.bootstrap_p10.le(scores.bootstrap_p90).all()
labels=read_judgments(out,news)
assert set(scores.loc[scores.session_reviewed,'news_id'])==set(labels.news_id)
assert scores.loc[scores.session_reviewed,'score_source'].eq('direct_Codex_session_judgment').all()
assert scores.loc[~scores.session_reviewed,'score_source'].eq('ai_supervised_semantic_model_prediction').all()
fm=json.loads((cache/'features_manifest.json').read_text())
assert fm['news_parquet_sha256']==sha256(corpus/'news.parquet')
assert fm['features_sha256']==sha256(cache/'features.npy')
feature_rows=pd.read_csv(cache/'feature_rows.csv')
assert feature_rows.news_id.tolist()==news.news_id.tolist()
assert feature_rows.body_sha256.tolist()==news.body_sha256.tolist()
seen=set();covered=0;windows=0;last_character_gaps=[]
from tokenizers import Tokenizer
tokenizer=Tokenizer.from_file(str(Path(a.encoder)/'tokenizer.json'))
tokenizer.enable_truncation(max_length=256,stride=32)
untruncated=Tokenizer.from_file(str(Path(a.encoder)/'tokenizer.json'))
untruncated.no_truncation()
for r in news.itertuples():
    body=gzip.decompress((raw/'bodies'/(r.body_sha256+'.txt.gz')).read_bytes()).decode()
    assert hashlib.sha256(body.encode()).hexdigest()==r.body_sha256
    text='passage: '+r.title+'\n'+body
    encoded=tokenizer.encode(text)
    chunks=[encoded]+list(encoded.overflowing)
    restored=[]
    for index,chunk in enumerate(chunks):
        content=[token for token,special in zip(chunk.ids,chunk.special_tokens_mask) if not special]
        restored.extend(content if index==0 else content[32:])
    full=untruncated.encode(text)
    assert restored==[token for token,special in zip(full.ids,full.special_tokens_mask) if not special]
    record=json.loads((cache/'evidence'/(r.body_sha256+'.json')).read_text())
    assert record['chunks']==len(chunks) and record['all_body_tokens_processed']
    assert (cache/'articles'/(r.body_sha256+'.npz')).exists()
    # Every untruncated content-token ID occurs in the ordered overflow windows.
    # Concatenated overlap windows must reach the original last token offset.
    max_offset=max(b for c in chunks for _,b in c.offsets)
    if max_offset<len(text.rstrip()):last_character_gaps.append({'news_id':r.news_id,'trailing_chars':len(text.rstrip())-max_offset})
    windows+=len(chunks);covered+=1
for round_number in [1,2,3]:
    folder=out/f'round_{round_number}'
    training=json.loads((folder/'training.json').read_text())
    snapshot=pd.read_csv(folder/'development_judgments.tsv',sep='\t')
    assert training['development_snapshot_sha256']==sha256(folder/'development_judgments.tsv')
    assert set(snapshot.case_id)==set(training['development_case_ids'])
    assert snapshot.role.eq('development').all()
    assert not set(snapshot.event_group)&set(labels.loc[~labels.role.eq('development'),'event_group'])
    predicted=pd.read_parquet(folder/'predictions.parquet')
    assert predicted.news_id.tolist()==news.news_id.tolist()
    assert predicted.score_source.eq('ai_supervised_semantic_model_prediction').all()
    assert not predicted.session_reviewed.any()
    assert len(predicted)==8128
    assert seen.issubset(set(snapshot.case_id))
    seen=set(snapshot.case_id)
    assert training['audit_labels_used_for_training_or_tuning']==0
csv=pd.read_csv(out/'news_risk_scores.csv')
assert csv.news_id.tolist()==scores.news_id.tolist()
assert np.allclose(csv.danger_score,scores.danger_score)
corrections=pd.read_csv(out/'input_corrections.csv')
assert len(corrections)==180
effective_windows=windows
cm=json.loads((out/'input_correction_manifest.json').read_text())
assert cm['corrected_features_sha256']==sha256(cache/'scoring_features_corrected.npy')
assert cm['corrected_narrative_sha256']==sha256(cache/'scoring_narrative_corrected.npy')
assert not set(corrections.news_id)&set(labels.loc[labels.role.eq('development'),'news_id'])
for r in corrections.itertuples():
    body=gzip.decompress((cache/'corrected_bodies'/(r.assessment_body_sha256+'.txt.gz')).read_bytes()).decode()
    assert hashlib.sha256(body.encode()).hexdigest()==r.assessment_body_sha256
    original_html=gzip.decompress((raw/'http/responses'/(r.html_sha256+'.gz')).read_bytes())
    before,reextracted,n_widgets=remove_promotion(original_html)
    assert reextracted==body and n_widgets==r.widgets_removed
    assert hashlib.sha256(before.encode()).hexdigest()==r.original_body_sha256
    assert '서울시의회 바로가기' not in body
    row=scores.set_index('news_id').loc[r.news_id]
    assert row.assessment_body_sha256==r.assessment_body_sha256
    encoded=tokenizer.encode('passage: '+row.title+'\n'+body)
    chunks=[encoded]+list(encoded.overflowing)
    assert len(chunks)==r.complete_scoring_windows
    restored=[]
    for i,chunk in enumerate(chunks):
        content=[t for t,special in zip(chunk.ids,chunk.special_tokens_mask) if not special]
        restored.extend(content if i==0 else content[32:])
    full=untruncated.encode('passage: '+row.title+'\n'+body)
    assert restored==[t for t,special in zip(full.ids,full.special_tokens_mask) if not special]
    old_window_count=json.loads((cache/'evidence'/(r.original_body_sha256+'.json')).read_text())['chunks']
    effective_windows+=len(chunks)-old_window_count
for round_number,weight_hash in cm['frozen_model_hashes_unchanged'].items():
    assert sha256(out/f'round_{round_number}'/'ridge_model.joblib')==weight_hash
freeze=json.loads((out/'pre_audit_model_freeze.json').read_text())
for file,digest in freeze['files'].items():assert sha256(file)==digest
write_json(out/'verification.json',{'status':'passed','articles_checked':len(scores),'full_body_hashes_checked':covered,
           'complete_overflow_token_windows_checked':windows,'token_offset_trailing_gaps':last_character_gaps,
           'direct_AI_judgments':len(labels),'human_judgments':0,'rounds_audited':3,
           'audit_used_for_training':False,'csv_parquet_scores_identical':True,
           'native_promotion_repairs_verified':len(corrections),'affected_training_articles':0,
           'effective_promotion_free_scoring_windows_checked':effective_windows,
           'pre_audit_model_hashes_unchanged':True,'assessment_research_eligible_articles':int(scores.assessment_research_eligible.sum()),
           'meaning':'Provenance, numerical and partition checks only; not proof of semantic accuracy or real-world probabilities.'})
print(f'Audited {len(scores)} scores, {covered} body hashes, {windows} full-text windows, {len(labels)} direct AI judgments')
