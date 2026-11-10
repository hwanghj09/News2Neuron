"""Evaluate the sealed second attempt without changing its fitted model."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .provenance import sha256, write_json
from .risk_assessment import AXES, STATES, danger_index, risk_band
from .risk_v6_evaluation import paired_evidence, score_metrics
from .risk_v7 import OUT, PRIVATE, SEED


def judgments():
    selection=pd.read_csv(OUT/'audit_selection.csv')
    labels=pd.read_csv(OUT/'audit_judgments.tsv',sep='\t',keep_default_na=False)
    if set(selection.case_id)!=set(labels.case_id) or labels.case_id.duplicated().any():
        raise ValueError('Incomplete or duplicate actual AI judgments')
    a=labels[AXES].to_numpy(float)
    if not (np.isfinite(a).all() and ((a>=0)&(a<=4)&(a==np.floor(a))).all()):
        raise ValueError('Invalid ordinal axes')
    if not labels.state.isin(STATES).all() or not labels.rationale.str.len().ge(15).all():
        raise ValueError('Invalid state or rationale')
    if labels.loc[labels.state.isin(['none','resolved','retrospective']),['T','L','I']].to_numpy().any():
        raise ValueError('Closed state with residual risk')
    d=selection.merge(labels,on='case_id',validate='one_to_one')
    d['danger_score']=danger_index(d[['S','L','I','X']].to_numpy(float))
    d['judge']='current_Codex_session_AI_not_human'
    d['input_scope']='complete_stored_cleaned_body_and_title'
    return d


def paired_second_attempt(reference,baseline,candidate,sources,*,resamples=20000):
    r=paired_evidence(reference,baseline,candidate,sources,resamples=resamples,seed=SEED)
    y,b,p=map(np.asarray,[reference,baseline,candidate]);sources=np.asarray(sources)
    old=np.abs(y-b);new=np.abs(y-p);delta=old-new
    agreement=(new<=10).astype(float)-(old<=10).astype(float)
    rng=np.random.default_rng(SEED)
    draws=np.concatenate([rng.choice(ix,(resamples,len(ix)),replace=True)
        for s in sorted(set(sources)) for ix in [np.flatnonzero(sources==s)]],axis=1)
    r['MAE_reduction_97_5_percentile']=np.quantile(delta[draws].mean(axis=1),[.0125,.9875]).tolist()
    r['within10_change_97_5_percentile']=np.quantile(agreement[draws].mean(axis=1),[.0125,.9875]).tolist()
    r['primary_interval_level']=.975
    return r


def seal():
    if (OUT/'reference_freeze.json').exists():raise ValueError('Already sealed')
    audit=judgments()
    assert len(audit)==322 and audit.news_id.is_unique and audit.family.is_unique
    write_json(OUT/'reference_freeze.json',dict(articles=len(audit),
        labels_sha256=sha256(OUT/'audit_judgments.tsv'),
        packet_receipts={str(p):sha256(p) for p in sorted(OUT.glob('packet_*_receipt.json'))},
        individual_predictions_opened=False,labels_changed_after_predictions=False))


def evaluate():
    sealed=json.loads((OUT/'reference_freeze.json').read_text())
    assert sha256(OUT/'audit_judgments.tsv')==sealed['labels_sha256']
    frozen=json.loads((OUT/'pre_audit_freeze.json').read_text())
    for p,h in frozen['files'].items():
        if sha256(p)!=h:raise ValueError('Changed frozen artifact: '+p)
    assert sha256(Path(__file__).with_name('risk_v7.py'))==frozen['source_sha256']
    assert sha256(PRIVATE/'features.npy')==frozen['features_sha256']
    audit=judgments();dev=pd.read_csv(OUT/'development.csv')
    families=pd.read_csv(OUT/'corpus_families.csv').set_index('news_id')
    assert not set(families.loc[dev.news_id].family)&set(audit.family)
    y=audit.danger_score.to_numpy();primary=audit.cohort.eq('random').to_numpy();challenge=~primary
    assert primary.sum()==280 and challenge.sum()==42
    values={m:pd.read_parquet(f'data/risk/20261109_{m}/predictions.parquet').set_index('news_id').loc[audit.news_id].danger_score.to_numpy()
            for m in ['v5','v6','v7']}
    values['zero']=np.zeros(len(audit));rows=[];errors=[];per_source=[]
    for model,p in values.items():
        for cohort,mask in [('random',primary),('challenge',challenge),('all',np.ones(len(y),bool))]:
            rows.append(dict(model=model,cohort=cohort,**score_metrics(y[mask],p[mask])))
        for source in sorted(set(audit.source)):
            mask=primary&audit.source.eq(source).to_numpy()
            per_source.append(dict(model=model,source=source,**score_metrics(y[mask],p[mask])))
        errors.extend(dict(case_id=r.case_id,news_id=r.news_id,source=r.source,cohort=r.cohort,
            family=int(r.family),model=model,reference=float(t),prediction=float(v),absolute_error=float(abs(t-v)))
            for r,t,v in zip(audit.itertuples(),y,p))
    evidence={m:paired_second_attempt(y[primary],values[m][primary],values['v7'][primary],audit.source.to_numpy()[primary])
              for m in ['v5','v6']}
    get=lambda m,c:next(r for r in rows if r['model']==m and r['cohort']==c)
    checks=dict(MAE_superiority_both=all(r['MAE_reduction_97_5_percentile'][0]>0 for r in evidence.values()),
        within10_point_estimate_non_decrease_both=all(r['within10_change']>=0 for r in evidence.values()),
        high_recall_non_decrease_both_cohorts=all(get('v7',c)['high_recall']>=get(m,c)['high_recall']
            for m in ['v5','v6'] for c in ['random','challenge'] if get(m,c)['high_recall'] is not None),
        challenge_MAE_non_increase_both=all(get('v7','challenge')['MAE']<=get(m,'challenge')['MAE'] for m in ['v5','v6']),
        beats_constant_zero=get('v7','random')['MAE']<get('zero','random')['MAE'])
    passed=all(checks.values())
    write_json(OUT/'fresh_audit_metrics.json',dict(reference='single executing AI full-body residual-risk rubric; no independent human truth',
        primary_cohort='random',primary_articles=280,challenge_articles=42,primary_publishers=7,
        results=rows,paired_comparisons=evidence,preregistered_checks=checks,preregistered_evidence_pass=passed,
        prior_v6_failure_preserved=True,primary_interval_level=.975,no_posthoc_exclusions=True,
        fresh_audit_labels_used_for_training_or_selection=False,
        uncertainty_scope='publisher-stratified paired sampling conditional on this AI reference and family-purged seven-publisher pool; no label-truth or future-shift guarantee'))
    pd.DataFrame(errors).to_csv(OUT/'audit_errors.csv',index=False)
    pd.DataFrame(per_source).to_csv(OUT/'publisher_metrics.csv',index=False)
    # Only after pure-model comparisons may references replace display scores.
    prior=pd.read_csv('data/risk/20261109_v6/decision_audit.csv')
    decisions=pd.concat([prior,audit],ignore_index=True);assert decisions.news_id.is_unique and len(decisions)==1000
    final=pd.read_parquet(OUT/'predictions.parquet');final['model_danger_score']=final.danger_score
    final['decision_state']='unreviewed_model_prediction';final['rationale']='전체 본문 문맥을 학습한 모델의 예측이며 개별 직접 판정 아님.'
    direct=decisions.set_index('news_id')
    for i,r in final.iterrows():
        if r.news_id not in direct.index:continue
        d=direct.loc[r.news_id]
        for a in AXES:final.loc[i,a]=d[a]/4
        final.loc[i,'danger_score']=float(danger_index([d.S,d.L,d.I,d.X]));final.loc[i,'component_index']=final.loc[i,'danger_score']
        final.loc[i,'session_reviewed']=True;final.loc[i,'score_source']='direct_Codex_session_judgment'
        final.loc[i,'decision_state']=d.state;final.loc[i,'rationale']=d.rationale
    overlay=json.loads(Path('data/risk/20261109_v6/input_quality_overlay.json').read_text())
    draft_ids={r['news_id'] for r in overlay['records']}
    final['input_quality_flag']=np.where(final.news_id.isin(draft_ids),'unfilled_numeric_broadcast_draft','')
    final.loc[final.news_id.isin(draft_ids),'assessment_research_eligible']=False
    final['danger_band']=risk_band(final.danger_score.to_numpy())
    final.to_csv(OUT/'news_risk_scores.csv',index=False);final.to_parquet(OUT/'news_risk_scores.parquet',index=False)
    decisions.to_csv(OUT/'decision_audit.csv',index=False)
    write_json(OUT/'summary.json',dict(articles=len(final),direct_AI_judgments=int(final.session_reviewed.sum()),
        cumulative_new_full_body_AI_judgments=642,development_articles=len(dev),fresh_audit_articles=len(audit),
        primary_articles=280,challenge_articles=42,primary_publishers=7,human_raters=0,
        input_quality_flagged=len(draft_ids),preregistered_evidence_pass=passed))
    print(json.dumps(dict(checks=checks,paired_comparisons=evidence),ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','evaluate']);a=p.parse_args()
    seal() if a.stage=='seal' else evaluate()
