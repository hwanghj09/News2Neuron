"""Fixed-size paired evidence for v6; never selects or changes the fitted model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .provenance import sha256, write_json
from .risk_assessment import AXES, danger_index, risk_band
from .risk_v4 import load_news
from .risk_v6 import OUT, PREVIOUS, SEED, judgments


def paired_evidence(reference, baseline, candidate, sources, *, resamples=20000, seed=SEED):
    """Preserve publisher strata and pair the same articles in every resample."""
    reference, baseline, candidate = map(lambda a:np.asarray(a,dtype=float),
                                         [reference,baseline,candidate])
    sources=np.asarray(sources)
    if not (len(reference)==len(baseline)==len(candidate)==len(sources)) or not len(reference):
        raise ValueError('Nonempty equal-length paired arrays required')
    if not np.isfinite(np.concatenate([reference,baseline,candidate])).all():
        raise ValueError('Nonfinite scores')
    old=np.abs(reference-baseline);new=np.abs(reference-candidate)
    delta=old-new;agreement=(new<=10).astype(float)-(old<=10).astype(float)
    rng=np.random.default_rng(seed)
    strata=[np.flatnonzero(sources==s) for s in sorted(set(sources))]
    draw=np.concatenate([rng.choice(ix,(resamples,len(ix)),replace=True) for ix in strata],axis=1)
    mean_delta=delta[draw].mean(axis=1)
    agree_delta=agreement[draw].mean(axis=1)
    improved=int(((new<=10)&(old>10)).sum());regressed=int(((new>10)&(old<=10)).sum())
    discordant=improved+regressed
    wins=int((delta>1e-10).sum());losses=int((delta< -1e-10).sum())
    non_ties=wins+losses
    # The preregistered criterion is a paired bootstrap interval. These tests
    # supplement it; they cannot replace a failed primary endpoint.
    signs=rng.choice([-1.,1.],size=(resamples,len(delta)))
    perm=(signs*delta).mean(axis=1)
    return {
        'articles':len(reference), 'baseline_MAE':float(old.mean()),'candidate_MAE':float(new.mean()),
        'MAE_reduction':float(delta.mean()),'MAE_reduction_95_percentile':np.quantile(mean_delta,[.025,.975]).tolist(),
        'relative_MAE_reduction':float(delta.mean()/old.mean()) if old.mean()>0 else None,
        'baseline_within10':float((old<=10).mean()),'candidate_within10':float((new<=10).mean()),
        'within10_change':float(agreement.mean()),'within10_change_95_percentile':np.quantile(agree_delta,[.025,.975]).tolist(),
        'within10_improved_cases':improved,'within10_regressed_cases':regressed,
        'mcnemar_exact_one_sided_p':float(binomtest(improved,discordant,.5,alternative='greater').pvalue) if discordant else 1.,
        'paired_error_wins':wins,'paired_error_losses':losses,
        'sign_test_one_sided_p':float(binomtest(wins,non_ties,.5,alternative='greater').pvalue) if non_ties else 1.,
        'paired_sign_flip_one_sided_p':float((1+(perm>=delta.mean()-1e-12).sum())/(resamples+1)),
        'resamples':resamples,'seed':seed,
    }


def score_metrics(reference,prediction):
    y=np.asarray(reference);p=np.asarray(prediction);error=np.abs(y-p)
    high=y>=30;positive=y>0;zero=y==0
    return dict(articles=len(y),MAE=float(error.mean()),RMSE=float(np.sqrt((error**2).mean())),
        within10_AI_agreement=float((error<=10).mean()),
        reference_positive=int(positive.sum()), reference_high=int(high.sum()),
        high_recall=float((p[high]>=30).mean()) if high.any() else None,
        nonzero_recall=float((p[positive]>0).mean()) if positive.any() else None,
        zero_false_positive_rate=float((p[zero]>10).mean()) if zero.any() else None,
        error_gt30=int((error>30).sum()))


def evaluate():
    freeze=json.loads((OUT/'pre_audit_freeze.json').read_text())
    for path,digest in freeze['files'].items():
        if sha256(path)!=digest:raise ValueError('Frozen model/protocol changed: '+path)
    if sha256(Path(__file__).with_name('risk_v6.py')) != freeze['model_source_sha256']:
        raise ValueError('Frozen model source changed')
    audit=judgments('audit')
    assert len(audit)==192 and audit.news_id.is_unique and audit.family.is_unique
    dev=pd.read_csv(OUT/'development.csv')
    assert not set(dev.family)&set(audit.family)
    y=audit.danger_score.to_numpy()
    models={
        'v3':pd.read_parquet('data/risk/20261109_v3/round_3/input_corrected_predictions.parquet').set_index('news_id'),
        'v5':pd.read_parquet(PREVIOUS/'predictions.parquet').set_index('news_id'),
        'v6':pd.read_parquet(OUT/'predictions.parquet').set_index('news_id')}
    values={m:f.loc[audit.news_id].danger_score.to_numpy() for m,f in models.items()}
    values['zero']=np.zeros(len(y));metrics=[];errors=[];evidence={}
    for model,pred in values.items():
        for cohort in ['random','challenge','all']:
            mask=np.ones(len(y),bool) if cohort=='all' else audit.cohort.eq(cohort).to_numpy()
            metrics.append(dict(model=model,cohort=cohort,**score_metrics(y[mask],pred[mask])))
        errors.extend(dict(case_id=row.case_id,news_id=row.news_id,source=row.source,
            cohort=row.cohort,family=int(row.family),model=model,reference=float(target),
            prediction=float(p),absolute_error=float(abs(target-p)))
            for row,target,p in zip(audit.itertuples(),y,pred))
    primary=audit.cohort.eq('random').to_numpy();challenge=~primary
    for baseline in ['v3','v5']:
        evidence[baseline]=paired_evidence(y[primary],values[baseline][primary],values['v6'][primary],audit.source.to_numpy()[primary])
    superiority=all(evidence[m]['MAE_reduction_95_percentile'][0]>0 for m in evidence)
    agreement_guard=all(evidence[m]['within10_change']>=0 for m in evidence)
    challenge_mae=float(np.abs(y[challenge]-values['v6'][challenge]).mean())
    challenge_guard=all(challenge_mae<=float(np.abs(y[challenge]-values[m][challenge]).mean()) for m in ['v3','v5'])
    zero_mae=float(np.abs(y[primary]).mean());new_mae=float(np.abs(y[primary]-values['v6'][primary]).mean())
    evidence_pass=bool(superiority and agreement_guard and challenge_guard and new_mae<zero_mae)
    write_json(OUT/'fresh_audit_metrics.json',dict(
        reference='same executing AI residual-risk rubric; not independent human or empirical danger truth',
        primary_cohort='random',primary_articles=int(primary.sum()),challenge_articles=int(challenge.sum()),
        results=metrics,paired_comparisons=evidence,
        preregistered_checks=dict(MAE_superiority_both=superiority,
            within10_point_estimate_non_decrease_both=agreement_guard,
            challenge_MAE_non_increase_both=challenge_guard,beats_constant_zero=new_mae<zero_mae),
        preregistered_evidence_pass=evidence_pass,
        no_posthoc_exclusions=True,fresh_audit_labels_used_for_training_or_selection=False,
        uncertainty_scope='publisher-stratified paired resampling conditional on this single AI reference and family-purged corpus pool; does not measure label truth or future domain shift'))
    pd.DataFrame(errors).to_csv(OUT/'audit_errors.csv',index=False)
    per_source=[]
    for source in sorted(set(audit.source)):
        mask=primary&audit.source.eq(source).to_numpy()
        for model,p in values.items():per_source.append(dict(source=source,model=model,**score_metrics(y[mask],p[mask])))
    pd.DataFrame(per_source).to_csv(OUT/'publisher_metrics.csv',index=False)
    final=pd.read_parquet(OUT/'predictions.parquet')
    final['model_danger_score']=final.danger_score
    final['decision_state']='unreviewed_model_prediction'
    final['rationale']='전체 본문 의미·문자 문맥을 결합한 모델의 예측. 개별 직접 판정 아님.'
    decisions=pd.concat([pd.read_csv(PREVIOUS/'decision_audit.csv'),judgments('development'),audit],ignore_index=True)
    assert decisions.news_id.is_unique
    direct=decisions.set_index('news_id')
    for i,row in final.iterrows():
        if row.news_id not in direct.index:continue
        label=direct.loc[row.news_id]
        for a in AXES:final.loc[i,a]=label[a]/4
        final.loc[i,'danger_score']=float(danger_index([label.S,label.L,label.I,label.X]))
        final.loc[i,'component_index']=final.loc[i,'danger_score']
        final.loc[i,'session_reviewed']=True
        final.loc[i,'score_source']='direct_Codex_session_judgment'
        final.loc[i,'decision_state']=label.state;final.loc[i,'rationale']=label.rationale
    final['danger_band']=risk_band(final.danger_score.to_numpy())
    final.to_csv(OUT/'news_risk_scores.csv',index=False)
    final.to_parquet(OUT/'news_risk_scores.parquet',index=False)
    decisions.to_csv(OUT/'decision_audit.csv',index=False)
    write_json(OUT/'summary.json',dict(articles=len(final),direct_AI_judgments=int(final.session_reviewed.sum()),
        new_full_body_AI_judgments=320,development_articles=len(dev),fresh_audit_articles=len(audit),
        primary_articles=int(primary.sum()),challenge_articles=int(challenge.sum()),human_raters=0,
        preregistered_evidence_pass=evidence_pass))
    print(json.dumps({'evidence_pass':evidence_pass,'checks':json.loads((OUT/'fresh_audit_metrics.json').read_text())['preregistered_checks'],
                      'paired_comparisons':evidence},ensure_ascii=False,indent=2))


if __name__=='__main__':evaluate()
