"""Publish model-gated scores, keeping every failed experiment reviewable."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .provenance import sha256, write_json
from .risk_assessment import AXES, danger_index, risk_band
from .risk_v7_evaluation import paired_second_attempt

ROOT=Path('data/risk')
OUT=ROOT/'20261110_release'


def select_model(experiments):
    for candidate in ['v7','v6']:
        if experiments[candidate]['preregistered_evidence_pass']:
            return candidate
    return 'v5'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    experiments={v:json.loads((ROOT/f'20261109_{v}'/'fresh_audit_metrics.json').read_text()) for v in ['v6','v7']}
    # v5 was the previously published model. A failed candidate does not
    # become the default just because one of its point estimates looks better.
    selected=select_model(experiments)
    final=pd.read_parquet(ROOT/f'20261109_{selected}'/'predictions.parquet')
    final['model_danger_score']=final.danger_score
    final['decision_state']='unreviewed_model_prediction'
    final['rationale']='사전 개선 기준을 통과한 후속 모델이 없어 기존 v5 예측을 유지한 기사. 개별 직접 판정 아님.'
    direct=pd.read_csv(ROOT/'20261109_v7'/'decision_audit.csv').set_index('news_id')
    for i,r in final.iterrows():
        if r.news_id not in direct.index:continue
        d=direct.loc[r.news_id]
        for a in AXES:final.loc[i,a]=d[a]/4
        final.loc[i,'danger_score']=float(danger_index([d.S,d.L,d.I,d.X]));final.loc[i,'component_index']=final.loc[i,'danger_score']
        final.loc[i,'session_reviewed']=True;final.loc[i,'score_source']='direct_Codex_session_judgment'
        final.loc[i,'decision_state']=d.state;final.loc[i,'rationale']=d.rationale
    overlay=json.loads((ROOT/'20261109_v6'/'input_quality_overlay.json').read_text())
    draft={r['news_id'] for r in overlay['records']}
    final['input_quality_flag']=np.where(final.news_id.isin(draft),'unfilled_numeric_broadcast_draft','')
    final.loc[final.news_id.isin(draft),'assessment_research_eligible']=False
    final['danger_band']=risk_band(final.danger_score.to_numpy())
    final.to_csv(OUT/'news_risk_scores.csv',index=False);final.to_parquet(OUT/'news_risk_scores.parquet',index=False)
    direct.reset_index().to_csv(OUT/'decision_audit.csv',index=False)
    write_json(OUT/'release_policy.json',dict(model=selected,previous_published_model='v5',
        promotion='candidate must pass all of its preregistered checks; otherwise retain prior published model',
        candidate_pass={v:m['preregistered_evidence_pass'] for v,m in experiments.items()},
        new_direct_judgments=642,total_direct_judgments=len(direct),human_raters=0,
        new_full_body_judgments=642,older_358_scope='title/lead/evidence windows plus selected whole bodies; not all individually read in full',
        incomplete_drafts_flagged=len(draft),frozen_models_retrained_after_audit=False,
        claimed_confirmatory_accuracy_improvement=False,
        caveat='Direct references replace display scores only. Agreement with those same references is not held-out accuracy evidence.',
        evidence_reports={'v6':'../20261109_v6/report.html','v7':'../20261109_v7/report.html'}))
    print(json.dumps(dict(articles=len(final),selected_model=selected,direct=len(direct),drafts=len(draft))))


def sensitivity():
    """Known related-case exclusions, fixed before predictions, secondary only."""
    results={}
    for version,oldnames in [('v6',['v3','v5']),('v7',['v5','v6'])]:
        p=ROOT/f'20261109_{version}';obs=json.loads((p/'pre_prediction_related_observations.json').read_text())
        exclude={c for link in obs['links'] if link.get('development_cases') for c in link['cases']}
        if version=='v6':exclude.add('v6_a_108')
        e=pd.read_csv(p/'audit_errors.csv');e=e[e.cohort.eq('random')&~e.case_id.isin(exclude)]
        new=e[e.model.eq(version)].set_index('case_id');comparisons={}
        for old in oldnames:
            b=e[e.model.eq(old)].set_index('case_id').loc[new.index]
            comparisons[old]=paired_second_attempt(new.reference,b.prediction,new.prediction,new.source)
        all_errors=pd.read_csv(p/'audit_errors.csv');all_errors=all_errors[all_errors.cohort.eq('random')]
        candidate=all_errors[all_errors.model.eq(version)].set_index('case_id')
        ids=list(candidate.index);parent={i:i for i in ids}
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for link in obs['links']:
            present=[c for c in link['cases'] if c in parent]
            for c in present[1:]:parent[find(c)]=find(present[0])
        groups=np.array([find(i) for i in ids]);unique=np.unique(groups)
        rng=np.random.default_rng(20261113)
        draws=rng.integers(0,len(unique),(20000,len(unique)))
        sizes=np.array([(groups==g).sum() for g in unique]);clustered={}
        for old in oldnames:
            b=all_errors[all_errors.model.eq(old)].set_index('case_id').loc[ids]
            delta=np.abs(candidate.reference.to_numpy()-b.prediction.to_numpy())-np.abs(candidate.reference.to_numpy()-candidate.prediction.to_numpy())
            sums=np.array([delta[groups==g].sum() for g in unique])
            boot=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
            clustered[old]=dict(MAE_reduction=float(delta.mean()),MAE_reduction_97_5_percentile=np.quantile(boot,[.0125,.9875]).tolist())
        results[version]=dict(secondary_only=True,excluded_primary_cases=sorted(set(exclude)&set(pd.read_csv(p/'audit_selection.csv').query("cohort=='random'").case_id)),
            primary_articles_retained=len(new),comparisons=comparisons,
            all_primary_known_campaign_cluster_bootstrap=dict(clusters=len(unique),comparisons=clustered,
                scheme='resample known connected campaign components, ratio of article-weighted error sums; whole clusters can cross publishers so publisher proportions vary',resamples=20000,seed=20261113),
            caveat='Known-link exclusions only; no claim all surviving stories are independent. Main endpoint unchanged. 97.5% intervals used conservatively for both supplemental analyses.')
    write_json(OUT/'known_relation_sensitivity.json',results)


def report():
    """Reuse the tested explorer, with explicit candidate and release wording."""
    from .risk_evidence_report import report as old_report
    p=ROOT/'20261109_v7'
    old_report(p,version='v7')
    # The release explorer is built from release data, while all accuracy
    # tables continue to point at the untouched pure candidate evaluation.
    s=json.loads((p/'summary.json').read_text())
    s.update(selected_model='v5',selected_model_development_articles=321,experiment_development_articles=673)
    write_json(OUT/'summary.json',s)
    (OUT/'fresh_audit_metrics.json').write_text((p/'fresh_audit_metrics.json').read_text())
    old_report(OUT,version='v7')
    html=(OUT/'report.html').read_text()
    html=html.replace('v7 후보 검증 · 기본 모델 유지','News2Neuron · 최신 적용 점수와 검증 결과')
    html=html.replace('<h2>전체 기사 평가</h2>','<h2>최신 적용 점수 · v5 모델 + 직접 AI 판정 1,000건</h2>')
    html=html.replace('잔존 위험을 더 정확하게,<br>개선 근거는 검증 가능하게.','뉴스 위험 지수와<br>모델 검증 결과를 함께.')
    html=html.replace('이 페이지 전체 기사 점수는 실험 후보의 결과이며 최종 적용 점수는 최신 배포 보고서를 보십시오.',
        '아래 기사 점수는 기존 v5 모델에 누적 직접 판정 1,000건을 적용한 최종 배포본입니다. 표의 v7 수치는 직접 판정 적용 전 후보 모델 검증 결과이며 배포 모델의 성능으로 표시하지 않습니다.')
    html=html.replace('<a href="../20261110_release/report.html">최신 적용 점수와 두 실험 결과</a>',
        '<a href="../20261109_v6/report.html">v6 실패 기록</a> · <a href="../20261109_v7/report.html">v7 후보 검증</a>')
    html=html.replace('href="protocol.json"','href="../20261109_v7/protocol.json"').replace('href="audit_errors.csv"','href="../20261109_v7/audit_errors.csv"').replace('href="publisher_metrics.csv"','href="../20261109_v7/publisher_metrics.csv"')
    html=html.replace('href="../../../docs/risk-evidence-v7.md"','href="../../../docs/risk-evidence-v7.md"')
    html=html.replace("+'건 · 개발 '+D.summary.development_articles", "+'건 · v7 후보 개발 '+D.summary.development_articles")
    (OUT/'report.html').write_text(html)


if __name__=='__main__':build();sensitivity();report()
