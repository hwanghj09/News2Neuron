"""Family-balanced, zero-aware candidate after retiring the opened v4 audit."""
from pathlib import Path
import hashlib,json,re
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

from .risk_v4 import load_news,body_for,features as v4_features,named_families,near_matrix,family_groups,AXES,STATES,V3,SEED
from .risk_assessment import danger_index,risk_band
from .provenance import sha256,write_json

OUT=Path('data/risk/20261109_v5')
PRIVATE=Path('data/raw/risk/20261109_v5')
V4=Path('data/risk/20261109_v4')
EXTRA={
 'exercise':r'훈련|모의|가정한|시연|대응연습|실전.{0,4}대비',
 'legal_past':r'각하|불송치|무죄|재판|구형|항소심|공판|과거.{0,4}사건',
 'negative_breach':r'침해.{0,12}없|유출.{0,10}확인되지|피해.{0,10}발생하지|거짓.{0,4}경보|허위.{0,4}경보',
 'recovery_remaining':r'중태|심정지|중환자|위중|결장|회복.{0,5}중|장애|미회복|돌려받지|회수.{0,5}못',
 'benign_service':r'출시|서비스.{0,5}개발|무료.{0,4}운영|상견례|방문.{0,4}예정|박람회|특별전|개최',
 'current_cost':r'생계비|보증금|주거비|수익.{0,4}(끊|중단|감소)|소득.{0,5}감소|운행.{0,4}중단|인력난',
}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.csv').exists():raise ValueError('Selection already frozen')
    news=load_news();dev=pd.read_csv(V4/'decision_audit.csv');dev=dev[dev.role.ne('quality')].copy()
    dev['previous_role']=dev.role;dev['role']='development';dev.to_csv(OUT/'development.csv',index=False)
    bodies=[body_for(r) for r in news.itertuples()]
    extra=np.array([[min(1,np.log1p(len(re.findall(p,z)))/np.log(5)) for z in [r.title,b[:500],b[-700:],b] for p in EXTRA.values()] for r,b in zip(news.itertuples(),bodies)],dtype=np.float32)
    x=np.concatenate([v4_features(),extra*.5],axis=1);np.save(PRIVATE/'features.npy',x)
    write_json(OUT/'feature_manifest.json',{'v4_features_sha256':sha256('data/raw/risk/20261109_v4/features.npy'),'extra_patterns':EXTRA,'shape':list(x.shape),'features_sha256':sha256(PRIVATE/'features.npy')})
    protected=np.flatnonzero(news.news_id.isin(dev.news_id))
    pool=np.flatnonzero(news.assessment_research_eligible&~news.news_id.isin(set(dev.news_id)|set(pd.read_csv(V3/'decision_audit.csv').news_id)))
    pool=pool[~near_matrix(news,pool,protected).any(axis=1)]
    # The body-level pass catches event names hidden by indirect headlines.
    families=[named_families(r.title+' '+b) for r,b in zip(news.itertuples(),bodies)]
    known=set().union(*(families[i] for i in protected))
    pool=np.array([i for i in pool if not(families[i]&known)])
    baseline=pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet')
    rng=np.random.default_rng(SEED+1);rows=[];used=set()
    for source in sorted(news.source.unique()):
        choices=np.array([i for i in pool if news.source.iloc[i]==source and news.event_group.iloc[i] not in used])
        random=rng.choice(choices,2,replace=False);remaining=np.array([i for i in choices if i not in set(random)])
        priority=baseline.danger_score.to_numpy()[remaining]+(baseline.bootstrap_p90-baseline.bootstrap_p10).to_numpy()[remaining]
        stress=rng.choice(remaining[np.argsort(-priority)[:min(20,len(remaining))]],2,replace=False)
        for cohort,indices in [('random',random),('stress',stress)]:
            for i in indices:
                r=news.iloc[i];rows.append({'case_id':f'v5_a_{len(rows)+1:03d}','news_id':r.news_id,'body_sha256':r.body_sha256,'assessment_body_sha256':r.assessment_body_sha256,'source':source,'cohort':cohort,'event_group':r.event_group});used.add(r.event_group)
    pd.DataFrame(rows).to_csv(OUT/'selection.csv',index=False)
    write_json(OUT/'protocol.json',{'development_articles':len(dev),'previous_v4_holdout_retired':True,'new_audit_articles':32,'random_articles':16,'stress_articles':16,
        'purge':'candidate groups, headline lexical/semantic neighbors and named families in complete cleaned bodies',
        'selection_before_model_fit_and_fresh_judgments':True,'sampling_does_not_use_v5_predictions':True,'selection_sha256':sha256(OUT/'selection.csv'),
        'model_selection_loss':'event-family-balanced MAE + 4 * (1 - within10 agreement) on development OOF predictions',
        'reference':'same-session AI rubric; not independent human accuracy or temporal forecasting'})


class ZeroAware:
    def __init__(self,C=2.,gamma=.25,gate=True,floor=0.,family_weight=True):
        self.C,self.gamma,self.use_gate,self.floor,self.family_weight=C,gamma,gate,floor,family_weight

    def fit(self,x,y,groups):
        _,inv,counts=np.unique(groups,return_inverse=True,return_counts=True)
        w=1/np.sqrt(counts[inv]) if self.family_weight else np.ones(len(y));w=w/w.mean()
        self.heads=Ridge(alpha=1.,solver='lsqr').fit(x,y,sample_weight=w)
        target=danger_index(y[:,3:])/100
        self.reg=SVR(C=self.C,gamma=self.gamma,epsilon=.03).fit(x,target,sample_weight=w)
        presence=target>0;self.constant=bool(presence[0]) if len(np.unique(presence))==1 else None
        if self.use_gate and self.constant is None:self.gate=LogisticRegression(C=4.,max_iter=1000).fit(x,presence,sample_weight=w)
        return self

    def predict(self,x):
        axes=np.clip(self.heads.predict(x),0,4);risk=np.clip(self.reg.predict(x),0,1)*100
        if self.use_gate:
            positive=np.full(len(x),self.constant) if self.constant is not None else self.gate.predict(x)
            risk=np.where(positive,risk,0)
        return axes,np.where(risk<self.floor,0,risk)


def train():
    if (OUT/'model.joblib').exists() or (OUT/'pre_audit_freeze.json').exists():raise ValueError('Model immutable')
    if (OUT/'session_judgments.tsv').exists():raise ValueError('Fresh labels already opened; use a new experiment')
    news=load_news();x=np.load(PRIVATE/'features.npy');d=pd.read_csv(OUT/'development.csv')
    lookup={n:i for i,n in enumerate(news.news_id)};ix=np.array([lookup[n] for n in d.news_id]);y=d[AXES].to_numpy(float);target=danger_index(y[:,3:])
    groups=family_groups(news,ix);folds=list(GroupKFold(n_splits=4).split(ix,groups=groups))
    _,inv,counts=np.unique(groups,return_inverse=True,return_counts=True);eval_w=1/counts[inv]
    results=[]
    for C in [2.,8.]:
      for gamma in [.25,1.]:
       for gate in [True,False]:
        for floor in [0.,8.]:
         for family_weight in [True,False]:
          params=dict(C=C,gamma=gamma,gate=gate,floor=floor,family_weight=family_weight);oof=np.zeros(len(ix))
          for tr,va in folds:
            model=ZeroAware(**params).fit(x[ix[tr]],y[tr],groups[tr]);_,oof[va]=model.predict(x[ix[va]])
          err=np.abs(target-oof);mae=float(np.average(err,weights=eval_w));agreement=float(np.average(err<=10,weights=eval_w))
          results.append({'parameters':params,'family_balanced_MAE':mae,'within10':agreement,'loss':mae+4*(1-agreement)})
    best=min(results,key=lambda r:r['loss']);model=ZeroAware(**best['parameters']).fit(x[ix],y,groups);axes,risk=model.predict(x)
    pred=news[['news_id','source','url','title','summary','published_date','event_group','body_sha256','assessment_body_sha256','assessment_research_eligible','input_correction']].copy()
    for j,a in enumerate(AXES):pred[a]=axes[:,j]/4
    pred['danger_score']=risk;pred['component_index']=danger_index(axes[:,3:]);pred['danger_band']=risk_band(risk);pred['session_reviewed']=False;pred['score_source']='AI_supervised_v5_model_prediction'
    pred.to_parquet(OUT/'predictions.parquet',index=False);joblib.dump(model,OUT/'model.joblib',compress=3)
    pd.DataFrame({'news_id':d.news_id,'family':groups}).to_csv(OUT/'development_families.csv',index=False)
    write_json(OUT/'training.json',{'development_articles':len(d),'candidate_configurations':results,'selected':best,'fresh_audit_labels_used':0,
        'development_sha256':sha256(OUT/'development.csv'),'features_sha256':sha256(PRIVATE/'features.npy')})
    write_json(OUT/'pre_audit_freeze.json',{'files':{str(p):sha256(p) for p in [OUT/'model.joblib',OUT/'predictions.parquet',OUT/'development.csv',OUT/'selection.csv',PRIVATE/'features.npy']},'fresh_reference_judgments':0})
    print('Frozen v5 model:',best,flush=True)


def packet():
    if not (OUT/'pre_audit_freeze.json').exists():raise ValueError('Freeze model before fresh audit')
    path=PRIVATE/'fresh_packet.json'
    if path.exists():
        if sha256(path)!=json.loads((OUT/'fresh_packet_receipt.json').read_text())['sha256']:raise ValueError('Frozen packet changed')
        return json.loads(path.read_text())
    s=pd.read_csv(OUT/'selection.csv');news=load_news().set_index('news_id');records=[]
    for r in s.itertuples():
        n=news.loc[r.news_id];body=body_for(n);sentences=re.split(r'\n|(?<=[.!?])\s+',body)
        pattern=re.compile(r'실종|중태|심정지|위험|피해|사망|완진|검거|대피|회복')
        evidence=max(sentences,key=lambda z:len(pattern.findall(z)),default='')
        records.append({'case_id':r.case_id,'title':n.title,'lead':body[:130],'evidence':evidence[:145],'tail':body[-135:],'assessment_body_sha256':r.assessment_body_sha256})
    write_json(path,records);write_json(OUT/'fresh_packet_receipt.json',{'sha256':sha256(path),'input_scope':'title, lead130, selected evidence145, tail135; additional reads recorded separately'})
    return records


def evaluate():
    news=load_news();s=pd.read_csv(OUT/'selection.csv');j=pd.read_csv(OUT/'session_judgments.tsv',sep='\t')
    if set(j.case_id)!=set(s.case_id) or j.case_id.duplicated().any():raise ValueError('Audit incomplete')
    for a in AXES:
        if not j[a].map(lambda v:type(v) is int and 0<=v<=4).all():raise ValueError('Invalid integer ordinal judgment')
    if not j.state.isin(STATES).all():raise ValueError('Invalid state')
    if not j.rationale.str.len().ge(10).all():raise ValueError('Missing actual AI rationale')
    for r in j.itertuples():
        if r.state in ['none','resolved','retrospective'] and (r.T!=0 or r.L!=0):raise ValueError('Residual contradiction')
    audit=s.merge(j,on='case_id',validate='one_to_one');target=danger_index(audit[['S','L','I','X']].to_numpy(float))
    checked=news.set_index('news_id')
    if any(r.assessment_body_sha256!=checked.loc[r.news_id].assessment_body_sha256 for r in audit.itertuples()):raise ValueError('Judged body changed')
    related=json.loads((OUT/'known_related_audit_links.json').read_text());excluded={r['case_id'] for r in related['links']}
    novel=~audit.case_id.isin(excluded).to_numpy()
    for p,h in json.loads((OUT/'pre_audit_freeze.json').read_text())['files'].items():
        if sha256(p)!=h:raise ValueError('Frozen model/input changed')
    models={'v3':pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet').set_index('news_id'),'v4':pd.read_parquet(V4/'round_2/predictions.parquet').set_index('news_id'),'v5':pd.read_parquet(OUT/'predictions.parquet').set_index('news_id')}
    results=[];errors=[]
    for name,pred in models.items():
        values=pred.loc[audit.news_id].danger_score.to_numpy();error=np.abs(values-target)
        for cohort in ['all','random','stress','novel_all','novel_random','novel_stress']:
            mask=np.ones(len(audit),bool) if cohort in ['all','novel_all'] else audit.cohort.eq(cohort.replace('novel_','')).to_numpy()
            if cohort.startswith('novel_'):mask=mask&novel
            results.append({'model':name,'cohort':cohort,'articles':int(mask.sum()),'MAE':float(error[mask].mean()),'within10_AI_agreement':float((error[mask]<=10).mean())})
        errors.extend({'case_id':r.case_id,'news_id':r.news_id,'source':r.source,'cohort':r.cohort,'model':name,'reference':float(y),'prediction':float(p),'absolute_error':float(e)} for r,y,p,e in zip(audit.itertuples(),target,values,error))
    old_error=np.abs(models['v3'].loc[audit.news_id].danger_score.to_numpy()-target);new_error=np.abs(models['v5'].loc[audit.news_id].danger_score.to_numpy()-target)
    rng=np.random.default_rng(SEED+1);boot=[]
    for _ in range(2000):
        sample=rng.choice(np.flatnonzero(novel),int(novel.sum()),replace=True);boot.append(float((old_error[sample]-new_error[sample]).mean()))
    write_json(OUT/'fresh_audit_metrics.json',{'reference':'same-session AI rubric, not independent human truth','new_audit_used_for_training_or_selection':False,'results':results,
        'primary_cohort':'novel_all','primary_articles':int(novel.sum()),'related_followups_reported_separately':len(excluded),
        'paired_error_reduction':float((old_error[novel]-new_error[novel]).mean()),'paired_bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),
        'interval_scope':'descriptive paired resampling of this small same-AI sample; not independent human accuracy'})
    error_frame=pd.DataFrame(errors);error_frame['included_in_primary']=~error_frame.case_id.isin(excluded);error_frame.to_csv(OUT/'audit_errors.csv',index=False)
    final=pd.read_parquet(OUT/'predictions.parquet');final['model_danger_score']=final.danger_score
    final['decision_state']='unreviewed_model_prediction';final['rationale']='본문 의미와 종결·예방·진행 문맥에서 예측한 지수. 개별 직접 판정 아님.'
    decisions=pd.concat([pd.read_csv(V4/'decision_audit.csv'),audit],ignore_index=True).drop_duplicates('news_id',keep='last').set_index('news_id')
    for i,row in final.iterrows():
        if row.news_id not in decisions.index:continue
        d=decisions.loc[row.news_id]
        for a in AXES:final.loc[i,a]=d[a]/4
        final.loc[i,'danger_score']=float(danger_index([d.S,d.L,d.I,d.X]));final.loc[i,'component_index']=final.loc[i,'danger_score'];final.loc[i,'danger_band']=risk_band(final.loc[i,'danger_score'])
        final.loc[i,'session_reviewed']=True;final.loc[i,'score_source']='direct_Codex_session_judgment';final.loc[i,'decision_state']=d.state;final.loc[i,'rationale']=d.rationale
    final.to_csv(OUT/'news_risk_scores.csv',index=False);final.to_parquet(OUT/'news_risk_scores.parquet',index=False);decisions.reset_index().to_csv(OUT/'decision_audit.csv',index=False)
    write_json(OUT/'summary.json',{'articles':len(final),'direct_AI_judgments':int(final.session_reviewed.sum()),'development_articles':len(pd.read_csv(OUT/'development.csv')),'new_audit_articles':len(audit),'primary_novel_audit_articles':int(novel.sum()),'human_raters':0,'model_selection_reference':'development grouped CV only','index':'same rubric anchors; direct index estimation and a learned zero gate; not probability','published_bands':final.danger_band.value_counts().to_dict()})
