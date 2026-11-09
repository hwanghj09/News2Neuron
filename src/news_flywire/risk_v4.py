"""New, source-balanced development and a never-before-judged v4 AI audit.

The v3 holdout was already opened. It is development data here, never a new
test. Reference indices retain the v3 rubric; their nonlinear target is learned
directly rather than multiplying marginal point estimates.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC, SVR

from .provenance import sha256, write_json
from .risk_assessment import AXES, STATES, danger_index, risk_band, scaled_features

DEFAULT_OUT=Path('data/risk/20261109_v4')
V3=Path('data/risk/20261109_v3')
CACHE=Path('data/raw/risk/20261109_v3')
RAW=Path('data/raw/corpus/20261109_v2')
SEED=20261110
CUES={
 'fire_ended':r'완진|진화(?:를|가)?\s*(?:완료|마쳤|됐|되었|됐다)|불(?:길)?을?\s*잡았|구조.{0,8}수색.{0,8}마쳤|구조.{0,5}완료|진압.{0,5}완료',
 'release_or_restoration':r'경보.{0,6}해제|통제.{0,6}해제|복구.{0,6}완료|안전하게.{0,6}돌아|무사히.{0,6}(귀가|발견|구조)|모두.{0,6}구조',
 'custody':r'현행범.{0,5}체포|긴급체포|구속.{0,5}송치|구속.{0,5}기소|피의자.{0,10}검거|직무.{0,3}정지|분리.{0,3}조치',
 'unresolved_people':r'실종자|실종돼|행방.{0,5}(모르|불명)|수색.{0,8}(중|계속|진행)|구조.{0,6}(중|계속)|위중|중태|생사.{0,5}(확인|불명)',
 'unresolved_hazards':r'여전히|계속.{0,6}(사용|발생|쏟아|피해)|검사.{0,8}않|방호.{0,5}(없|미설치)|안전.{0,5}(미흡|미비)|추가.{0,5}(공격|피해|사망)|이행.{0,5}확인',
 'urgent_warning':r'대피령|대피.{0,6}명령|즉시.{0,8}대피|호우경보|한파주의보|한파경보|태풍경보|통제선|접근.{0,5}금지|국가소방동원령',
 'ongoing_attack':r'공격.{0,6}(이어|반복|계속|재개)|공습|미사일.{0,8}공격|민간.{0,6}피해|충돌.{0,5}격화|경고사격|군사.{0,5}행동',
 'structural_health':r'필수의료|인력난|전문의|괴롭힘|태움|자살률|고위험군|산업재해|산재|노동자.{0,6}사망|작업자',
 'material_livelihood':r'주거비|월세|생계|소득|고용.{0,4}불안|실직|부채|재정.{0,5}(적자|부담)|공급.{0,4}부족|피해금|환급|미환수',
 'privacy_ongoing':r'개인정보.{0,5}유출|계정.{0,4}탈취|해킹|비인가|불법.{0,4}접근|사칭|송금|유출.{0,4}확인|유출.{0,4}조사',
 'historical':r'\d+년\s*전|20세기|19세기|추모|별세|기념.{0,4}전시|사진전|과거.{0,5}(사건|전쟁)|유골|국제추모',
 'prevention_service':r'예방.{0,5}(교육|캠페인|지원|대책)|시연|안전키트|안전.{0,5}교육|구호.{0,4}물품|지원.{0,5}확대|조기.{0,5}(예측|진단)|연구.{0,4}성과',
 'hypothetical':r'가능성|우려|경우|가정|전망|검토|만약|\d+년.{0,3}뒤',
 'unverified':r'주장|의혹|추정|부인|확인되지|미확인|반박|혐의|사실.{0,4}아니',
 'negated_harm':r'인명\s*피해.{0,10}없|다친\s*사람.{0,8}없|사망자.{0,6}없|위험.{0,6}없|피해.{0,6}발생하지',
 'positive_excited':r'우승|연승|정상에|신기록|축제|공연|할인|전시|기부|수상',
 'figurative':r'관심\s*폭발|분노\s*폭발|외무성.{0,12}부상|장관급|전쟁나면|전쟁이.{0,4}일어났을\s*때',
}


def load_news():
    n=pd.read_parquet('data/corpus/20261109_v2/news.parquet').fillna('')
    s=pd.read_parquet(V3/'news_risk_scores.parquet')
    assert n.news_id.tolist()==s.news_id.tolist()
    for col in ['assessment_body_sha256','assessment_research_eligible','input_correction']:
        n[col]=s[col]
    n['summary']=s.summary
    return n


def body_for(row):
    path=(CACHE/'corrected_bodies' if row.assessment_body_sha256!=row.body_sha256 else RAW/'bodies')/(row.assessment_body_sha256+'.txt.gz')
    text=gzip.decompress(path.read_bytes()).decode()
    if hashlib.sha256(text.encode()).hexdigest()!=row.assessment_body_sha256:raise ValueError('Scoring body hash mismatch')
    return text


def extended_context(title,body):
    patterns=[re.compile(p) for p in CUES.values()]
    values=[min(1,np.log1p(len(p.findall(zone)))/np.log(5)) for zone in [title,body[:500],body[-700:],body] for p in patterns]
    sentences=re.split(r'[.!?]\s|\n',body)
    harm=re.compile(r'사망|부상|화재|폭발|공격|위험|실종|피해|살해')
    close=re.compile(CUES['fire_ended']+'|'+CUES['release_or_restoration']+'|'+CUES['custody'])
    ongoing=re.compile(CUES['unresolved_people']+'|'+CUES['unresolved_hazards'])
    values += [min(1,sum(bool(harm.search(s) and p.search(s)) for s in sentences)/3) for p in [close,ongoing,re.compile(CUES['negated_harm']),re.compile(CUES['hypothetical'])]]
    return np.array(values,dtype=np.float32)


def features(out=DEFAULT_OUT):
    out=Path(out);private=Path('data/raw/risk')/out.name;private.mkdir(parents=True,exist_ok=True)
    m=private/'feature_manifest.json'
    settings={'news_sha256':sha256('data/corpus/20261109_v2/news.parquet'),'semantic_sha256':sha256(CACHE/'scoring_features_corrected.npy'),'narrative_sha256':sha256(CACHE/'scoring_narrative_corrected.npy'),'cues':CUES,'zones':['title','lead500','tail700','fullbody'],'cue_weight':.5}
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if m.exists():
        d=json.loads(m.read_text())
        if d['fingerprint']!=fingerprint or d['sha256']!=sha256(private/'features.npy'):raise ValueError('Frozen features changed')
        return np.load(private/'features.npy')
    news=load_news()
    extra=np.array([extended_context(r.title,body_for(r)) for r in news.itertuples()])
    x=np.concatenate([scaled_features(np.load(CACHE/'scoring_features_corrected.npy')),np.load(CACHE/'scoring_narrative_corrected.npy')*.5,extra*.5],axis=1)
    np.save(private/'features.npy',x)
    write_json(m,{**settings,'fingerprint':fingerprint,'sha256':sha256(private/'features.npy'),'shape':list(x.shape)})
    return x


def near_matrix(news,left,right):
    """Conservative article-family purge; no claim of perfect semantic identity."""
    if not len(left) or not len(right):return np.zeros((len(left),len(right)),bool)
    tf=TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=50000,dtype=np.float32).fit(news.title)
    t=tf.transform(news.title)
    lexical=(t[left]@t[right].T).toarray()
    emb=np.load(CACHE/'titles.npy');sem=emb[left]@emb[right].T
    dates=pd.to_datetime(news.published_date,utc=True).astype('int64').to_numpy()/1e9
    close=np.abs(dates[left,None]-dates[None,right])<=14*86400
    oldgroups=news.event_group.to_numpy()
    same=oldgroups[left,None]==oldgroups[None,right]
    families=[named_families(t) for t in news.title]
    named=np.array([[bool(families[i]&families[j]) for j in right] for i in left])
    return same|named|((lexical>=.42)&close)|((sem>=.94)&(lexical>=.2)&close)


def named_families(title):
    """Conservative campaigns, not claims of exact event identity."""
    result=set()
    for tag,pattern in {
        'nepal_flood':r'네팔.*(?:홍수|실종|수색|이재민)',
        'geoje_rain':r'(?:거제|통영).*(?:호우|폭우|홍수|산사태|물폭탄|mm)',
        'dmz_mines':r'(?:DMZ|서부전선|비무장지대).*(?:지뢰|폭발|부상)',
        'jeju_missing':r'(?:제주.*실종|실종.*허위.?종결|장 씨.*(?:발인|실종))',
        'binance_hack':r'바이낸스.*(?:해킹|탈취|피해)',
        'nvidia_scandal':r'젠슨\s?황.*(?:스파이|불륜|리스크|스캔들)',
        'yemen_campaign':r'후티|예멘|리야드',
        'hormuz_campaign':r'호르무즈',
        'ebola_outbreak':r'에볼라',
        'guryong_fire':r'구룡마을',
    }.items():
        if re.search(pattern,title,re.I):result.add(tag)
    return result


def strengthen_fresh_purge(out=DEFAULT_OUT):
    """Protocol revision before any model or fresh-reference label is opened."""
    out=Path(out)
    if list(out.glob('round_*/training.json')):raise ValueError('Cannot revise the audit after training')
    s=pd.read_csv(out/'selection.csv');d=pd.read_csv(out/'session_judgments.tsv',sep='\t')
    if any(d.case_id.str.startswith('v4_a_')):raise ValueError('Fresh labels already opened')
    s.to_csv(out/'selection_initial_before_named_family_purge.csv',index=False)
    news=load_news();legacy=pd.read_csv(out/'legacy_development.csv');dev=s[s.role.eq('development')]
    protected=set(legacy.news_id)|set(dev.news_id)
    left=np.flatnonzero(news.assessment_research_eligible&~news.news_id.isin(protected))
    right=np.flatnonzero(news.news_id.isin(protected))
    allowed=left[~near_matrix(news,left,right).any(axis=1)]
    baseline=pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet');rng=np.random.default_rng(SEED)
    records=[];used_families=set();used_groups=set()
    for source in sorted(news.source.unique()):
        pool=np.array([i for i in allowed if news.source.iloc[i]==source and news.event_group.iloc[i] not in used_groups and not(named_families(news.title.iloc[i])&used_families)])
        uniform=rng.choice(pool,3,replace=False)
        remaining=np.array([i for i in pool if i not in set(uniform)])
        priority=baseline.danger_score.to_numpy()[remaining]+(baseline.bootstrap_p90-baseline.bootstrap_p10).to_numpy()[remaining]
        hard=rng.choice(remaining[np.argsort(-priority)[:min(30,len(remaining))]],3,replace=False)
        for cohort,indices in [('fresh_random',uniform),('fresh_stress',hard)]:
            for i in indices:
                r=news.iloc[i];records.append(dict(case_id=f'v4_a_{len(records)+1:03d}',news_id=r.news_id,body_sha256=r.body_sha256,assessment_body_sha256=r.assessment_body_sha256,event_group=r.event_group,source=source,role=cohort,round=0))
                used_families.update(named_families(r.title));used_groups.add(r.event_group)
    fresh=pd.DataFrame(records);pd.concat([fresh,dev],ignore_index=True).to_csv(out/'selection.csv',index=False)
    write_json(out/'audit_selection_revision.json',{'before_sha256':sha256(out/'selection_initial_before_named_family_purge.csv'),'after_sha256':sha256(out/'selection.csv'),
        'reason':'Conservatively block recurring named disaster/crime/military campaign families across the full 90-day range',
        'before_any_training':True,'before_fresh_reference_decisions':True,'development_case_ids_unchanged':True})


def setup(out=DEFAULT_OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.csv').exists():raise ValueError('v4 selection already frozen')
    news=load_news();legacy=pd.read_csv(V3/'decision_audit.csv')
    eligible=np.flatnonzero(news.assessment_research_eligible)
    old_indices=np.flatnonzero(news.news_id.isin(legacy.news_id))
    audit_candidates=eligible[~near_matrix(news,eligible,old_indices).any(axis=1)]
    old=pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet')
    rng=np.random.default_rng(SEED);chosen=[];records=[]
    for source in sorted(news.source.unique()):
        pool=np.array([i for i in audit_candidates if news.source.iloc[i]==source])
        uniform=rng.choice(pool,3,replace=False)
        stress=np.array([i for i in pool if i not in set(uniform)])
        priority=old.danger_score.to_numpy()[stress]+(old.bootstrap_p90-old.bootstrap_p10).to_numpy()[stress]
        stress=stress[np.argsort(-priority)[:min(30,len(stress))]]
        hard=rng.choice(stress,3,replace=False)
        for cohort,indices in [('fresh_random',uniform),('fresh_stress',hard)]:
            for i in indices:
                chosen.append(int(i));r=news.iloc[i]
                records.append(dict(case_id=f'v4_a_{len(records)+1:03d}',news_id=r.news_id,body_sha256=r.body_sha256,assessment_body_sha256=r.assessment_body_sha256,event_group=r.event_group,source=source,role=cohort,round=0))
    allowed=eligible[~near_matrix(news,eligible,np.array(chosen)).any(axis=1)&~news.news_id.iloc[eligible].isin(legacy.news_id).to_numpy()]
    for source in sorted(news.source.unique()):
        pool=np.array([i for i in allowed if news.source.iloc[i]==source])
        # Half hard cases / half random controls, all publishers represented.
        priority=(old.bootstrap_p90-old.bootstrap_p10).to_numpy()[pool]+old.danger_score.to_numpy()[pool]*.5
        hard=pool[np.argsort(-priority)[:4]]
        rest=np.array([i for i in pool if i not in set(hard)])
        picks=np.concatenate([hard,rng.choice(rest,4,replace=False)])
        for i in picks:
            r=news.iloc[i];records.append(dict(case_id=f'v4_d_{sum(x["role"]=="development" for x in records)+1:03d}',news_id=r.news_id,body_sha256=r.body_sha256,assessment_body_sha256=r.assessment_body_sha256,event_group=r.event_group,source=source,role='development',round=1))
    pd.DataFrame(records).to_csv(out/'selection.csv',index=False)
    legacy=legacy[legacy.role.ne('quality')].copy()
    legacy['previous_role']=legacy.role;legacy['role']='legacy_development'
    legacy.to_csv(out/'legacy_development.csv',index=False)
    write_json(out/'protocol.json',{'seed':SEED,'rubric':'risk-rubric-v3 (unchanged reference anchors and index)',
        'legacy_judgments_reclassified_as_development':len(legacy),'fresh_random_audit':24,'fresh_stress_audit':24,
        'initial_new_training':64,'second_active_training':16,'new_test_labels_used_for_tuning':False,
        'evaluation_target':'cross-sectional scoring of this 90-day corpus; no temporal forecast claim',
        'known_limitations':'same-session AI reference, not independent human accuracy; approximate event-family purge',
        'selection_sha256':sha256(out/'selection.csv'),'legacy_snapshot_sha256':sha256(out/'legacy_development.csv')})
    print('Frozen 48 fresh audit articles; selected 64 new development cases')


def packet(role,round_number=1,out=DEFAULT_OUT):
    out=Path(out);private=Path('data/raw/risk')/out.name;private.mkdir(parents=True,exist_ok=True)
    path=private/f'packet_{role}_{round_number}.json'
    if role.startswith('fresh_') and not (out/'pre_audit_freeze.json').exists():raise ValueError('Freeze the final model before opening fresh AI reference inputs')
    if path.exists():return json.loads(path.read_text())
    news=load_news().set_index('news_id');s=pd.read_csv(out/'selection.csv')
    s=s[s.role.eq(role)&s['round'].eq(round_number)]
    records=[]
    for r in s.itertuples():
        n=news.loc[r.news_id];body=body_for(n)
        sentences=re.split(r'(?<=[.!?])\s+|\n',body)
        terms=re.compile('|'.join(CUES[k] for k in ['unresolved_people','fire_ended','custody','unresolved_hazards','structural_health','ongoing_attack']))
        salience=max(sentences,key=lambda t:len(terms.findall(t)),default='')
        records.append({'case_id':r.case_id,'title':n.title,'lead':body[:130],'evidence':salience[:145],'tail':body[-135:],
                        'assessment_body_sha256':r.assessment_body_sha256})
    write_json(path,records);write_json(out/f'packet_{role}_{round_number}_receipt.json',{'sha256':sha256(path),'cases':len(records),'input_scope':'title, first130 chars, selected evidence145 chars, last135 chars; not a full individual read claim'})
    return records


def judgments(out=DEFAULT_OUT,include_audit=False):
    out=Path(out);news=load_news().set_index('news_id');s=pd.read_csv(out/'selection.csv')
    d=pd.read_csv(out/'session_judgments.tsv',sep='\t',keep_default_na=False)
    if d.case_id.duplicated().any() or not set(d.case_id).issubset(set(s.case_id)):raise ValueError('Duplicate/unselected judgments')
    for a in AXES:
        if not d[a].map(lambda v:type(v) is int and 0<=v<=4).all():raise ValueError('Invalid ordinal judgment')
    if not d.state.isin(STATES).all() or not d.rationale.str.len().ge(10).all():raise ValueError('Invalid decision record')
    merged=s.merge(d,on='case_id',validate='one_to_one')
    for r in merged.itertuples():
        if r.assessment_body_sha256!=news.loc[r.news_id].assessment_body_sha256:raise ValueError('Judged body changed')
        if r.state in ['none','resolved','retrospective'] and (r.T!=0 or r.L!=0):raise ValueError('Residual-risk contradiction')
    if not include_audit:merged=merged[merged.role.eq('development')]
    return merged


def family_groups(news,indices):
    links=near_matrix(news,indices,indices);parent=list(range(len(indices)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in zip(*np.where(np.triu(links,1))):parent[find(int(j))]=find(int(i))
    return np.array([find(i) for i in range(len(indices))])


class IndexModel:
    def __init__(self,kind,C=2.,gamma=.25,alpha=1.):
        self.kind,self.C,self.gamma,self.alpha=kind,C,gamma,alpha

    def fit(self,x,y):
        self.heads=Ridge(alpha=self.alpha,solver='lsqr').fit(x,y)
        target=danger_index(y[:,3:])/100
        if self.kind=='ordinal':return self
        if self.kind=='ridge':self.reg=Ridge(alpha=self.alpha,solver='lsqr').fit(x,target)
        else:
            self.reg=SVR(C=self.C,gamma=self.gamma,epsilon=.03)
            if self.kind=='hurdle':
                positive=target>0
                self.gate_constant=bool(positive[0]) if len(np.unique(positive))==1 else None
                if self.gate_constant is None:self.gate=SVC(C=2.,gamma=self.gamma,class_weight='balanced').fit(x,positive)
                if positive.any():self.reg.fit(x[positive],target[positive])
            else:self.reg.fit(x,target)
        return self

    def predict(self,x):
        axes=np.clip(self.heads.predict(x),0,4)
        if self.kind=='ordinal':r=danger_index(axes[:,3:])
        else:
            if self.kind=='hurdle' and self.gate_constant is False:return axes,np.zeros(len(x))
            r=np.clip(self.reg.predict(x),0,1)*100
            if self.kind=='hurdle' and self.gate_constant is None:r=np.where(self.gate.predict(x),r,0.)
        return axes,r


def train(round_number,out=DEFAULT_OUT):
    out=Path(out);folder=out/f'round_{round_number}'
    if folder.exists():raise ValueError('Training round is immutable')
    if (out/'pre_audit_freeze.json').exists():raise ValueError('Final model already frozen; start a new experiment')
    recorded=pd.read_csv(out/'session_judgments.tsv',sep='\t')
    if recorded.case_id.str.startswith('v4_a_').any():raise ValueError('Cannot train after fresh-reference labels are opened')
    news=load_news();x=features(out);new=judgments(out)
    selection=pd.read_csv(out/'selection.csv');required=selection[selection.role.eq('development')&selection['round'].le(round_number)]
    if not set(required.case_id).issubset(set(new.case_id)):raise ValueError('Real AI decisions required for selected cases')
    dev=pd.concat([pd.read_csv(out/'legacy_development.csv'),new[new['round'].le(round_number)]],ignore_index=True)
    if dev.news_id.duplicated().any():raise ValueError('Duplicate development article')
    lookup={n:i for i,n in enumerate(news.news_id)};indices=np.array([lookup[n] for n in dev.news_id])
    audit_indices=np.array([lookup[n] for n in selection[selection.role.str.startswith('fresh_')].news_id])
    if near_matrix(news,indices,audit_indices).any():raise ValueError('Development/fresh audit event-family overlap')
    groups=family_groups(news,indices);y=dev[AXES].to_numpy(float);target=danger_index(y[:,3:])
    folds=list(GroupKFold(n_splits=4).split(indices,groups=groups))
    candidates=[dict(kind=k,alpha=a) for k in ['ordinal','ridge'] for a in [.1,1.,10.]]
    candidates += [dict(kind=k,C=c,gamma=g,alpha=1.) for k in ['svr','hurdle'] for c in [2.,8.] for g in [.25,1.]]
    results=[]
    for params in candidates:
        oof=np.zeros(len(indices))
        for tr,va in folds:
            model=IndexModel(**params).fit(x[indices[tr]],y[tr]);_,oof[va]=model.predict(x[indices[va]])
        mae=float(mean_absolute_error(target,oof));positive=target>0
        pos_mae=float(mean_absolute_error(target[positive],oof[positive]))
        results.append({'parameters':params,'MAE':mae,'nonzero_MAE':pos_mae,'selection_loss':.7*mae+.3*pos_mae})
    winner=min(results,key=lambda r:r['selection_loss']);model=IndexModel(**winner['parameters']).fit(x[indices],y)
    axes,risk=model.predict(x)
    prediction=news[['news_id','source','url','title','summary','published_date','event_group','body_sha256','assessment_body_sha256','assessment_research_eligible','input_correction']].copy()
    for j,a in enumerate(AXES):prediction[a]=axes[:,j]/4
    prediction['danger_score']=risk;prediction['component_index']=danger_index(axes[:,3:]);prediction['danger_band']=risk_band(risk)
    prediction['score_source']='AI_supervised_v4_model_prediction';prediction['session_reviewed']=False
    folder.mkdir();prediction.to_parquet(folder/'predictions.parquet',index=False);dev.to_csv(folder/'development.csv',index=False)
    pd.DataFrame({'news_id':dev.news_id,'event_family':groups}).to_csv(folder/'cv_families.csv',index=False)
    joblib.dump(model,folder/'model.joblib',compress=3)
    write_json(folder/'training.json',{'round':round_number,'development_articles':len(dev),'event_families':len(set(groups)),
        'fresh_audit_labels_used':0,'cv_candidates':results,'selected':winner,'cv_primary_loss':'.7 all-case MAE + .3 nonzero-reference MAE',
        'development_sha256':sha256(folder/'development.csv'),'selection_sha256':sha256(out/'selection.csv'),
        'model_sha256':sha256(folder/'model.joblib'),'feature_sha256':sha256(Path('data/raw/risk')/out.name/'features.npy'),
        'cross_sectional_not_temporal_forecast':True})
    print('Round',round_number,'development',len(dev),'winner',winner,flush=True)
    return prediction


def active_select(out=DEFAULT_OUT):
    out=Path(out);news=load_news();s=pd.read_csv(out/'selection.csv')
    if (s['round']==2).any():raise ValueError('Active round already selected')
    dev=pd.read_csv(out/'round_1/development.csv');pred=pd.read_parquet(out/'round_1/predictions.parquet')
    old=pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet')
    eligible=np.flatnonzero(news.assessment_research_eligible&~news.news_id.isin(set(s.news_id)|set(dev.news_id)))
    protected=np.flatnonzero(news.news_id.isin(s.loc[s.role.str.startswith('fresh_'),'news_id']))
    eligible=eligible[~near_matrix(news,eligible,protected).any(axis=1)]
    priority=np.abs(pred.danger_score-old.danger_score).to_numpy()+pred.danger_score.to_numpy()*.15
    rows=[]
    for source in sorted(news.source.unique()):
        pool=np.array([i for i in eligible if news.source.iloc[i]==source]);picks=pool[np.argsort(-priority[pool])[:2]]
        for i in picks:
            r=news.iloc[i];rows.append(dict(case_id=f'v4_e_{len(rows)+1:03d}',news_id=r.news_id,body_sha256=r.body_sha256,assessment_body_sha256=r.assessment_body_sha256,event_group=r.event_group,source=source,role='development',round=2))
    pd.concat([s,pd.DataFrame(rows)],ignore_index=True).to_csv(out/'selection.csv',index=False)
    write_json(out/'active_selection.json',{'cases':16,'rule':'two per publisher with largest v3/v4 disagreement plus .15 predicted index','audit_labels_used':0,'case_ids':[r['case_id'] for r in rows]})


def finalize(out=DEFAULT_OUT):
    out=Path(out);news=load_news();s=pd.read_csv(out/'selection.csv');all_new=judgments(out,True)
    audit=all_new[all_new.role.str.startswith('fresh_')]
    freeze=json.loads((out/'pre_audit_freeze.json').read_text())
    for filename,digest in freeze['files'].items():
        if sha256(filename)!=digest:raise ValueError('Pre-audit frozen model/input changed')
    if set(audit.case_id)!=set(s[s.role.str.startswith('fresh_')].case_id):raise ValueError('Incomplete fresh AI audit')
    target=danger_index(audit[['S','L','I','X']].to_numpy(float));baseline=pd.read_parquet(V3/'round_3/input_corrected_predictions.parquet').set_index('news_id')
    models={'v3_frozen_baseline':baseline}
    models.update({f'v4_round_{r}':pd.read_parquet(out/f'round_{r}/predictions.parquet').set_index('news_id') for r in [1,2]})
    related=json.loads((out/'known_related_audit_links.json').read_text())
    excluded={item['audit_case'] for item in related['links']}
    novel=~audit.case_id.isin(excluded).to_numpy()
    rows=[];errors=[]
    for name,p in models.items():
        chosen=p.loc[audit.news_id];error=np.abs(target-chosen.danger_score.to_numpy())
        for cohort in ['all','fresh_random','fresh_stress','novel_all','novel_random','novel_stress']:
            base_cohort=cohort.replace('novel_','fresh_')
            mask=np.ones(len(audit),bool) if cohort in ['all','novel_all'] else audit.role.eq(base_cohort).to_numpy()
            if cohort.startswith('novel_'):mask=mask&novel
            rows.append({'model':name,'cohort':cohort,'articles':int(mask.sum()),'danger_MAE':float(error[mask].mean()),'within10_AI_agreement':float((error[mask]<=10).mean()),
                         'NTA_MAE':{a:float(mean_absolute_error(audit.loc[mask,a],chosen.loc[mask,a]*4)) for a in ['N','T','A']}})
        for r,actual,estimate,e in zip(audit.itertuples(),target,chosen.danger_score,error):errors.append({'case_id':r.case_id,'news_id':r.news_id,'source':r.source,'cohort':r.role,'model':name,'AI_reference':actual,'prediction':estimate,'absolute_error':e})
    before=np.abs(target-models['v3_frozen_baseline'].loc[audit.news_id].danger_score.to_numpy())
    after=np.abs(target-models['v4_round_2'].loc[audit.news_id].danger_score.to_numpy())
    rng=np.random.default_rng(SEED);strata=[]
    for source in audit.source.unique():
        for cohort in ['fresh_random','fresh_stress']:
            indices=np.flatnonzero(novel&audit.source.eq(source).to_numpy()&audit.role.eq(cohort).to_numpy())
            if len(indices):strata.append(indices)
    boot=[]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);boot.append(float((before[indices]-after[indices]).mean()))
    write_json(out/'fresh_audit_metrics.json',{'reference':'same-session AI rubric, not independent human truth','old_audit_retired':True,'fresh_audit_used_for_model_selection':False,'results':rows,
        'primary_cohort':'novel_all','primary_articles':int(novel.sum()),'related_articles_separately_reported':len(excluded),
        'paired_error_reduction':float((before[novel]-after[novel]).mean()),'paired_bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap_scope':'2000 paired resamples within publisher/cohort strata; describes this same-AI sample, not a guarantee for human truth or all future news'})
    error_frame=pd.DataFrame(errors);error_frame['included_in_primary']=~error_frame.case_id.isin(excluded);error_frame.to_csv(out/'audit_errors.csv',index=False)
    final=pd.read_parquet(out/'round_2/predictions.parquet');final['model_danger_score']=final.danger_score
    final['v3_model_danger_score']=baseline.loc[final.news_id,'danger_score'].to_numpy()
    final['requires_further_AI_review']=(np.abs(final.danger_score-final.v3_model_danger_score)>20)|~final.assessment_research_eligible
    final['decision_state']='unreviewed_model_prediction';final['rationale']='전체 본문 의미·종결/지속 문맥으로 예측한 AI 기준 위험 지수. 직접 판정 아님.'
    legacy=pd.read_csv(V3/'decision_audit.csv');decisions=pd.concat([legacy,all_new],ignore_index=True).drop_duplicates('news_id',keep='last').set_index('news_id')
    for i,r in final.iterrows():
        if r.news_id not in decisions.index:continue
        d=decisions.loc[r.news_id]
        for a in AXES:final.loc[i,a]=d[a]/4
        final.loc[i,'danger_score']=float(danger_index([d.S,d.L,d.I,d.X]));final.loc[i,'component_index']=final.loc[i,'danger_score']
        final.loc[i,'danger_band']=risk_band(final.loc[i,'danger_score']);final.loc[i,'session_reviewed']=True
        final.loc[i,'score_source']='direct_Codex_session_judgment';final.loc[i,'decision_state']=d.state;final.loc[i,'rationale']=d.rationale
        final.loc[i,'requires_further_AI_review']=False
    final.to_parquet(out/'news_risk_scores.parquet',index=False);final.to_csv(out/'news_risk_scores.csv',index=False)
    decisions.reset_index().to_csv(out/'decision_audit.csv',index=False)
    write_json(out/'summary.json',{'articles':len(final),'direct_AI_judgments':int(final.session_reviewed.sum()),'model_predictions':int((~final.session_reviewed).sum()),'development_articles':len(pd.read_csv(out/'round_2/development.csv')),
        'fresh_audit_articles':len(audit),'primary_novel_audit_articles':int(novel.sum()),'known_related_audit_articles':len(excluded),'human_judgments':0,'index':'direct estimate of the unchanged AI rubric index; component_index is separate for unreviewed model outputs',
        'model_choice':'predeclared final round 2 selected by development CV only','published_bands':final.danger_band.value_counts().to_dict()})
    return final


def freeze(out=DEFAULT_OUT):
    out=Path(out)
    recorded=pd.read_csv(out/'session_judgments.tsv',sep='\t')
    if recorded.case_id.str.startswith('v4_a_').any():raise ValueError('Fresh references were already opened')
    required=pd.read_csv(out/'selection.csv').query("role == 'development'")
    if not set(required.case_id).issubset(set(recorded.case_id)):raise ValueError('Missing actual development judgments')
    files=[out/f'round_{r}'/name for r in [1,2] for name in ['model.joblib','predictions.parquet','development.csv','training.json']]
    files += [out/'selection.csv',Path('data/raw/risk')/out.name/'features.npy']
    write_json(out/'pre_audit_freeze.json',{'files':{str(p):sha256(p) for p in files},'fresh_reference_judgments_at_freeze':0,
        'final_round_predeclared':2,'development_cases':273,'audit_article_ids':pd.read_csv(out/'selection.csv').query("role != 'development'").news_id.tolist()})
