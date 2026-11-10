"""Second, separately registered experiment after v6 failed superiority.

v6 reference labels are now development. Its failed result remains published.
The next audit is a new fixed sample with a stricter 97.5% interval; it cannot
be enlarged or retuned after opening its reference judgments.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC,SVR

from .provenance import sha256,write_json
from .risk_assessment import AXES,STATES,danger_index
from .risk_v4 import body_for,load_news
from .risk_input_quality import incomplete_numeric_draft

OUT=Path('data/risk/20261109_v7')
PRIVATE=Path('data/raw/risk/20261109_v7')
PREVIOUS=Path('data/risk/20261109_v6')
SEED=20261112
CUES={
 'current_treatment':r'치료받고|치료를 받고|치료 중|입원 중|입원.{0,5}치료|병원으로.{0,5}이송',
 'current_injury':r'부상으로.{0,12}(?:빠|결장|이탈)|결장|인대.{0,4}파열|앓고 있|완치.{0,4}(?:안|어려)|활동.{0,4}제한',
 'severe_medical':r'중상|중태|위중|의식.{0,4}없|생명.{0,4}위험|절단|회복.{0,4}불가',
 'nonfatal':r'생명.{0,8}지장.{0,4}없|큰.{0,4}이상.{0,4}없|경상|경미한.{0,4}부상',
 'successful_recovery':r'일상으로.{0,6}(?:돌아|복귀)|무사히.{0,6}(?:출산|분만|풀려|석방|복귀)|건강.{0,4}양호|뇌 손상 없이|전액.{0,4}반환|모두.{0,4}갚',
 'remaining_repair':r'미복구|복구.{0,8}(?:진행 중|추진|예정)|보강.{0,8}(?:진행|이뤄)|임시.{0,5}(?:숙소|숙박|거처)|거처.{0,8}옮',
 'unrecovered_finance':r'돌려받지 못|회수하지 못|미회수|상환.{0,4}부담|채무.{0,4}부담|보증금.{0,8}손실|체납.{0,8}(?:장기|남|해소되지)',
 'active_market_loss':r'(?:주가|지수|코인).{0,16}(?:급락|폭락|하락)|손실.{0,8}(?:계속|누적|확대)|급락세.{0,6}(?:계속|이어)',
 'profit_decline':r'(?:이익|매출|순이익).{0,18}(?:감소|줄|하락)|감익|실적.{0,8}(?:둔화|부진)|적자.{0,6}(?:전환|돌아)',
 'ongoing_cost':r'물가.{0,10}(?:상승|압력)|비용.{0,8}(?:상승|부담)|난방비.{0,5}부담|주거비.{0,5}부담|전셋값.{0,15}(?:올|상승)|원가.{0,5}부담',
 'actual_shortage':r'공급.{0,5}부족|생산.{0,5}중단|운행.{0,5}중단|의약품.{0,8}(?:미보급|보급받지)|진료.{0,8}(?:차질|미운영)|인력난',
 'actual_uncontrolled_hazard':r'긴급.{0,5}조치.{0,5}필요|방호.{0,6}(?:없|부재)|지하.{0,4}공동|안전검사.{0,12}(?:않|미)|도주 중|실종.{0,8}(?:수색|찾|명)',
 'completed_control':r'작업중지|출입.{0,4}금지|체포|구속.{0,4}송치|전원.{0,4}대피|완전히 꺼|완진|수색.{0,4}종료|우려.{0,5}해소',
 'closed_reference':r'형이 확정|판결을 유지|형을 선고|법원.{0,6}판결|\d+년 전|지난해.{0,10}사건|과거.{0,6}사건',
 'fiction':r'영화|뮤지컬|오페라|만화|드라마|배우|줄거리|연기|작품|소설',
 'ceremony':r'기념식|표창|수상|우표|공로|기념행사|장학금|봉사|업무협약|세미나',
 'advice_condition':r'가능성|경우|시나리오|조건|예상|전망|우려|가정|권고|예방',
 'explicit_continuing':r'현재|여전히|계속|이어지고|지속|진행 중|남아 있|해소되지|늘고 있',
}


def features(news):
    path=PRIVATE/'features.npy'
    if path.exists():return np.load(path)
    values=[]
    for row in news.itertuples():
        body=body_for(row)
        values.append([min(1,np.log1p(len(re.findall(p,z)))/np.log(5))
            for z in [row.title,body[:600],body[-800:],body] for p in CUES.values()])
    dense=np.load('data/raw/risk/20261109_v5/features.npy')
    result=np.concatenate([dense,np.asarray(values,dtype=np.float32)],axis=1)
    np.save(path,result)
    write_json(OUT/'feature_manifest.json',{'old_sha256':sha256('data/raw/risk/20261109_v5/features.npy'),
        'cue_patterns':CUES,'shape':list(result.shape),'features_sha256':sha256(path),
        'scores_not_set_by_patterns':True})
    return result


def prepare():
    if (OUT/'protocol.json').exists():raise ValueError('Protocol immutable')
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    news=load_news();d=pd.read_csv(PREVIOUS/'decision_audit.csv')
    dev=d[d.role.ne('quality')].copy();dev['previous_role']=dev.role;dev['role']='development'
    dev.to_csv(OUT/'development.csv',index=False)
    f=pd.read_csv(PREVIOUS/'corpus_families.csv');families=f.family.to_numpy()
    # Merge observed event relationships that escaped the old automatic pass.
    parent={int(i):int(i) for i in set(families)}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    labels=d.set_index('case_id')
    observations=json.loads((PREVIOUS/'pre_prediction_related_observations.json').read_text())
    lookup={n:i for i,n in enumerate(news.news_id)}
    for link in observations['links']:
        ids=[c for c in link['cases']+link.get('development_cases',[]) if c in labels.index]
        indices=[lookup[labels.loc[c].news_id] for c in ids]
        if indices:
            root=find(int(families[indices[0]]))
            for i in indices:parent[find(int(families[i]))]=root
    families=np.array([find(int(i)) for i in families])
    pd.DataFrame({'news_id':news.news_id,'family':families}).to_csv(OUT/'corpus_families.csv',index=False)
    seen=set(families[news.news_id.isin(d.news_id)])
    quality=np.array([incomplete_numeric_draft(body_for(r)) for r in news.itertuples()])
    allowed=np.flatnonzero(news.assessment_research_eligible.to_numpy()&~quality&~np.isin(families,list(seen)))
    old=pd.read_parquet(PREVIOUS/'predictions.parquet');rng=np.random.default_rng(SEED)
    records=[];used=set()
    sources=sorted(s for s in news.source.unique() if s!='SBS')
    for cohort,count in [('random',40),('challenge',6)]:
        for source in sources:
            choices=np.array([i for i in allowed if news.source.iloc[i]==source and families[i] not in used])
            if cohort=='challenge':choices=choices[np.argsort(-old.danger_score.to_numpy()[choices],kind='stable')[:min(30,len(choices))]]
            chosen=[]
            for i in rng.permutation(choices):
                if families[i] in used:continue
                used.add(families[i]);chosen.append(int(i))
                if len(chosen)==count:break
            if len(chosen)!=count:raise ValueError(f'Insufficient novel families for {source}/{cohort}')
            for i in chosen:
                r=news.iloc[i];records.append(dict(case_id=f'v7_a_{len(records)+1:03d}',news_id=r.news_id,
                    source=source,cohort=cohort,role='audit',family=int(families[i]),assessment_body_sha256=r.assessment_body_sha256))
    pd.DataFrame(records).to_csv(OUT/'audit_selection.csv',index=False)
    write_json(OUT/'protocol.json',dict(seed=SEED,prior_v6_superiority_failed=True,
        v6_all_audit_retired_to_development=True,development_articles=len(dev),primary_articles=280,challenge_articles=42,
        primary_population='seven publishers, 40 uniform random articles per source, excluding all previously judged conservative families and incomplete drafts',
        SBS_not_in_confirmatory_audit='only four eligible unjudged articles remained after family purge; training still includes all eight sources',
        primary_endpoint='AI-reference MAE, paired comparisons against both v5 AND v6 fixed original models',
        superiority='both paired stratified 97.5% percentile lower bounds for MAE reduction > 0',
        alpha='two-sided .025 for this second confirmatory attempt; v6 failure remains published, not pooled or erased',
        guards='within10 point estimate, high-risk recall (reference >=30) and challenge MAE must not worsen against either comparison baseline',
        within10_significance_reported_separately=True,bootstrap_resamples=20000,
        stop_rule='one fixed 280+42 audit; no extra samples, retuning, outcome-based exclusions or new primary endpoints after opening',
        model_selection='family grouped five-fold CV on development only; MAE + 4*(1-within10) + 8*high-risk miss rate',
        reference='one executing AI full-body rubric, not human truth, physical risk probability or whole-corpus representative accuracy',
        features='whole-body E5 semantics, char context, explicit present treatment/repair/finance/negation cues as learned inputs',
        files={str(p):sha256(p) for p in [OUT/'development.csv',OUT/'corpus_families.csv',OUT/'audit_selection.csv']}))
    features(news)
    print('Frozen v7 sampling protocol: 280 primary + 42 challenge; 673 development')


class ContextTree:
    def __init__(self,loss='absolute_error',quantile=None,high_weight=1.,full_weight=1.,leaves=7):
        self.loss,self.quantile,self.high_weight,self.full_weight,self.leaves=loss,quantile,high_weight,full_weight,leaves

    def transform(self,x):
        return np.concatenate([self.pca.transform(x[:,:1152]),x[:,1152:]],axis=1)

    def fit(self,x,y,groups,full):
        self.pca=PCA(n_components=min(24,len(x)-1),svd_solver='randomized',random_state=SEED).fit(x[:,:1152])
        target=danger_index(y[:,3:]);_,inv,n=np.unique(groups,return_inverse=True,return_counts=True)
        w=(1/np.sqrt(n[inv]))*np.where(full,self.full_weight,1.)*np.where(target>=30,self.high_weight,1.);w/=w.mean()
        self.reg=HistGradientBoostingRegressor(loss=self.loss,quantile=self.quantile,max_iter=220,
            learning_rate=.06,max_leaf_nodes=self.leaves,min_samples_leaf=10,l2_regularization=1.,
            early_stopping=False,random_state=SEED).fit(self.transform(x),target,sample_weight=w)
        return self

    def predict(self,x):return np.clip(self.reg.predict(self.transform(x)),0,100)


class KernelRisk:
    def __init__(self,lexical_weight=.25,high_weight=1.,mode='plain',full_weight=1.):
        self.lexical_weight,self.high_weight,self.mode,self.full_weight=lexical_weight,high_weight,mode,full_weight

    def fit(self,x,texts,y,groups,full):
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=60000,min_df=2,sublinear_tf=True)
        self.lex=self.vectorizer.fit_transform(texts);self.x=x.copy();k=self.kernel(x,self.lex)
        target=danger_index(y[:,3:])/100;_,inv,n=np.unique(groups,return_inverse=True,return_counts=True)
        w=1/np.sqrt(n[inv])*np.where(full,self.full_weight,1.)*np.where(target>=.3,self.high_weight,1.);w/=w.mean()
        self.reg=SVR(C=8.,epsilon=.02,kernel='precomputed').fit(k,target,sample_weight=w)
        present=target>0;self.constant=bool(present[0]) if len(np.unique(present))==1 else None
        if self.mode=='gated' and self.constant is None:self.gate=SVC(C=4.,kernel='precomputed').fit(k,present,sample_weight=w)
        return self

    def kernel(self,x,lex):return (1-self.lexical_weight)*rbf_kernel(x,self.x,gamma=.2)+self.lexical_weight*(lex@self.lex.T).toarray()

    def predict(self,x,texts):
        k=self.kernel(x,self.vectorizer.transform(texts));p=np.clip(self.reg.predict(k),0,1)*100
        if self.mode=='gated':
            present=np.full(len(x),self.constant) if self.constant is not None else self.gate.predict(k)
            p=np.where(present,p,0.)
        return p


class Blend:
    def __init__(self,tree,kernel,weight):self.tree,self.kernel,self.weight=tree,kernel,weight
    def predict(self,x,texts):return self.weight*self.tree.predict(x)+(1-self.weight)*self.kernel.predict(x,texts)


def train():
    if (OUT/'pre_audit_freeze.json').exists() or (OUT/'audit_judgments.tsv').exists():raise ValueError('Frozen/opened experiment')
    news=load_news();d=pd.read_csv(OUT/'development.csv');f=pd.read_csv(OUT/'corpus_families.csv').set_index('news_id')
    groups=f.loc[d.news_id].family.to_numpy();test=pd.read_csv(OUT/'audit_selection.csv');assert not set(groups)&set(test.family)
    x=features(news);lookup={n:i for i,n in enumerate(news.news_id)};ix=np.array([lookup[n] for n in d.news_id])
    y=d[AXES].to_numpy(float);target=danger_index(y[:,3:]);full=d.input_scope.fillna('').eq('complete_stored_cleaned_body_and_title').to_numpy()
    texts=np.array([(r.title+'\n')*2+body_for(r) for r in news.itertuples()])
    folds=list(GroupKFold(n_splits=5).split(ix,groups=groups));_,inv,counts=np.unique(groups,return_inverse=True,return_counts=True);evalw=1/counts[inv]
    tree_configs=[dict(loss=loss,quantile=q,high_weight=h,full_weight=2.,leaves=l)
        for loss,q in [('absolute_error',None),('quantile',.65),('squared_error',None)] for h in [1.,2.] for l in [7,15]]
    kernel_configs=[dict(lexical_weight=l,high_weight=h,mode=m,full_weight=2.) for l in [.25,.5] for h in [1.,2.] for m in ['plain','gated']]
    write_json(OUT/'training_plan.json',{'trees':tree_configs,'kernels':kernel_configs,'blend_tree_weights':[0.,.25,.5,.75,1.],
        'folds':5,'source_sha256':sha256(__file__),'audit_labels_used':0,'loss':'family-balanced MAE + 4*(1-within10) + 8*high-risk miss rate'})
    tree_oof=[];kernel_oof=[]
    for params in tree_configs:
        p=np.empty(len(ix))
        for tr,va in folds:
            model=ContextTree(**params).fit(x[ix[tr]],y[tr],groups[tr],full[tr]);p[va]=model.predict(x[ix[va]])
        tree_oof.append(p);print('tree',params,float(np.abs(target-p).mean()),flush=True)
    fold_kernels=[]
    for tr,va in folds:
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=60000,min_df=2,sublinear_tf=True)
        lt=vectorizer.fit_transform(texts[ix[tr]]);lv=vectorizer.transform(texts[ix[va]])
        fold_kernels.append((tr,va,rbf_kernel(x[ix[tr]],gamma=.2),rbf_kernel(x[ix[va]],x[ix[tr]],gamma=.2),(lt@lt.T).toarray(),(lv@lt.T).toarray()))
    for params in kernel_configs:
        p=np.empty(len(ix));weight=params['lexical_weight']
        for tr,va,st,sv,lt,lv in fold_kernels:
            ktr=(1-weight)*st+weight*lt;kva=(1-weight)*sv+weight*lv
            _,inverse,n=np.unique(groups[tr],return_inverse=True,return_counts=True)
            w=1/np.sqrt(n[inverse])*np.where(full[tr],2.,1.)*np.where(target[tr]>=30,params['high_weight'],1.);w/=w.mean()
            reg=SVR(C=8.,epsilon=.02,kernel='precomputed').fit(ktr,target[tr]/100,sample_weight=w)
            pred=np.clip(reg.predict(kva),0,1)*100
            if params['mode']=='gated':
                present=SVC(C=4.,kernel='precomputed').fit(ktr,target[tr]>0,sample_weight=w).predict(kva)
                pred=np.where(present,pred,0.)
            p[va]=pred
        kernel_oof.append(p);print('kernel',params,float(np.abs(target-p).mean()),flush=True)
    results=[];oofs=[]
    for ti,tp in enumerate(tree_oof):
      for ki,kp in enumerate(kernel_oof):
       for weight in [0.,.25,.5,.75,1.]:
        pred=weight*tp+(1-weight)*kp;error=np.abs(pred-target)
        mae=float(np.average(error,weights=evalw));agree=float(np.average(error<=10,weights=evalw));high=target>=30
        miss=float(np.average(pred[high]<30,weights=evalw[high]))
        results.append(dict(tree=ti,kernel=ki,tree_weight=weight,MAE=mae,within10=agree,high_miss=miss,loss=mae+4*(1-agree)+8*miss));oofs.append(pred)
    selected_index=min(range(len(results)),key=lambda i:results[i]['loss']);best=results[selected_index]
    tree=ContextTree(**tree_configs[best['tree']]).fit(x[ix],y,groups,full)
    kernel=KernelRisk(**kernel_configs[best['kernel']]).fit(x[ix],texts[ix],y,groups,full)
    model=Blend(tree,kernel,best['tree_weight']);heads=Ridge(alpha=1.,solver='lsqr').fit(x[ix],y)
    pred=news[['news_id','source','url','title','summary','published_date','event_group','body_sha256','assessment_body_sha256','assessment_research_eligible','input_correction']].copy()
    risk=np.concatenate([model.predict(x[i:i+128],texts[i:i+128]) for i in range(0,len(news),128)])
    axes=np.clip(heads.predict(x),0,4)
    for j,a in enumerate(AXES):pred[a]=axes[:,j]/4
    pred['danger_score']=risk;pred['component_index']=danger_index(axes[:,3:]);pred['session_reviewed']=False;pred['score_source']='AI_supervised_v7_model_prediction'
    pred.to_parquet(OUT/'predictions.parquet',index=False);joblib.dump({'risk':model,'heads':heads},OUT/'model.joblib',compress=3)
    pd.DataFrame({'news_id':d.news_id,'family':groups,'reference':target,'out_of_fold_prediction':oofs[selected_index]}).to_csv(OUT/'development_oof.csv',index=False)
    write_json(OUT/'training.json',dict(selected=best,selected_tree=tree_configs[best['tree']],selected_kernel=kernel_configs[best['kernel']],candidates=results,development_articles=len(d),audit_labels_used=0))
    paths=[OUT/p for p in ['protocol.json','audit_selection.csv','corpus_families.csv','development.csv','training_plan.json','training.json','feature_manifest.json','model.joblib','predictions.parquet']]
    write_json(OUT/'pre_audit_freeze.json',{'files':{str(p):sha256(p) for p in paths},'source_sha256':sha256(__file__),'features_sha256':sha256(PRIVATE/'features.npy'),'audit_judgments_opened':0})
    print('FROZEN',json.dumps(best),flush=True)


def packet(start=0,count=6):
    if not (OUT/'pre_audit_freeze.json').exists():raise ValueError('Freeze model before audit')
    selection=pd.read_csv(OUT/'audit_selection.csv').iloc[start:start+count];news=load_news().set_index('news_id');records=[]
    for r in selection.itertuples():
        n=news.loc[r.news_id];records.append(dict(case_id=r.case_id,title=n.title,body=body_for(n),assessment_body_sha256=r.assessment_body_sha256))
    path=PRIVATE/f'packet_audit_{start:03d}.json'
    if path.exists():assert json.loads(path.read_text())==records
    else:
        write_json(path,records);write_json(OUT/f'packet_audit_{start:03d}_receipt.json',dict(sha256=sha256(path),cases=selection.case_id.tolist(),scope='complete cleaned body and title; no model scores'))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','packet']);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=6);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train()
    else:print(json.dumps(packet(a.start,a.count),ensure_ascii=False))


if __name__=='__main__':main()
