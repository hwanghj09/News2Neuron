"""Remove a discovered native promotion widget without tuning frozen models.

The immutable v2 corpus and pre-audit predictions stay intact. None of the 180
affected articles belongs to development, so all training inputs and model
parameters remain identical. Corrected inference is a separately named sidecar.
"""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import re

import joblib
from lxml import html
import numpy as np
import pandas as pd

from news_flywire.corpus.extract import SELECTORS, clean_body
from news_flywire.provenance import sha256,write_json
from news_flywire.risk_assessment import AXES,NARRATIVE_CUES,danger_index,fit_predict,risk_band,scaled_features
from news_flywire.risk_features import Encoder,PROBES,normalize


def remove_promotion(payload):
    tree=html.fromstring(payload.decode('utf-8'))
    candidates=tree.xpath(SELECTORS['Seoul'])
    if not candidates:
        raise ValueError('Missing original Seoul article container')
    node=copy.deepcopy(max(candidates,key=lambda e:len(e.text_content())))
    before=clean_body(node)[0]
    widgets=node.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," sportLightPromotion ") or contains(concat(" ",normalize-space(@class)," ")," spotLightPromotion ")]')
    for widget in widgets:
        widget.drop_tree()
    return before,clean_body(node)[0],len(widgets)


def main():
    out=Path('data/risk/20261109_v3');cache=Path('data/raw/risk/20261109_v3')
    raw=Path('data/raw/corpus/20261109_v2')
    news=pd.read_parquet('data/corpus/20261109_v2/news.parquet')
    selected=pd.read_csv(out/'selected_cases.csv')
    encoder=Encoder('external/multilingual-e5-small')
    probes=encoder.single(['query: '+text for text in PROBES])
    features=np.load(cache/'features.npy');titles=np.load(cache/'titles.npy')
    narrative=np.load(cache/'narrative.npy');patterns=[re.compile(v) for v in NARRATIVE_CUES.values()]
    corrections=[];changed=[];window_total=0
    for i,r in enumerate(news.itertuples()):
        if r.source!='Seoul':continue
        payload=gzip.decompress((raw/'http/responses'/(r.html_sha256+'.gz')).read_bytes())
        if b'sportLightPromotion' not in payload and b'spotLightPromotion' not in payload:continue
        before,body,widgets=remove_promotion(payload)
        if not widgets:continue
        assert hashlib.sha256(before.encode()).hexdigest()==r.body_sha256
        assert 80<=len(body)<len(before)
        digest=hashlib.sha256(body.encode()).hexdigest()
        target=cache/'corrected_bodies'/(digest+'.txt.gz');target.parent.mkdir(exist_ok=True)
        target.write_bytes(gzip.compress(body.encode(),mtime=0))
        # Same tokenization/pooling as the frozen encoder; only the unrelated
        # publisher promotion is removed. Original article prose remains.
        text='passage: '+r.title+'\n'+body
        encoded=encoder.tokenizer.encode(text);chunks=[encoded]+list(encoded.overflowing)
        vectors=np.concatenate([encoder.encode_chunks(chunks[j:j+16]) for j in range(0,len(chunks),16)])
        similarities=vectors@probes.T
        chosen=int(np.argmax(similarities[:,:5].max(axis=1)-similarities[:,5:].max(axis=1)))
        features[i]=np.concatenate([titles[i],normalize(vectors.mean(axis=0)),vectors[chosen],similarities.max(axis=0),similarities.mean(axis=0)])
        narrative[i]=[min(1,np.log1p(len(p.findall(zone)))/np.log(6)) for zone in [r.title,body[:500],body] for p in patterns]
        budget=max(0,25-len(r.title.split()))
        corrections.append({'news_id':r.news_id,'html_sha256':r.html_sha256,'original_body_sha256':r.body_sha256,
                            'assessment_body_sha256':digest,'removed_chars':len(before)-len(body),
                            'assessment_body_chars':len(body),'widgets_removed':widgets,'complete_scoring_windows':len(chunks),
                            'assessment_research_eligible':len(body)>=400 and len(re.findall(r'(?:다[.!?]|요[.!?]|[.!?]\s)',body))>=3,
                            'summary':' '.join(body.split()[:budget]),'method':'drop_only_Seoul_sportLightPromotion_DOM_subtree'})
        changed.append(i);window_total+=len(chunks)
        if len(changed)%30==0:print('Corrected and encoded promotion-free articles',len(changed),flush=True)
    table=pd.DataFrame(corrections);table.to_csv(out/'input_corrections.csv',index=False)
    affected_training=selected[selected.role.eq('development') & selected.news_id.isin(table.news_id)]
    assert affected_training.empty,'A new development experiment is needed if training inputs change'
    np.save(cache/'scoring_features_corrected.npy',features);np.save(cache/'scoring_narrative_corrected.npy',narrative)
    x=np.concatenate([scaled_features(features),narrative*.5],axis=1)
    old_x=np.concatenate([scaled_features(np.load(cache/'features.npy')),np.load(cache/'narrative.npy')*.5],axis=1)
    lookup={n:i for i,n in enumerate(news.news_id)}
    fixed_weights={}
    for round_number in [1,2,3]:
        folder=out/f'round_{round_number}'
        original=pd.read_parquet(folder/'predictions.parquet');final=original.copy()
        training=pd.read_csv(folder/'development_judgments.tsv',sep='\t')
        indices=np.array([lookup[n] for n in training.news_id])
        assert np.array_equal(x[indices],old_x[indices])
        model=joblib.load(folder/'ridge_model.joblib')
        before_hash=sha256(folder/'ridge_model.joblib')
        settings=json.loads((folder/'training.json').read_text())
        recreated,pred,ensemble=fit_predict(x,indices,training[AXES].to_numpy(float),settings['alpha'],training.event_group.to_numpy())
        assert np.allclose(model.coef_,recreated.coef_,rtol=0,atol=1e-12)
        assert np.allclose(model.intercept_,recreated.intercept_,rtol=0,atol=1e-12)
        pred=np.clip(model.predict(x),0,4)
        risk=danger_index(pred[:,3:]);bootstrap=danger_index(ensemble[:,:,3:])
        sensitivity=[]
        for multiplier,floor in [(0.75,.25),(1.,.5),(1.25,.75)]:
            s,l,t,e=pred[:,3:].T
            sensitivity.append(np.minimum(100,100*np.minimum(1,s/4*multiplier)*(l/4)*(floor+(1-floor)*t/4)*(.8+.2*e/4)))
        for j,axis in enumerate(AXES):
            final.loc[changed,axis+'_ordinal_prediction']=pred[changed,j]
            final.loc[changed,axis]=pred[changed,j]/4
        final.loc[changed,'danger_score']=risk[changed]
        final.loc[changed,'danger_band']=risk_band(risk[changed])
        final.loc[changed,'bootstrap_p10']=np.quantile(bootstrap,.1,axis=0)[changed]
        final.loc[changed,'bootstrap_p90']=np.quantile(bootstrap,.9,axis=0)[changed]
        final.loc[changed,'rubric_sensitivity_min']=np.min(sensitivity,axis=0)[changed]
        final.loc[changed,'rubric_sensitivity_max']=np.max(sensitivity,axis=0)[changed]
        final['assessment_body_sha256']=news.body_sha256
        final['input_correction']='none'
        final['assessment_research_eligible']=news.research_subset_selected
        for i,record in zip(changed,corrections):
            final.loc[i,'assessment_body_sha256']=record['assessment_body_sha256']
            final.loc[i,'input_correction']='removed_Seoul_sportLightPromotion'
            final.loc[i,'assessment_research_eligible']=bool(record['assessment_research_eligible']) and bool(news.research_subset_selected.iloc[i])
            final.loc[i,'summary']=record['summary']
            final.loc[i,'rationale']=f"홍보 삽입물을 제거한 본문 의미 예측: 심각도 {pred[i,3]:.1f}/4, 잔존 위해 근거 {pred[i,4]:.1f}/4, 임박성 {pred[i,5]:.1f}/4, 범위 {pred[i,6]:.1f}/4. 직접 세션 판정 아님."
        final['requires_further_ai_review']=(final.bootstrap_p90-final.bootstrap_p10).gt(20)|final.nearest_title_similarity.lt(.72)|~final.assessment_research_eligible
        final.to_parquet(folder/'input_corrected_predictions.parquet',index=False)
        assert before_hash==sha256(folder/'ridge_model.joblib')
        fixed_weights[str(round_number)]=before_hash
    write_json(out/'input_correction_manifest.json',{'articles_corrected':len(table),'training_articles_changed':0,
               'assessment_body_windows':window_total,'corrections_sha256':sha256(out/'input_corrections.csv'),
               'corrected_features_sha256':sha256(cache/'scoring_features_corrected.npy'),
               'corrected_narrative_sha256':sha256(cache/'scoring_narrative_corrected.npy'),
               'frozen_model_hashes_unchanged':fixed_weights,
               'reason':'Native Seoul promotion widget inserted unrelated prose within an otherwise valid article',
               'evaluation_caveat':'Structural input repair discovered during AI audit, applied by identical DOM rule to every Seoul article. No labels, model parameters or hyperparameters were changed; this is not a wholly untouched preprocessing holdout.'})
    print('Repaired scoring inputs:',len(table),'articles; frozen training rows and parameters unchanged',flush=True)


if __name__=='__main__':main()
