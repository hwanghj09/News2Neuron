"""Recheck every released row against private text hashes and cached publisher responses."""
import argparse
import gzip
import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from news_flywire.corpus.extract import canonicalize,extract_article
from news_flywire.corpus.report import report
from news_flywire.provenance import sha256,write_json

p=argparse.ArgumentParser()
p.add_argument('--out',default='data/corpus/20261109_v2')
p.add_argument('--raw',default='data/raw/corpus/20261109_v2')
a=p.parse_args()
out,raw=Path(a.out),Path(a.raw)
config=json.loads((out/'resolved_config.json').read_text())
news=pd.read_parquet(out/'news.parquet')
ledger=pd.read_parquet(out/'extraction_ledger.parquet').fillna('')
splits=pd.read_csv(out/'splits.csv')
assert news.news_id.is_unique and news.url.is_unique
assert news.quality_status.eq('accepted').all()
assert news.exclusion_reasons.fillna('').eq('').all()
assert pd.to_numeric(news.body_chars).ge(config['min_body_chars']).all()
assert news.date_precision.eq('timestamp').all()
dates=pd.to_datetime(news.published_date,utc=True)
assert dates.dt.strftime('%Y-%m-%d').between(config['start_date'],config['end_date']).all()
assert news.korean_letter_ratio.ge(config['min_korean_letter_ratio']).all()
assert max(len(r.title.split())+len(r.summary.split()) for r in news.itertuples())<=25
assert splits.news_id.is_unique and set(splits.news_id)==set(news.news_id)
assert splits.groupby('event_group').split.nunique().max()==1
for name,minimum,maximum in [('train',config['start_date'],config['train_end']),('validation','2026-10-12',config['validation_end']),('test','2026-10-26',config['end_date'])]:
    subset=splits[splits.split.eq(name)]
    assert pd.to_datetime(subset.published_date,utc=True).dt.strftime('%Y-%m-%d').between(minimum,maximum).all()
groups=news.groupby('event_group').published_date.agg(['min','max'])
assert ((pd.to_datetime(groups['max'],utc=True)-pd.to_datetime(groups['min'],utc=True)).dt.total_seconds()<=config['event_max_span_hours']*3600).all()
dedup=pd.read_parquet(out/'deduplication_ledger.parquet')
assert set(dedup.loc[dedup.duplicate_of.ne(''),'duplicate_of']).issubset(set(news.news_id))
publishers={s['name']:s for s in config['publishers']}
candidate=pd.read_parquet(out/'fetch_plan.parquet').set_index('candidate_id')
receipt_count=0;recomputed=0;hashes=set();flags=Counter();unicode_ratios=[]
for r in news.to_dict('records'):
    body=gzip.decompress((raw/'bodies'/(r['body_sha256']+'.txt.gz')).read_bytes()).decode()
    assert hashlib.sha256(body.encode()).hexdigest()==r['body_sha256']
    assert len(body)==r['body_chars']
    unicode_ratio=sum('가'<=c<='힣' for c in body)/max(1,sum(c.isalpha() for c in body))
    assert unicode_ratio>=config['min_korean_letter_ratio']
    unicode_ratios.append(unicode_ratio)
    assert r['body_sha256'] not in hashes
    hashes.add(r['body_sha256'])
    for noise in ['구글에서 서울신문 먼저 보기','이미지 확대 보기','◎공감언론 뉴시스','☞공감언론 뉴시스']:
        assert noise not in body,f'Known contamination remains: {r["news_id"]} {noise}'
    requested=r['requested_url']
    receipt_path=raw/'http/receipts'/(hashlib.sha256(requested.encode()).hexdigest()+'.json')
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']==200 and receipt['sha256']==r['html_sha256']
    receipt_count+=1
    payload=gzip.decompress((raw/'http/responses'/(r['html_sha256']+'.gz')).read_bytes())
    assert hashlib.sha256(payload).hexdigest()==r['html_sha256']
    source=candidate.loc[r['candidate_id']].to_dict()
    source['candidate_id']=r['candidate_id']
    source['allowed_hosts']=publishers[r['source']]['allowed_hosts']
    row,body2=extract_article(payload,source,receipt,config)
    for field in ['news_id','url','title','summary','published_date','modified_date','date_precision','publication_source',
                  'body_sha256','quality_status','quality_flags','body_extractor','category','quoted_words','korean_letter_ratio']:
        assert row[field]==r[field],f'Extracted field changed: {r["news_id"]} {field}'
    recomputed+=1
    flags.update(filter(None,r['quality_flags'].split('|')))
    if recomputed%500==0:print(f'Verified original HTML and clean-body hashes {recomputed}/{len(news)}',flush=True)
selected=news[news.research_subset_selected]
assert selected.source.value_counts(normalize=True).max()<=config['max_research_source_share']+1e-12
assert selected.split.isin(['train','validation','test']).all()
assert not selected.known_legacy_pilot_overlap.any()
for rater in range(1,4):
    sheet=pd.read_csv(out/f'blind_rater_{rater}_template.csv')
    assert sheet[['N','T','A']].isna().all().all()
    assert not any(c in sheet for c in ['N_ai','T_ai','A_ai','N_rule','T_rule'])
manifest=json.loads((out/'manifest.json').read_text())
for path,digest in manifest['files'].items():assert sha256(path)==digest,f'Run input/code hash mismatch: {path}'
result={'status':'passed','unique_articles_checked':len(news),'publisher_responses_hash_verified':receipt_count,
    'auditor_sha256':sha256(__file__),
    'articles_reextracted_offline_with_identical_results':recomputed,
    'published_window':'2026-08-12 through 2026-11-09 UTC','known_site_controls_remaining':0,
    'computed_event_group_split_leakage':0,'event_group_max_hours':config['event_max_span_hours'],
    'research_source_share_cap':config['max_research_source_share'],'body_hashes_unique':True,
    'minimum_hangul_ratio_among_all_unicode_letters':min(unicode_ratios),
    'three_independent_rater_templates_blank':True,'independent_human_judgments_completed':0,
    'limitations':'These are automated provenance/consistency checks. They do not validate semantic event identity, news claims, labels or biological responses.'}
write_json(out/'verification.json',result)
print(report(out))
