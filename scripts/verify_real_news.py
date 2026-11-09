"""Audit real article inputs, recorded judgments, group splits and shared simulation links."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from news_flywire.news_ingestion import parse_feed
from news_flywire.provenance import sha256, write_json

p=argparse.ArgumentParser()
p.add_argument('--require-source-cache',action='store_true')
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
dataset=ROOT/'data/real_news/2026-11-09'
out=ROOT/'results/real_news_20261109'
news=pd.read_csv(dataset/'news.csv',dtype=str,keep_default_na=False)
assert len(news)==240 and news.news_id.nunique()==240 and news.url.nunique()==240
assert news.data_source.eq('real_publisher_rss').all()
assert news.groupby('category').size().eq(40).all()
assert len(news.source.unique())==3
assert max(len(r.title.split())+len(r.summary.split()) for r in news.itertuples())<=25
meta=json.loads((dataset/'manifest.json').read_text())
assert sha256(dataset/'news.csv')==meta['news_csv_sha256']
snapshot=json.loads((dataset/'ratings/input_snapshot.json').read_text())
payload=json.dumps(news[snapshot['columns']].to_dict('records'),ensure_ascii=False,separators=(',',':'))
assert hashlib.sha256(payload.encode()).hexdigest()==snapshot['ordered_content_sha256']
audit=pd.read_csv(dataset/'ratings/decision_audit.csv')
assert audit.news_id.tolist()==news.news_id.tolist()
for i,row in news.iterrows():
    assert hashlib.sha256((row.title+'\n'+row.summary).encode()).hexdigest()==audit.evaluated_text_sha256[i]
scores=pd.read_csv(ROOT/'data/derived/real_news_scores.csv')
assert scores.news_id.tolist()==news.news_id.tolist()
for axis in 'NTA':np.testing.assert_array_equal(scores[axis]*4,audit[axis])
assert scores.rationale.notna().all() and scores.confidence.isna().all()
splits=pd.read_csv(ROOT/'data/labels/real_news_20261109/splits.csv')
assert set(splits.news_id)==set(news.news_id)
assert splits.groupby('event_group').split.nunique().eq(1).all()
ratings=pd.read_csv(ROOT/'data/labels/real_news_20261109/ratings_template.csv')
assert len(ratings)==720 and ratings[['N','T','A']].isna().all().all()
linked=pd.read_csv(out/'article_responses.csv')
assert len(linked)==7200 and linked.groupby('news_id').seed.nunique().eq(30).all()
assert linked.shared_level_simulation.all()
original=pd.read_csv(ROOT/'results/v783/metrics.csv')
reference=original.set_index(['condition_id','seed']).delta_rate_hz
for row in linked.itertuples():assert np.isclose(row.delta_rate_hz,reference[row.condition_id,row.seed])
observed_inputs={}
cache_files=[]
for path,digest in meta['files'].items():
    cache=ROOT/path if not Path(path).is_absolute() else Path(path)
    if cache.exists():
        assert sha256(cache)==digest,f'Feed cache changed: {path}'
        cache_files.append(cache)
if a.require_source_cache:
    assert len(cache_files)==len(meta['files']),'Original RSS responses are required for this audit'
if len(cache_files)==len(meta['files']):
    logs=json.loads((dataset/'collection_log.json').read_text())
    by_hash={sha256(path):path for path in cache_files}
    for log in logs:
        if log['status']!='ok':continue
        rows,_=parse_feed(by_hash[log['sha256']].read_bytes(),log['source'],log['category'],log['url'],
            datetime.fromisoformat(meta['collection_utc']),meta['max_quoted_words_per_article'])
        for row in rows:observed_inputs[row['feed_url'],row['news_id']]=row
    for row in news.to_dict('records'):
        original=observed_inputs[row['feed_url'],row['news_id']]
        for key in ['title','summary','url','published_date','source']:
            assert row[key]==original[key],f'Article differs from actual RSS response: {row["news_id"]} {key}'
write_json(out/'verification.json',{
    'status':'passed','actual_articles':240,'official_publishers':3,'categories':6,
    'all_240_inputs_matched_to_cached_official_rss':bool(observed_inputs),
    'rss_snapshots_verified':len(cache_files),'snapshot_and_decision_hashes':'verified',
    'contextual_scores_with_rationales':240,'independent_human_ratings':0,
    'event_group_split_leakage':0,'test_articles':int(splits.split.eq('test').sum()),
    'linked_model_rows':len(linked),'new_independent_simulation_trials':0,
    'simulation_values_match_existing_30_seed_results':True,
    'scope':'Input provenance and output consistency; not model accuracy or biological validity'})
print('240 real inputs, recorded ratings, group splits and 7,200 shared-model links verified.')
