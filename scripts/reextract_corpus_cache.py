"""Apply audited extractor fixes to every saved response without network access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from news_flywire.corpus.build import fetch_and_extract
from news_flywire.corpus.http import CachedHTTP, FetchFailure
from news_flywire.provenance import write_json,sha256

p=argparse.ArgumentParser()
p.add_argument('--out',default='data/corpus/20261109_v2')
p.add_argument('--raw',default='data/raw/corpus/20261109_v2')
a=p.parse_args()
out,raw=Path(a.out),Path(a.raw)
config=json.loads((out/'resolved_config.json').read_text())
plan=pd.read_parquet(out/'fetch_plan.parquet')
state=raw/'extraction_results.jsonl'
rows=state.read_text().splitlines()
if len(rows)!=len(plan):raise SystemExit('Initial extraction state is incomplete')
backup=raw/'stage1_extraction_results.jsonl.gz'
if backup.exists():raise SystemExit('Offline re-extraction already started; preserve the stage1 snapshot')
backup.write_bytes(gzip.compress(state.read_bytes(),mtime=0))
state.unlink()
class OfflineHTTP(CachedHTTP):
    def _request(self,url,max_bytes):
        receipt=self.root/'receipts'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
        if not receipt.exists():raise FetchFailure('original_response_cache_unavailable')
        return super()._request(url,max_bytes)

result=fetch_and_extract(plan,config,OfflineHTTP(config,raw/'http'),out,raw)
write_json(out/'offline_reextraction.json',{'status':'complete','network_requests':0,
  'original_response_count':len(plan),'initial_state_sha256':sha256(backup),
  'extractor_sha256':sha256('src/news_flywire/corpus/extract.py'),
  'reason':'AI quality spotcheck found photo captions, site controls, related-article cards, bylines and video-only teasers; deterministic cleanup applied to all cached inputs',
  'phase2_accepted_before_deduplication':int(result.quality_status.eq('accepted').sum())})
