"""Bind this session's recorded decisions to an immutable real-news input snapshot.

This formats already recorded AI judgments; it does not call an LLM API or
represent independent human annotation.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from news_flywire.news_scoring import import_llm_scores
from news_flywire.provenance import manifest, sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
dataset=ROOT/'data/real_news/2026-11-09'
news_path=dataset/'news.csv'
decisions_path=dataset/'ratings/contextual_decisions.tsv'
news=pd.read_csv(news_path,dtype=str,keep_default_na=False)
decisions=pd.read_csv(decisions_path,sep='\t',keep_default_na=False)
snapshot=json.loads((dataset/'ratings/input_snapshot.json').read_text())
payload=json.dumps(news[snapshot['columns']].to_dict('records'),ensure_ascii=False,separators=(',',':'))
assert hashlib.sha256(payload.encode()).hexdigest()==snapshot['ordered_content_sha256'],'Article inputs changed: these decisions cannot be applied to another collection'
assert news.news_id.tolist()==snapshot['ordered_news_ids'],'Article order differs from the evaluated snapshot'
assert len(news)==240
assert decisions['index'].tolist()==list(range(1,len(news)+1)),'Missing/duplicated recorded article decisions'
assert decisions[['N','T','A']].isin([0,1,2,3,4]).all().all()
assert decisions.rationale.str.len().min()>8
before=sha256(news_path)

# Preserve automatic candidate links, then conservatively merge explicitly
# identified follow-up coverage. Broad multi-event bulletins connect all tags.
parent=list(range(len(news)))
def find(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]]
        i=parent[i]
    return i
by_auto={}
for i,group in enumerate(news.event_group):
    if group in by_auto:parent[find(i)]=find(by_auto[group])
    else:by_auto[group]=i
by_tag={}
for i,row in decisions.iterrows():
    for tag in row.event_key.split(';'):
        if tag.startswith('nvidia_'):tag='nvidia_related_20261109'
        if tag=='basketball_standings_20261109':tag='samsung_kt_basketball_20261109'
        if tag in by_tag:parent[find(i)]=find(by_tag[tag])
        else:by_tag[tag]=i
members={}
for i in range(len(news)):members.setdefault(find(i),[]).append(news.news_id[i])
labels={root:'event_'+hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest()[:12] for root,ids in members.items()}
news['event_group']=[labels[find(i)] for i in range(len(news))]
news.to_csv(news_path,index=False)
audit=decisions.copy()
audit['news_id']=news.news_id
audit['event_group']=news.event_group
audit['evaluated_text_sha256']=[hashlib.sha256((r.title+'\n'+r.summary).encode()).hexdigest() for r in news.itertuples()]
audit.to_csv(dataset/'ratings/decision_audit.csv',index=False)
jsonl=dataset/'ratings/codex_nta_v1.jsonl'
with jsonl.open('w',encoding='utf-8') as f:
    for i,row in decisions.iterrows():
        item={'news_id':news.news_id[i],**{axis:int(row[axis]) for axis in 'NTA'},'confidence':None,'rationale':row.rationale}
        f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
model='codex_session_nta_v1_20261109'
scored=import_llm_scores(jsonl,news,model)
scores_path=ROOT/'data/derived/real_news_scores.csv'
scored.to_csv(scores_path,index=False)
write_json(str(scores_path)+'.manifest.json',manifest([news_path,jsonl,decisions_path,dataset/'ratings/input_snapshot.json'],method=model,
    evaluated_against_humans=False,scoring_mode='interactive_Codex_session_content_judgment',
    exact_model_snapshot='not exposed to this session',temperature='not exposed; no API decoding setting claimed',
    scoring_reference='docs/labeling-rubric.md; raw N/T/A integer anchors 0..4',
    temporal_reference='article publication narrative, not a current real-world alert',
    source_scope='Title and short RSS excerpt only; no full article or fact-check claim',
    confidence='null; neither confidence calibration nor accuracy measured',
    decision_formatting_script='Reformats recorded judgments; does not rerun an LLM',
    human_raters=0))
collection=json.loads((dataset/'manifest.json').read_text())
collection.update(news_csv_sha256=sha256(news_path),n_events=int(news.event_group.nunique()),
    contextual_event_group_review={'reviewer':'current Codex session; not independent human review',
        'before_news_sha256':before,'decisions_sha256':sha256(decisions_path),
        'method':'union of automatic similar pairs and explicitly identified conservative related-event tags; multi-event bulletins merge all related tags',
        'limitation':'Conservative topic clusters can merge different events; independent human audit remains pending'})
write_json(dataset/'manifest.json',collection)
print(f'Recorded contextual AI scores for {len(scored)} real articles; {news.event_group.nunique()} conservative event groups.')
