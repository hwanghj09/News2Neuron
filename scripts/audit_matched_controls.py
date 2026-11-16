"""Replay matched controls and verify matching balance and paired raw events."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from news_flywire.article_controls import OUT,verify,signatures,subset_events,match_inputs
from news_flywire.article_stimulus import event_hash
from news_flywire.brain_simulation import simulate,response_metrics,SimulationResult
from news_flywire.flywire_graph import Circuit
from news_flywire.provenance import sha256,write_json

p=verify(OUT);parent=Path(p['parent']);c=Circuit.load(p['circuit']);s=signatures(c)
selection=pd.read_csv(OUT/'selection.csv');controls=json.loads((OUT/'controls.json').read_text());sim=json.loads((parent/'resolved_config.json').read_text())['simulation']
m=pd.read_csv(OUT/'metrics.csv');manifest=json.loads((OUT/'run_manifest.json').read_text())
assert len(m)==120 and len(selection)==8 and not m.duplicated(['news_id','seed','condition']).any()
for path,digest in {**manifest['files'],**manifest['receipts']}.items():assert sha256(path)==digest
for path in manifest['receipts']:
 for file,digest in json.loads(Path(path).read_text())['files'].items():assert sha256(file)==digest
keys=['GF_hops','DNp06_hops','top_nt']
for control in controls:
 again=match_inputs(c,control['seed'])
 for key in ['common_target_ids','common_control_ids','approximate_control_ids']:assert again[key]==control[key]
 a,b=control['common_target_ids'],control['common_control_ids']
 assert len(a)==len(b)==len(set(b))==178
 assert not set(b)&set(c.cells(role='input')+c.cells(role='output'))
 assert np.array_equal(s.loc[a,keys],s.loc[b,keys])
 assert len(control['approximate_control_ids'])==len(set(control['approximate_control_ids']))==314
 assert s.loc[control['approximate_control_ids'],'top_nt'].eq('acetylcholine').all()
 assert sum(r['common_n'] for r in control['feasibility'])==178
checked=0
for article in selection.itertuples():
 folder=OUT/'trials'/article.news_id
 e=pd.read_parquet(folder/'events.parquet');sp=pd.read_parquet(folder/'spikes.parquet')
 pe=pd.read_parquet(parent/'trials'/article.news_id/'events.parquet')
 for control in controls:
  seed=control['computational_seed'];raw=pe[pe.seed.eq(seed)&pe.drive.eq('R')][['step','input_slot']].to_numpy(dtype=np.int64)
  sub=subset_events(raw,control['common_original_slots'])
  for kind,expected in [('full',raw),('common',sub)]:assert np.array_equal(e[e.seed.eq(seed)&e.schedule.eq(kind)][['step','input_slot']].to_numpy(),expected)
  rows=m[m.news_id.eq(article.news_id)&m.seed.eq(seed)]
  common=rows[rows.condition.str.startswith('common')]
  assert common.schedule_sha256.nunique()==1 and common.external_input_events.nunique()==1
  for r in rows.itertuples():
   used=raw if r.condition=='ACh314_approx' else sub
   assert r.schedule_sha256==event_hash(used) and r.external_input_events==len(used)
   spikes=sp[sp.seed.eq(seed)&sp.condition.eq(r.condition)][['time_ms','root_id']]
   metrics,_,_=response_metrics(SimulationResult(spikes,r.effective_delay_ms,len(used)),c,sim)
   for key,value in metrics.items():assert np.isclose(value,getattr(r,key),equal_nan=True)
   checked+=1
 # Replay all three conditions for one article/seed against exact raw spikes.
 if article.news_id==selection.news_id.iloc[0]:
  control=controls[0];seed=control['computational_seed'];raw=pe[pe.seed.eq(seed)&pe.drive.eq('R')][['step','input_slot']].to_numpy(dtype=np.int64);sub=subset_events(raw,control['common_original_slots'])
  for condition,ids,events in [('ACh314_approx',control['approximate_control_ids'],raw),('common178_target',control['common_target_ids'],sub),('common178_control',control['common_control_ids'],sub)]:
   result=simulate(c,ids,events,sim);ref=sp[sp.seed.eq(seed)&sp.condition.eq(condition)][['time_ms','root_id']].reset_index(drop=True)
   pd.testing.assert_frame_equal(result.spikes,ref,check_dtype=False)
combined=pd.read_csv(OUT/'all_output_types.csv');assert len(combined)==400
assert combined[combined.fresh_simulation].shape[0]==240 and combined[~combined.fresh_simulation].shape[0]==160
write_json(OUT/'integrity_audit.json',dict(frozen_hashes_verified=True,new_trials=120,reused_parent_trials=80,
 matching_reproduced_seeds=5,exact_support_cells=178,full_control_cells=314,
 paired_article_seed_arrays_verified=40,raw_spike_metric_sets_verified=checked,exact_spike_replays=3,
 no_sensory_type_matching_claim=True))
print('Verified 120 new runs, 40 paired schedules, five matching assignments, and three exact spike replays.')
