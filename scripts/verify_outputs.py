"""Audit saved provenance and recompute the primary metric from every raw trial."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from news_flywire.provenance import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'results/v783'
metrics=pd.read_csv(run/'metrics.csv')
neurons=pd.read_csv(run/'circuit/neurons.csv',dtype={'root_id':str})
outputs=neurons[neurons.role.eq('output')].root_id.tolist()
config=json.loads((run/'resolved_config.json').read_text())['simulation']
start,end=config['baseline_ms'],config['baseline_ms']+config['stimulus_ms']
for row in metrics.itertuples():
    path=run/'spikes'/row.condition_id/f'seed{row.seed}.parquet'
    spikes=pd.read_parquet(path)
    output=spikes[spikes.root_id.isin(outputs)]
    pre=(output.time_ms<start).sum()/len(outputs)*1000/start
    stim=((output.time_ms>=start)&(output.time_ms<end)).sum()/len(outputs)*1000/config['stimulus_ms']
    np.testing.assert_allclose(stim-pre,row.delta_rate_hz,atol=1e-10,rtol=0)
counts=metrics.pivot(index='seed',columns='condition_id',values='external_input_events')
for condition in ['E4','E5','E6','E7']:
    np.testing.assert_array_equal(counts.E3,counts[condition])
rates=metrics.pivot(index='seed',columns='condition_id',values='delta_rate_hz')
np.testing.assert_array_equal(rates.E9a,rates.E9b)
manifest=json.loads((run/'manifest.json').read_text())
for path,digest in manifest['files'].items():
    assert sha256(path)==digest,f'Code changed since run: {path}'
source_manifest=json.loads((ROOT/'data/derived/source_manifest.json').read_text())
for path,digest in source_manifest['files'].items():
    assert sha256(path)==digest,f'Source changed: {path}'
for path in [ROOT/'notebooks/01_explore_results.ipynb']:
    notebook=json.loads(path.read_text())
    namespace={}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            exec(compile(''.join(cell['source']),str(path),'exec'),namespace)
write_json(ROOT/'results/verification.json',{
    'status':'passed','primary_metric_recomputed_from_raw_trials':len(metrics),
    'raw_spike_files':len(list((run/'spikes').glob('*/*.parquet'))),
    'matched_external_event_counts':'E3/E4/E5/E6/E7 identical per seed',
    'E9_invariance':'exact equality per seed',
    'code_and_source_checksums':'verified',
    'notebook_code_cells':'executed successfully',
    'scope':'Saved computational output consistency, not biological validity'})
print('All 330 raw trials, controls, provenance and notebook cells verified.')
