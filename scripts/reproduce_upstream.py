"""Run the upstream example's first v630 activation with explicit trial count."""
import argparse, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--trials',type=int,default=1)
p.add_argument('--duration-ms',type=float,default=1000)
p.add_argument('--out',default='results/upstream_reproduction')
a=p.parse_args()
if a.trials<1 or a.duration_ms<=0:
    p.error('trials and duration must be positive')
root=Path(__file__).resolve().parents[1]
upstream=root/'external/Drosophila_brain_model'
sys.path.insert(0,str(upstream))
import brian2 as b
import pandas as pd
from model import run_exp, default_params
from news_flywire.provenance import write_json, manifest, git_commit
notebook=json.loads((upstream/'example.ipynb').read_text())
namespace={}
cell=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('neu_sugar ='))
exec(compile(cell,'upstream_example_neu_sugar','exec'),namespace)
out=root/a.out
out.mkdir(parents=True,exist_ok=True)
if (out/'sugarR_reproduction.parquet').exists():
    p.error('Output already exists. Choose a fresh --out to preserve the original run.')
params=default_params.copy()
params.update(n_run=a.trials,t_run=a.duration_ms*b.ms)
b.prefs.codegen.target='numpy'
b.seed(20261109)
t=time.time()
run_exp(exp_name='sugarR_reproduction',neu_exc=namespace['neu_sugar'],path_res=str(out),
        path_comp=str(upstream/'2023_03_23_completeness_630_final.csv'),
        path_con=str(upstream/'2023_03_23_connectivity_630_final.parquet'),
        params=params,n_proc=1,force_overwrite=True)
spikes=pd.read_parquet(out/'sugarR_reproduction.parquet')
reference=pd.read_parquet(upstream/'results/example/sugarR.parquet')
write_json(out/'reproduction.json',manifest([upstream/'model.py',upstream/'example.ipynb'],
 upstream_commit=git_commit(upstream),connectome_version='FAFB_v630',seed=20261109,
 duration_ms=a.duration_ms,trials=a.trials,elapsed_seconds=time.time()-t,
 input_rate_hz=float(params['r_poi']/b.Hz),
 n_spikes=len(spikes),active_neurons=int(spikes.flywire_id.nunique()),
 reference_spikes=len(reference),reference_trials=int(reference.trial.nunique()),
 reference_active_neurons=int(reference.flywire_id.nunique()),
 status='upstream_first_example_executed',
 limitation='Original code execution check; no claim of paper-wide or threat-circuit validation, or parameter-equivalent reproduction of the archived output'))
print((out/'reproduction.json').read_text())
