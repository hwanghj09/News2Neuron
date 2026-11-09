"""Explicit sensitivity settings from the proposal, with computational seeds only."""
import argparse
import copy
from pathlib import Path
import pandas as pd
import yaml

from news_flywire.flywire_graph import extract_v783, Circuit
from news_flywire.experiments import run_experiments
from news_flywire.provenance import write_json

p=argparse.ArgumentParser()
p.add_argument('--seeds',type=int,default=3)
p.add_argument('--out',default='results/sensitivity')
p.add_argument('--max-neurons',type=int,default=10000)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
graph_config=yaml.safe_load((root/'configs/connectome_v783.yaml').read_text())
sim_config=yaml.safe_load((root/'configs/stimulation_v1.yaml').read_text())
out=root/a.out
out.mkdir(parents=True,exist_ok=True)
baseline=Circuit.load(root/'data/derived/flywire_v783')
rows=[]
for name,threshold,rate_scale,dt in [
    ('synapses_1',1,1.0,.5),('synapses_3',3,1.0,.5),('synapses_5',5,1.0,.5),
    ('rate_075',5,.75,.5),('rate_125',5,1.25,.5),('dt_01',5,1.0,.1)]:
    c=copy.deepcopy(sim_config)
    c['rates_hz']=[v*rate_scale for v in c['rates_hz']]
    c['simulation']['dt_ms']=dt
    graph=copy.deepcopy(graph_config)
    graph.update(min_synapses=threshold,max_neurons=a.max_neurons)
    run_out=out/name
    if (run_out/'metrics.csv').exists():
        raise SystemExit(f'Preserve previous sensitivity results: choose a fresh --out ({run_out} exists)')
    circuit=baseline if threshold==5 else extract_v783(graph)
    metrics=run_experiments(circuit,c,run_out,seeds=a.seeds)
    circuit.save(run_out/'circuit')
    for condition,group in metrics.groupby('condition_id'):
        rows.append(dict(setting=name,min_synapses=threshold,rate_scale=rate_scale,dt_ms=dt,
            neurons=len(circuit.ids),edges=len(circuit.edges),condition_id=condition,
            delta_mean_hz=float(group.delta_rate_hz.mean()),computational_seeds=a.seeds))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
write_json(out/'scope.json',{'status':'pilot_sensitivity','seeds':a.seeds,
 'scope':'Computational repeatability and parameter dependence; not biological inference',
 'settings':'synapse thresholds 1/3/5, rate scaling .75/1/1.25, dt .1/.5 ms; fixed before running this sensitivity sweep'})
print(out/'summary.csv')
