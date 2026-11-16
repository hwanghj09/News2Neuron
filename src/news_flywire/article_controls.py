"""Output-path and transmitter-matched input controls for the article bridge."""
from pathlib import Path
import json

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from .article_stimulus import event_hash
from .brain_simulation import simulate, response_metrics
from .flywire_graph import Circuit
from .provenance import sha256, write_json

OUT = Path('results/control_matching_v1')
CONFIG = Path('configs/control_matching_v1.yaml')
CONDITIONS = ['ACh314_approx', 'common178_target', 'common178_control']


def signatures(circuit):
    data = circuit.neurons.copy().set_index('root_id', drop=False)
    graph = circuit.graph(); reverse = graph.reverse(copy=False)
    for cell_type in ['GF', 'DNp06']:
        targets = circuit.cells(cell_type=cell_type)
        if not targets:
            raise ValueError('Output cell type missing: '+cell_type)
        distance = nx.multi_source_dijkstra_path_length(reverse, targets, weight=None)
        data[cell_type+'_hops'] = data.root_id.map(distance).fillna(len(circuit.ids)+1).astype(int)
    data['in_degree'] = [graph.in_degree(i) for i in data.index]
    data['out_degree'] = [graph.out_degree(i) for i in data.index]
    data['in_abs_weight'] = circuit.edges.assign(w=circuit.edges.signed_synapses.abs()).groupby('post').w.sum().reindex(data.index, fill_value=0)
    data['out_abs_weight'] = circuit.edges.assign(w=circuit.edges.signed_synapses.abs()).groupby('pre').w.sum().reindex(data.index, fill_value=0)
    return data


def match_inputs(circuit, seed, penalty=10.):
    data = signatures(circuit)
    target = circuit.cells(role='input')
    candidates = sorted(set(circuit.ids)-set(target)-set(circuit.cells(role='output')))
    keys = ['GF_hops','DNp06_hops','top_nt']
    feature_names = ['in_degree','out_degree','in_abs_weight','out_abs_weight']
    features = np.log1p(data[feature_names].to_numpy(float))
    features /= np.maximum(features.std(axis=0), 1e-12)
    features = pd.DataFrame(features, index=data.index)
    rng = np.random.default_rng(seed)
    groups = data.loc[target].groupby(keys).groups
    common_target, common_control, feasibility = [], [], []
    for key, ids in sorted(groups.items()):
        ids = list(ids)
        pool = data.loc[candidates]
        pool = pool[(pool[keys] == pd.Series(key, index=keys)).all(axis=1)].index.tolist()
        n = min(len(ids), len(pool))
        feasibility.append(dict(GF_hops=int(key[0]), DNp06_hops=int(key[1]), top_nt=key[2],
            target_n=len(ids), candidate_n=len(pool), common_n=n, deficit=len(ids)-n))
        if not n:
            continue
        chosen = sorted(rng.choice(ids, n, replace=False).tolist())
        cost = np.linalg.norm(features.loc[chosen].to_numpy()[:, None]-features.loc[pool].to_numpy()[None, :], axis=2)
        row, col = linear_sum_assignment(cost+rng.uniform(0, 1e-8, cost.shape))
        common_target.extend(chosen[i] for i in row)
        common_control.extend(pool[j] for j in col)
    if not common_target:
        raise ValueError('No common support for matched input controls')
    if data.loc[target, 'top_nt'].nunique() != 1:
        raise ValueError('Full-size transmitter matching requires homogeneous source inputs')
    pool = [i for i in candidates if data.loc[i,'top_nt'] == data.loc[target[0],'top_nt']]
    if len(pool) < len(target):
        raise ValueError('Not enough transmitter-matched control neurons')
    degree_cost = np.linalg.norm(features.loc[target].to_numpy()[:, None]-features.loc[pool].to_numpy()[None, :],axis=2)
    hops = ['GF_hops','DNp06_hops']
    hop_cost = np.abs(data.loc[target,hops].to_numpy()[:, None]-data.loc[pool,hops].to_numpy()[None, :]).sum(axis=2)
    cost = degree_cost+penalty*hop_cost
    row, col = linear_sum_assignment(cost+rng.uniform(0, 1e-8, cost.shape))
    approximate = [pool[j] for j in col]
    # Put paired common-support controls into original input-slot order.
    ordered = sorted(zip(common_target, common_control), key=lambda pair:target.index(pair[0]))
    common_target, common_control = map(list, zip(*ordered))
    def mismatch(a,b):
        h = data.loc[a,hops].to_numpy()-data.loc[b,hops].to_numpy()
        return dict(hop_pair_exact_fraction=float((h == 0).all(axis=1).mean()),
            mean_absolute_hop_mismatch=float(np.abs(h).sum(axis=1).mean()),
            mean_standardized_log_feature_distance=float(np.linalg.norm(features.loc[a].to_numpy()-features.loc[b].to_numpy(),axis=1).mean()),
            transmitter_match_fraction=float((data.loc[a,'top_nt'].to_numpy()==data.loc[b,'top_nt'].to_numpy()).mean()))
    return dict(seed=seed, full_target_ids=target, approximate_control_ids=approximate,
        common_target_ids=common_target, common_control_ids=common_control,
        common_original_slots=[target.index(i) for i in common_target],
        common_mismatch=mismatch(common_target,common_control), approximate_mismatch=mismatch(target,approximate),
        common_target_cell_types=data.loc[common_target].cell_type.value_counts().to_dict(), feasibility=feasibility)


def subset_events(events, original_slots):
    events = np.asarray(events, dtype=np.int64).reshape(-1,2)
    if len(set(original_slots)) != len(original_slots):
        raise ValueError('Repeated target input slots')
    mapping = {slot:i for i,slot in enumerate(original_slots)}
    result = [(int(step),mapping[int(slot)]) for step,slot in events if int(slot) in mapping]
    return np.asarray(result,dtype=np.int64).reshape(-1,2)


def prepare(out=OUT, config=CONFIG):
    out, config = Path(out), Path(config)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    cfg = yaml.safe_load(config.read_text()); parent = Path(cfg['parent'])
    p = json.loads((parent/'protocol.json').read_text())
    circuit = Circuit.load(p['circuit_path'])
    if circuit.metadata['connectome_version'] != 'FAFB_v783':
        raise ValueError('Measured v783 circuit required')
    sim = json.loads((parent/'resolved_config.json').read_text())
    selection = pd.read_csv(parent/'selection.csv'); selection = selection[selection.cohort.eq(cfg['cohort'])]
    controls = [match_inputs(circuit, seed+cfg['matching_seed_offset'],cfg['hop_mismatch_penalty']) | dict(computational_seed=seed)
        for seed in sim['computational_seeds']]
    out.mkdir(parents=True,exist_ok=True)
    selection.to_csv(out/'selection.csv',index=False)
    write_json(out/'controls.json',controls)
    signatures(circuit).to_csv(out/'neuron_signatures.csv',index=False)
    write_json(out/'resolved_config.json',cfg)
    files = [config, Path('src/news_flywire/article_controls.py'), Path('src/news_flywire/brain_simulation.py'),
        Path('src/news_flywire/article_stimulus.py'), parent/'protocol.json',parent/'metrics.csv',parent/'cell_selections.json',
        parent/'resolved_config.json', Path(p['circuit_path'])/'neurons.csv',Path(p['circuit_path'])/'edges.parquet',
        Path(p['circuit_path'])/'metadata.json',out/'selection.csv',out/'controls.json',out/'neuron_signatures.csv',out/'resolved_config.json']
    for aid in selection.news_id:
        files.extend(parent/'trials'/aid/name for name in ['events.parquet','cells.parquet'])
    write_json(out/'protocol.json',dict(parent=str(parent), circuit=p['circuit_path'],
        real_articles=len(selection), new_conditions=CONDITIONS,
        expected_new_runs=len(selection)*len(controls)*len(CONDITIONS),
        reused_conditions=['R_intact','R_random_input'],
        primary='Within common-support subset: target minus matched control output delta Hz, by GF and DNp06 separately.',
        secondary='Full314 approximate transmitter/path/degree/weight control vs original intact and random; residual mismatch disclosed.',
        support='Exact GF/DNp06 shortest-hop pair and predicted transmitter; degree and absolute synaptic strength approximately matched within strata.',
        subset='Sample target neurons within deficient strata with fixed seed before new activity; no replacement or invented neurons.',
        input_pairing='Subset original parent R events and remap slots once; reuse exact subset events for both common-support runs. Full314 uses original R array.',
        analysis_unit='Article after averaging five computational seeds; selected-set descriptive variability, not biological replicates.',
        limitations=['Still intermediate rather than sensory-type-matched controls', 'Short paths include inhibitory intermediate routes',
            'Predicted transmitter is not measured receptor physiology', 'Common support changes target population; no generalization to all314',
            'Degree/strength matching remains approximate; no selection by new output activity'],
        files={str(f):sha256(f) for f in files}))
    print('Frozen controls:',len(controls[0]['common_target_ids']),'common-support neurons;',len(selection)*len(controls)*3,'new runs')


def verify(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    for path,digest in p['files'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen input or source changed: '+path)
    return p


def run(out=OUT):
    out = Path(out); p = verify(out)
    if (out/'metrics.csv').exists():
        raise ValueError('Completed experiment exists')
    parent = Path(p['parent']); sim = json.loads((parent/'resolved_config.json').read_text())['simulation']
    circuit = Circuit.load(p['circuit']); selection = pd.read_csv(out/'selection.csv')
    controls = json.loads((out/'controls.json').read_text()); metrics_all=[]; types_all=[]
    for a in selection.itertuples():
        folder=out/'trials'/a.news_id;folder.mkdir(parents=True,exist_ok=True)
        parent_events=pd.read_parquet(parent/'trials'/a.news_id/'events.parquet')
        spikes_all=[];events_all=[];rows=[];type_rows=[]
        for control in controls:
            seed=control['computational_seed']
            events=parent_events[parent_events.seed.eq(seed)&parent_events.drive.eq('R')][['step','input_slot']].to_numpy(dtype=np.int64)
            subset=subset_events(events,control['common_original_slots'])
            for kind,e in [('full',events),('common',subset)]:
                events_all.append(pd.DataFrame(dict(step=e[:,0],input_slot=e[:,1],schedule=kind,seed=seed)))
            for condition,ids,e in [('ACh314_approx',control['approximate_control_ids'],events),
                ('common178_target',control['common_target_ids'],subset),('common178_control',control['common_control_ids'],subset)]:
                result=simulate(circuit,ids,e,sim)
                m,cells,_=response_metrics(result,circuit,sim)
                rows.append(dict(news_id=a.news_id,source=a.source,seed=seed,condition=condition,input_neurons=len(ids),
                    schedule_sha256=event_hash(e),fresh_simulation=True,**m))
                sp=result.spikes.copy();sp['condition']=condition;sp['seed']=seed;spikes_all.append(sp)
                for typ in ['GF','DNp06']:
                    delta=cells[cells.root_id.isin(circuit.cells(cell_type=typ))].delta_rate_hz.mean()
                    type_rows.append(dict(news_id=a.news_id,seed=seed,condition=condition,cell_type=typ,delta_rate_hz=float(delta),fresh_simulation=True))
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
        pd.concat(spikes_all,ignore_index=True).to_parquet(folder/'spikes.parquet',compression='zstd',index=False)
        pd.concat(events_all,ignore_index=True).to_parquet(folder/'events.parquet',compression='zstd',index=False)
        write_json(folder/'receipt.json',dict(files={str(f):sha256(f) for f in sorted(folder.iterdir()) if f.name!='receipt.json'}))
        metrics_all.extend(rows);types_all.extend(type_rows)
        print('New matched-control runs:',a.news_id,15,flush=True)
    pd.DataFrame(metrics_all).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(types_all).to_csv(out/'output_types.csv',index=False)
    write_json(out/'run_manifest.json',dict(fresh_runs=len(metrics_all),reused_parent_runs=len(selection)*len(controls)*2,
        files={str(f):sha256(f) for f in [out/'protocol.json',out/'metrics.csv',out/'output_types.csv']},
        receipts={str(f):sha256(f) for f in sorted((out/'trials').glob('*/receipt.json'))}))
