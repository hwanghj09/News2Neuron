"""Fresh article-specific simulations on the measured FAFB v783 subcircuit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .article_stimulus import profile, schedules, event_hash
from .brain_simulation import simulate, response_metrics
from .flywire_graph import Circuit, degree_matched_control
from .provenance import sha256, write_json, manifest
from .risk_review_updates import current_corpus_scores

DEFAULT_OUT = Path('results/article_bridge_v1')
DEFAULT_CONFIG = Path('configs/article_bridge_v1.yaml')
DEFAULT_CIRCUIT = Path('data/derived/flywire_v783')
CONDITIONS = {
    'R_intact': ('R', 'intact'),
    'R_random_input': ('R', 'random_input'),
    'R_LPLC2_cut': ('R', 'LPLC2'),
    'R_LC4_cut': ('R', 'LC4'),
    'R_degree_cut': ('R', 'degree'),
    'T_intact': ('T', 'intact'),
    'N_intact': ('N', 'intact'),
}


def source_files(config, circuit):
    return [Path(config), Path(circuit)/'neurons.csv', Path(circuit)/'edges.parquet',
        Path(circuit)/'metadata.json', Path('data/risk/review_catalogs.json'),
        Path('data/risk/20261110_followup/news_risk_scores.parquet'),
        Path('data/risk/20261109_v7/corpus_families.csv'),
        *[Path(e['path']) for e in json.loads(Path('data/risk/review_catalogs.json').read_text())['catalogs']],
        *[Path(__file__).with_name(n+'.py') for n in ['article_bridge', 'article_stimulus',
            'brain_simulation', 'flywire_graph', 'risk_review_updates']]]


def encode_table(scores, config):
    needed = ['news_id', 'source', 'url', 'title', 'published_date', 'assessment_body_sha256',
        'assessment_research_eligible', 'input_quality_flag', 'score_source', 'session_reviewed',
        'decision_state', 'N', 'T', 'A', 'danger_score']
    data = scores[needed].copy()
    if data.news_id.duplicated().any() or not data.url.str.startswith('https://').all():
        raise ValueError('Unique real article IDs and publisher URLs required')
    values = data[['N', 'T', 'A']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Normalized N/T/A must be finite and in [0,1]')
    if not np.isfinite(data.danger_score).all() or not data.danger_score.between(0, 100).all():
        raise ValueError('Remaining harm index must be in [0,100]')
    data['R'] = data.danger_score/100
    data['simulation_allowed'] = data.assessment_research_eligible & data.input_quality_flag.fillna('').eq('')
    for axis in ['R', 'T', 'N']:
        data[axis+'_mean_rate_hz'] = np.where(data.simulation_allowed,
            data[axis]*config['maximum_mean_rate_hz'], np.nan)
    p = config['physical']
    data['R_l_over_v_ms'] = np.where(data.simulation_allowed,
        p['slow_l_over_v_ms']+(p['fast_l_over_v_ms']-p['slow_l_over_v_ms'])*data.R, np.nan)
    return data


def select_articles(table, families, seed):
    eligible = table.simulation_allowed & table.session_reviewed
    masks = {
        'minimal': table.N.le(.25) & table.danger_score.lt(10),
        'negative_closed': table.N.ge(.75) & table.danger_score.eq(0) & table.decision_state.isin(['resolved', 'retrospective']),
        'ongoing_harm': table.danger_score.ge(30) & table['T'].ge(.5),
    }
    rng = np.random.default_rng(seed); used = set(); rows = []
    for cohort, mask in masks.items():
        for source in sorted(table.source.unique()):
            candidates = table[eligible & mask & table.source.eq(source)]
            chosen = None
            for i in rng.permutation(candidates.index):
                row = table.loc[i]; family = int(families.loc[row.news_id])
                if family not in used:
                    chosen = row.to_dict(); chosen.update(cohort=cohort, family=family)
                    used.add(family); break
            if chosen is None:
                raise ValueError(f'No eligible independently grouped article for {cohort}/{source}')
            rows.append(chosen)
    return pd.DataFrame(rows)


def prepare(out=DEFAULT_OUT, config=DEFAULT_CONFIG, circuit=DEFAULT_CIRCUIT):
    out, config, circuit = Path(out), Path(config), Path(circuit)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation output must be empty; preserve earlier runs')
    c = yaml.safe_load(config.read_text())
    for s in np.linspace(0, 1, 101):
        profile(float(s), c)
    scores = current_corpus_scores()
    table = encode_table(scores, c)
    families = pd.read_csv('data/risk/20261109_v7/corpus_families.csv').set_index('news_id').family
    selected = select_articles(table, families, c['selection_seed'])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/'article_inputs.csv', index=False)
    table.to_parquet(out/'article_inputs.parquet', index=False)
    selected.to_csv(out/'selection.csv', index=False)
    write_json(out/'resolved_config.json', c)
    files = source_files(config, circuit)+[out/'article_inputs.csv', out/'article_inputs.parquet',
        out/'selection.csv', out/'resolved_config.json']
    write_json(out/'protocol.json', dict(stage='fresh_article_specific_neural_simulation',
        corpus_articles=len(table), eligible_articles=int(table.simulation_allowed.sum()),
        real_articles=len(selected), computational_repeats=len(c['computational_seeds']),
        conditions=CONDITIONS, expected_new_trials=len(selected)*len(c['computational_seeds'])*len(CONDITIONS)+len(c['computational_seeds']),
        selection='One directly AI-reviewed eligible article per source and stratum, distinct automatic families; fixed seed, before neural outcomes.',
        strata={k: v for k, v in [('minimal','N<=.25 and R<10'), ('negative_closed','N>=.75 and R=0 and resolved/retrospective'), ('ongoing_harm','R>=30 and T>=.5')]},
        primary_mapping='R /100 -> continuous mean drive 0..150 Hz; T and N mappings are separate sensitivity comparators.',
        geometry='Constant-speed approach 20..80 degrees, radius .01m, l/v linearly 80..20ms, end-aligned to400ms; zero drive renders a static disk.',
        drive='80% uniform plus20% normalized angular expansion; fixed expected count per mean rate. Defined hypothesis, no measured visual-neuron calibration.',
        paired_controls='Identical complete R event array for intact, random-input, LPLC2/LC4 cut, degree-matched cut. R/T/N share article-specific Poisson proposals.',
        baseline='Five fresh no-input runs, shared baseline; deterministic resting network, no spontaneous background input.',
        statistical_unit='Article for paired effects, average its five computational seeds first. Seeds are not biological replicates.',
        predefined_comparisons=['R intact minus random-input', 'degree-cut minus LPLC2-cut', 'R intact minus T intact', 'N intact minus R intact'],
        negative_closed_prediction='R=0 implies no external events and no response in this resting model, by design; not a finding about understanding news.',
        limitations=['Reduced 1367-neuron subcircuit, not whole brain', 'No new physiological measurements',
            'Random input neurons are not sensory-type matched', 'Connection removal is not measured biological inhibition',
            'News AI references and virtual-drive mapping remain uncertain; R/T/N differences depend on this mapping'],
        files={str(p): sha256(p) for p in files}, circuit_path=str(circuit)))
    print(f'Frozen {len(table)} article inputs; {len(selected)} articles, {len(CONDITIONS)} conditions.')


def check_freeze(out):
    protocol = json.loads((Path(out)/'protocol.json').read_text())
    for path, digest in protocol['files'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen bridge input/source changed: '+path)
    return protocol


def run(out=DEFAULT_OUT):
    out = Path(out)
    protocol = check_freeze(out)
    if (out/'metrics.csv').exists():
        raise ValueError('Completed run exists; use report/audit without rerunning')
    c = json.loads((out/'resolved_config.json').read_text()); sim = c['simulation']
    circuit = Circuit.load(protocol['circuit_path'])
    if circuit.metadata['connectome_version'] != 'FAFB_v783':
        raise ValueError('This research run requires measured FAFB v783 connectivity')
    articles = pd.read_csv(out/'selection.csv')
    inputs, outputs = circuit.cells(role='input'), circuit.cells(role='output')
    lplc2, lc4 = circuit.cells(cell_type='LPLC2'), circuit.cells(cell_type='LC4')
    if not inputs or not outputs or not lplc2 or not lc4:
        raise ValueError('Required cells missing')
    pool = sorted(set(circuit.ids)-set(inputs)-set(outputs))
    selections = []; controls = {}
    for seed in c['computational_seeds']:
        random_inputs = np.random.default_rng(seed).choice(pool, len(inputs), replace=False).tolist()
        degree, mismatch = degree_matched_control(circuit, lplc2, set(inputs)|set(outputs), seed)
        controls[seed] = dict(random_inputs=random_inputs, degree=degree, mismatch=mismatch)
        selections.append(dict(seed=seed, input_ids=inputs, output_ids=outputs, random_input_ids=random_inputs,
            LPLC2_cut_ids=lplc2, LC4_cut_ids=lc4, degree_cut_ids=degree, degree_matching_mean_log_distance=mismatch))
    write_json(out/'cell_selections.json', selections)
    base_rows = []
    for seed in c['computational_seeds']:
        result = simulate(circuit, inputs, np.empty((0, 2), int), sim)
        m, _, _ = response_metrics(result, circuit, sim)
        base_rows.append(dict(seed=seed, condition='no_input', shared_baseline=True, **m))
    pd.DataFrame(base_rows).to_csv(out/'baseline_metrics.csv', index=False)
    collectors = []
    for j, article in enumerate(articles.to_dict('records')):
        aid = article['news_id']; folder = out/'trials'/aid; receipt = folder/'receipt.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for path, digest in saved['files'].items():
                if sha256(path) != digest:
                    raise ValueError('Corrupt trial checkpoint: '+path)
            collectors.append(pd.read_csv(folder/'metrics.csv'))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        event_frames, spike_frames, cell_frames, curve_frames, rows = [], [], [], [], []
        profile_rows = []
        for drive in ['R', 'T', 'N']:
            values, frames = profile(float(article[drive]), c)
            frames['drive'] = drive
            frames['rate_hz'] = np.r_[values, values[-1]]
            profile_rows.append(frames)
        pd.concat(profile_rows).to_parquet(folder/'stimulus_profiles.parquet', index=False)
        for seed in c['computational_seeds']:
            generated = schedules(aid, seed, {k:float(article[k]) for k in ['R', 'T', 'N']}, len(inputs), c)
            for drive, events in generated.items():
                event_frames.append(pd.DataFrame(dict(step=events[:, 0], input_slot=events[:, 1], drive=drive, seed=seed)))
            for condition, (drive, action) in CONDITIONS.items():
                events = generated[drive]
                stimulated = controls[seed]['random_inputs'] if action == 'random_input' else inputs
                blocked = {'LPLC2':lplc2, 'LC4':lc4, 'degree':controls[seed]['degree']}.get(action, [])
                result = simulate(circuit, stimulated, events, sim, blocked)
                m, cells, curve = response_metrics(result, circuit, sim)
                rows.append(dict(news_id=aid, cohort=article['cohort'], source=article['source'], seed=seed,
                    condition=condition, drive=drive, normalized_score=article[drive],
                    mean_rate_hz=article[drive]*c['maximum_mean_rate_hz'], schedule_sha256=event_hash(events),
                    expected_input_events=len(inputs)*article[drive]*c['maximum_mean_rate_hz']*sim['stimulus_ms']/1000,
                    degree_matching_mean_log_distance=controls[seed]['mismatch'] if action == 'degree' else np.nan,
                    shared_level_simulation=False, **m))
                for frame, collector in [(result.spikes, spike_frames), (cells, cell_frames), (curve, curve_frames)]:
                    frame['condition'], frame['seed'] = condition, seed
                    collector.append(frame)
        pd.DataFrame(rows).to_csv(folder/'metrics.csv', index=False)
        for name, frames in [('events',event_frames), ('spikes',spike_frames), ('cells',cell_frames), ('curves',curve_frames)]:
            pd.concat(frames, ignore_index=True).to_parquet(folder/(name+'.parquet'), index=False, compression='zstd')
        files = sorted(folder.glob('*.parquet'))+[folder/'metrics.csv']
        write_json(receipt, dict(news_id=aid, trials=len(rows), files={str(p):sha256(p) for p in files}))
        collectors.append(pd.DataFrame(rows))
        print(f'Fresh neural trials: article {j+1}/{len(articles)}, {len(rows)} runs', flush=True)
    metrics = pd.concat(collectors, ignore_index=True)
    if len(metrics) != protocol['expected_new_trials']-len(c['computational_seeds']):
        raise ValueError('Incomplete experimental run')
    metrics.to_csv(out/'metrics.csv', index=False)
    write_json(out/'run_manifest.json', manifest([out/'protocol.json', out/'metrics.csv', out/'baseline_metrics.csv'],
        news_articles=len(articles), fresh_trials=len(metrics)+len(base_rows), shared_level_lookup_used=False,
        connectivity_version='FAFB_v783', engine='reduced_lif_analytic_v1',
        simulation_only=True, physiological_validation=False,
        trial_receipts={str(p):sha256(p) for p in sorted((out/'trials').glob('*/receipt.json'))}))
    print(f'Completed {len(metrics)+len(base_rows)} new simulations.')
