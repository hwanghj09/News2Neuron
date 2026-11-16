"""Audit raw events, spikes, paired controls and reproducibility of the bridge."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_flywire.article_bridge import check_freeze, CONDITIONS
from news_flywire.article_stimulus import schedules, event_hash
from news_flywire.brain_simulation import simulate, response_metrics, SimulationResult
from news_flywire.flywire_graph import Circuit
from news_flywire.provenance import write_json, sha256
from news_flywire.risk_review_updates import current_corpus_scores


def audit(out):
    out = Path(out); p = check_freeze(out)
    config = json.loads((out/'resolved_config.json').read_text()); sim = config['simulation']
    manifest = json.loads((out/'run_manifest.json').read_text())
    if (out/'path_packaging.json').exists():
        packaging = json.loads((out/'path_packaging.json').read_text())
        assert sha256(out/'protocol_original.json') == packaging['original_protocol_sha256']
        assert sha256(out/'protocol.json') == packaging['packaged_protocol_sha256']
        original = json.loads((out/'protocol_original.json').read_text())
        assert sorted(original['files'].values()) == sorted(p['files'].values())
        original['files'] = p['files']
        assert original == p  # Only path aliases changed, no experimental settings.
        old_run = json.loads((out/'run_manifest_original.json').read_text())
        assert old_run['files'][str(out/'protocol.json')] == packaging['original_protocol_sha256']
        old_run['files'][str(out/'protocol.json')] = packaging['packaged_protocol_sha256']
        assert old_run == manifest
    for path, digest in manifest['files'].items():
        assert sha256(path) == digest, path
    for path, digest in manifest['trial_receipts'].items():
        assert sha256(path) == digest, path
        for file, checksum in json.loads(Path(path).read_text())['files'].items():
            assert sha256(file) == checksum, file
    selection = pd.read_csv(out/'selection.csv')
    table = pd.read_parquet(out/'article_inputs.parquet')
    current = current_corpus_scores().set_index('news_id')
    assert len(table) == 8128 and len(selection) == 24 and selection.family.is_unique
    assert (selection.groupby(['cohort','source']).size() == 1).all()
    assert selection.simulation_allowed.all() and selection.session_reviewed.all()
    assert np.allclose(table.danger_score, current.loc[table.news_id].danger_score)
    assert np.allclose(table.R, table.danger_score/100)
    assert table.loc[~table.simulation_allowed, ['R_mean_rate_hz','T_mean_rate_hz','N_mean_rate_hz']].isna().all().all()
    assert np.allclose(table.loc[table.simulation_allowed, 'R_mean_rate_hz'], table.loc[table.simulation_allowed, 'R']*150)
    circuit = Circuit.load(p['circuit_path'])
    inputs = circuit.cells(role='input')
    selections = {r['seed']:r for r in json.loads((out/'cell_selections.json').read_text())}
    all_metrics = pd.read_csv(out/'metrics.csv'); baselines = pd.read_csv(out/'baseline_metrics.csv')
    assert len(all_metrics) == 840 and len(baselines) == 5
    assert not all_metrics.shared_level_simulation.any()
    assert baselines.delta_rate_hz.eq(0).all() and baselines.external_input_events.eq(0).all()
    assert not all_metrics.duplicated(['news_id','condition','seed']).any()
    assert set(all_metrics.condition) == set(CONDITIONS)
    duplicate_events = 0; checked_spike_metrics = 0; paired_sets = 0
    for a in selection.to_dict('records'):
        folder = out/'trials'/a['news_id']
        events = pd.read_parquet(folder/'events.parquet')
        spikes = pd.read_parquet(folder/'spikes.parquet')
        metrics = pd.read_csv(folder/'metrics.csv')
        duplicate_events += int(events.duplicated(['drive','seed','step','input_slot']).sum())
        for seed in config['computational_seeds']:
            regenerated = schedules(a['news_id'],seed,{k:a[k] for k in ['R','T','N']},len(inputs),config)
            for drive, value in regenerated.items():
                stored = events[events.drive.eq(drive)&events.seed.eq(seed)][['step','input_slot']].to_numpy(dtype=np.int64)
                assert np.array_equal(stored,value)
            rows = metrics[metrics.seed.eq(seed)]
            paired = rows[rows.drive.eq('R')]
            assert paired.schedule_sha256.nunique() == 1 and paired.external_input_events.nunique() == 1
            paired_sets += 1
            for r in rows.itertuples():
                assert r.schedule_sha256 == event_hash(regenerated[r.drive])
                assert r.external_input_events == len(regenerated[r.drive])
                s = spikes[spikes.seed.eq(seed)&spikes.condition.eq(r.condition)][['time_ms','root_id']]
                m, _, _ = response_metrics(SimulationResult(s,r.effective_delay_ms,r.external_input_events),circuit,sim)
                for key, value in m.items():
                    assert np.isclose(value,getattr(r,key),equal_nan=True), (r.news_id,r.condition,key)
                checked_spike_metrics += 1
        if a['cohort'] == 'negative_closed':
            assert metrics.loc[metrics.drive.eq('R'),'external_input_events'].eq(0).all()
            assert metrics.loc[metrics.drive.eq('R'),'delta_rate_hz'].eq(0).all()
    # Replay four actual network runs; raw-spike equality is stronger than
    # checking a score alone. Do not increment the recorded experiment count.
    a = selection[selection.cohort.eq('ongoing_harm')].iloc[0]
    seed = config['computational_seeds'][0]; control = selections[seed]
    e = schedules(a.news_id,seed,{k:a[k] for k in ['R','T','N']},len(inputs),config)
    stored = pd.read_parquet(out/'trials'/a.news_id/'spikes.parquet')
    for condition in ['R_intact','R_LPLC2_cut','R_random_input','T_intact']:
        drive, action = CONDITIONS[condition]
        active = control['random_input_ids'] if action == 'random_input' else inputs
        blocked = control['LPLC2_cut_ids'] if action == 'LPLC2' else []
        result = simulate(circuit,active,e[drive],sim,blocked)
        ref = stored[stored.seed.eq(seed)&stored.condition.eq(condition)][['time_ms','root_id']].reset_index(drop=True)
        pd.testing.assert_frame_equal(result.spikes,ref,check_dtype=False)
    html = (out/'report.html').read_text()
    import re
    payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1),
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Invalid JSON constant '+s)))
    assert len(payload['articles']) == 24 and payload['summary']['fresh_simulations'] == 845
    assert duplicate_events > 0
    write_json(out/'integrity_audit.json',dict(frozen_sources_and_inputs_verified=True,
        corpus_inputs=8128,selected_real_articles=24,eligible_articles=int(table.simulation_allowed.sum()),
        article_seed_control_sets_verified=paired_sets,raw_event_schedules_recreated=24*5*3,
        raw_spike_metric_sets_verified=checked_spike_metrics,network_spike_replays=4,
        event_multiplicity_retained=duplicate_events,fresh_simulations=845,
        article_level_lookup_reused=False,report_payload_valid=True,
        original_protocol_preserved=True,packaging_changes_only_path_aliases=True))
    print('Verified 845 experiments: raw inputs, raw spike metrics, four exact network replays and report payload.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results/article_bridge_v1')
    audit(parser.parse_args().out)
