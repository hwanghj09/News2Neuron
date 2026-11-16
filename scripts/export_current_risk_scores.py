"""Rebuild the current complete table from a frozen snapshot and review patches."""
import argparse
from pathlib import Path

import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, risk_band
from news_flywire.risk_review_updates import load_review_catalogs, input_key


def export(out):
    snapshot = Path('data/risk/20261110_followup/news_risk_scores.parquet')
    data = pd.read_parquet(snapshot)
    catalog = load_review_catalogs()
    applied = set()
    for i, row in data.iterrows():
        key = input_key(row.title, row.assessment_body_sha256)
        if key not in catalog:
            continue
        r = catalog[key]; applied.add(key)
        for axis in AXES:
            data.loc[i, axis] = r[axis]/4
        data.loc[i, 'danger_score'] = r['danger_score']
        data.loc[i, 'component_index'] = r['danger_score']
        data.loc[i, 'session_reviewed'] = True
        data.loc[i, 'score_source'] = 'direct_Codex_session_judgment'
        data.loc[i, 'decision_state'] = r['state']
        data.loc[i, 'rationale'] = r['rationale']
    if applied != catalog.keys():
        raise ValueError('Review version missing from corpus snapshot')
    data['danger_band'] = risk_band(data.danger_score.to_numpy())
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for filename in ['news_risk_scores.csv', 'news_risk_scores.parquet', 'manifest.json']:
        if (out/filename).exists():
            raise ValueError('Output already exists; choose a new directory')
    data.to_csv(out/'news_risk_scores.csv', index=False)
    data.to_parquet(out/'news_risk_scores.parquet', index=False)
    write_json(out/'manifest.json', dict(articles=len(data), direct_reviews=len(applied),
        model='v5', source_snapshot_sha256=sha256(snapshot),
        catalog_manifest_sha256=sha256('data/risk/review_catalogs.json'),
        output_sha256=sha256(out/'news_risk_scores.parquet'),
        exact_version_review_application=True, accuracy_evaluation=False))
    print(f'{len(data)} articles, {len(applied)} direct reviews -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    export(parser.parse_args().out)
