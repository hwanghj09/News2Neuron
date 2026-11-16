"""Rebuild the current complete table from a frozen snapshot and review patches."""
import argparse
from pathlib import Path

import pandas as pd

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_assessment import AXES, risk_band
from news_flywire.risk_review_updates import current_corpus_scores


def export(out):
    snapshot = Path('data/risk/20261110_followup/news_risk_scores.parquet')
    data = current_corpus_scores(snapshot)
    applied = int(data.session_reviewed.sum())
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for filename in ['news_risk_scores.csv', 'news_risk_scores.parquet', 'manifest.json']:
        if (out/filename).exists():
            raise ValueError('Output already exists; choose a new directory')
    data.to_csv(out/'news_risk_scores.csv', index=False)
    data.to_parquet(out/'news_risk_scores.parquet', index=False)
    write_json(out/'manifest.json', dict(articles=len(data), direct_reviews=applied,
        model='v5', source_snapshot_sha256=sha256(snapshot),
        catalog_manifest_sha256=sha256('data/risk/review_catalogs.json'),
        output_sha256=sha256(out/'news_risk_scores.parquet'),
        exact_version_review_application=True, accuracy_evaluation=False))
    print(f'{len(data)} articles, {applied} direct reviews -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    export(parser.parse_args().out)
