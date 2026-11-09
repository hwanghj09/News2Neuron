"""A pre-registered, family-purged comparison with an unopened AI audit.

The reference is a rubric judgment by the executing AI, not human ground truth.
Publisher prose stays in the ignored cache. All evaluation uses pure model
predictions frozen before audit packets can be opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from .provenance import sha256, write_json
from .risk_assessment import AXES, STATES, danger_index
from .risk_v4 import body_for, load_news, named_families

OUT = Path('data/risk/20261109_v6')
PRIVATE = Path('data/raw/risk/20261109_v6')
PREVIOUS = Path('data/risk/20261109_v5')
SEED = 20261111


def corpus_families(news, bodies):
    """Connected components of conservative, outcome-blind corpus relations."""
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                                 max_features=60000, dtype=np.float32)
    titles = vectorizer.fit_transform(news.title)
    vectors = np.load('data/raw/risk/20261109_v3/titles.npy')
    times = pd.to_datetime(news.published_date, utc=True).astype('int64').to_numpy()/1e9
    edges = []
    for start in range(0, len(news), 128):
        indices = np.arange(start, min(start + 128, len(news)))
        lex = (titles[indices] @ titles.T).toarray()
        sem = vectors[indices] @ vectors.T
        close = np.abs(times[indices, None] - times[None, :]) <= 21*86400
        linked = close & ((lex >= .40) | ((sem >= .91) & (lex >= .15)))
        a, b = np.where(linked)
        edges.extend(zip(indices[a].tolist(), b.tolist()))
    groups = {}
    for i, (row, body) in enumerate(zip(news.itertuples(), bodies)):
        tags = {'candidate:' + str(row.event_group)}
        tags.update('campaign:' + f for f in named_families(row.title + ' ' + body))
        for tag in tags:
            if tag in groups:
                edges.append((i, groups[tag]))
            else:
                groups[tag] = i
    a, b = np.asarray(edges).T
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)),
                      shape=(len(news), len(news)))
    return connected_components(graph, directed=False)[1]


def prepare():
    if (OUT/'protocol.json').exists():
        raise ValueError('Protocol already frozen; create a new experiment')
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    news = load_news()
    bodies = [body_for(row) for row in news.itertuples()]
    legacy = pd.read_csv(PREVIOUS/'decision_audit.csv')
    dev = legacy[legacy.role.ne('quality')].copy()
    dev['previous_role'] = dev.role
    dev['role'] = 'legacy_development'
    dev.to_csv(OUT/'legacy_development.csv', index=False)
    families = corpus_families(news, bodies)
    pd.DataFrame({'news_id': news.news_id, 'family': families}).to_csv(
        OUT/'corpus_families.csv', index=False)
    legacy_families = set(families[news.news_id.isin(legacy.news_id)])
    eligible = news.assessment_research_eligible.to_numpy()
    pool = np.flatnonzero(eligible & ~np.isin(families, list(legacy_families)))
    rng = np.random.default_rng(SEED)
    baseline = pd.read_parquet(PREVIOUS/'predictions.parquet')
    old = pd.read_parquet('data/risk/20261109_v3/round_3/input_corrected_predictions.parquet')
    records, used = [], set()
    # First select the primary random cohort, without reading new references.
    for cohort, number in [('random', 16), ('challenge', 8)]:
        for source in sorted(news.source.unique()):
            choices = np.array([i for i in pool if news.source.iloc[i] == source
                                and families[i] not in used])
            if cohort == 'challenge':
                priority = np.maximum(baseline.danger_score.to_numpy()[choices],
                                      old.danger_score.to_numpy()[choices])
                choices = choices[np.argsort(-priority, kind='stable')[:max(number, min(40, len(choices)))]]
            chosen = []
            for i in rng.permutation(choices):
                if families[i] in used:
                    continue
                chosen.append(int(i)); used.add(families[i])
                if len(chosen) == number:
                    break
            if len(chosen) != number:
                raise ValueError(f'Insufficient purged families for {source}/{cohort}')
            for i in chosen:
                row = news.iloc[i]
                records.append(dict(case_id=f'v6_a_{len(records)+1:03d}',
                    news_id=row.news_id, source=source, cohort=cohort, role='audit',
                    family=int(families[i]), assessment_body_sha256=row.assessment_body_sha256))
    audit_families = set(used)
    audit = pd.DataFrame(records)
    audit.to_csv(OUT/'audit_selection.csv', index=False)
    # Development is selected only after all test families have been removed.
    candidates = np.flatnonzero(eligible & ~news.news_id.isin(legacy.news_id).to_numpy()
                               & ~np.isin(families, list(audit_families)))
    records = []
    for source in sorted(news.source.unique()):
        choices = np.array([i for i in candidates if news.source.iloc[i] == source])
        priority = (np.abs(old.danger_score.to_numpy()[choices]-baseline.danger_score.to_numpy()[choices])
                    + .3*np.maximum(old.danger_score.to_numpy()[choices],baseline.danger_score.to_numpy()[choices]))
        hard = choices[np.argsort(-priority, kind='stable')[:8]]
        remaining = np.array([i for i in choices if i not in set(hard)])
        selected = np.concatenate([hard, rng.choice(remaining, 8, replace=False)])
        for i in selected:
            row = news.iloc[i]
            records.append(dict(case_id=f'v6_d_{len(records)+1:03d}',
                news_id=row.news_id, source=source, role='development',
                family=int(families[i]), assessment_body_sha256=row.assessment_body_sha256))
    pd.DataFrame(records).to_csv(OUT/'development_selection.csv', index=False)
    write_json(OUT/'protocol.json', {
        'seed': SEED, 'rubric': 'docs/risk-rubric-v3.md (unchanged)',
        'primary': '128 source-stratified random articles, 16 per publisher, one per purged family',
        'secondary': '64 challenge articles, 8 per publisher; do not pool into primary',
        'new_development_articles': 128, 'legacy_development_articles': len(dev),
        'sampling_population': 'research-eligible articles outside all previously judged conservative families',
        'sampling_not_representative_of_unrestricted_corpus': True,
        'reference': 'same executing AI rubric; not independent human/empirical danger ground truth',
        'reference_blinding': 'audit packets omit previous and candidate scores; opened only after final freeze',
        'primary_endpoint': 'mean absolute error, index points /100, against AI rubric',
        'comparison_baselines': ['v3 original pure model', 'v5 original pure model', 'constant zero diagnostic'],
        'superiority_rule': 'paired mean error reduction 95% percentile interval lower bound > 0 against BOTH v3 and v5 on random cohort',
        'agreement_guard': 'within10 point estimate must not decrease versus either v3 or v5; report paired interval and exact McNemar test separately',
        'challenge_guard': 'challenge MAE point estimate must not increase versus either v3 or v5',
        'all_zero_guard': 'report nonzero/high-risk reference recall and constant-zero MAE; no accuracy claim from zero-majority collapse',
        'bootstrap': '20000 stratified paired resamples within publisher, seed 20261111; descriptive conditional AI-reference uncertainty',
        'two_primary_comparisons': 'require BOTH comparisons to pass (intersection-union claim); no alternative endpoint promotion',
        'post_opening_rule': 'do not tune weights, thresholds, exclusions, labels or primary cohort after viewing final audit predictions',
        'no_optional_stopping': 'one fixed audit size; publish failure if criteria not met',
        'family_construction': {'headline_char_tfidf_cosine': .40, 'semantic_cosine': .91,
            'minimum_lexical_for_semantic': .15, 'days': 21,
            'other': 'candidate event ID + pre-existing named campaigns in entire bodies; connected components'},
        'families_total': int(len(set(families))), 'purged_random_pool_articles': int(len(pool)),
        'files': {str(p):sha256(p) for p in [OUT/'audit_selection.csv',OUT/'development_selection.csv',OUT/'legacy_development.csv',OUT/'corpus_families.csv']}
    })
    print(json.dumps({'families':len(set(families)), 'audit_pool':len(pool),
                      'audit':len(audit),'new_development':len(records)}))


def packet(role, start=0, count=8):
    if role == 'audit' and not (OUT/'pre_audit_freeze.json').exists():
        raise ValueError('Freeze model before opening audit bodies')
    selection = pd.read_csv(OUT/f'{role}_selection.csv').iloc[start:start+count]
    news = load_news().set_index('news_id')
    records = []
    for row in selection.itertuples():
        n = news.loc[row.news_id]
        body = body_for(n)
        records.append({'case_id':row.case_id,'title':n.title,'body':body,
                        'assessment_body_sha256':row.assessment_body_sha256})
    path = PRIVATE/f'packet_{role}_{start:03d}.json'
    if path.exists():
        if json.loads(path.read_text()) != records:
            raise ValueError('Packet changed')
    else:
        write_json(path, records)
        write_json(OUT/f'packet_{role}_{start:03d}_receipt.json',{
            'sha256':sha256(path),'cases':selection.case_id.tolist(),
            'scope':'complete cleaned stored article body and title, no prediction scores'})
    return records


def judgments(role):
    selection = pd.read_csv(OUT/f'{role}_selection.csv')
    labels = pd.read_csv(OUT/f'{role}_judgments.tsv',sep='\t',keep_default_na=False)
    if set(selection.case_id) != set(labels.case_id) or labels.case_id.duplicated().any():
        raise ValueError('Incomplete/duplicate actual AI judgments')
    values = labels[AXES].to_numpy(float)
    if not (np.isfinite(values).all() and ((values >= 0)&(values <= 4)&(values == np.floor(values))).all()):
        raise ValueError('Invalid ordinal judgment')
    if not labels.state.isin(STATES).all() or not labels.rationale.str.len().ge(15).all():
        raise ValueError('Invalid state/rationale')
    closed = labels.state.isin(['none','retrospective','resolved'])
    if labels.loc[closed,['T','L','I']].to_numpy().any():
        raise ValueError('Residual-risk contradiction')
    merged = selection.merge(labels,on='case_id',validate='one_to_one')
    merged['danger_score'] = danger_index(merged[['S','L','I','X']].to_numpy(float))
    merged['judge'] = 'current_Codex_session_AI_not_human'
    merged['input_scope'] = 'complete_stored_cleaned_body_and_title'
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','packet'])
    parser.add_argument('--role',choices=['development','audit'],default='development')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=8)
    args=parser.parse_args()
    if args.stage == 'prepare':prepare()
    else:print(json.dumps(packet(args.role,args.start,args.count),ensure_ascii=False))


if __name__ == '__main__':main()
