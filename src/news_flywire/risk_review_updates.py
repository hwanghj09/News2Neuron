"""Apply explicit AI reviews to exact article versions, preserving pure-model audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256, write_json
from .risk_assessment import AXES, STATES, danger_index, risk_band
from .risk_inference import score_articles, validate_article

ROOT = Path('data/risk')
OUT = ROOT/'20261110_followup'
DIAGNOSTIC = ROOT/'20261110_v8'


def input_key(title, body_sha256):
    return hashlib.sha256(json.dumps([title, body_sha256], ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def load_catalog(path):
    catalog = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        key = input_key(r['title'], r['assessment_body_sha256'])
        if key != r['input_sha256'] or key in catalog:
            raise ValueError('Invalid or duplicate reviewed article fingerprint')
        axes = np.array([r[k] for k in AXES], dtype=float)
        if not np.isfinite(axes).all() or not np.isin(axes, [0, 1, 2, 3, 4]).all():
            raise ValueError('Reviewed axes must be integer rubric values 0..4')
        if r['state'] not in STATES or not r['rationale'].strip():
            raise ValueError('Reviewed state and rationale are required')
        if r['state'] in {'none', 'resolved', 'retrospective'} and any(r[k] for k in ['T', 'L', 'I']):
            raise ValueError('Closed review cannot retain T/L/I')
        if not np.isclose(danger_index(axes[3:]), r['danger_score']):
            raise ValueError('Reviewed index differs from S/L/I/X formula')
        catalog[key] = r
    return catalog


def score_with_reviews(articles, catalog_path=OUT/'review_catalog.jsonl',
                       model_path=ROOT/'20261109_v5/model.joblib',
                       model_dir='external/multilingual-e5-small'):
    catalog = load_catalog(catalog_path)
    catalog_digest = sha256(catalog_path)
    results, pending, positions = [None]*len(articles), [], []
    for i, a in enumerate(articles):
        validate_article(a['title'], a['body'])
        body_hash = hashlib.sha256(a['body'].encode()).hexdigest()
        r = catalog.get(input_key(a['title'], body_hash))
        if r is None:
            pending.append(a)
            positions.append(i)
            continue
        results[i] = dict(case_id=a.get('case_id', str(i)), title=a['title'],
            origin=a.get('origin', 'user_supplied_text'), body_sha256=body_hash,
            prediction_kind='direct_Codex_session_judgment', reference_case_id=r['case_id'],
            danger_score=r['danger_score'], risk_band=risk_band(r['danger_score']),
            state=r['state'], rationale=r['rationale'],
            axes={k: r[k] for k in AXES}, review_input_sha256=r['input_sha256'],
            reference_catalog_sha256=catalog_digest,
            score_scope='AI rubric judgment of this exact article version; not independent accuracy evidence')
    if pending:
        for i, r in zip(positions, score_articles(pending, model_path, model_dir)):
            # Preserve original positions even when supplied inputs omit IDs.
            r['case_id'] = articles[i].get('case_id', str(i))
            results[i] = r
    return results


def build():
    if (OUT/'release_policy.json').exists():
        raise ValueError('Follow-up release already exists; do not overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'20261110_release'
    final = pd.read_parquet(previous/'news_risk_scores.parquet')
    old = pd.read_csv(previous/'decision_audit.csv')
    fresh = pd.read_csv(DIAGNOSTIC/'real_reference.csv')
    assert len(fresh) == 32 and not set(fresh.news_id) & set(old.news_id)
    fresh['input_scope'] = 'complete_stored_cleaned_body_and_title'
    fresh['reference_origin'] = 'direct_current_session_AI_before_predictions'
    direct = pd.concat([old, fresh], ignore_index=True)
    assert len(direct) == 1032 and direct.news_id.is_unique
    lookup = final.set_index('news_id')
    for r in fresh.itertuples():
        assert lookup.loc[r.news_id].assessment_body_sha256 == r.assessment_body_sha256
        i = final.index[final.news_id.eq(r.news_id)][0]
        for k in AXES:
            final.loc[i, k] = getattr(r, k)/4
        final.loc[i, 'danger_score'] = float(danger_index([r.S, r.L, r.I, r.X]))
        final.loc[i, 'component_index'] = final.loc[i, 'danger_score']
        final.loc[i, 'session_reviewed'] = True
        final.loc[i, 'score_source'] = 'direct_Codex_session_judgment'
        final.loc[i, 'decision_state'] = r.state
        final.loc[i, 'rationale'] = r.rationale
    final['danger_band'] = risk_band(final.danger_score.to_numpy())
    final.to_csv(OUT/'news_risk_scores.csv', index=False)
    final.to_parquet(OUT/'news_risk_scores.parquet', index=False)
    direct.to_csv(OUT/'decision_audit.csv', index=False)
    records = []
    for r in direct.itertuples():
        row = lookup.loc[r.news_id]
        records.append(dict(news_id=r.news_id, case_id=r.case_id, title=row.title,
            assessment_body_sha256=row.assessment_body_sha256,
            input_sha256=input_key(row.title, row.assessment_body_sha256),
            **{k: int(getattr(r, k)) for k in AXES}, state=r.state,
            rationale=r.rationale, danger_score=float(danger_index([r.S, r.L, r.I, r.X])),
            reference_kind='direct_Codex_session_judgment'))
    catalog = OUT/'review_catalog.jsonl'
    catalog.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    assert len(load_catalog(catalog)) == 1032
    # The v8 candidate failed its real-article guard. Preserve its untouched
    # test scores and retain the previously published v5 model everywhere else.
    evidence = json.loads((DIAGNOSTIC/'test_metrics.json').read_text())
    assert not evidence['targeted_guard_passed']
    write_json(OUT/'release_policy.json', dict(model='v5', articles=8128,
        previous_release='../20261110_release', new_real_direct_reviews=32,
        total_real_direct_reviews=1032, synthetic_training_cases=32,
        synthetic_cases_in_corpus_or_review_catalog=0, failed_candidate='v8',
        candidate_reason='Real held-out MAE and within10 worsened; synthetic within10 also worsened.',
        candidate_metrics='../20261110_v8/test_metrics.json',
        no_pure_model_accuracy_gain_claimed=True,
        reviews_are_not_held_out_accuracy=True,
        exact_title_and_cleaned_body_hash_required_for_review_reuse=True,
        unchanged_unreviewed_model_predictions=7096,
        catalog_sha256=sha256(catalog)))
    summary = json.loads((previous/'summary.json').read_text())
    summary['direct_AI_judgments'] = 1032
    write_json(OUT/'summary.json', summary)
    payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',
        (previous/'report.html').read_text(), re.S).group(1))
    payload['summary'] = summary
    columns = list(payload['articles'][0])
    payload['articles'] = final[columns].fillna('').to_dict('records')
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    html = (previous/'report.html').read_text().replace('1,000건', '1,032건')
    html = re.sub(r'(<script id="payload" type="application/json">).*?(</script>)',
        lambda m: m.group(1)+encoded+m.group(2), html, flags=re.S)
    note = '<p class="note">새 실제 기사 32건을 직접 판정해 반영했습니다. 합성 64건은 실제 뉴스·직접 판정 원장에 넣지 않았습니다. v8 후보는 무해 기사 과대평가로 채택하지 않았습니다. <a href="../../../docs/new-article-diagnostics-v8.md">새 기사·합성 테스트와 남은 오류</a></p>'
    html = html.replace('<div class="cards">', note+'<div class="cards">', 1)
    # Keep historical v7 evidence links pointed at the frozen source artifacts.
    html = html.replace('href="fresh_audit_metrics.json"', 'href="../20261109_v7/fresh_audit_metrics.json"')
    (OUT/'report.html').write_text(html)
    shutil.copytree(previous/'figures', OUT/'figures', dirs_exist_ok=True)
    for svg in (OUT/'figures').glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(f'Published {len(final)} articles, {len(direct)} real AI reviews; v5 retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL with title and complete cleaned body')
    parser.add_argument('--catalog', type=Path, default=OUT/'review_catalog.jsonl')
    parser.add_argument('--model', type=Path, default=ROOT/'20261109_v5/model.joblib')
    parser.add_argument('--encoder', default='external/multilingual-e5-small')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    articles = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    results = score_with_reviews(articles, args.catalog, args.model, args.encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))


if __name__ == '__main__':
    main()
