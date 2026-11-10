"""Score new, cleaned article text with the same complete-body features as v5.

Raw HTML is deliberately not accepted: callers must use the corpus extractor
first. No labels, article IDs, or instructions inside an article affect inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np

from .risk_assessment import NARRATIVE_CUES, scaled_features, risk_band
from .risk_features import Encoder, PROBES, normalize
from .risk_input_quality import incomplete_numeric_draft
from .risk_v4 import extended_context
from .risk_v5 import EXTRA


def validate_article(title, body):
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError('title and body must be strings')
    if not title.strip() or not body.strip():
        raise ValueError('title and complete cleaned body are required')
    if re.search(r'<(?:html|body|script|article)\b', body, re.I):
        raise ValueError('Extract cleaned article prose before scoring HTML')
    if incomplete_numeric_draft(body):
        raise ValueError('Incomplete numeric draft must be quarantined before scoring')


def legacy_context(title, body):
    narrative = [min(1, np.log1p(len(re.findall(p, zone)))/np.log(6))
                 for zone in [title, body[:500], body] for p in NARRATIVE_CUES.values()]
    extra = [min(1, np.log1p(len(re.findall(p, zone)))/np.log(5))
             for zone in [title, body[:500], body[-700:], body] for p in EXTRA.values()]
    return np.concatenate([np.asarray(narrative, np.float32)*.5,
                           extended_context(title, body)*.5,
                           np.asarray(extra, np.float32)*.5])


class ArticleFeatures:
    def __init__(self, model_dir='external/multilingual-e5-small'):
        self.encoder = Encoder(model_dir)
        self.probes = self.encoder.single(['query: '+p for p in PROBES])

    def transform(self, articles):
        values, receipts = [], []
        for article in articles:
            title, body = article['title'], article['body']
            validate_article(title, body)
            text = 'passage: '+title+'\n'+body
            encoded = self.encoder.tokenizer.encode(text)
            chunks = [encoded]+list(encoded.overflowing)
            vectors = np.concatenate([self.encoder.encode_chunks(chunks[i:i+16])
                                      for i in range(0, len(chunks), 16)])
            similarity = vectors @ self.probes.T
            selected = int(np.argmax(similarity[:, :5].max(axis=1)-similarity[:, 5:].max(axis=1)))
            semantic = np.concatenate([self.encoder.single(['passage: '+title])[0],
                normalize(vectors.mean(axis=0)), vectors[selected],
                similarity.max(axis=0), similarity.mean(axis=0)]).astype(np.float32)
            values.append(np.concatenate([scaled_features(semantic[None])[0], legacy_context(title, body)]))
            receipts.append(dict(body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                windows=len(chunks), input_chars=len(text), all_body_tokens_processed=True))
        return np.asarray(values, dtype=np.float32).reshape(-1, 1297), receipts


def score_articles(articles, model_path, model_dir='external/multilingual-e5-small'):
    """Only load trusted local joblib artifacts; never accept a remote model URL."""
    x, receipts = ArticleFeatures(model_dir).transform(articles)
    if not len(x):
        return []
    model = joblib.load(model_path)
    if hasattr(model, 'predict_articles'):
        scores = model.predict_articles(x, articles)
    else:
        _, scores = model.predict(x)
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    return [dict(case_id=a.get('case_id', str(i)), origin=a.get('origin', 'user_supplied_text'),
                 title=a['title'], danger_score=float(s), risk_band=risk_band(float(s)),
                 score_scope='learned remaining-harm index at the time described; not a probability',
                 prediction_kind='pure_model_prediction', model_sha256=digest,
                 **receipt) for i, (a, s, receipt) in enumerate(zip(articles, scores, receipts))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL containing title and complete cleaned body')
    parser.add_argument('--model', type=Path, default=Path('data/risk/20261109_v5/model.joblib'))
    parser.add_argument('--encoder', default='external/multilingual-e5-small')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    articles = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    results = score_articles(articles, args.model, args.encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))


if __name__ == '__main__':
    main()
