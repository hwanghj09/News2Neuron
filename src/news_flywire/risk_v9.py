"""Fixed median calibration of v8; old artifacts and the default stay intact."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from .provenance import sha256, write_json
from .risk_v8 import metrics

OUT = Path('data/risk/20261116_v9')
BASE = Path('data/risk/20261110_v8/model.joblib')


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum()/2, side='left')])


class MedianCalibration:
    """L1 isotonic pool-adjacent-violators; tied inputs share one value."""
    def fit(self, scores, targets, weights=None):
        x, y = np.asarray(scores, float), np.asarray(targets, float)
        w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
        if x.ndim != 1 or x.shape != y.shape or x.shape != w.shape or not len(x):
            raise ValueError('Nonempty equal one-dimensional calibration arrays required')
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()) or (w <= 0).any():
            raise ValueError('Calibration arrays must be finite with positive weights')
        if (x < 0).any() or (x > 100).any() or (y < 0).any() or (y > 100).any():
            raise ValueError('Calibration indices must be in [0,100]')
        self.x = np.unique(x)
        blocks = []
        for i, value in enumerate(self.x):
            mask = x == value
            blocks.append([i, i+1, y[mask], w[mask], weighted_median(y[mask], w[mask])])
            while len(blocks) > 1 and blocks[-2][4] > blocks[-1][4]:
                b, a = blocks.pop(), blocks.pop()
                vals, weights = np.concatenate([a[2], b[2]]), np.concatenate([a[3], b[3]])
                blocks.append([a[0], b[1], vals, weights, weighted_median(vals, weights)])
        self.y = np.empty(len(self.x))
        for start, end, _, _, value in blocks:
            self.y[start:end] = value
        return self

    def predict(self, scores):
        x = np.asarray(scores, float)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Finite one-dimensional scores required')
        return np.interp(x, self.x, self.y)


class CalibratedRisk:
    def __init__(self, calibration, base_path=BASE):
        self.calibration, self.base_path = calibration, str(base_path)
        self.base_sha256 = sha256(base_path)

    def predict_articles(self, x, articles):
        if sha256(self.base_path) != self.base_sha256:
            raise ValueError('Frozen base model changed')
        raw = joblib.load(self.base_path).predict_articles(x, articles)
        return self.calibration.predict(raw)


def freeze():
    if (OUT/'model.joblib').exists() or (OUT/'judgments.tsv').exists():
        raise ValueError('Frozen or reference-opened experiment')
    d = pd.read_csv('data/risk/20261110_v8/development_oof.csv')
    assert len(d) == 1043
    weights = 1/d.groupby('family').family.transform('size').to_numpy()
    calibration = MedianCalibration().fit(d.v8_oof_score, d.reference_score, weights)
    model = CalibratedRisk(calibration)
    joblib.dump(model, OUT/'model.joblib')
    pd.DataFrame(dict(raw_score=calibration.x, calibrated_score=calibration.y)).to_csv(OUT/'mapping.csv', index=False)
    write_json(OUT/'model_freeze.json', dict(model_sha256=sha256(OUT/'model.joblib'),
        source_sha256=sha256(__file__), base_sha256=sha256(BASE),
        development_sha256=sha256('data/risk/20261110_v8/development_oof.csv'),
        protocol_sha256=sha256(OUT/'protocol.json'), selection_sha256=sha256(OUT/'selection.csv'),
        development=1043, fresh_labels_opened=False, no_parameter_search=True))


def evaluate():
    if (OUT/'test_results.csv').exists():
        raise ValueError('Test already opened; no retuning')
    from .risk_assessment import AXES, danger_index
    from .risk_inference import ArticleFeatures
    from .risk_v4 import load_news, body_for
    f = json.loads((OUT/'model_freeze.json').read_text())
    assert f['source_sha256'] == sha256(__file__)
    assert f['model_sha256'] == sha256(OUT/'model.joblib')
    assert f['selection_sha256'] == sha256(OUT/'selection.csv')
    d = pd.read_csv(OUT/'selection.csv').merge(pd.read_csv(OUT/'judgments.tsv', sep='\t'), on='case_id', validate='one_to_one')
    assert len(d) == 14
    n = load_news()
    articles = [dict(title=r.title, body=body_for(n.iloc[r.row_index])) for r in d.itertuples()]
    x, receipts = ArticleFeatures().transform(articles)
    private = Path('data/raw/risk/20261116_v9')
    np.save(private/'test_features.npy', x)
    reference = danger_index(d[AXES].to_numpy(float)[:, 3:])
    _, v5 = joblib.load('data/risk/20261109_v5/model.joblib').predict(x)
    v8 = joblib.load(BASE).predict_articles(x, articles)
    v9 = joblib.load(OUT/'model.joblib').predict_articles(x, articles)
    d['reference_score'] = reference
    scores = {'v5': v5, 'v8': v8, 'v9': v9}
    results = {}
    for version, p in scores.items():
        d[version+'_score'], d[version+'_error'] = p, np.abs(reference-p)
        results[version] = metrics(reference, p)
        benign = reference == 0
        results[version]['benign_n'] = int(benign.sum())
        results[version]['benign_over10'] = int((p[benign] > 10).sum())
    def no_worse(old):
        a, b = results['v9'], results[old]
        return a['MAE'] <= b['MAE'] and a['within10'] >= b['within10'] and (
            a['high_risk_recall'] is None or a['high_risk_recall'] >= b['high_risk_recall'])
    d.to_csv(OUT/'test_results.csv', index=False)
    write_json(OUT/'test_metrics.json', dict(results=results, guard_passed=no_worse('v5') and no_worse('v8'),
        population_accuracy_claim=False, corpus_default='v5 unchanged',
        reference='Same AI full-body rubric, 14 new stored articles; no human ground truth.',
        input_receipts=receipts, feature_sha256=sha256(private/'test_features.npy')))
    print(json.dumps(results, ensure_ascii=False, indent=2))
