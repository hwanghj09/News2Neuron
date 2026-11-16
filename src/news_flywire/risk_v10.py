"""Context-sensitive residual-harm calibration; all older experiments stay frozen."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .provenance import sha256, write_json
from .risk_assessment import AXES, danger_index
from .risk_v8 import context_features, dataset, metrics
from .risk_v9 import BASE, MedianCalibration

OUT = Path('data/risk/20261116_v10')
PRIVATE = Path('data/raw/risk/20261116_v10')


class ContextCalibration:
    def __init__(self, C=.1, mode='soft'):
        if C <= 0 or mode not in {'soft', 'hard'}:
            raise ValueError('Positive C and soft/hard mode required')
        self.C, self.mode = C, mode

    @staticmethod
    def inputs(raw, context):
        raw, context = np.asarray(raw, float), np.asarray(context, float)
        if raw.ndim != 1 or context.ndim != 2 or len(raw) != len(context):
            raise ValueError('Scores and context rows must align')
        if not np.isfinite(raw).all() or not np.isfinite(context).all() or (raw < 0).any() or (raw > 100).any():
            raise ValueError('Finite context and raw indices in [0,100] required')
        return np.column_stack([raw/100, context])

    def fit(self, raw, context, target, groups):
        z = self.inputs(raw, context)
        y = np.asarray(target, float)
        if y.shape != (len(z),) or not len(y) or not np.isfinite(y).all() or (y < 0).any() or (y > 100).any() or len(groups) != len(y):
            raise ValueError('Aligned nonempty targets in [0,100] and families required')
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1/np.sqrt(counts[inverse]); weights /= weights.mean()
        positive = y > 0
        self.constant = bool(positive[0]) if len(np.unique(positive)) == 1 else None
        self.scaler = StandardScaler().fit(z, sample_weight=weights)
        if self.constant is None:
            self.classifier = LogisticRegression(C=self.C, max_iter=3000).fit(
                self.scaler.transform(z), positive, sample_weight=weights)
        self.curve = MedianCalibration().fit(np.asarray(raw)[positive], y[positive], weights[positive]) if positive.any() else None
        return self

    def support(self, raw, context):
        z = self.inputs(raw, context)
        if self.constant is not None:
            return np.full(len(z), float(self.constant))
        return self.classifier.predict_proba(self.scaler.transform(z))[:, 1]

    def predict(self, raw, context):
        support = self.support(raw, context)
        if self.curve is None:
            return np.zeros(len(support))
        conditional = self.curve.predict(raw)
        return conditional*(support if self.mode == 'soft' else (support >= .5))


class ContextRisk:
    def __init__(self, calibration):
        self.calibration = calibration
        self.base_path, self.base_sha256 = str(BASE), sha256(BASE)

    def predict_articles(self, x, articles):
        if sha256(self.base_path) != self.base_sha256:
            raise ValueError('Frozen base model changed')
        raw = joblib.load(self.base_path).predict_articles(x, articles)
        return self.calibration.predict(raw, context_features(articles))


def development():
    from .risk_v4 import load_news, body_for
    d, _, articles = dataset(False)
    old_oof = pd.read_csv('data/risk/20261110_v8/development_oof.csv').set_index('case_id').v8_oof_score
    old_test = pd.read_csv('data/risk/20261110_v8/test_results.csv').set_index('case_id').v8_score
    scores = pd.concat([old_oof, old_test])
    raw = scores.loc[d.case_id].to_numpy()
    recent = pd.read_csv('data/risk/20261116_v9/test_results.csv')
    n = load_news()
    for r in recent.itertuples():
        articles.append(dict(title=r.title, body=body_for(n.iloc[r.row_index])))
    add = recent[['case_id', *AXES]].copy()
    add['family'] = 'real_'+recent.family.astype(str)
    add['origin'] = 'retired_v9_real'
    d = pd.concat([d, add], ignore_index=True)
    raw = np.r_[raw, recent.v8_score]
    assert len(d) == 1105 and d.case_id.is_unique
    return d, raw, context_features(articles)


def train():
    if (OUT/'model.joblib').exists() or (OUT/'judgments.tsv').exists():
        raise ValueError('Frozen or reference-opened experiment')
    d, raw, context = development()
    groups = d.family.to_numpy(); target = danger_index(d[AXES].to_numpy(float)[:, 3:])
    selection = pd.read_csv(OUT/'selection.csv')
    assert not set('real_'+str(i) for i in selection.family) & set(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1/counts[inverse]
    results, outputs = [], []
    for C in [.1, 1.]:
        for mode in ['soft', 'hard']:
            p = np.empty(len(d))
            for tr, va in GroupKFold(n_splits=4).split(raw, groups=groups):
                model = ContextCalibration(C=C, mode=mode).fit(raw[tr], context[tr], target[tr], groups[tr])
                p[va] = model.predict(raw[va], context[va])
            error = np.abs(target-p)
            loss = np.average(error, weights=weights)+4*np.average(error > 10, weights=weights)+8*np.average(p[target >= 30] < 30, weights=weights[target >= 30])
            results.append(dict(C=C, mode=mode, loss=float(loss), **metrics(target, p)))
            outputs.append(p)
    i = min(range(len(results)), key=lambda j: results[j]['loss'])
    params = {k: results[i][k] for k in ['C', 'mode']}
    calibration = ContextCalibration(**params).fit(raw, context, target, groups)
    joblib.dump(ContextRisk(calibration), OUT/'model.joblib')
    d['reference_score'], d['raw_v8_score'], d['calibration_oof_score'] = target, raw, outputs[i]
    d.to_csv(OUT/'development_oof.csv', index=False)
    write_json(OUT/'training.json', dict(development=1105, configurations=results, selected=params,
        fresh_test_used=0, note='Calibration CV on historical base OOF plus retired tests; not fully nested base-model CV.'))
    write_json(OUT/'model_freeze.json', dict(model_sha256=sha256(OUT/'model.joblib'),
        source_sha256=sha256(__file__), base_sha256=sha256(BASE),
        protocol_sha256=sha256(OUT/'protocol.json'), selection_sha256=sha256(OUT/'selection.csv'),
        development_sha256=sha256(OUT/'development_oof.csv'), fresh_judgments_opened=False))
    print(json.dumps(results, indent=2)); print('Frozen selected', params)


def evaluate():
    from .risk_inference import ArticleFeatures
    from .risk_v4 import load_news, body_for
    if (OUT/'test_results.csv').exists():
        raise ValueError('Final audit already opened; no retuning')
    freeze = json.loads((OUT/'model_freeze.json').read_text())
    ref = json.loads((OUT/'reference_freeze.json').read_text())
    for key, path in [('source_sha256', __file__), ('model_sha256', OUT/'model.joblib'),
                      ('protocol_sha256', OUT/'protocol.json'), ('selection_sha256', OUT/'selection.csv')]:
        assert sha256(path) == freeze[key], path
    assert sha256(OUT/'judgments.tsv') == ref['judgments_sha256']
    d = pd.read_csv(OUT/'selection.csv').merge(pd.read_csv(OUT/'judgments.tsv', sep='\t'), on='case_id', validate='one_to_one')
    assert len(d) == 28
    n = load_news()
    articles = [dict(title=r.title, body=body_for(n.iloc[r.row_index])) for r in d.itertuples()]
    x, receipts = ArticleFeatures().transform(articles)
    np.save(PRIVATE/'test_features.npy', x)
    reference = danger_index(d[AXES].to_numpy(float)[:, 3:])
    d['reference_score'] = reference
    _, v5 = joblib.load('data/risk/20261109_v5/model.joblib').predict(x)
    v8 = joblib.load(BASE).predict_articles(x, articles)
    v9 = joblib.load('data/risk/20261116_v9/model.joblib').calibration.predict(v8)
    model = joblib.load(OUT/'model.joblib'); context = context_features(articles)
    v10 = model.calibration.predict(v8, context)
    d['presence_support'] = model.calibration.support(v8, context)
    for version, p in [('v5', v5), ('v8', v8), ('v9', v9), ('v10', v10)]:
        d[version+'_score'], d[version+'_error'] = p, np.abs(reference-p)
    results = {}
    for cohort, part in d.groupby('cohort'):
        results[cohort] = {}
        for version in ['v5', 'v8', 'v9', 'v10']:
            prediction = part[version+'_score'].to_numpy()
            result = metrics(part.reference_score, prediction)
            benign, harmed = part.reference_score.eq(0), part.reference_score.gt(0)
            result['benign_n'] = int(benign.sum()); result['benign_over10'] = int((prediction[benign] > 10).sum())
            result['remaining_harm_n'] = int(harmed.sum()); result['remaining_harm_zeroed'] = int((prediction[harmed] == 0).sum())
            results[cohort][version] = result
    def guard(cohort, old):
        a, b = results[cohort]['v10'], results[cohort][old]
        return a['MAE'] <= b['MAE'] and (cohort != 'random' or a['within10'] >= b['within10']) and (a['high_risk_recall'] is None or a['high_risk_recall'] >= b['high_risk_recall'])
    d.to_csv(OUT/'test_results.csv', index=False)
    write_json(OUT/'test_metrics.json', dict(cohorts=results,
        guard_passed=all(guard(c, v) for c in results for v in ['v5', 'v9']),
        corpus_default='v5 unchanged; no population accuracy claim',
        receipts=receipts, feature_sha256=sha256(PRIVATE/'test_features.npy')))
    print(json.dumps(results, ensure_ascii=False, indent=2))
