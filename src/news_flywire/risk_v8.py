"""Targeted new-article diagnostics; synthetic tests are functional specifications.

This experiment does not replace the published v5 corpus release. All previous
AI references are development here. Final scenario families are held out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

from .provenance import sha256, write_json
from .risk_assessment import AXES, danger_index
from .risk_v4 import load_news, body_for
from .risk_v7 import CUES

OUT = Path('data/risk/20261110_v8')
PRIVATE = Path('data/raw/risk/20261110_v8')
ADDITIONAL = {
    'rehabilitation': r'재활|복귀.{0,8}(?:기다|못|대기)|출전.{0,8}못|출전.{0,8}어렵',
    'critical_unresolved': r'의식.{0,8}(?:회복하지 못|찾지 못|없)|생사.{0,8}확인되지|중태|위독|중환자',
    'unpaid_or_unrepaired': r'퇴거|체납|미지급|반환되지|돌려받지|복구.{0,8}끝나지|수습.{0,8}않|밸브.{0,8}잠그지',
    'complete_recovery': r'모두.{0,8}(?:퇴원|귀가)|복귀.{0,8}(?:승인|허가)|전액.{0,10}(?:돌려|반환)|재활.{0,8}마쳤|후유증.{0,5}없',
    'simulation_only': r'가상|모형|모의|훈련|실제.{0,12}(?:없|아니|발생하지)|배우가.{0,8}연기|특수효과',
    'actual_contrast': r'실제.{0,8}(?:사고|폭발|부상|누출|화재|넘쳐)|하지만|그러나|그런데|반면|불구하고',
}


def context_features(articles):
    patterns = list(CUES.values())+list(ADDITIONAL.values())
    values = []
    for a in articles:
        body = a['body']
        row = [min(1, np.log1p(len(re.findall(p, zone)))/np.log(5))
               for zone in [a['title'], body[:600], body[-800:], body] for p in patterns]
        # Learned co-occurrence, not a rule assigning risk. A closed fire and
        # an unconscious patient can both be present in the same article.
        clauses = re.split(r'[.!?]\s|\n|(?:하지만|그러나|그런데)', body)
        closing = re.compile(CUES['completed_control']+'|'+ADDITIONAL['complete_recovery'])
        ongoing = re.compile(CUES['current_treatment']+'|'+ADDITIONAL['critical_unresolved']+'|'+ADDITIONAL['unpaid_or_unrepaired'])
        simulation = re.compile(ADDITIONAL['simulation_only'])
        row += [min(1, sum(bool(p.search(c)) for c in clauses)/3)
                for p in [closing, ongoing, simulation]]
        row += [float(bool(closing.search(body)) and bool(ongoing.search(body))),
                float(bool(simulation.search(body)) and bool(re.search(ADDITIONAL['actual_contrast'], body)))]
        values.append(row)
    return np.asarray(values, dtype=np.float32).reshape(-1, len(patterns)*4+5)


def article_texts(articles):
    return [(a['title']+'\n')*2+a['body'] for a in articles]


class EvidenceKernel:
    """Separate semantic/context kernels avoid drowning semantics in cue counts."""
    def __init__(self, cue_weight=.35, high_weight=2., floor=3.):
        self.cue_weight, self.high_weight, self.floor = cue_weight, high_weight, floor

    def kernel(self, x, context, lex):
        sem = rbf_kernel(x[:, :1152], self.x[:, :1152], gamma=1.)
        cue = rbf_kernel(context, self.context, gamma=.06)
        return (1-self.cue_weight-.25)*sem+self.cue_weight*cue+.25*(lex@self.lex.T).toarray()

    def fit(self, x, articles, y, groups):
        self.x = x.copy()
        self.context = np.concatenate([x[:, 1152:], context_features(articles)], axis=1)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5),
            max_features=60000, min_df=2, sublinear_tf=True)
        self.lex = self.vectorizer.fit_transform(article_texts(articles))
        target = danger_index(y[:, 3:])/100
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = np.where(target >= .3, self.high_weight, 1.)/np.sqrt(counts[inv])
        weights /= weights.mean()
        self.reg = SVR(C=12., epsilon=.015, kernel='precomputed').fit(
            self.kernel(x, self.context, self.lex), target, sample_weight=weights)
        return self

    def predict_articles(self, x, articles):
        context = np.concatenate([x[:, 1152:], context_features(articles)], axis=1)
        scores = np.clip(self.reg.predict(self.kernel(x, context,
            self.vectorizer.transform(article_texts(articles)))), 0, 1)*100
        return np.where(scores < self.floor, 0., scores)


class EvidenceTree:
    def __init__(self, high_weight=2., leaves=7):
        self.high_weight, self.leaves = high_weight, leaves

    def transform(self, x, articles):
        return np.concatenate([self.pca.transform(x[:, :1152]), x[:, 1152:],
                               context_features(articles)], axis=1)

    def fit(self, x, articles, y, groups):
        self.pca = PCA(n_components=min(24, len(x)-1), svd_solver='randomized', random_state=20261114).fit(x[:, :1152])
        target = danger_index(y[:, 3:])
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        w = np.where(target >= 30, self.high_weight, 1.)/np.sqrt(counts[inv])
        w /= w.mean()
        self.reg = HistGradientBoostingRegressor(loss='squared_error', max_iter=200,
            learning_rate=.05, max_leaf_nodes=self.leaves, min_samples_leaf=8,
            l2_regularization=2., early_stopping=False, random_state=20261114).fit(
                self.transform(x, articles), target, sample_weight=w)
        return self

    def predict_articles(self, x, articles):
        return np.clip(self.reg.predict(self.transform(x, articles)), 0, 100)


class EvidenceBlend:
    def __init__(self, kernel, tree, tree_weight):
        self.kernel, self.tree, self.tree_weight = kernel, tree, tree_weight

    def predict_articles(self, x, articles):
        return (1-self.tree_weight)*self.kernel.predict_articles(x, articles)+self.tree_weight*self.tree.predict_articles(x, articles)


def prepare_data():
    from .risk_inference import ArticleFeatures
    if (PRIVATE/'dataset.npz').exists():
        raise ValueError('Prepared data immutable')
    news = load_news()
    old = pd.read_csv('data/risk/20261110_release/decision_audit.csv')
    old = old[old.role.ne('quality')].copy()
    assert len(old) == 995
    families = pd.read_csv('data/risk/20261109_v7/corpus_families.csv').set_index('news_id').family
    lookup = {n: i for i, n in enumerate(news.news_id)}
    metadata, articles, blocks = [], [], []
    xold = np.load('data/raw/risk/20261109_v5/features.npy')
    for r in old.itertuples():
        i = lookup[r.news_id]
        metadata.append(dict(case_id=r.case_id, news_id=r.news_id, role='development',
            origin='legacy_real', family='real_'+str(families.loc[r.news_id]),
            **{k: getattr(r, k) for k in AXES}))
        articles.append(dict(title=news.title.iloc[i], body=body_for(news.iloc[i])))
        blocks.append(xold[i].copy())
    real = pd.read_csv(OUT/'real_reference.csv')
    for r in real.itertuples():
        metadata.append(dict(case_id=r.case_id, news_id=r.news_id, role=r.role,
            origin='new_real', family='real_'+str(r.family), **{k: getattr(r, k) for k in AXES}))
        articles.append(dict(title=r.title, body=body_for(news.iloc[r.row_index])))
        blocks.append(xold[r.row_index].copy())
    x = np.asarray(blocks, np.float32)
    # Recompute titles individually: dynamic quantization otherwise makes a
    # title depend on unrelated batch peers. Full-body windows are unchanged.
    encoder = ArticleFeatures()
    for i, a in enumerate(articles):
        x[i, :384] = encoder.encoder.single(['passage: '+a['title']])[0]*.8
        if (i+1) % 200 == 0:
            print('Canonical titles', i+1, flush=True)
    synthetic = [json.loads(line) for line in (OUT/'synthetic_cases.jsonl').read_text().splitlines()]
    for r in synthetic:
        metadata.append({k: r[k] for k in ['case_id', 'role', 'family', *AXES]} | dict(origin='synthetic'))
        articles.append(dict(title=r['title'], body=r['body']))
    x = np.concatenate([x, np.load(PRIVATE/'synthetic_features.npy')])
    d = pd.DataFrame(metadata)
    development = d.role.eq('development').to_numpy()
    assert development.sum() == 1043 and (~development).sum() == 48
    assert not set(d.family[development]) & set(d.family[~development])
    np.savez_compressed(PRIVATE/'dataset.npz', x=x)
    d.to_csv(OUT/'diagnostic_rows.csv', index=False)
    (PRIVATE/'articles.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False)+'\n' for a in articles))
    write_json(OUT/'dataset_manifest.json', dict(rows=len(d), development=int(development.sum()), test=48,
        feature_sha256=sha256(PRIVATE/'dataset.npz'), articles_sha256=sha256(PRIVATE/'articles.jsonl'),
        metadata_sha256=sha256(OUT/'diagnostic_rows.csv'), source_sha256=sha256(__file__),
        canonical_title_policy='one title per encoder call; same during training and inference',
        no_test_families_in_training=True))


def dataset(development_only=True):
    d = pd.read_csv(OUT/'diagnostic_rows.csv')
    x = np.load(PRIVATE/'dataset.npz')['x']
    articles = [json.loads(line) for line in (PRIVATE/'articles.jsonl').read_text().splitlines()]
    mask = d.role.eq('development').to_numpy() if development_only else np.ones(len(d), bool)
    return d[mask].reset_index(drop=True), x[mask], [a for a, m in zip(articles, mask) if m]


def metrics(reference, prediction):
    error = np.abs(np.asarray(reference)-prediction)
    high = np.asarray(reference) >= 30
    return dict(n=len(error), MAE=float(error.mean()), within10=float((error <= 10).mean()),
        high_risk_n=int(high.sum()), high_risk_recall=float((prediction[high] >= 30).mean()) if high.any() else None,
        max_error=float(error.max()))


def development_baseline():
    d, x, articles = dataset()
    target = danger_index(d[AXES].to_numpy()[:, 3:])
    _, p = joblib.load('data/risk/20261109_v5/model.joblib').predict(x)
    d['reference_score'] = target
    d['v5_score'] = p
    d['absolute_error'] = np.abs(target-p)
    fresh = d.origin.ne('legacy_real')
    d[fresh].to_csv(OUT/'development_baseline.csv', index=False)
    print(d[fresh].sort_values('absolute_error', ascending=False)[['case_id', 'origin', 'reference_score', 'v5_score', 'absolute_error']].head(20).to_string(index=False), flush=True)


def train():
    if (OUT/'model.joblib').exists() or (OUT/'test_results.csv').exists():
        raise ValueError('Frozen or opened experiment; do not retune')
    d, x, articles = dataset()
    y = d[AXES].to_numpy(float)
    target = danger_index(y[:, 3:])
    groups = d.family.to_numpy()
    folds = list(GroupKFold(n_splits=4).split(x, groups=groups))
    kernels = [dict(cue_weight=c, high_weight=h, floor=3.) for c in [.15, .4] for h in [1., 2.]]
    trees = [dict(high_weight=h, leaves=l) for h in [1., 2.] for l in [7, 15]]
    write_json(OUT/'training_plan.json', dict(kernels=kernels, trees=trees, tree_weights=[0., .5, 1.],
        objective='0.5 * old-family-balanced MAE + 0.5 * fresh development MAE + 4 * overall high-risk miss rate',
        dev_only=True, test_labels_used=0, source_sha256=sha256(__file__),
        functional_promotion='For these diagnostics only: test real MAE and within10 do not worsen, synthetic MAE improves. No corpus-default promotion or population superiority claim.'))
    oofs = []
    for cls, configs in [(EvidenceKernel, kernels), (EvidenceTree, trees)]:
        values = []
        for config in configs:
            p = np.empty(len(d))
            for tr, va in folds:
                model = cls(**config).fit(x[tr], [articles[i] for i in tr], y[tr], groups[tr])
                p[va] = model.predict_articles(x[va], [articles[i] for i in va])
            values.append(p)
            print(cls.__name__, config, metrics(target, p), flush=True)
        oofs.append(values)
    legacy = d.origin.eq('legacy_real').to_numpy()
    _, inv, counts = np.unique(groups[legacy], return_inverse=True, return_counts=True)
    weights = 1/counts[inv]
    comparisons = []
    for ki, kp in enumerate(oofs[0]):
        for ti, tp in enumerate(oofs[1]):
            for w in [0., .5, 1.]:
                p = (1-w)*kp+w*tp
                error = np.abs(target-p)
                loss = .5*np.average(error[legacy], weights=weights)+.5*error[~legacy].mean()+4*(p[target >= 30] < 30).mean()
                comparisons.append(dict(kernel=ki, tree=ti, tree_weight=w, loss=float(loss),
                    **metrics(target, p), fresh_development=metrics(target[~legacy], p[~legacy])))
    selected = min(comparisons, key=lambda r: r['loss'])
    kernel = EvidenceKernel(**kernels[selected['kernel']]).fit(x, articles, y, groups)
    tree = EvidenceTree(**trees[selected['tree']]).fit(x, articles, y, groups)
    model = EvidenceBlend(kernel, tree, selected['tree_weight'])
    joblib.dump(model, OUT/'model.joblib')
    p = (1-selected['tree_weight'])*oofs[0][selected['kernel']]+selected['tree_weight']*oofs[1][selected['tree']]
    d['reference_score'], d['v8_oof_score'] = target, p
    d.to_csv(OUT/'development_oof.csv', index=False)
    write_json(OUT/'training.json', dict(selected=selected, comparisons=comparisons,
        development_articles=len(d), test_articles_used=0))
    write_json(OUT/'model_freeze.json', dict(model_sha256=sha256(OUT/'model.joblib'),
        source_sha256=sha256(__file__), inference_sha256=sha256(Path(__file__).with_name('risk_inference.py')),
        dataset_sha256=sha256(PRIVATE/'dataset.npz'), test_predictions_not_opened=True))
    print('FROZEN', selected, flush=True)


def evaluate():
    if (OUT/'test_results.csv').exists():
        raise ValueError('Final test already opened; do not rerun or retune')
    frozen = json.loads((OUT/'model_freeze.json').read_text())
    assert frozen['source_sha256'] == sha256(__file__)
    assert frozen['model_sha256'] == sha256(OUT/'model.joblib')
    d, x, articles = dataset(False)
    mask = d.role.eq('test').to_numpy()
    d = d[mask].copy()
    x = x[mask]
    articles = [a for a, m in zip(articles, mask) if m]
    target = danger_index(d[AXES].to_numpy()[:, 3:])
    _, baseline = joblib.load('data/risk/20261109_v5/model.joblib').predict(x)
    p = joblib.load(OUT/'model.joblib').predict_articles(x, articles)
    d['reference_score'], d['v5_score'], d['v8_score'] = target, baseline, p
    d['v5_error'], d['v8_error'] = np.abs(target-baseline), np.abs(target-p)
    d['v8_within10'] = d.v8_error <= 10
    d.to_csv(OUT/'test_results.csv', index=False)
    results = {o: {'v5': metrics(target[d.origin.eq(o)], baseline[d.origin.eq(o)]),
                    'v8': metrics(target[d.origin.eq(o)], p[d.origin.eq(o)])} for o in d.origin.unique()}
    real, synthetic = results['new_real'], results['synthetic']
    passed = (real['v8']['MAE'] <= real['v5']['MAE'] and real['v8']['within10'] >= real['v5']['within10']
              and synthetic['v8']['MAE'] < synthetic['v5']['MAE'])
    write_json(OUT/'test_metrics.json', dict(cohorts=results, targeted_guard_passed=passed,
        default_corpus_model='v5 remains unchanged; small same-AI diagnostics are not population evidence',
        unresolved_test_errors_over10=d.loc[~d.v8_within10, 'case_id'].tolist(),
        reference='One AI evaluator; synthetic specifications and real article references reported separately.',
        model_sha256=sha256(OUT/'model.joblib')))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
