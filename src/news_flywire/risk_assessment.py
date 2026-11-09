"""AI-supervised ordinal news-risk model with immutable rounds and held-out audit.

Session judgments and model extrapolations are different score sources. An AI
audit measures agreement with this AI's rubric; it never measures human truth.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from .provenance import manifest, sha256, write_json

AXES = ["N", "T", "A", "S", "L", "I", "X"]
STATES = {"none", "hypothetical", "ongoing", "imminent", "resolved", "retrospective", "prevention", "allegation"}
NARRATIVE_CUES = {
    "judicial_control":r"검거|구속|체포|재판|구형|징역",
    "emergency_resolved":r"완진|진화 완료|모두 구조|상황 종료|해제|복구 완료|안전.{0,8}확인|재가동 승인",
    "past_or_memorial":r"추모|별세|회고|당시|지난해|수년 전|회복|기념",
    "continuing":r"확산|이어지고|계속.{0,10}(발생|피해)|진행 중|증가하고|추가.{0,6}(피해|사망|감염)",
    "immediate_action":r"즉시.{0,6}대피|대피.{0,6}명령|대피령|호우경보|태풍경보|긴급.{0,8}구조|실종자",
    "conditional":r"경우|가능성|우려|가정|전망|시나리오|만약",
    "uncertain_claim":r"주장|의혹|추정|부인|확인되지|미확인|반박",
    "explicit_no_casualty":r"인명피해.{0,12}없|인명 피해.{0,12}없|다친 사람.{0,12}없|사망자.{0,8}없|위험.{0,8}없",
    "prevention_or_benefit":r"예방|안전.{0,4}교육|부담.{0,6}줄|치료.{0,6}개선|안전대책|재발방지|지원.{0,4}확대",
    "realized_harm":r"사망|숨졌|살해|부상|피해액|실직|파산|폭발|폭행",
    "celebration":r"우승|축제|수상|관람|공연|박람회|신제품|채용|출시",
}


def build_narrative_features(corpus, raw, cache):
    """Learned context cues complement embeddings; never hard-assign scores.

    Separate title/lead/body features keep a historical reference in a tail
    from being treated identically to an actual emergency in the lead.
    """
    corpus, raw, cache = map(Path, (corpus, raw, cache))
    settings={"news_sha256":sha256(corpus/"news.parquet"),"patterns":NARRATIVE_CUES,
              "zones":["title","first_500_body_chars","complete_body"],"version":"context_cues_v1"}
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    receipt=cache/"narrative_manifest.json"
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved["fingerprint"]!=fingerprint or saved["sha256"]!=sha256(cache/"narrative.npy"):
            raise ValueError("Context feature cache changed; use a fresh cache")
        return np.load(cache/"narrative.npy")
    patterns=[re.compile(value) for value in NARRATIVE_CUES.values()]
    rows=[]
    for r in pd.read_parquet(corpus/"news.parquet").itertuples():
        body=gzip.decompress((raw/"bodies"/(r.body_sha256+".txt.gz")).read_bytes()).decode()
        if hashlib.sha256(body.encode()).hexdigest()!=r.body_sha256:
            raise ValueError("Narrative input body mismatch")
        rows.append([min(1,np.log1p(len(pattern.findall(zone)))/np.log(6)) for zone in [r.title,body[:500],body] for pattern in patterns])
    values=np.asarray(rows,dtype=np.float32)
    np.save(cache/"narrative.npy",values)
    write_json(receipt,{**settings,"fingerprint":fingerprint,"sha256":sha256(cache/"narrative.npy"),"shape":list(values.shape)})
    return values


def danger_index(values):
    """A rubric index, not a probability. No remaining risk => L=0 => index=0.

    N and A do not enter this equation: grief/excitement are not future harm.
    Values may be continuous ordinal predictions; raw judgments are integers.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim==0 or x.shape[-1] != 4 or not np.isfinite(x).all() or (x < 0).any() or (x > 4).any():
        raise ValueError("S/L/I/X must be finite and in [0,4]")
    s, l, i, exposure = np.moveaxis(x, -1, 0)
    return 100 * (s / 4) * (l / 4) * (0.5 + 0.5 * i / 4) * (0.8 + 0.2 * exposure / 4)


def risk_band(score):
    return np.array(["minimal", "low", "moderate", "high", "critical"])[np.searchsorted([10, 30, 60, 85], score, side="right")]


def read_judgments(out, news):
    out = Path(out)
    selections = pd.read_csv(out / "selected_cases.csv", keep_default_na=False)
    decisions = pd.read_csv(out / "session_judgments.tsv", sep="\t", keep_default_na=False)
    if selections.case_id.duplicated().any() or selections.news_id.duplicated().any():
        raise ValueError("Selection case and article IDs must be unique")
    if decisions.case_id.duplicated().any() or not set(decisions.case_id).issubset(set(selections.case_id)):
        raise ValueError("Duplicate or unselected session judgment")
    for axis in AXES:
        if not decisions[axis].map(lambda n: type(n) is int and 0 <= n <= 4).all():
            raise ValueError(axis + " judgments must be integer 0..4")
    if not decisions.state.isin(STATES).all() or not decisions.rationale.str.len().ge(10).all():
        raise ValueError("Missing decision state or rationale")
    merged = selections.merge(decisions, on="case_id", validate="one_to_one")
    source = news.set_index("news_id")
    for r in merged.itertuples():
        if source.loc[r.news_id].body_sha256 != r.body_sha256:
            raise ValueError("Article body changed after review selection")
        if r.role == "development" and source.loc[r.news_id].split != "train":
            raise ValueError("Only temporal training articles may refine the model")
        if r.state in {"none", "resolved", "retrospective"} and (r.L != 0 or r.T != 0):
            raise ValueError("No residual risk states must have L=0 and T=0")
    if set(merged[merged.role.eq("development")].event_group) & set(merged[~merged.role.eq("development")].event_group):
        raise ValueError("Development/audit event candidate group overlap")
    return merged


def scaled_features(features):
    # All title/body embeddings already have unit norm. Probe cosines get a
    # small weight, so query wording cannot dominate the learned assessment.
    x = features.copy()
    x[:, :384] *= 0.8
    x[:, 384:768] *= 1.0
    x[:, 768:1152] *= 0.6
    x[:, 1152:] *= 0.2
    return x


def fit_predict(x, indices, labels, alpha, groups, seed=20261109):
    model = Ridge(alpha=alpha, solver="svd").fit(x[indices], labels)
    prediction = np.clip(model.predict(x), 0, 4)
    # Dispersion is model sensitivity, not an accuracy/confidence interval.
    rng = np.random.default_rng(seed)
    ensemble = []
    unique_groups = np.unique(groups)
    for _ in range(12):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        sample = np.concatenate([np.flatnonzero(np.asarray(groups)==group) for group in sampled_groups])
        learner = Ridge(alpha=alpha, solver="lsqr").fit(x[np.asarray(indices)[sample]], labels[sample])
        ensemble.append(np.clip(learner.predict(x), 0, 4))
    return model, prediction, np.asarray(ensemble)


def train_round(corpus, cache, out, round_number):
    corpus, cache, out = map(Path, (corpus, cache, out))
    news = pd.read_parquet(corpus / "news.parquet")
    feature_manifest = json.loads((cache / "features_manifest.json").read_text())
    if feature_manifest["news_parquet_sha256"] != sha256(corpus / "news.parquet") or feature_manifest["features_sha256"] != sha256(cache / "features.npy"):
        raise ValueError("Frozen feature input mismatch")
    x = scaled_features(np.load(cache / "features.npy"))
    narrative_manifest=json.loads((cache/"narrative_manifest.json").read_text())
    if narrative_manifest['news_sha256']!=sha256(corpus/'news.parquet') or narrative_manifest['sha256']!=sha256(cache/'narrative.npy'):
        raise ValueError("Narrative feature input mismatch")
    x=np.concatenate([x,np.load(cache/'narrative.npy')*.5],axis=1)
    judgments = read_judgments(out, news)
    selections = pd.read_csv(out / "selected_cases.csv")
    required = selections[selections.role.eq("development") & selections["round"].le(round_number)]
    if not set(required.case_id).issubset(set(judgments.case_id)):
        raise ValueError("Unjudged development cases remain; obtain actual AI decisions first")
    round_path = out / f"round_{round_number}"
    if round_path.exists():
        raise ValueError("A completed training round is immutable; use a new round/output")
    dev = judgments[judgments.role.eq("development") & judgments["round"].le(round_number)].copy()
    audit = judgments[judgments.role.eq("audit")].copy()
    lookup = {n: i for i, n in enumerate(news.news_id)}
    indices = np.array([lookup[n] for n in dev.news_id])
    y = dev[AXES].to_numpy(dtype=float)
    # Hyperparameters use grouped development folds only. Frozen audit labels
    # are not used here; their metrics are written only by finalize().
    folds = list(GroupKFold(n_splits=min(4, dev.event_group.nunique())).split(indices, groups=dev.event_group))
    tuning = []
    for alpha in [0.1, 1.0, 10.0]:
        errors = []
        danger_errors = []
        for tr, va in folds:
            reg = Ridge(alpha=alpha, solver="lsqr").fit(x[indices[tr]], y[tr])
            estimates = np.clip(reg.predict(x[indices[va]]), 0, 4)
            errors.append(float(mean_absolute_error(y[va], estimates)))
            danger_errors.append(float(mean_absolute_error(danger_index(y[va,3:]), danger_index(estimates[:,3:]))))
        tuning.append({"alpha": alpha, "grouped_dev_mae": float(np.mean(errors)),
                       "grouped_dev_danger_mae":float(np.mean(danger_errors)),
                       "selection_loss":.5*float(np.mean(errors))+.5*float(np.mean(danger_errors))/25})
    alpha = min(tuning, key=lambda item: item["selection_loss"])["alpha"]
    model, prediction, ensemble = fit_predict(x, indices, y, alpha, dev.event_group.to_numpy())
    risk = danger_index(prediction[:, 3:])
    bootstrap_risk = danger_index(ensemble[:, :, 3:])
    mean = prediction[:, 3:].copy()
    derivatives = []
    for s_multiplier, temporal_floor in [(0.75, 0.25), (1.0, 0.5), (1.25, 0.75)]:
        s, l, i, exposure = mean.T
        derivatives.append(np.minimum(100, 100 * np.minimum(1, s / 4 * s_multiplier) * (l / 4) * (temporal_floor + (1-temporal_floor) * i/4) * (.8+.2*exposure/4)))
    title_sim = x[:, :384] @ x[indices, :384].T / 0.8**2
    nearest = title_sim.argmax(axis=1)
    result = news[["news_id", "title", "summary", "source", "url", "published_date", "body_sha256", "event_group", "split", "research_subset_selected"]].copy()
    for j, axis in enumerate(AXES):
        result[axis + "_ordinal_prediction"] = prediction[:, j]
        result[axis] = prediction[:, j] / 4
    result["danger_score"] = risk
    result["danger_band"] = risk_band(risk)
    result["bootstrap_p10"] = np.quantile(bootstrap_risk, .1, axis=0)
    result["bootstrap_p90"] = np.quantile(bootstrap_risk, .9, axis=0)
    result["rubric_sensitivity_min"] = np.min(derivatives, axis=0)
    result["rubric_sensitivity_max"] = np.max(derivatives, axis=0)
    result["nearest_development_news_id"] = dev.news_id.to_numpy()[nearest]
    result["nearest_title_similarity"] = title_sim.max(axis=1)
    result["score_source"] = "ai_supervised_semantic_model_prediction"
    result["session_reviewed"] = False
    result["decision_state"] = "unreviewed_model_prediction"
    result["rationale"] = [f"본문 의미 모델 예측: 심각도 {s:.1f}/4, 잔존 위해 근거 {l:.1f}/4, 시간 {i:.1f}/4, 범위 {e:.1f}/4. 직접 세션 판정 아님." for s,l,i,e in prediction[:,3:]]
    result["model_round"] = round_number
    result["requires_further_ai_review"] = (result.bootstrap_p90-result.bootstrap_p10).gt(20) | result.nearest_title_similarity.lt(.72)
    # No external test-set score gets silently substituted into model metrics.
    round_path.mkdir(parents=True, exist_ok=True)
    result.to_parquet(round_path / "predictions.parquet", index=False)
    dev.to_csv(round_path / "development_judgments.tsv", sep="\t", index=False)
    write_json(round_path / "training.json", {"round": round_number, "development_articles": len(dev),
               "audit_labels_used_for_training_or_tuning": 0, "alpha": alpha, "development_tuning": tuning,
               "judgment_file_sha256": sha256(out / "session_judgments.tsv"),
               "development_snapshot_sha256": sha256(round_path / "development_judgments.tsv"),
               "development_case_ids": dev.case_id.tolist(), "frozen_audit_case_ids": selections.loc[selections.role.eq('audit'),'case_id'].tolist(),
               "frozen_challenge_case_ids":selections.loc[selections.role.eq('challenge'),'case_id'].tolist(),
               "bootstrap":"12 candidate-event-group resamples, sensitivity not a confidence interval",
               "narrative_fingerprint":narrative_manifest['fingerprint'],
               "model_input_dimensions":x.shape[1],
               "feature_fingerprint": feature_manifest["fingerprint"]})
    joblib.dump(model, round_path / "ridge_model.joblib", compress=3)
    print(f"Round {round_number}: {len(dev)} AI development judgments -> {len(result)} predictions", flush=True)
    return result


def finalize(corpus, out, round_number, cache=None):
    corpus, out = map(Path, (corpus, out))
    news = pd.read_parquet(corpus / "news.parquet")
    judgments = read_judgments(out, news)
    audit = judgments[judgments.role.eq("audit")]
    challenge = judgments[judgments.role.eq("challenge")]
    expected_audit = pd.read_csv(out / "selected_cases.csv").query("role == 'audit'")
    if set(audit.case_id) != set(expected_audit.case_id):
        raise ValueError("The entire frozen AI audit must have real session decisions")
    expected_challenge = pd.read_csv(out / "selected_cases.csv").query("role == 'challenge'")
    if set(challenge.case_id) != set(expected_challenge.case_id):
        raise ValueError("The entire targeted AI challenge must have real session decisions")
    audits = []
    challenges = []
    original_audits = []
    baseline = pd.read_csv(corpus / "provisional_rule_scores.csv").set_index("news_id")
    matched_baseline=baseline.copy()
    if cache is not None and (out/'input_corrections.csv').exists():
        from .news_scoring import rule_score
        for r in pd.read_csv(out/'input_corrections.csv').itertuples():
            body=gzip.decompress((Path(cache)/'corrected_bodies'/(r.assessment_body_sha256+'.txt.gz')).read_bytes()).decode()
            text=news.set_index('news_id').loc[r.news_id].title+' '+body
            result=rule_score(text)
            for axis in ['N','T','A']:matched_baseline.loc[r.news_id,axis]=result[axis]
    for r in range(1, round_number + 1):
        folder=out/f"round_{r}"
        prediction_path=folder/'input_corrected_predictions.parquet' if (folder/'input_corrected_predictions.parquet').exists() else folder/'predictions.parquet'
        predicted = pd.read_parquet(prediction_path).set_index("news_id")
        original_predicted = pd.read_parquet(folder/'predictions.parquet').set_index('news_id').loc[audit.news_id]
        p = predicted.loc[audit.news_id]
        metrics = {axis: float(mean_absolute_error(audit[axis], p[axis+"_ordinal_prediction"])) for axis in AXES}
        target = danger_index(audit[["S", "L", "I", "X"]].to_numpy(dtype=float))
        audits.append({"round": r, "audit_articles": len(audit), "ordinal_mae_0_to_4": metrics,
                       "danger_index_mae_0_to_100": float(mean_absolute_error(target, p.danger_score)),
                       "reference": "held_out_same_session_AI_judgments_not_independent_humans"})
        original_audits.append({'round':r,'danger_index_mae_0_to_100':float(mean_absolute_error(target,original_predicted.danger_score)),
                               'ordinal_mae_0_to_4':{axis:float(mean_absolute_error(audit[axis],original_predicted[axis+'_ordinal_prediction'])) for axis in AXES}})
        cp = predicted.loc[challenge.news_id]
        ct = danger_index(challenge[["S", "L", "I", "X"]].to_numpy(dtype=float))
        challenges.append({"round": r, "articles":len(challenge),
                           "danger_index_mae_0_to_100":float(mean_absolute_error(ct,cp.danger_score)),
                           "ordinal_mae_0_to_4":{axis:float(mean_absolute_error(challenge[axis],cp[axis+"_ordinal_prediction"])) for axis in AXES}})
    # The audit is opened only after the final model/configuration is frozen.
    write_json(out / "frozen_ai_audit.json", {"independent_human_labels": 0,
               "model_selection_used_audit_metrics": False, "rounds": audits, "targeted_challenge_rounds": challenges,
               "original_pre_input_repair_rounds":original_audits,
               "input_repair_caveat":"Native promotion discovered during audit; identical DOM repair applied to 180 rows, zero training rows affected, model parameters unchanged. Original sealed predictions retained; preprocessing was not entirely untouched by audit.",
               "legacy_rule_baseline_ordinal_mae_0_to_4": {axis:float(mean_absolute_error(audit[axis],baseline.loc[audit.news_id,axis]*4)) for axis in ["N","T","A"]},
               "input_matched_rule_baseline_ordinal_mae_0_to_4": {axis:float(mean_absolute_error(audit[axis],matched_baseline.loc[audit.news_id,axis]*4)) for axis in ["N","T","A"]},
               "scope": "Small stratified temporal holdout; measures agreement with this session's decisions, not correctness or real-world danger probability."})
    folder=out/f"round_{round_number}"
    final = pd.read_parquet(folder/'input_corrected_predictions.parquet' if (folder/'input_corrected_predictions.parquet').exists() else folder/'predictions.parquet')
    if 'assessment_body_sha256' not in final:
        final['assessment_body_sha256']=final.body_sha256
        final['assessment_research_eligible']=final.research_subset_selected
        final['input_correction']='none'
    final["model_danger_score"] = final.danger_score
    final["scoring_rubric"] = "article_remaining_danger_v3"
    final["temporal_reference"] = "article_publication_narrative_not_current_live_alert"
    by_id = judgments.set_index("news_id")
    for i, row in final.iterrows():
        if row.news_id not in by_id.index:
            continue
        d = by_id.loc[row.news_id]
        for axis in AXES:
            final.loc[i, axis] = d[axis] / 4
        final.loc[i, "danger_score"] = float(danger_index([d.S, d.L, d.I, d.X]))
        final.loc[i, "danger_band"] = risk_band(final.loc[i, "danger_score"])
        final.loc[i, "score_source"] = "direct_Codex_session_judgment"
        final.loc[i, "session_reviewed"] = True
        final.loc[i, "decision_state"] = d.state
        final.loc[i, "rationale"] = d.rationale
        final.loc[i, "requires_further_ai_review"] = False
    final.to_parquet(out / "news_risk_scores.parquet", index=False)
    final.to_csv(out / "news_risk_scores.csv", index=False)
    queue = final[~final.session_reviewed].copy()
    queue["review_priority"] = queue.bootstrap_p90-queue.bootstrap_p10+(1-queue.nearest_title_similarity)*20
    queue.sort_values("review_priority", ascending=False).head(200).to_csv(out / "next_ai_review_queue.csv", index=False)
    labels = judgments.copy()
    labels["judge"] = "current_Codex_session_AI_not_human"
    labels["input_scope"] = "title_and_logged_body_evidence_windows_not_all_body_claim"
    labels["danger_score"] = danger_index(labels[["S", "L", "I", "X"]].to_numpy(dtype=float))
    labels['assessment_body_sha256']=labels.news_id.map(final.set_index('news_id').assessment_body_sha256)
    labels.to_csv(out / "decision_audit.csv", index=False)
    counts = final.danger_band.value_counts().to_dict()
    write_json(out / "summary.json", {"articles": len(final), "full_body_model_features": len(final),
               "direct_session_judgments": len(judgments), "development_judgments": int(judgments.role.eq("development").sum()),
               "held_out_ai_audit_judgments": len(audit), "scoring_rounds": round_number,
               "targeted_challenge_judgments":len(challenge),
               "additional_quality_judgments":int(judgments.role.eq('quality').sum()),
               "assessment_research_eligible_articles":int(final.assessment_research_eligible.sum()),
               "native_promotion_corrections":int(final.input_correction.ne('none').sum()),
               "danger_band_counts": counts, "further_ai_review_candidates": int(final.requires_further_ai_review.sum()),
               "independent_human_labels": 0, "perfect_accuracy_claim": False,
               "danger_score_meaning": "rubric index 0..100, not probability, actual stress or a real-time alert"})
    return final


def select_cases(corpus, out, cache, round_number=1, count=32):
    """Freeze audit first; active rounds never query the temporal audit partition."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus, out, cache = map(Path, (corpus, out, cache))
    news = pd.read_parquet(corpus / "news.parquet").fillna("")
    selected_path = out / "selected_cases.csv"
    selected = pd.read_csv(selected_path, keep_default_na=False) if selected_path.exists() else pd.DataFrame()
    if len(selected) and (selected["round"] == round_number).any():
        raise ValueError("Selection round is already frozen")
    rng = np.random.default_rng(20261109 + round_number)
    eligible = news[news.split.eq("train") & news.research_subset_selected]
    if round_number == 1:
        rows = []
        # Six holdout articles per publisher, drawn without examining model scores.
        for source, group in news[news.split.isin(["validation", "test"]) & news.research_subset_selected].groupby("source"):
            take = group.sample(n=min(6, len(group)), random_state=20261109)
            rows.extend((i, "audit") for i in take.index)
        # Eight farthest-first titles per publisher, training only. Unlike
        # sparse mini-batch k-means this cannot silently return empty clusters.
        for source, group in eligible.groupby("source"):
            v = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=15000)
            x = v.fit_transform(group.title)
            chosen = [int(rng.integers(len(group)))]
            similarities = (x @ x[chosen[0]].T).toarray().ravel()
            for _ in range(min(8, len(group))-1):
                similarities[chosen] = np.inf
                nxt = int(np.argmin(similarities))
                chosen.append(nxt)
                similarities = np.maximum(similarities, (x @ x[nxt].T).toarray().ravel())
            # Rare severe harm must appear in development, alongside benign
            # controls. This selects candidates only; the AI can reject a hit.
            hazard = group.title.str.count(r"사망|살해|화재|붕괴|대피|호우|폭발|공습|사상|해고|피해액")
            remaining = [j for j in range(len(group)) if j not in chosen]
            pool = sorted(remaining, key=lambda j: (-int(hazard.iloc[j]), j))[:50]
            chosen.extend(int(j) for j in rng.choice(pool, min(5, len(pool)), replace=False))
            rows.extend((group.index[i], "development") for i in chosen)
    else:
        old = pd.read_parquet(out / f"round_{round_number-1}" / "predictions.parquet")
        excluded = set(selected.news_id)
        groups = set(selected.event_group)
        candidates = old[old.split.eq("train") & old.research_subset_selected & ~old.news_id.isin(excluded) & ~old.event_group.isin(groups)].copy()
        candidates["priority"] = candidates.bootstrap_p90 - candidates.bootstrap_p10 + (1-candidates.nearest_title_similarity)*20
        # Balance uncertainty exploration and very high/very low score checks.
        rows = []
        source_groups = list(candidates.groupby("source"))
        for position, (source, group) in enumerate(source_groups):
            n = count // len(source_groups) + int(position < count % len(source_groups))
            if n == 0:
                continue
            top = group.sort_values("priority", ascending=False).head(max(1, n//2))
            rest = group[~group.news_id.isin(top.news_id)]
            extremes = rest.sort_values("danger_score", ascending=False).head(1)
            random = rest[~rest.news_id.isin(extremes.news_id)].sample(n=max(0, n-len(top)-len(extremes)), random_state=20261109+round_number)
            ids = pd.concat([top, extremes, random]).news_id
            rows.extend((int(news.index[news.news_id.eq(nid)][0]), "development") for nid in ids)
    records = []
    for index, role in rows:
        row = news.loc[index]
        records.append({"case_id": f"r{round_number}_{role[0]}_{len(records)+1:03d}", "news_id": row.news_id,
                        "body_sha256": row.body_sha256, "event_group": row.event_group, "source": row.source,
                        "role": role, "round": round_number})
    fresh = pd.DataFrame(records)
    if fresh.news_id.duplicated().any():
        raise ValueError("Selection is not unique")
    selected = pd.concat([selected, fresh], ignore_index=True)
    selected.to_csv(selected_path, index=False)
    write_json(out / f"selection_round_{round_number}.json", {"case_ids": fresh.case_id.tolist(),
               "selection_sha256": sha256(selected_path), "seed": 20261109+round_number,
               "audit_selection": "source_stratified_random_temporal_holdout_frozen_before_training",
               "development_selection": "source_title_farthest_first_diversity" if round_number==1 else "uncertainty_extremes_random_training_only"})
    print(f"Selected {len(fresh)} cases for round {round_number}")
    return fresh


def select_challenge_cases(corpus, out):
    """Freeze an explicit severe/false-positive stress sample before training."""
    corpus,out=map(Path,(corpus,out))
    if list(out.glob('round_*/training.json')) or (out/'challenge_selection.csv').exists():
        raise ValueError('Challenge sample must be frozen once, before any model training')
    selected=pd.read_csv(out/'selected_cases.csv')
    news=pd.read_parquet(corpus/'news.parquet').fillna('')
    candidates=news[news.split.isin(['test','validation']) & news.research_subset_selected & ~news.news_id.isin(selected.news_id) & ~news.event_group.isin(selected.event_group)]
    rows=[]
    for source,group in candidates.groupby('source'):
        severity=group.title.str.count(r'사망|살해|붕괴|대피|호우|폭발|공습|에볼라|실종')
        top=group.loc[severity.sort_values(ascending=False).index].iloc[0]
        resolved=group[~group.news_id.eq(top.news_id) & group.title.str.contains(r'검거|추모|별세|삼성화재|안전교육|예방|진화',regex=True)]
        neutral=resolved.sample(n=1,random_state=20261109).iloc[0] if len(resolved) else group[~group.news_id.eq(top.news_id)].sample(n=1,random_state=20261109).iloc[0]
        for r in [top,neutral]:
            rows.append(dict(case_id=f'c1_c_{len(rows)+1:03d}',news_id=r.news_id,body_sha256=r.body_sha256,event_group=r.event_group,source=source,role='challenge',round=1))
    fresh=pd.DataFrame(rows)
    pd.concat([selected,fresh],ignore_index=True).to_csv(out/'selected_cases.csv',index=False)
    fresh.to_csv(out/'challenge_selection.csv',index=False)
    write_json(out/'challenge_selection.json',{'cases':len(fresh),'selection_sha256':sha256(out/'challenge_selection.csv'),
               'selection':'two_per_source: maximum_severe_headline_terms_and_random_resolved_or_prevention_headline',
               'purpose':'targeted_AI_hazard_and_false_positive_audit_not_population_estimate','frozen_before_any_training':True})
    return fresh


def review_packet(corpus, raw, out, cache, role, round_number):
    import re
    corpus, raw, out, cache = map(Path, (corpus, raw, out, cache))
    news = pd.read_parquet(corpus / "news.parquet").set_index("news_id")
    selected = pd.read_csv(out / "selected_cases.csv")
    selected = selected[selected.role.eq(role) & selected["round"].eq(round_number)]
    path = cache / f"review_packet_{role}_{round_number}.json"
    receipt_path=out / f"review_packet_{role}_{round_number}_receipt.json"
    if path.exists():
        stored=json.loads(path.read_text())
        if [p['case_id'] for p in stored]!=selected.case_id.tolist():
            raise ValueError('Frozen review packet selection changed')
        if not receipt_path.exists() or json.loads(receipt_path.read_text())['sha256']!=sha256(path):
            raise ValueError('Frozen review packet hash mismatch')
        return stored
    packets = []
    for r in selected.itertuples():
        row = news.loc[r.news_id]
        body = gzip.decompress((raw / "bodies" / (r.body_sha256 + ".txt.gz")).read_bytes()).decode()
        private_evidence = cache / "evidence" / (r.body_sha256 + ".json")
        if private_evidence.exists():
            evidence = json.loads(private_evidence.read_text())["salient_window"]
            selection = "lead_plus_semantic_salient_window"
        else:
            sentences = re.split(r"(?<=[.!?다])\s+|\n", body)
            pattern = re.compile(r"사망|부상|피해|위험|경보|화재|대피|전쟁|감염|살해|해고|붕괴|중단|우려|사기|폭행")
            evidence = max(sentences, key=lambda s: len(pattern.findall(s)), default="")
            selection = "lead_plus_lexical_evidence_candidate_not_a_score"
        packet = {"case_id": r.case_id, "title": row.title, "lead": body[:180], "evidence": evidence[:220],
                  "body_sha256": r.body_sha256, "selection": selection}
        packets.append(packet)
    write_json(path, packets)
    # Only hashes, never the private full review text, enter the public audit.
    write_json(out / f"review_packet_{role}_{round_number}_receipt.json", {"sha256": sha256(path), "articles": len(packets),
               "case_input_sha256": {p["case_id"]: hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for p in packets}})
    return packets
