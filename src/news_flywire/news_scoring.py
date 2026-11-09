"""Transparent rule baseline and strict import of externally generated LLM ratings."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LEXICON = {
    "N": {"사망": 4, "부상": 3, "피해": 3, "붕괴": 4, "대피": 3, "실직": 3,
          "침체": 3, "해고": 3, "사별": 4, "추모": 3, "별세": 4, "폐업": 3,
          "추락": 3, "감염": 3, "적자": 2, "우려": 2, "감소": 2, "사고": 3,
          "경보": 3, "위험": 3, "태풍": 3, "홍수": 3, "불안": 2},
    "T": {"즉시 대피": 4, "대피 명령": 4, "접근": 3, "경보": 4, "임박": 4,
          "붕괴 위험": 4, "확산": 3, "태풍": 3, "홍수": 3, "위험": 3,
          "감염": 2, "해고": 2, "실직": 2, "침체": 1, "우려": 1},
    "A": {"즉시": 4, "경보": 4, "대피": 4, "사망": 3, "사별": 3, "별세": 3,
          "추모": 3, "태풍": 3, "위험": 3, "해고": 2, "침체": 2,
          "우승": 3, "축제": 2, "성공": 2, "기부": 2},
}
REQUIRED_NEWS = {"news_id", "event_group", "title", "summary", "category", "data_source"}


def load_news(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not REQUIRED_NEWS.issubset(df):
        raise ValueError(f"Missing news columns: {sorted(REQUIRED_NEWS - set(df))}")
    if df.news_id.duplicated().any() or (df.news_id == "").any() or (df.event_group == "").any():
        raise ValueError("news_id must be unique; news_id and event_group must be nonempty")
    return df


def rule_score(text: str):
    result = {}
    rationales = []
    for axis, lexicon in LEXICON.items():
        hits = {word: weight for word, weight in lexicon.items() if word in text}
        result[axis] = max(hits.values(), default=0) / 4.0
        rationales.append(f"{axis}: {', '.join(hits) or '일치 어휘 없음'}")
    return {**result, "confidence": None, "rationale": "; ".join(rationales),
            "scorer": "rule_baseline_v1", "score_source": "rule_prediction"}


def score_news(df):
    scores = pd.DataFrame([rule_score(row.title + " " + row.summary) for row in df.itertuples()])
    return pd.concat([df.reset_index(drop=True), scores], axis=1)


def import_llm_scores(path, news, model: str, prompt_version="nta_v1"):
    """Input is JSONL with integer 0..4 N/T/A. Confidence is never accuracy."""
    if not model.strip():
        raise ValueError("An explicit model/version is required")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        allowed = {"news_id", "N", "T", "A", "confidence", "rationale"}
        if not {"news_id", "N", "T", "A", "rationale"}.issubset(item) or set(item) - allowed:
            raise ValueError("LLM output must follow the documented JSON schema")
        for axis in "NTA":
            value = item[axis]
            if type(value) is not int or not 0 <= value <= 4:
                raise ValueError(f"{axis} must be an integer from 0 through 4")
            item[axis] = value / 4.0
        c = item.get("confidence")
        if c is not None and (isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1):
            raise ValueError("confidence must be null or a finite 0..1 number")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError("rationale must be nonempty text")
        item.update(scorer=model, prompt_version=prompt_version, score_source="llm_prediction")
        rows.append(item)
    ratings = pd.DataFrame(rows)
    if ratings.empty or ratings.news_id.duplicated().any() or set(ratings.news_id) != set(news.news_id):
        raise ValueError("LLM ratings must contain each news_id exactly once")
    return news.merge(ratings, on="news_id", validate="one_to_one")


def event_split(news, fraction=0.25, seed=20261109):
    if news.event_group.nunique() < 2:
        raise ValueError("At least two independent event groups are required")
    _, test = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(news, groups=news.event_group))
    out = news[["news_id", "event_group"]].copy()
    out["split"] = "development"
    out.loc[out.index[test], "split"] = "test"
    return out


def label_template(news):
    return pd.DataFrame([{"news_id": n, "rater_id": r, "N": "", "T": "", "A": "", "notes": ""}
                         for n in news.news_id for r in ["rater_1", "rater_2", "rater_3"]])
