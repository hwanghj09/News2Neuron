"""Reproducible AI refinement stages; never fabricates missing AI decisions."""
import argparse
import json

from news_flywire.risk_assessment import finalize, review_packet, select_cases, select_challenge_cases, train_round, build_narrative_features

p = argparse.ArgumentParser()
p.add_argument("stage", choices=["features", "select", "challenge-select", "packet", "train", "finalize"])
p.add_argument("--corpus", default="data/corpus/20261109_v2")
p.add_argument("--raw", default="data/raw/corpus/20261109_v2")
p.add_argument("--out", default="data/risk/20261109_v3")
p.add_argument("--cache", default="data/raw/risk/20261109_v3")
p.add_argument("--encoder", default="external/multilingual-e5-small")
p.add_argument("--round", type=int, default=1)
p.add_argument("--role", choices=["development", "audit", "challenge"], default="development")
p.add_argument("--count", type=int, default=32)
a = p.parse_args()
if a.stage == "features":
    from news_flywire.risk_features import build_features
    build_features(a.corpus, a.raw, a.out, a.cache, a.encoder)
    build_narrative_features(a.corpus,a.raw,a.cache)
elif a.stage == "select":
    select_cases(a.corpus, a.out, a.cache, a.round, a.count)
elif a.stage == "challenge-select":
    select_challenge_cases(a.corpus,a.out)
elif a.stage == "packet":
    print(json.dumps(review_packet(a.corpus, a.raw, a.out, a.cache, a.role, a.round), ensure_ascii=False, indent=2))
elif a.stage == "train":
    train_round(a.corpus, a.cache, a.out, a.round)
else:
    finalize(a.corpus, a.out, a.round, a.cache)
    from news_flywire.risk_reporting import report
    print(report(a.out))
