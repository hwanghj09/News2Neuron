from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

from .news_scoring import load_news, score_news, event_split, label_template, import_llm_scores
from .flywire_graph import Circuit, demo_circuit, extract_v783
from .experiments import run_experiments
from .reporting import make_report
from .evaluation import evaluate_ratings, compare_observations
from .provenance import write_json, manifest

REAL_NEWS = "data/real_news/2026-11-09/news.csv"

def config(path):
    with open(path,encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser=argparse.ArgumentParser(description="뉴스 위협성 → 가상 입력 → FlyWire LIF 연구 파이프라인")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("collect-news",help="Collect real articles from official RSS feeds, without a synthetic fallback")
    p.add_argument("--config",default="configs/news_sources_v1.yaml")
    p.add_argument("--out",required=True)
    p=sub.add_parser("analyze-news",help="Report actual article ratings, baseline differences and shared-level model responses")
    p.add_argument("--news",default="data/real_news/2026-11-09/news.csv")
    p.add_argument("--scores",default="data/derived/real_news_scores.csv")
    p.add_argument("--simulation",default="results/v783")
    p.add_argument("--out",required=True)
    p=sub.add_parser("prepare",help="Validate articles, freeze event split, write blank 3-rater sheet")
    p.add_argument("--news",default=REAL_NEWS)
    p.add_argument("--out",default="data/labels/real_news_20261109")
    p.add_argument("--config",default="configs/scoring_v1.yaml")
    p=sub.add_parser("score",help="Rule baseline or strict LLM JSONL import")
    p.add_argument("--news",default=REAL_NEWS)
    p.add_argument("--llm-jsonl")
    p.add_argument("--model")
    p.add_argument("--out",default="data/derived/real_news_rule_scores.csv")
    p.add_argument("--config",default="configs/scoring_v1.yaml")
    p=sub.add_parser("extract",help="Extract confirmed paths from measured v783 connectivity")
    p.add_argument("--config",default="configs/connectome_v783.yaml")
    p.add_argument("--out",default="data/derived/flywire_v783")
    for command in ["demo","run"]:
        p=sub.add_parser(command,help="Run E0–E9 and produce figures and an HTML report")
        p.add_argument("--config",default="configs/stimulation_v1.yaml")
        p.add_argument("--scores",default="data/derived/real_news_scores.csv" if command=="run" else "data/derived/news_scores.csv")
        p.add_argument("--seeds",type=int)
        p.add_argument("--out",default=f"results/{command}")
        if command=="run":
            p.add_argument("--circuit",default="data/derived/flywire_v783")
    p=sub.add_parser("report",help="Rebuild report from stored run outputs")
    p.add_argument("--run",required=True)
    p.add_argument("--circuit",default="data/derived/flywire_v783")
    p.add_argument("--scores",default="data/derived/news_scores.csv")
    p=sub.add_parser("evaluate",help="Held-out human-vs-model ratings, ordinal alpha, bootstrap CIs")
    p.add_argument("--scores",default="data/derived/news_scores.csv")
    p.add_argument("--labels",required=True)
    p.add_argument("--splits",default="data/labels/splits.csv")
    p.add_argument("--out",default="results/news_evaluation")
    p=sub.add_parser("validate",help="Compare independently measured cell-type response directions")
    p.add_argument("--observed",required=True)
    p.add_argument("--run",default="results/v783")
    p.add_argument("--circuit",default="data/derived/flywire_v783")
    p.add_argument("--out",default="results/validation.csv")
    args=parser.parse_args(argv)
    try:
        if args.command=="collect-news":
            from .news_ingestion import collect_news
            news=collect_news(config(args.config),args.out)
            print(f"{len(news)} actual articles → {args.out}/news.csv")
        elif args.command=="analyze-news":
            from .news_analysis import analyze_news
            print(analyze_news(args.news,args.scores,args.out,args.simulation))
        elif args.command=="prepare":
            news=load_news(args.news)
            scoring=config(args.config)
            out=Path(args.out)
            out.mkdir(parents=True,exist_ok=True)
            if any((out/f).exists() for f in ["splits.csv","ratings_template.csv"]):
                raise ValueError("Split/ratings files already exist; choose a fresh --out to avoid overwriting ratings")
            fraction,seed=scoring["split"]["test_fraction"],scoring["split"]["seed"]
            event_split(news,fraction,seed).to_csv(out/"splits.csv",index=False)
            label_template(news).to_csv(out/"ratings_template.csv",index=False)
            write_json(out/"split_manifest.json",manifest([args.news,args.config],seed=seed,test_fraction=fraction,unit="event_group"))
            print(out)
        elif args.command=="score":
            news=load_news(args.news)
            scoring=config(args.config)
            if scoring["version"]!="scoring_v1" or scoring["scale"]!=[0,4]:
                raise ValueError("This scorer requires scoring_v1 with raw scale 0..4")
            if args.llm_jsonl and not args.model:
                raise ValueError("--model must identify the LLM and version")
            scored=import_llm_scores(args.llm_jsonl,news,args.model) if args.llm_jsonl else score_news(news)
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            scored.to_csv(args.out,index=False)
            write_json(str(args.out)+".manifest.json",manifest([args.news,args.config]+([args.llm_jsonl] if args.llm_jsonl else []),
                method=args.model if args.llm_jsonl else "rule_baseline_v1", evaluated_against_humans=False))
            print(args.out)
        elif args.command=="extract":
            circuit=extract_v783(config(args.config))
            circuit.save(args.out)
            print(f"{len(circuit.ids)} neurons / {len(circuit.edges)} edges → {args.out}")
        elif args.command in ["demo","run"]:
            if args.command=="run" and not Path(args.scores).exists():
                raise FileNotFoundError(f"Scores file does not exist: {args.scores}. Run score first.")
            scores=pd.read_csv(args.scores,keep_default_na=False,dtype={"news_id":str,"event_group":str}) if Path(args.scores).exists() else score_news(load_news("data/sample_news.csv"))
            c=config(args.config)
            circuit=demo_circuit() if args.command=="demo" else Circuit.load(args.circuit)
            run_experiments(circuit,c,args.out,seeds=args.seeds,news_scores=scores)
            circuit.save(Path(args.out)/"circuit")
            input_files=[args.config]+([args.scores] if Path(args.scores).exists() else ["data/sample_news.csv"])
            write_json(Path(args.out)/"input_manifest.json",manifest(input_files))
            print(make_report(args.out,circuit,scores,c))
        elif args.command=="report":
            c=json.loads((Path(args.run)/"resolved_config.json").read_text())
            print(make_report(args.run,Circuit.load(args.circuit),pd.read_csv(args.scores),c))
        elif args.command=="evaluate":
            out=Path(args.out)
            out.mkdir(parents=True,exist_ok=True)
            result,consensus,subgroups=evaluate_ratings(pd.read_csv(args.scores,dtype={"news_id":str,"event_group":str}),pd.read_csv(args.labels,dtype={"news_id":str,"rater_id":str}),pd.read_csv(args.splits,dtype={"news_id":str,"event_group":str}))
            write_json(out/"evaluation.json",result)
            consensus.to_csv(out/"consensus_and_predictions.csv",index=False)
            subgroups.to_csv(out/"subgroup_errors.csv",index=False)
            print(out/"evaluation.json")
        elif args.command=="validate":
            observations=pd.read_csv(args.observed,keep_default_na=False)
            cells=pd.read_parquet(Path(args.run)/"cell_metrics.parquet")
            comparison=compare_observations(observations,Circuit.load(args.circuit),cells)
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            comparison.to_csv(args.out,index=False)
            print(args.out)
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.exit(2,f"Error: {error}\n")


if __name__=="__main__":
    main()
