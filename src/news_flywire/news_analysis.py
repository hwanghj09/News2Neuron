"""Evaluate real article inputs and link them to previously simulated drive levels."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import spearman
from .news_scoring import load_news, score_news
from .provenance import manifest, sha256, write_json
from .stimulus_encoder import encode


def validate_real_news(news):
    required={"url","source","published_date","collected_at","input_scope","summary_source"}
    if not required.issubset(news):
        raise ValueError(f"Real news is missing provenance columns: {sorted(required-set(news))}")
    if not news.data_source.eq("real_publisher_rss").all():
        raise ValueError("This analysis requires actual publisher RSS articles, with no synthetic mixture")
    if news.news_id.duplicated().any() or news.url.duplicated().any():
        raise ValueError("Duplicate real article IDs or URLs")
    if not news.url.str.match(r"https?://[^/]+/").all():
        raise ValueError("Actual article URLs are required")
    dates=pd.to_datetime(news.published_date,utc=True,errors="raise")
    collection=pd.to_datetime(news.collected_at,utc=True,errors="raise")
    if dates.isna().any() or collection.isna().any() or (dates>collection).any():
        raise ValueError("Publication dates must be known and no later than collection")
    if news.source.eq("").any() or news.summary.str.strip().eq("").any():
        raise ValueError("Publisher and evaluated excerpt are required")


def link_article_responses(scores, simulation_dir):
    """Reuse valid level simulations: articles at one level are not independent trials."""
    run=Path(simulation_dir)
    c=json.loads((run/"resolved_config.json").read_text())
    metrics=pd.read_csv(run/"metrics.csv")
    rows=[]
    for article in scores.to_dict("records"):
        stimulus=encode(article,c)
        condition={"L":"E1","M":"E2","H":"E3"}[stimulus.level]
        selected=metrics[metrics.condition_id.eq(condition)]
        if selected.empty:
            raise ValueError(f"Stored simulation has no {condition}")
        for response in selected.to_dict("records"):
            rows.append({**response,"news_id":article["news_id"],"event_group":article["event_group"],
                "N_ai":article["N"],"T_ai":article["T"],"A_ai":article["A"],
                "article_data_source":article["data_source"],"stimulus_level":stimulus.level,
                "article_url":article["url"],"shared_level_simulation":True,
                "simulation_source":str(run),"simulation_manifest_sha256":sha256(run/"manifest.json")})
    return pd.DataFrame(rows)


def analyze_news(news_path,scores_path,out_dir,simulation_dir="results/v783"):
    news=load_news(news_path)
    validate_real_news(news)
    scores=pd.read_csv(scores_path,dtype={"news_id":str,"event_group":str},keep_default_na=False)
    if scores.news_id.duplicated().any() or set(scores.news_id)!=set(news.news_id):
        raise ValueError("Each actual article must have exactly one score record")
    source=news.set_index("news_id")
    for field in ["title","summary","url","source","event_group","data_source"]:
        if field not in scores or not scores[field].equals(scores.news_id.map(source[field])):
            raise ValueError(f"Score record does not match evaluated article metadata: {field}")
    for axis in "NTA":
        scores[axis]=pd.to_numeric(scores[axis],errors="raise")
        if not np.isfinite(scores[axis]).all() or not scores[axis].between(0,1).all():
            raise ValueError("Normalized scores must be finite and between 0 and 1")
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Analysis output already exists; choose a fresh directory")
    out.mkdir(parents=True,exist_ok=True)
    baseline=score_news(news)
    comparison=scores.merge(baseline[["news_id","N","T","A"]].rename(columns={k:f"{k}_rule" for k in "NTA"}),on="news_id",validate="one_to_one")
    for axis in "NTA":
        comparison[f"{axis}_difference"]=comparison[axis]-comparison[f"{axis}_rule"]
    comparison.to_csv(out/"article_scores.csv",index=False)
    baseline.to_csv(out/"rule_baseline.csv",index=False)
    linked=link_article_responses(scores,simulation_dir)
    linked.to_csv(out/"article_responses.csv",index=False)
    levels=linked.groupby("news_id").agg(stimulus_level=("stimulus_level","first"),model_delta_mean_hz=("delta_rate_hz","mean"),computational_seeds=("seed","count"))
    table=comparison.merge(levels,left_on="news_id",right_index=True,validate="one_to_one")
    table.to_csv(out/"article_summary.csv",index=False)
    category=table.groupby("category").agg(articles=("news_id","count"),N_mean=("N","mean"),T_mean=("T","mean"),A_mean=("A","mean"))
    category.to_csv(out/"category_summary.csv")
    agreement={axis:{"spearman_between_scorers":spearman(comparison[axis],comparison[f"{axis}_rule"]),
                     "mean_absolute_difference":float(np.abs(comparison[f"{axis}_difference"]).mean())} for axis in "NTA"}
    summary={"articles":len(news),"events":int(news.event_group.nunique()),"sources":news.source.value_counts().to_dict(),
        "categories":news.category.value_counts().to_dict(),"levels":table.stimulus_level.value_counts().to_dict(),
        "scorers":sorted(scores.scorer.unique()),"score_sources":sorted(scores.score_source.unique()),
        "published_min":news.published_date.min(),"published_max":news.published_date.max(),
        "agreement_with_rule_baseline":agreement,"human_accuracy_evaluated":False,
        "simulation_reused_from":str(simulation_dir),"new_simulation_trials":0,
        "simulation_scope":"30 computational seeds per predefined level, shared across articles",
        "input_scope":"Title plus short RSS excerpt only; no full-article claim"}
    write_json(out/"summary.json",summary)
    write_json(out/"manifest.json",manifest([news_path,scores_path,Path(simulation_dir)/"manifest.json"],
        data_source="real_news_and_ai_ratings",human_reference_available=False,report_module_sha256=sha256(__file__)))
    _report(out,table,category,summary)
    return out/"report.html"


def _report(out,table,category,summary):
    colors=["#287a9b","#ce7742","#528f68","#92549f","#bd4b62","#657487"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    rng=np.random.default_rng(20261109)
    for i,(name,group) in enumerate(table.groupby("category")):
        axes[0].scatter(group["T"]+rng.uniform(-.012,.012,len(group)),group["N"]+rng.uniform(-.012,.012,len(group)),s=24,alpha=.65,label=name,color=colors[i%6])
    axes[0].set(xlabel="Threat (T)",ylabel="Negativity (N)",xlim=(-.05,1.05),ylim=(-.05,1.05),title="Real news · contextual AI ratings")
    axes[0].legend(fontsize=7)
    axes[1].scatter(table.T_rule,table["T"],s=22,alpha=.5,color=colors[0])
    axes[1].plot([0,1],[0,1],ls="--",color="#8896a0")
    axes[1].set(xlabel="Keyword baseline T",ylabel="Contextual AI T",title="Scorer disagreement · not human accuracy")
    fig.tight_layout()
    fig.savefig(out/"news_evaluation.png",dpi=180)
    fig.savefig(out/"news_evaluation.svg")
    plt.close(fig)
    def esc(value):return html.escape(str(value),quote=True)
    rows=[]
    for article in table.to_dict("records"):
        rows.append(f'''<tr data-category="{esc(article['category'])}" data-source="{esc(article['source'])}" data-threat="{article['T']}"><td><a href="{esc(article['url'])}" target="_blank" rel="noopener noreferrer">{esc(article['title'])}</a><p>{esc(article['summary'])}</p><small>{esc(article['news_id'])}</small></td><td>{esc(article['source'])}<br><small>{esc(article['published_date'])}</small></td><td>{esc(article['category'])}</td><td>{article['N']:.2f}</td><td>{article['T']:.2f}</td><td>{article['A']:.2f}</td><td>{article['T_rule']:.2f}</td><td>{esc(article['rationale'])}</td><td>{esc(article['stimulus_level'])}<br>{article['model_delta_mean_hz']:.2f} Hz</td></tr>''')
    options=''.join(f'<option value="{esc(x)}">{esc(x)}</option>' for x in sorted(table.category.unique()))
    source_options=''.join(f'<option value="{esc(x)}">{esc(x)}</option>' for x in sorted(table.source.unique()))
    category_html=category.round(3).to_html(border=0)
    total=len(table)
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>News2Neuron · 실제 뉴스 평가 {total}건</title>
<style>*{{box-sizing:border-box}}body{{font:15px/1.7 system-ui,sans-serif;margin:0;background:#eff4f7;color:#23384c}}main{{max-width:1500px;margin:auto;padding:40px 24px}}h1{{font-size:clamp(27px,4vw,43px);line-height:1.3}}h2{{font-size:22px}}.tag{{font-size:12px;color:#2a7c85;letter-spacing:.12em}}section{{background:white;border:1px solid #dae4eb;border-radius:12px;padding:24px;margin:20px 0}}.notice{{background:#fff7e8;border-left:4px solid #b97d2c;padding:18px}}.cards{{display:flex;gap:15px;flex-wrap:wrap;margin:24px 0}}.card{{background:white;padding:18px 25px;border-radius:10px;border:1px solid #dce5ed}}.card strong{{display:block;font-size:29px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e3eaf0;padding:12px;text-align:left;vertical-align:top}}th{{background:#f2f6fa;position:sticky;top:0}}td p{{font-size:12px;color:#637587;max-width:500px;margin:8px 0}}small{{font-size:10px;color:#6d7e8c}}a{{color:#236d92}}.overflow{{overflow:auto}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}input,select,button{{font:inherit;padding:8px 12px;border:1px solid #bfccd7;border-radius:7px;max-width:100%}}button{{cursor:pointer;background:#edf5f8}}.downloads a{{display:inline-block;margin:6px 16px 6px 0}}#articles{{min-width:1150px}}#articles td:first-child{{min-width:270px}}#articles td:nth-child(8){{min-width:210px}}</style>
<main><div class="tag">NEWS2NEURON · ACTUAL PUBLISHER NEWS</div><h1>실제 뉴스 {total}건의<br>부정성·위협성·예상 각성 평가</h1><p>언론사 공식 RSS의 제목과 짧은 공개 발췌문을 평가했습니다. 기사별 원문 링크, AI 근거, 규칙 baseline과의 차이를 제공합니다.</p>
<div class="notice">점수는 현재 Codex 세션의 문맥 기반 AI 판단입니다. 원문 전체를 읽었다는 뜻이 아니며, 독립된 사람 정답·정확도·실제 생리 반응은 아직 없습니다. AI 자체 판단을 사람 평가로 취급하지 않습니다.</div>
<div class="cards"><div class="card"><strong>{total}</strong>실제 기사</div><div class="card"><strong>{len(summary['sources'])}</strong>언론사</div><div class="card"><strong>{len(summary['categories'])}</strong>매체 주제 분야</div><div class="card"><strong>{summary['events']}</strong>보수적 사건 그룹</div></div>
<section><h2>점수의 의미와 입력 범위</h2><p>N=부정성, T=위협성, A=예상 정서적 각성. 각 0~4 정수를 0~1로 정규화했습니다. 개인적 관련성이나 실제 스트레스 확률은 추정하지 않습니다. 위협성은 기사 발행 당시의 서술을 기준으로 판단하며, 현재 유효한 재난 경보나 의료 권고를 제공하는 서비스가 아닙니다.</p><p>평가 범위는 기사 제목과 짧은 RSS 발췌문입니다. 기사별 인용은 합계 25개 공백 단위 단어 이내로 제한했습니다. 분야는 언론사의 RSS 분류이며 긍정·중립·부정의 정답 레이블이 아닙니다. 이번 표본은 제공된 피드의 편의표본입니다. 유사 기사와 확인한 후속 보도를 묶었으나 사건 그룹은 보수적 근사이며 독립 사람의 검토는 아직 없습니다.</p><p>발행 범위(UTC): {esc(summary['published_min'])} ~ {esc(summary['published_max'])}</p></section>
<section><h2>문맥 평가와 규칙 baseline 비교</h2><img src="news_evaluation.png" alt="실제 뉴스 N/T 분포와 두 평가기의 차이"><p>점 겹침을 줄이기 위해 왼쪽 그림에 작은 시각적 흔들림을 주었습니다. 저장된 점수는 바꾸지 않았습니다. 두 평가기의 차이는 사람 정답에 대한 오류가 아닙니다.</p><div class="overflow">{category_html}</div></section>
<section><h2>기사별 평가</h2><div class="filters"><input id="query" aria-label="기사 검색" placeholder="제목·요약·근거 검색"><select id="category" aria-label="분야"><option value="">모든 분야</option>{options}</select><select id="source" aria-label="언론사"><option value="">모든 언론사</option>{source_options}</select><button id="high" aria-pressed="false">T ≥ 0.5만 보기</button><span id="count">{total}건</span></div><div class="overflow"><table id="articles"><thead><tr><th>제목·평가 발췌문·기사 ID</th><th>출처·발행일(UTC)</th><th>매체 분야</th><th>N</th><th>T</th><th>A</th><th>T 규칙</th><th>AI 평가 근거</th><th>가상 입력 단계<br>모델 Δ rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>모델 출력 연결의 의미</h2><p>사전에 계산한 E1/E2/E3의 30개 시드 결과를 점수 단계별로 연결했습니다. 새로운 기사별 뇌 시뮬레이션을 독립적으로 실행한 결과가 아닙니다. 같은 T 단계의 기사는 같은 모델 출력 분포를 공유합니다.</p><p>T &lt; 1/3 → L(30 Hz), T &lt; 2/3 → M(80 Hz), 나머지 → H(150 Hz). 따라서 T=0도 L 입력에 대응하며 무입력 E0와 다릅니다. 모델의 출력 Hz는 사람의 뉴스 스트레스 수치가 아닙니다.</p></section>
<section class="downloads"><h2>자료 다운로드</h2>{''.join(f'<a href="{name}">{name}</a>' for name in ['article_scores.csv','article_summary.csv','rule_baseline.csv','article_responses.csv','category_summary.csv','summary.json','manifest.json'])}</section></main>
<script>let highOnly=false;const rows=[...document.querySelectorAll('#articles tbody tr')];function apply(){{const q=document.querySelector('#query').value.toLocaleLowerCase();const cat=document.querySelector('#category').value;const source=document.querySelector('#source').value;let n=0;for(const row of rows){{const show=(!q||row.textContent.toLocaleLowerCase().includes(q))&&(!cat||row.dataset.category===cat)&&(!source||row.dataset.source===source)&&(!highOnly||Number(row.dataset.threat)>=.5);row.hidden=!show;if(show)n++;}}document.querySelector('#count').textContent=n+'건';}}for(const id of ['query','category','source'])document.querySelector('#'+id).addEventListener('input',apply);document.querySelector('#high').addEventListener('click',function(){{highOnly=!highOnly;this.setAttribute('aria-pressed',String(highOnly));this.textContent=highOnly?'전체 위협 점수 보기':'T ≥ 0.5만 보기';apply();}});</script></html>'''
    (out/"report.html").write_text(document,encoding="utf-8")
    (out/"report.md").write_text(f"# 실제 뉴스 평가 {total}건\n\n공식 RSS 제목·짧은 발췌문, {len(summary['sources'])}개 언론사, 6개 매체 분야.\n\n평가: 현재 Codex 세션의 N/T/A 문맥 판단 및 규칙 baseline. 독립 사람 평가 및 정확도 검증 없음.\n\n기사별 URL·점수·근거: article_scores.csv\n\n기존 30시드 단계별 모델 결과를 공유하여 연결: article_responses.csv. 새로운 기사별 독립 시뮬레이션이 아님.\n",encoding="utf-8")
