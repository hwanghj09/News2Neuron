from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def report(out_dir):
    out=Path(out_dir)
    summary=json.loads((out/"summary.json").read_text())
    news=pd.read_parquet(out/"news.parquet")
    coverage=pd.read_csv(out/"source_day_coverage.csv")
    figures=out/"figures";figures.mkdir(exist_ok=True)
    colors=["#277f9c","#7b5ba7","#de8e3b","#4c9e77","#bb6073","#6b8298","#77a0a5","#ac855d"]
    def save(fig,name):
        fig.savefig(figures/(name+".png"),dpi=160,bbox_inches="tight")
        fig.savefig(figures/(name+".svg"),bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=["Candidate sample","Pages attempted","Verified before dedup","Unique accepted","Research subset"]
    values=[summary[k] for k in ["sampled_candidates","pages_attempted","body_verified_before_deduplication","accepted_unique_articles","research_subset_articles"]]
    axes[0].barh(labels[::-1],values[::-1],color=colors[:5])
    axes[0].set(title="Processing funnel · measured counts",xlabel="Records")
    axes[1].hist(news.body_chars,bins=40,color=colors[0])
    axes[1].set(title="Accepted full-body length",xlabel="Characters",ylabel="Articles")
    save(fig,"01_funnel_body_quality")
    matrix=coverage.pivot(index="day",columns="source",values="articles").fillna(0)
    full=pd.date_range(summary["configured_start"],summary["configured_end"]).strftime("%Y-%m-%d")
    matrix=matrix.reindex(full,fill_value=0)
    fig,ax=plt.subplots(figsize=(12,4.5))
    matrix.index=pd.to_datetime(matrix.index)
    for i,column in enumerate(matrix):
        ax.plot(matrix.index,matrix[column],label=column,color=colors[i%8],lw=1.1)
    ax.set(title="Publisher × day coverage · gaps are retained",ylabel="Accepted articles per UTC day",xlabel="Publication date")
    ax.legend(ncol=4,fontsize=8);fig.autofmt_xdate();save(fig,"02_source_date_coverage")
    reasons=pd.Series(summary["quarantine_reasons"]).sort_values(ascending=False).head(12)
    fig,ax=plt.subplots(figsize=(12,5))
    ax.barh(reasons.index[::-1],reasons.to_numpy()[::-1],color=colors[3])
    ax.set(title="Recorded quarantine reasons · multiple reasons per record possible",xlabel="Flag occurrences")
    save(fig,"03_quarantine_reasons")
    groups=pd.read_csv(out/"event_group_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].hist(groups.articles,bins=25,color=colors[1]);axes[0].set_yscale("log")
    axes[0].set(title="Candidate event-group size",xlabel="Articles per group",ylabel="Groups (log)")
    counts=news.split.value_counts();axes[1].bar(counts.index,counts.values,color=colors[:len(counts)])
    axes[1].tick_params(axis="x",labelrotation=15)
    axes[1].set(title="Time-based partitions with boundary purge",ylabel="Unique accepted articles")
    save(fig,"04_groups_partitions")
    source=news.groupby("source").agg(articles=("news_id","count"),first_publication=("published_date","min"),
        last_publication=("published_date","max"),median_body_chars=("body_chars","median"))
    source["covered_days"]=coverage.groupby("source").day.nunique()
    source.to_csv(out/"publisher_quality_summary.csv")
    source_html=source.round(1).to_html(border=0)
    checks_path=out/"verification.json"
    checks=json.loads(checks_path.read_text()) if checks_path.exists() else {"status":"pending"}
    public=news[["news_id","source","url","title","summary","published_date","body_chars","quality_flags","event_group","split","research_subset_selected"]].to_dict("records")
    payload=json.dumps(public,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
    options="".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sorted(news.source.unique()))
    split_options="".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sorted(news.split.unique()))
    goals=summary["goals"]
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>News2Neuron · 대규모 뉴스 정제 보고서</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eff4f7;color:#20384b;font:15px/1.75 system-ui,sans-serif}}main{{max-width:1440px;margin:auto;padding:42px 24px}}h1{{font-size:clamp(28px,4vw,44px);line-height:1.25}}.tag{{font-size:12px;color:#2e7c8d;letter-spacing:.12em}}.notice{{background:#fff7e7;border-left:4px solid #b97c27;padding:18px}}section{{background:white;border:1px solid #dce6ec;border-radius:12px;padding:24px;margin:22px 0}}.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}.card{{border:1px solid #dae5ed;border-radius:10px;background:white;padding:17px 22px}}.card strong{{font-size:29px;display:block}}.card span{{font-size:12px;color:#607889}}img{{width:100%;height:auto}}.overflow{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:10px;border-bottom:1px solid #e2eaf0;text-align:left;vertical-align:top}}th{{background:#f2f6fa}}input,select,button{{font:inherit;padding:8px 11px;border:1px solid #bdccd7;border-radius:7px;max-width:100%}}button{{background:#edf5f8;cursor:pointer}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:15px}}a{{color:#246e92}}#records{{min-width:1080px}}#records td:first-child{{min-width:300px;max-width:500px}}td p{{color:#647c8d;font-size:11px}}small{{color:#758b99}}.download a{{display:inline-block;margin:7px 16px 7px 0}}li{{margin:8px 0}}</style>
<main><div class="tag">NEWS2NEURON · CORPUS QUALITY AUDIT V2</div><h1>최근 90일 한국어 뉴스<br>수집·본문 확인·중복 제거·분할 감사</h1><p>설정 기간(UTC): {summary['configured_start']} ~ {summary['configured_end']}. 공식 매체 사이트맵으로 후보를 찾고 실제 기사 응답에서 본문과 발행일 근거를 확인했습니다.</p>
<div class="notice">대규모 자료의 점수는 검증 전 단어 규칙 baseline입니다. 기존 240건의 세션 판단을 새 기사의 정답으로 옮기지 않았습니다. 사건 묶음은 자동 후보이며 사람 검증은 아직 없습니다.</div>
<div class="cards">{''.join(f'<div class="card"><strong>{v:,}</strong><span>{label}</span></div>' for label,v in [('본문 확인·정제 후 고유 기사',summary['accepted_unique_articles']),('확인 시도한 페이지',summary['pages_attempted']),('확인 전 후보 표본',summary['sampled_candidates']),('매체',len(summary['publishers'])),('자료가 있는 UTC 날짜',summary['covered_utc_days']),('확정 전 사건 후보',summary['event_groups'])])}</div>
<section><h2>서로 다른 수를 구분해 기록했습니다</h2><p>사이트맵에서 발견한 전체 고유 URL은 {summary['discovered_unique_urls']:,}개입니다. 이 목록은 날짜 미확인·기간 밖 후보를 포함하며 검증된 기사 수가 아닙니다. URL/사이트맵 날짜 힌트에 따른 범위 후보는 {summary['discovery_date_hint_in_window']:,}개이며 실제 발행일은 기사 본문 메타데이터에서 다시 확인합니다.</p><p>후보 표본 {summary['sampled_candidates']:,}건에서 {summary['pages_attempted']:,}건을 확인했고, {summary['body_verified_before_deduplication']:,}건이 본문·날짜 기준을 통과했습니다. 중복 {summary['duplicates_removed']:,}건을 제거한 뒤 {summary['accepted_unique_articles']:,}건을 보존했습니다.</p><p>경계 사건·후기 수정·기존 평가 중복을 제외하고 매체 비중을 제한한 연구용 부분집합은 {summary['research_subset_articles']:,}건입니다. 원자료를 삭제하지 않고 선택 여부를 별도 필드에 남겼습니다. 목표 3천 건 달성: {'예' if goals['accepted_goal_met'] else '아니오'} · 매체 목표 달성: {'예' if goals['publisher_goal_met'] else '아니오'}.</p><img src="figures/01_funnel_body_quality.png" alt="정제 단계별 건수와 본문 길이"></section>
<section><h2>매체·시간 분포</h2><p>날짜별 제공 범위가 매체마다 다릅니다. 빈 날짜를 채우거나 한두 매체의 시계열을 전체 한국 뉴스로 일반화하지 않았습니다.</p><img src="figures/02_source_date_coverage.png" alt="매체와 날짜별 뉴스 자료 분포"><div class="overflow">{source_html}</div></section>
<section><h2>제외와 보류 근거</h2><p>본문 부족·발행 시각 충돌·접근 제한·비기사 자료 등을 서로 구분했습니다. 한 기사에 사유가 여럿일 수 있어 아래 빈도의 합은 제외 기사 수와 다를 수 있습니다.</p><img src="figures/03_quarantine_reasons.png" alt="보류 사유별 빈도"></section>
<section><h2>중복과 사건, 시간 순서 분할</h2><p>완전히 같은 본문과 거의 같은 재송고는 중복으로 묶었습니다. 제목이 같다는 이유만으로 다른 본문을 삭제하지 않았고, 수치가 달라진 후속 보도는 보존했습니다. 사건 후보는 최대 72시간 범위 안에서 묶었습니다.</p><p>학습은 10월 11일까지, 검증은 10월 25일까지, 테스트는 이후 기간입니다. 같은 사건 후보가 시간 경계를 넘으면 양쪽에 나누지 않고 경계 보류로 분리했습니다. 명시된 수정 시점이 구간을 넘은 기사도 보류하며, 확인된 기존 240건 평가와 겹치는 기사·연결 그룹은 평가용 부분집합에서 제외했습니다.</p><p>자동 사건 묶음이 모든 의미상 누출을 찾았다는 뜻은 아닙니다. 수정 시각이 없는 경우를 표시했고 현재 수집한 본문이 과거 발행 당시와 같다고 보증하지 않습니다. 발행일 분할만으로 과거 시점 예측 성능이 입증되지는 않습니다.</p><img src="figures/04_groups_partitions.png" alt="사건 그룹과 분할 현황"></section>
<section><h2>기사 표본 탐색</h2><div class="filters"><input id="query" aria-label="기사 검색" placeholder="제목·짧은 발췌 검색"><select id="source" aria-label="매체"><option value="">모든 매체</option>{options}</select><select id="split" aria-label="분할"><option value="">모든 분할</option>{split_options}</select><button id="flagged" aria-pressed="false">주의 플래그만</button><span id="count"></span></div><div class="overflow"><table id="records"><thead><tr><th>제목·공개 발췌</th><th>매체·발행일 UTC</th><th>본문 길이</th><th>분할</th><th>주의 플래그</th></tr></thead><tbody></tbody></table></div><div class="filters"><button id="prev">이전</button><span id="page"></span><button id="next">다음</button></div></section>
<section><h2>현재 검증과 남은 일</h2><ul><li>자동 일관성 감사 상태: {checks.get('status','pending')}. 원문은 비공개 캐시에 보존하고 해시로 추적합니다.</li><li>공개 파일에는 짧은 발췌만 제공하며 전문·개인 생리 자료를 배포하지 않습니다.</li><li>사람 독립 품질검수 양식 {summary['qc_template_articles']}건과 별도 사람 라벨 후보 240건을 준비했습니다. 빈 양식은 완료한 검수나 사람 레이블이 아닙니다.</li><li>단어 규칙의 점수는 모델 개발 전 진단용입니다. AI/사람 정확도, 위협 회로의 생물학적 타당성, 인간 스트레스는 입증하지 않았습니다.</li></ul></section>
<section class="download"><h2>자료와 감사 기록</h2>{''.join(f'<a href="{name}">{name}</a>' for name in ['news.parquet','news.csv','summary.json','manifest.json','verification.json','extraction_ledger.parquet','duplicate_pairs.csv','event_candidate_pairs.csv','splits.csv','source_day_coverage.csv','provisional_rule_scores.csv','independent_qc_template.csv','human_annotation_candidates.csv'])}</section></main>
<script id="data" type="application/json">{payload}</script><script>const all=JSON.parse(document.querySelector('#data').textContent);let filtered=all,page=0,flagged=false;const size=40;function cell(row,value){{const td=document.createElement('td');td.textContent=String(value);row.appendChild(td);return td;}}function draw(){{const tbody=document.querySelector('#records tbody');tbody.replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){{const tr=document.createElement('tr');const td=cell(tr,'');const a=document.createElement('a');a.href=r.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=r.title;td.append(a);const p=document.createElement('p');p.textContent=r.summary;td.append(p);cell(tr,r.source+' · '+r.published_date);cell(tr,r.body_chars);cell(tr,r.split);cell(tr,r.quality_flags||'없음');tbody.append(tr);}}document.querySelector('#count').textContent=filtered.length.toLocaleString()+'건';document.querySelector('#page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/size));document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=(page+1)*size>=filtered.length;}}function apply(){{const q=document.querySelector('#query').value.toLocaleLowerCase(),s=document.querySelector('#source').value,t=document.querySelector('#split').value;filtered=all.filter(r=>(!q||(r.title+' '+r.summary).toLocaleLowerCase().includes(q))&&(!s||r.source===s)&&(!t||r.split===t)&&(!flagged||r.quality_flags));page=0;draw();}}for(const id of ['query','source','split'])document.querySelector('#'+id).addEventListener('input',apply);document.querySelector('#flagged').onclick=function(){{flagged=!flagged;this.setAttribute('aria-pressed',String(flagged));apply();}};document.querySelector('#prev').onclick=()=>{{page--;draw();}};document.querySelector('#next').onclick=()=>{{page++;draw();}};draw();</script></html>'''
    (out/"report.html").write_text(text,encoding="utf-8")
    return out/"report.html"
