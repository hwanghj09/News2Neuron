"""Standalone article explorer and preregistered evidence, including failures."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .risk_v6 import OUT


def report(out=OUT):
    out=Path(out)
    metrics=json.loads((out/'fresh_audit_metrics.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    data=pd.read_parquet(out/'news_risk_scores.parquet')
    rows=metrics['results'];names=['v3','v5','v6','zero']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    colors=['#8f9ca1','#d09a64','#16796f','#d6dbdf']
    for ax,cohort,title in zip(axes,['random','challenge'],['Primary: 128 random articles','Secondary: 64 challenge articles']):
        vals=[next(r['MAE'] for r in rows if r['model']==m and r['cohort']==cohort) for m in names]
        ax.bar(names,vals,color=colors);ax.set(title=title,ylabel='MAE against one AI rubric (0-100)')
        for i,value in enumerate(vals):ax.text(i,value+.3,f'{value:.2f}',ha='center',fontsize=9)
    (out/'figures').mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/'figures'/f'preregistered_comparison.{ext}',dpi=160)
    plt.close(fig)
    fields=['news_id','title','source','url','summary','published_date','danger_score',
            'model_danger_score','component_index','N','T','A','S','L','I','X',
            'session_reviewed','decision_state','rationale','assessment_research_eligible','assessment_body_sha256']
    payload=json.dumps(dict(metrics=metrics,summary=summary,articles=data[fields].fillna('').to_dict('records')),
                       ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>News2Neuron · v6 정확도 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f4;color:#193332;font:16px/1.7 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:40px 22px}h1{font-size:clamp(30px,5vw,48px);line-height:1.25}h2{font-size:23px}small{color:#5e7471}section{background:white;border:1px solid #dbe6e2;border-radius:18px;padding:26px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{padding:20px;border:1px solid #dbe6e2;border-radius:14px;background:white}.card b{display:block;font-size:28px;color:#16796f}a{color:#126c61}img{max-width:100%;height:auto}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px;border-bottom:1px solid #e4ece9;text-align:left;vertical-align:top}th{white-space:nowrap}.note{background:#fff4d9;padding:16px;border-radius:12px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input,select,button{font:inherit;padding:9px;border:1px solid #bbcdc6;border-radius:8px;background:white;color:inherit}input{flex:1;min-width:140px}button{cursor:pointer}td button{border:0;text-align:left;padding:0;color:#126c61}dialog{max-width:760px;width:calc(100% - 24px);max-height:85vh;border:0;border-radius:18px;padding:26px}dialog::backdrop{background:#102e2b99}.axes{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.axes div{padding:8px;background:#f2f5f4;border-radius:8px;font-size:13px}.axes b{display:block;font-size:20px}code{word-break:break-all}.links{display:flex;gap:20px;flex-wrap:wrap}footer{font-size:13px;color:#5e7471}@media(max-width:650px){main{padding:22px 14px}.cards{grid-template-columns:1fr}.card{padding:14px}section{padding:18px}.axes{grid-template-columns:repeat(2,1fr)}th,td{padding:9px;min-width:80px}td:first-child{min-width:220px}}
</style><main><small>NEWS2NEURON / REPRODUCIBLE AI REFERENCE EVALUATION / v6</small>
<h1>잔존 위험을 더 정확하게,<br>개선 근거는 검증 가능하게.</h1>
<p>실제 한국어 뉴스 8,128건을 대상으로 한 연구 지수입니다. 사건 발생 확률·사람의 스트레스·독립 사람 정답 정확도를 뜻하지 않습니다.</p>
<div class="cards"><div class="card">v5 → v6 평균 절대오차<b id="mae"></b><small>새 무작위 128건 · 100점 척도</small></div><div class="card">10점 이내 AI 일치율<b id="agreement"></b><small>같은 새 기사, 직접 덮기 전 모델 예측</small></div><div class="card">사전 검증 기준<b id="verdict"></b><small>모델·임계값·표본 고정 뒤 평가</small></div></div>
<section><h2>개선의 범위와 한계</h2><p id="result"></p><p class="note">기준 판정은 현재 Codex 한 AI가 내렸습니다. 통계 범위는 이 AI 기준에 대한 오차이며, 기준 자체의 진실성이나 미래 뉴스·사람 판단에 대한 정확도를 입증하지 않습니다. 이전 판정 사건군을 제외한 표본이므로 전체 뉴스 분포의 대표성도 보장하지 않습니다.</p><p>기존 평가 자료는 개발 자료로 전환했습니다. 새 학습 128건과 새 평가 192건은 저장된 정제 본문 전체를 읽고 판정했습니다. 주 평가는 매체별 무작위 16건씩 128건, 어려운 기사 64건은 별도 보조 평가입니다. 결과를 본 뒤 기사·기준·점수를 제외하거나 고치지 않았습니다.</p><div class="scroll"><table><thead><tr><th>비교 기준</th><th>평균 오차 감소</th><th>감소의 95% 범위</th><th>일치율 변화</th><th>일치율 변화 95% 범위</th></tr></thead><tbody id="intervals"></tbody></table></div></section>
<section><h2>같은 새 표본의 모델 비교</h2><img src="figures/preregistered_comparison.png" alt="기존 모델과 v6 및 항상 0인 기준 모델의 무작위·어려운 기사 평균 절대오차 비교"><div class="scroll"><table><thead><tr><th>표본</th><th>모델</th><th>기사 수</th><th>MAE</th><th>10점 이내 일치</th><th>30점 이상 위해 재현율</th><th>30점 넘는 큰 오차</th></tr></thead><tbody id="comparison"></tbody></table></div><p><small>항상 0으로 예측하는 기준도 포함합니다. 무위험 기사가 많다는 이유만으로 좋은 모델이라고 판단하지 않도록 위해 재현율과 큰 오차를 함께 공개합니다.</small></p></section>
<section><div class="row"><h2>전체 기사 평가</h2><b id="count"></b></div><div class="row"><input id="search" placeholder="기사·근거 검색" aria-label="기사·근거 검색"><select id="source" aria-label="매체"><option value="">모든 매체</option></select><select id="kind" aria-label="판정 종류"><option value="">모든 판정</option><option value="direct">직접 AI 판정</option><option value="model">모델 예측</option><option value="eligible">연구용 포함</option></select></div><div class="scroll"><table><thead><tr><th>기사 / 매체</th><th>위험 지수</th><th>N / T / A</th><th>판정</th></tr></thead><tbody id="articles"></tbody></table></div><div class="row"><button id="prev">이전</button><span id="page"></span><button id="next">다음</button></div></section>
<section><h2>재현 가능한 기록</h2><p id="counts"></p><div class="links"><a href="news_risk_scores.csv">전체 점수 CSV</a><a href="news_risk_scores.parquet">전체 점수 Parquet</a><a href="decision_audit.csv">직접 판정 원장</a><a href="protocol.json">사전 평가 기준</a><a href="fresh_audit_metrics.json">통계 결과</a><a href="audit_errors.csv">기사별 원래 예측·오차</a><a href="publisher_metrics.csv">매체별 결과</a><a href="../../../docs/risk-evidence-v6.md">방법·결과·한계</a><a href="../20261109_v5/report.html">이전 보고서</a></div></section>
<footer>8개 매체 · 2026-08-12~2026-11-09 발행 기사. 발행 당시 본문 서술에 남은 위해를 평가합니다. 전체 저장 본문과 특징 캐시는 Git에서 제외하고 해시를 보존합니다.</footer></main>
<dialog id="detail"><div class="row"><b>ARTICLE ASSESSMENT</b><button id="close">닫기</button></div><h2 id="title"></h2><small id="meta"></small><p id="excerpt"></p><div id="axes" class="axes"></div><p id="rationale"></p><p id="method" class="note"></p><p id="state"></p><a id="original" target="_blank" rel="noopener noreferrer">기사 원문 ↗</a><details><summary>평가 본문 해시</summary><code id="hash"></code></details></dialog>
<script id="payload" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);let page=0,filtered=[];const f=(m,c)=>D.metrics.results.find(r=>r.model===m&&r.cohort===c),old=f('v5','random'),now=f('v6','random');$('mae').textContent=old.MAE.toFixed(2)+' → '+now.MAE.toFixed(2);$('agreement').textContent=(old.within10_AI_agreement*100).toFixed(1)+' → '+(now.within10_AI_agreement*100).toFixed(1)+'%';$('verdict').textContent=D.metrics.preregistered_evidence_pass?'충족':'미충족';$('result').textContent=D.metrics.preregistered_evidence_pass?'사전에 정한 범위에서 AI 기준에 대한 평균 오차 감소를 확인했습니다. v3·v5 양쪽과 비교한 감소의 95% 재표집 범위가 모두 0보다 크며, 일치율과 보조 평가의 점추정도 사전 조건을 충족했습니다.':'사전에 정한 개선 입증 조건을 모두 충족하지 못했습니다. 감소한 지표와 악화한 지표, 불확실성을 아래에 그대로 공개합니다.';$('counts').textContent='전체 '+D.summary.articles.toLocaleString()+'건 · 직접 AI 판정 '+D.summary.direct_AI_judgments+'건 · 개발 '+D.summary.development_articles+'건 · 새 평가 '+D.summary.fresh_audit_articles+'건. 평가는 직접 판정으로 덮기 전의 순수 모델 예측으로 계산했습니다.';
function cells(body,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}
const pct=x=>(100*x).toFixed(1)+'%',interval=(a,scale=1)=>a.map(x=>(x*scale).toFixed(2)).join(' ~ ');
for(const [name,r] of Object.entries(D.metrics.paired_comparisons))cells($('intervals'),[name+' → v6',r.MAE_reduction.toFixed(2)+'점',interval(r.MAE_reduction_95_percentile)+'점',pct(r.within10_change),interval(r.within10_change_95_percentile,100)+'%p']);
for(const r of D.metrics.results.filter(r=>r.cohort!=='all'))cells($('comparison'),[r.cohort==='random'?'주 평가: 무작위':'보조: 어려운 기사',r.model,r.articles,r.MAE.toFixed(2),pct(r.within10_AI_agreement),r.high_recall===null?'해당 없음':pct(r.high_recall),r.error_gt30]);
for(const s of [...new Set(D.articles.map(a=>a.source))].sort()){const option=document.createElement('option');option.value=s;option.textContent=s;$('source').append(option)}
function detail(a){$('title').textContent=a.title;$('meta').textContent=a.source+' · '+a.published_date;$('excerpt').textContent=a.summary;$('rationale').textContent=a.rationale;$('axes').replaceChildren();for(const [key,label] of [['danger_score','위험 지수'],['N','부정성'],['T','잔존 위협'],['A','정서 강도'],['S','심각도'],['L','발생 근거'],['I','임박성'],['X','범위']]){const div=document.createElement('div');div.textContent=label;const b=document.createElement('b');b.textContent=(a[key]*(key==='danger_score'?1:4)).toFixed(1)+(key==='danger_score'?' /100':' /4');div.append(b);$('axes').append(div)}$('method').textContent=a.session_reviewed?'직접 AI 판정의 기준식으로 지수를 계산했습니다. 사람이 독립 검증한 정답은 아닙니다.':'AI 기준 지수를 학습한 모델의 예측입니다. 보조 축으로 계산한 값 '+a.component_index.toFixed(1)+'과 다를 수 있으며 실제 위해 확률이 아닙니다.';$('state').textContent=a.decision_state+' · '+(a.assessment_research_eligible?'연구용 포함':'연구용 제외');$('hash').textContent=a.assessment_body_sha256;$('original').href=a.url.startsWith('https://')?a.url:'#';$('detail').showModal()}
function render(){const q=$('search').value.toLowerCase(),source=$('source').value,kind=$('kind').value;filtered=D.articles.filter(a=>(!source||a.source===source)&&(!q||(a.title+' '+a.rationale).toLowerCase().includes(q))&&(!kind||(kind==='direct'&&a.session_reviewed)||(kind==='model'&&!a.session_reviewed)||(kind==='eligible'&&a.assessment_research_eligible))).sort((a,b)=>b.danger_score-a.danger_score);page=Math.min(page,Math.max(0,Math.ceil(filtered.length/40)-1));$('count').textContent=filtered.length.toLocaleString()+'건';$('articles').replaceChildren();for(const a of filtered.slice(page*40,(page+1)*40)){const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');button.textContent=a.title;button.onclick=()=>detail(a);const small=document.createElement('small');small.textContent=a.source+' · '+a.published_date.slice(0,10);td.append(button,document.createElement('br'),small);tr.append(td);for(const value of [a.danger_score.toFixed(1),[a.N,a.T,a.A].map(x=>(x*4).toFixed(1)).join(' / '),a.session_reviewed?'직접 AI':'모델']){const cell=document.createElement('td');cell.textContent=value;tr.append(cell)}$('articles').append(tr)}$('page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/40));$('prev').disabled=page===0;$('next').disabled=(page+1)*40>=filtered.length}
['search','source','kind'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()}));$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};$('close').onclick=()=>$('detail').close();render();
</script></html>'''
    (out/'report.html').write_text(html.replace('PAYLOAD',payload),encoding='utf-8')
    return out/'report.html'


if __name__=='__main__':print(report())
