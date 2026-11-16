"""Descriptive paired contrasts and offline explorer for fresh article trials."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .article_bridge import DEFAULT_OUT, CONDITIONS, check_freeze
from .flywire_graph import Circuit
from .provenance import write_json, sha256
from .risk_review_updates import load_review_catalogs, input_key

LABELS = {'minimal':'낮은 부정·위험', 'negative_closed':'부정적이지만 종결', 'ongoing_harm':'잔존 위해'}
PAIRS = {'targeted_vs_random':('R_intact','R_random_input'),
    'degree_vs_LPLC2_cut':('R_degree_cut','R_LPLC2_cut'),
    'R_vs_T':('R_intact','T_intact'), 'N_vs_R':('N_intact','R_intact')}


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def report(out=DEFAULT_OUT):
    out = Path(out); protocol = check_freeze(out)
    data = pd.read_csv(out/'metrics.csv')
    selected = pd.read_csv(out/'selection.csv').fillna('')
    config = json.loads((out/'resolved_config.json').read_text())
    circuit = Circuit.load(protocol['circuit_path'])
    means = data.groupby(['news_id','condition']).agg(delta_mean_hz=('delta_rate_hz','mean'),
        delta_sd_hz=('delta_rate_hz','std'), delta_min_hz=('delta_rate_hz','min'), delta_max_hz=('delta_rate_hz','max'),
        external_events_mean=('external_input_events','mean'), latency_mean_ms=('latency_ms','mean'),
        recruitment_mean=('recruitment_fraction','mean')).reset_index()
    means = means.merge(selected[['news_id','cohort','source','title']], on='news_id', validate='many_to_one')
    means.to_csv(out/'article_response_summary.csv', index=False)
    group = means.groupby(['cohort','condition']).agg(articles=('news_id','nunique'),
        mean_article_delta_hz=('delta_mean_hz','mean'), sd_article_delta_hz=('delta_mean_hz','std')).reset_index()
    group.to_csv(out/'cohort_summary.csv', index=False)
    wide = means.pivot(index='news_id', columns='condition', values='delta_mean_hz')
    comparison = []; rng = np.random.default_rng(20261118)
    for cohort, ids in [('all',selected.news_id.tolist())]+[(c, g.news_id.tolist()) for c,g in selected.groupby('cohort')]:
        for name, (a,b) in PAIRS.items():
            delta = (wide.loc[ids,a]-wide.loc[ids,b]).to_numpy()
            draws = rng.choice(delta, size=(10000,len(delta)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025,.975])
            comparison.append(dict(cohort=cohort, comparison=name, a=a, b=b, articles=len(delta),
                mean_paired_difference_hz=float(delta.mean()), descriptive_p025_hz=float(low), descriptive_p975_hz=float(high),
                resampling_unit='article after averaging computational seeds; descriptive selected-set variability, not biological CI'))
    pd.DataFrame(comparison).to_csv(out/'paired_comparisons.csv', index=False)
    articles = []; type_frames = []; reviews = load_review_catalogs()
    for article in selected.to_dict('records'):
        folder = out/'trials'/article['news_id']
        curves = pd.read_parquet(folder/'curves.parquet').groupby(['condition','time_ms']).rate_hz.mean().reset_index()
        profiles = pd.read_parquet(folder/'stimulus_profiles.parquet')
        profiles = profiles.groupby('drive',sort=False).apply(lambda f:f.iloc[::20],include_groups=False).reset_index(level=0)
        cells = pd.read_parquet(folder/'cells.parquet').merge(circuit.neurons[['root_id','role','cell_type']], on='root_id', validate='many_to_one')
        types = cells[cells.role.eq('output')].groupby(['condition','seed','cell_type']).delta_rate_hz.mean().reset_index()
        types['news_id'] = article['news_id']; types['cohort'] = article['cohort']; type_frames.append(types)
        r = reviews[input_key(article['title'],article['assessment_body_sha256'])]
        articles.append(dict(**article, rationale=r['rationale'], curves=records(curves),
            profiles=records(profiles[['drive','time_ms','angle_deg','rate_hz']]),
            responses=records(means[means.news_id.eq(article['news_id'])]),
            output_types=records(types.groupby(['condition','cell_type']).delta_rate_hz.mean().reset_index())))
    type_data = pd.concat(type_frames)
    type_data.to_csv(out/'output_type_responses.csv', index=False)
    type_cohorts = type_data.groupby(['news_id','cohort','condition','cell_type']).delta_rate_hz.mean().reset_index().groupby(['cohort','condition','cell_type']).delta_rate_hz.mean().reset_index()
    type_cohorts.to_csv(out/'output_type_cohort_summary.csv', index=False)
    baseline = pd.read_csv(out/'baseline_metrics.csv')
    summary = dict(corpus_articles=protocol['corpus_articles'], eligible_articles=protocol['eligible_articles'],
        simulated_real_articles=len(selected), publishers=int(selected.source.nunique()), computational_seeds=len(config['computational_seeds']),
        article_trials=len(data), new_baseline_trials=len(baseline), fresh_simulations=len(data)+len(baseline),
        nonzero_external_input_trials=int(data.external_input_events.gt(0).sum()), shared_level_lookup_used=False,
        stimulus_ms=config['simulation']['stimulus_ms'], total_ms=sum(config['simulation'][k] for k in ['baseline_ms','stimulus_ms','recovery_ms']), conditions=len(CONDITIONS),
        connectome_version='FAFB_v783', neurons=len(circuit.ids), directed_edges=len(circuit.edges),
        simulation_only=True, biological_replicates=0, human_physiology_measured=False,
        cohort_results=records(group), output_type_cohorts=records(type_cohorts), paired_comparisons=comparison)
    write_json(out/'summary.json', summary)
    figures = out/'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    pivot = group.pivot(index='cohort', columns='condition', values='mean_article_delta_hz')
    display = ['R_intact','N_intact','T_intact']
    pivot.loc[list(LABELS),display].plot.bar(ax=axes[0],rot=10,color=['#15685e','#b98346','#6c83ab'])
    axes[0].set(title='Article drive mapping',xlabel='',ylabel='Mean output delta rate (Hz)')
    chosen = pivot.loc['ongoing_harm',['R_intact','R_random_input','R_LPLC2_cut','R_LC4_cut','R_degree_cut']]
    chosen.plot.bar(ax=axes[1],rot=25,color=['#15685e','#8c9baa','#b98346','#6c83ab','#999999'])
    axes[1].set(title='Ongoing-harm articles: identical R input events',xlabel='',ylabel='Mean output delta rate (Hz)')
    fig.savefig(figures/'article_controls.png',dpi=170)
    fig.savefig(figures/'article_controls.svg'); plt.close(fig)
    svg=figures/'article_controls.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    payload = json.dumps(dict(summary=summary, articles=articles, labels=LABELS), ensure_ascii=False,
        separators=(',',':'),allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    (out/'report.html').write_text(HTML.replace('PAYLOAD',payload))
    write_json(out/'report_manifest.json',dict(summary_sha256=sha256(out/'summary.json'),
        metrics_sha256=sha256(out/'metrics.csv'), report_source_sha256=sha256(__file__),
        report_sha256=sha256(out/'report.html'), shared_level_lookup=False))
    return out/'report.html'


HTML = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>News2Neuron · 기사별 FlyWire 회로 실험</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f5;color:#16343d;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:40px 22px}h1{font-size:clamp(30px,5vw,48px);line-height:1.25}h2{font-size:23px}small{color:#52707b}section{background:white;border:1px solid #d9e3e6;border-radius:14px;padding:24px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:18px;background:white;border:1px solid #d9e3e6;border-radius:12px}.card b{display:block;font-size:29px;color:#15685e}.note{padding:16px;background:#fff2d7;border-radius:8px}a,button{color:#15685e}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px;border-bottom:1px solid #e0e8ea;text-align:left;vertical-align:top}input,select,button{font:inherit;background:white;border:1px solid #b2c6cd;border-radius:6px;padding:8px}button{cursor:pointer}td button{border:0;padding:0;text-align:left}.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}input{min-width:140px;flex:1}.charts{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}svg{width:100%;height:auto}.legend{font-size:12px}.links{display:flex;flex-wrap:wrap;gap:15px}code{overflow-wrap:anywhere}.meta{font-size:14px;color:#52707b}@media(max-width:650px){main{padding:20px 12px}.cards{grid-template-columns:1fr 1fr}.card{padding:12px}.charts{grid-template-columns:1fr}section{padding:16px}td:first-child{min-width:210px}}
</style><main><small>NEWS2NEURON / REAL ARTICLES → VIRTUAL DRIVE → MEASURED CONNECTIVITY</small>
<h1>뉴스 점수에서<br>기사별 회로 반응까지.</h1>
<p>최신 뉴스 점수를 가상 입력으로 변환하고, 실제 FAFB v783 연결을 사용하는 부분 회로에서 기사마다 새로 계산했습니다.</p>
<div class="cards"><div class="card">자극 파라미터<b id="inputs"></b>실제 기사</div><div class="card">새 계산 대상<b id="selected"></b>8개 매체</div><div class="card">새 시뮬레이션<b id="runs"></b>7조건 × 5시드 + 무입력</div><div class="card">실제 연결 기반<b id="neurons"></b>뉴런 · <span id="edges"></span>개 연결</div></div>
<p class="note">출력은 가상 자극을 넣은 계산 모델의 발화율입니다. 초파리가 뉴스를 이해하거나 사람의 스트레스·호르몬 반응을 측정한 결과가 아닙니다. 기사 점수에서 신경 입력으로 바꾸는 함수는 연구 가정이며, 생리 보정된 시각 인코더가 아닙니다.</p>
<section><h2>이번 단계에서 달라진 점</h2><p>세 단계 결과를 기사에 연결하던 방식에서, 연속적인 R/T/N 점수와 기사별 Poisson 이벤트를 사용하는 새 계산으로 확장했습니다. 같은 양수 점수인 기사도 다른 기사 ID의 입력 난수열을 갖습니다. 회로 대조 조건은 같은 R 이벤트 배열을 그대로 재사용합니다.</p><p>R=잔존 위해 지수/100, T=잔존 위협 축, N=부정성. 0점은 입력 0 Hz, 1점은 평균 150 Hz이며, 같은 400ms 안에서 접근 원판의 각도가 변합니다. 신경 입력의 80%는 균일하고 20%는 각도 확장에 따라 변합니다. 세 변환의 차이는 가정한 입력 함수의 차이이기도 합니다.</p><img src="figures/article_controls.png" alt="기사군별 입력 변환과 같은 입력 이벤트를 사용한 회로 대조 결과"><p class="meta">수치는 먼저 기사별 5개 시드를 평균한 뒤 기사들을 평균했습니다. 총 실행 수를 생물학적 표본 수로 세지 않습니다.</p></section>
<section><h2>출력 세포형도 나누어 확인</h2><p>전체 평균만 보면 GF와 DNp06의 다른 반응을 놓칠 수 있습니다. 아래는 잔존 위해 기사군의 세포형별 평균입니다. 무작위 입력에서 DNp06 반응이 더 커지는 결과도 포함되므로, 이 대조만으로 표적 입력의 생물학적 특이성이 입증됐다고 보지 않습니다.</p><div class="scroll"><table><thead><tr><th>조건</th><th>GF Δ Hz</th><th>DNp06 Δ Hz</th></tr></thead><tbody id="type-comparison"></tbody></table></div></section><section><h2>기사별 결과 탐색</h2><div class="filters"><input id="search" aria-label="기사 검색" placeholder="제목·근거 검색"><select id="cohort" aria-label="기사군"><option value="">모든 기사군</option></select><select id="source" aria-label="매체"><option value="">모든 매체</option></select><span id="count"></span></div><div class="scroll"><table><thead><tr><th>실제 기사 / 매체</th><th>R / T / N<br>0~1</th><th>R 평균 입력 Hz</th><th>R 출력 Δ Hz</th><th>LPLC2 차단 Δ Hz</th></tr></thead><tbody id="articles"></tbody></table></div></section>
<section id="detail"><small id="detail-cohort"></small><h2 id="title"></h2><p id="rationale"></p><a id="original" target="_blank" rel="noopener noreferrer">기사 원문 ↗</a><p class="meta" id="article-meta"></p><div class="charts"><div><h3>가상 원판의 각도</h3><svg id="geometry" role="img" aria-label="기사별 R T N 접근 원판 각도 곡선" viewBox="0 0 500 260"></svg></div><div><h3>가상 입력률</h3><svg id="drive" role="img" aria-label="기사별 R T N 신경 입력률 곡선" viewBox="0 0 500 260"></svg></div></div><h3>모델 출력 시간 곡선</h3><div id="response-legend" class="legend"></div><svg id="response" role="img" aria-label="조건별 모델 출력 발화율 시간 곡선" viewBox="0 0 700 270"></svg><p class="legend">R_intact: 잔존 위해 · R_random_input: 같은 이벤트를 다른 세포에 입력 · R_LPLC2_cut / R_LC4_cut: 해당 세포의 입출력 연결 제거 · R_degree_cut: 같은 수의 연결 정도 근사 매칭 세포 차단 · T_intact / N_intact: 대안 점수 변환. 곡선은 5시드 평균입니다.</p><div class="scroll"><table><thead><tr><th>조건</th><th>출력 평균 Δ Hz</th><th>5시드 최소~최대</th><th>GF Δ Hz</th><th>DNp06 Δ Hz</th><th>입력 이벤트 수 평균</th></tr></thead><tbody id="responses"></tbody></table></div><div id="raw-links" class="links"></div><details><summary>평가 본문 해시</summary><code id="body-hash"></code></details></section>
<section><h2>대조·해석의 한계</h2><p>무작위 입력 세포는 감각 세포형까지 맞춘 대조군이 아닙니다. 차단은 해당 뉴런의 입출력 연결을 제거하는 계산 조작이며 실제 생물학적 억제와 같지 않습니다. 연결 정도 매칭의 남은 오차도 저장했습니다. 부분 회로는 외부 입력과 전뇌 피드백을 생략합니다.</p><p>종결 기사에서 R 입력과 출력이 0인 것은 이 모델의 정의와 무입력 휴지 상태에서 따라오는 결과입니다. 부정성과 위험의 차이를 뇌가 이해했다는 증거가 아닙니다. 반복 시드는 계산상의 변동만 나타냅니다. 24건은 AI가 직접 판정한 의도적 균형 표본입니다. 모든 8,128건의 뇌 반응을 계산했다고 해석하지 않습니다.</p></section>
<section><h2>검증·원자료</h2><div class="links"><a href="article_inputs.csv">8,128건 자극 파라미터 CSV</a><a href="selection.csv">실제 기사 24건</a><a href="article_response_summary.csv">기사·조건별 반응</a><a href="output_type_responses.csv">세포형별 반응</a><a href="metrics.csv">840회 기사 실험 지표</a><a href="baseline_metrics.csv">무입력 5회</a><a href="paired_comparisons.csv">기사 단위 짝지은 차이</a><a href="protocol.json">사전 명세·고정 해시</a><a href="run_manifest.json">실행 원장</a><a href="integrity_audit.json">원자료 무결성 검사</a><a href="../../../docs/article-bridge-v1.md">방법·재현·한계</a></div><p>원래 스파이크·외부 이벤트·세포별 지표·물리 곡선은 <code>trials/기사ID/</code>의 Parquet 파일에 있습니다. 회로 대조는 같은 이벤트 파일의 R 배열을 공유합니다. 이벤트의 중복도 유지합니다. 본문 전문은 포함하지 않았습니다.</p></section></main>
<script id="payload" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),colors=['#126b61','#c0843d','#6b7fa6','#ad5a62','#9a70b1','#407d9b','#657a83'];
const colorBy={R:'#126b61',T:'#6b7fa6',N:'#c0843d',R_intact:'#126b61',T_intact:'#6b7fa6',N_intact:'#c0843d',R_random_input:'#7f939d',R_LPLC2_cut:'#ad5a62',R_LC4_cut:'#9a70b1',R_degree_cut:'#657a83'},color=(k,i)=>colorBy[k]||colors[i%colors.length];
$('neurons').textContent=D.summary.neurons.toLocaleString();$('edges').textContent=D.summary.directed_edges.toLocaleString();$('inputs').textContent=D.summary.corpus_articles.toLocaleString()+'건';$('selected').textContent=D.articles.length+'건';$('runs').textContent=D.summary.fresh_simulations.toLocaleString()+'회';
for(const [value,label] of Object.entries(D.labels)){const o=document.createElement('option');o.value=value;o.textContent=label;$('cohort').append(o)}for(const source of [...new Set(D.articles.map(a=>a.source))].sort()){const o=document.createElement('option');o.value=source;o.textContent=source;$('source').append(o)}
function el(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function chart(id,rows,key,value,maxX){const svg=$(id);svg.replaceChildren();const wide=id==='response',w=wide?700:500,h=wide?270:260,left=48,top=24,right=14,bottom=50,keys=[...new Set(rows.map(r=>r[key]))],maxY=Math.max(1,...rows.map(r=>r[value]));const sx=x=>left+(w-left-right)*x/maxX,sy=y=>h-bottom-(h-top-bottom)*y/maxY;
for(let i=0;i<=4;i++){let y=maxY*i/4;svg.append(el('line',{x1:left,x2:w-right,y1:sy(y),y2:sy(y),stroke:'#dfe8ea'}),el('text',{x:left-6,y:sy(y)+4,'text-anchor':'end','font-size':11,fill:'#52707b'},y.toFixed(0)))}
for(let i=0;i<=4;i++){const x=maxX*i/4;svg.append(el('text',{x:sx(x),y:h-bottom+18,'text-anchor':'middle','font-size':11},x.toFixed(0)))}
if(wide){$('response-legend').replaceChildren();keys.forEach((k,i)=>{const span=document.createElement('span');span.textContent=k+' ';span.style.color=color(k,i);span.style.marginRight='12px';$('response-legend').append(span)})}keys.forEach((k,i)=>{const data=rows.filter(r=>r[key]===k).sort((a,b)=>a.time_ms-b.time_ms),points=data.map(r=>sx(r.time_ms)+','+sy(r[value])).join(' ');svg.append(el('polyline',{points,fill:'none',stroke:color(k,i),'stroke-width':1.7}));if(!wide)svg.append(el('text',{x:left+i*90,y:14,fill:color(k,i),'font-size':12},k))});svg.append(el('text',{x:w/2,y:h-6,'text-anchor':'middle','font-size':12,fill:'#52707b'},'time (ms)'))}
function cell(tr,text){const td=document.createElement('td');td.textContent=text;tr.append(td)}
function detail(a){$('raw-links').replaceChildren();for(const [file,label] of [['spikes','원래 스파이크'],['events','입력 이벤트'],['cells','세포별 지표'],['stimulus_profiles','자극 곡선']]){const link=document.createElement('a');link.href='trials/'+encodeURIComponent(a.news_id)+'/'+file+'.parquet';link.textContent=label;$('raw-links').append(link)}$('title').textContent=a.title;$('detail-cohort').textContent=D.labels[a.cohort]+' · '+a.source;$('rationale').textContent=a.rationale;$('original').href=a.url;$('article-meta').textContent='발행 '+a.published_date+' · 직접 AI 판정 · 새 계산 '+(D.summary.computational_seeds*D.summary.conditions)+'회';$('body-hash').textContent=a.assessment_body_sha256;chart('geometry',a.profiles,'drive','angle_deg',D.summary.stimulus_ms);chart('drive',a.profiles,'drive','rate_hz',D.summary.stimulus_ms);chart('response',a.curves,'condition','rate_hz',D.summary.total_ms);$('responses').replaceChildren();for(const r of a.responses){const tr=document.createElement('tr');const t=name=>a.output_types.find(x=>x.condition===r.condition&&x.cell_type===name)?.delta_rate_hz;for(const v of [r.condition,r.delta_mean_hz.toFixed(2),r.delta_min_hz.toFixed(2)+' ~ '+r.delta_max_hz.toFixed(2),t('GF')?.toFixed(2)??'—',t('DNp06')?.toFixed(2)??'—',r.external_events_mean.toFixed(1)])cell(tr,v);$('responses').append(tr)}}
function render(){const q=$('search').value.toLowerCase(),cohort=$('cohort').value,source=$('source').value,rows=D.articles.filter(a=>(!q||(a.title+' '+a.rationale).toLowerCase().includes(q))&&(!cohort||a.cohort===cohort)&&(!source||a.source===source));$('count').textContent=rows.length+'건';$('articles').replaceChildren();for(const a of rows){const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');button.textContent=a.title;button.onclick=()=>{detail(a);$('detail').scrollIntoView({behavior:'smooth'})};td.append(button,document.createElement('br'),document.createTextNode(a.source+' · '+D.labels[a.cohort]));tr.append(td);const find=k=>a.responses.find(r=>r.condition===k).delta_mean_hz;for(const v of [[a.R,a.T,a.N].map(x=>x.toFixed(2)).join(' / '),a.R_mean_rate_hz.toFixed(2),find('R_intact').toFixed(2),find('R_LPLC2_cut').toFixed(2)])cell(tr,v);$('articles').append(tr)}}
for(const condition of ['R_intact','R_random_input','R_LPLC2_cut','R_LC4_cut','R_degree_cut']){const row=document.createElement('tr');cell(row,condition);for(const type of ['GF','DNp06'])cell(row,D.summary.output_type_cohorts.find(r=>r.cohort==='ongoing_harm'&&r.condition===condition&&r.cell_type===type).delta_rate_hz.toFixed(2));$('type-comparison').append(row)}['search','source','cohort'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',render));render();detail(D.articles.find(a=>a.cohort==='ongoing_harm')||D.articles[0]);
</script></html>'''
