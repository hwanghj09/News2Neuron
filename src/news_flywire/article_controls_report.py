"""Separate full-population approximate controls from exact common support."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .article_controls import OUT, verify
from .provenance import write_json, sha256


def report(out=OUT):
    out=Path(out);p=verify(out);parent=Path(p['parent'])
    selection=pd.read_csv(out/'selection.csv')
    current=pd.read_csv(out/'output_types.csv')
    old=pd.read_csv(parent/'output_type_responses.csv')
    old=old[old.news_id.isin(selection.news_id)&old.condition.isin(p['reused_conditions'])].copy()
    old['fresh_simulation']=False
    combined=pd.concat([current,old[current.columns]],ignore_index=True)
    combined.to_csv(out/'all_output_types.csv',index=False)
    article=combined.groupby(['news_id','condition','cell_type']).delta_rate_hz.mean().reset_index()
    summary=article.groupby(['condition','cell_type']).agg(mean_delta_hz=('delta_rate_hz','mean'),articles=('news_id','size')).reset_index()
    summary.to_csv(out/'output_summary.csv',index=False)
    rng=np.random.default_rng(20261119);comparisons=[]
    for typ,part in article.groupby('cell_type'):
        wide=part.pivot(index='news_id',columns='condition',values='delta_rate_hz')
        for label,a,b in [('common_support','common178_control','common178_target'),
            ('full_approx','ACh314_approx','R_intact'),('original_random','R_random_input','R_intact')]:
            delta=(wide[a]-wide[b]).to_numpy()
            interval=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1),[.025,.975])
            comparisons.append(dict(comparison=label,cell_type=typ,articles=len(delta),control_minus_target_hz=float(delta.mean()),
                descriptive_p025_hz=float(interval[0]),descriptive_p975_hz=float(interval[1]),
                scope='Article means over5 computational seeds; selected-set descriptive resampling, no biological CI'))
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    controls=json.loads((out/'controls.json').read_text())
    balance=[]
    for c in controls:
        for kind in ['common','approximate']:
            balance.append(dict(seed=c['computational_seed'],kind=kind,**c[kind+'_mismatch']))
    pd.DataFrame(balance).to_csv(out/'balance.csv',index=False)
    pd.DataFrame(controls[0]['feasibility']).to_csv(out/'common_support.csv',index=False)
    pair_article=article[article.condition.str.startswith('common')].pivot(index=['news_id','cell_type'],columns='condition',values='delta_rate_hz').reset_index()
    pair_article=pair_article.merge(selection[['news_id','source','title','url']],on='news_id')
    pair_article['control_minus_target_hz']=pair_article.common178_control-pair_article.common178_target
    pair_article.to_csv(out/'article_pairs.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,typ in enumerate(['GF','DNp06']):
        for col,(conditions,title) in enumerate([
            (['R_intact','R_random_input','ACh314_approx'],'Full314; matching remains approximate'),
            (['common178_target','common178_control'],'Common178; exact path/transmitter strata')]):
            values=summary[summary.cell_type.eq(typ)].set_index('condition').loc[conditions].mean_delta_hz.to_numpy()
            ax=axes[row,col];ax.bar(conditions,values,color=['#19796a','#7699b8','#c28e52'][:len(values)])
            ax.set(title=typ+' / '+title,ylabel='Mean output delta rate (Hz)')
            ax.tick_params(axis='x',labelsize=8,rotation=8)
            for i,v in enumerate(values):ax.text(i,v+max(values.max()*.02,.2),f'{v:.2f}',ha='center',fontsize=9)
            ax.set_ylim(0,max(values.max()*1.2,1))
    fig.savefig(out/'matched_controls.png',dpi=170);fig.savefig(out/'matched_controls.svg');plt.close(fig)
    f=out/'matched_controls.svg';f.write_text('\n'.join(l.rstrip() for l in f.read_text().splitlines())+'\n')
    result=dict(new_simulations=120,reused_parent_simulations=80,real_articles=8,computational_seeds=5,
        original_input_population=314,common_support_population=178,full_exact_matching_feasible=False,
        subset_target_types=[c['common_target_cell_types'] for c in controls],
        output_results=summary.to_dict('records'),paired_comparisons=comparisons,
        no_biological_specificity_claim=True,algorithm_changes_after_outcomes=False)
    write_json(out/'summary.json',result)
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>News2Neuron · 입력 대조군 보강</title><style>
body{font:16px/1.7 system-ui,sans-serif;color:#1e3946;background:#f0f5f6;margin:0}main{max-width:1100px;margin:auto;padding:28px 18px}section{background:white;border:1px solid #d7e3e7;padding:22px;border-radius:12px;margin:20px 0}h1{font-size:clamp(28px,5vw,42px);line-height:1.3}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #e0e8eb;text-align:left}.note{background:#fff1d7;padding:16px;border-radius:8px}a{color:#157065}.links{display:flex;flex-wrap:wrap;gap:16px}</style><main><h1>무작위 입력 대조군을<br>경로·전달물질에 맞춰 다시 비교.</h1><p>잔존 위해 기사 8건 · 5시드 · 새 계산 120회. 이전 정상/무작위 조건 80회는 재사용으로 따로 표시했습니다.</p>
<section><h2>314개 전부를 정확히 맞출 수 있는가?</h2><p>원래 입력은 모두 아세틸콜린 계열입니다. GF까지 1-hop, DNp06까지 2-hop인 입력은 150개지만 같은 조건의 다른 후보는 14개뿐입니다. 뉴런을 중복하거나 지어내지 않고 정확하게 맞출 수 있는 공통 부분집합은 178개입니다.</p>FEASIBILITY<p class="note">왼쪽의 314개 근사 매칭과 오른쪽의 178개 정확 층 매칭은 입력 인구가 다릅니다. 178개 비교를 314개 전체의 결과로 일반화하지 않습니다. 공통 부분집합에서는 LPLC2/LC4 비중도 바뀝니다.</p><img src="matched_controls.png" alt="전체314 근사 대조와 공통178 정확 경로 전달물질 대조를 분리한 GF DNp06 반응 비교"></section>
<section><h2>출력 세포형별 결과</h2>SUMMARY<p>각 기사의 5시드 평균을 먼저 구한 뒤 기사 8건을 평균했습니다. Hz는 계산 모델 출력이며 실제 초파리 측정이나 사람 스트레스 수치가 아닙니다.</p></section>
<section><h2>매칭 잔여 오차</h2>BALANCE<p>정확한 항목은 예측 전달물질과 두 출력까지의 최단 hop 층입니다. 연결 수와 절대 시냅스 강도는 근사 매칭이며 세포형·수용체·상세 경로는 일치하지 않습니다. 최단 경로는 억제성 중간 뉴런도 통과할 수 있습니다.</p></section>
<section><h2>기사별 공통 부분집합 비교</h2>ARTICLES</section>
<section><h2>자료와 재현</h2><div class="links"><a href="metrics.csv">새120회 지표</a><a href="all_output_types.csv">새 결과와 재사용 표시</a><a href="paired_comparisons.csv">기사 단위 짝지은 차이</a><a href="controls.json">뉴런ID·매칭</a><a href="protocol.json">사전 명세</a><a href="integrity_audit.json">감사 결과</a><a href="../../../docs/control-matching-v1.md">방법·결과·한계</a><a href="../article_bridge_v1/report.html">이전 기사별 실험</a></div><p>새 원래 스파이크와 외부 이벤트는 trials/기사ID 아래에 보존했습니다. 외부 이벤트 배열은 각 짝에서 정확히 같습니다. 세포형을 감각 뉴런까지 맞춘 대조군은 아니며 생물학적 특이성 입증으로 해석하지 않습니다.</p></section></main></html>'''
    def table(frame):return '<div class="scroll">'+frame.to_html(index=False,escape=True,border=0,float_format=lambda x:f'{x:.3f}')+'</div>'
    html=html.replace('FEASIBILITY',table(pd.DataFrame(controls[0]['feasibility'])))
    html=html.replace('SUMMARY',table(summary)).replace('BALANCE',table(pd.DataFrame(balance))).replace('ARTICLES',table(pair_article.drop(columns=['news_id','url'])))
    (out/'report.html').write_text(html)
    return out/'report.html'
