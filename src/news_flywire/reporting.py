from __future__ import annotations
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .evaluation import summarize_experiments
from .stimulus_encoder import encode, looming

COLORS = ["#3678bd", "#0e967a", "#db8e32", "#b9505d", "#6a64a4", "#55646e"]


def _save(fig, path):
    fig.savefig(str(path)+".png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path)+".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_report(out_dir, circuit, scores, config):
    out = Path(out_dir)
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    metrics = pd.read_csv(out/"metrics.csv")
    curves = pd.read_csv(out/"output_curves.csv")
    summary, contrasts = summarize_experiments(metrics)
    summary.to_csv(out/"condition_summary.csv", index=False)
    contrasts.to_csv(out/"paired_contrasts.csv", index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False,
                         "axes.titleweight":"bold", "axes.labelcolor":"#354353", "axes.edgecolor":"#bec8d2"})
    synthetic=scores.data_source.astype(str).str.contains("synthetic").all()
    scorer_name=", ".join(sorted(scores.scorer.astype(str).unique())) if "scorer" in scores else "imported scores"
    news_description="직접 작성한 가상 기사" if synthetic else ("실제 뉴스의 제목·RSS 발췌문" if scores.data_source.eq("real_publisher_rss").all() else "입력 기사")
    score_description="규칙 baseline 예측" if scorer_name=="rule_baseline_v1" else f"{scorer_name} 예측"
    fig, ax=plt.subplots(figsize=(8.4,5.5))
    for i,(category, group) in enumerate(scores.groupby("category")):
        ax.scatter(group["T"],group["N"],s=65,label=category,color=COLORS[i%len(COLORS)],alpha=.8,edgecolors="white")
    ax.set(xlabel="Threat prediction T (0–1)",ylabel="Negativity prediction N (0–1)",xlim=(-.07,1.07),ylim=(-.07,1.07),
           title=f"News scores · {scorer_name}"+ (" / synthetic articles" if synthetic else ""))
    ax.legend(fontsize=8,loc="lower right")
    ax.grid(alpha=.15)
    _save(fig, figures/"01_news_scores")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,score in enumerate([.1,.5,.9]):
        s=encode({"N":.5,"T":score,"A":.5},config)
        physical=looming(s,config)
        physical.to_csv(out/f"looming_{s.level}.csv",index=False)
        axes[0].plot(physical.time_ms/1000,physical.angle_deg,label=f"{s.level}: l/v={s.l_over_v_ms:g} ms",color=COLORS[i])
    axes[0].set(xlabel="Physical renderer time (s)",ylabel="Angular size (degrees)",title="Physical approach · common window")
    axes[0].legend(fontsize=8)
    seed=int(metrics.seed.min())
    inp=pd.read_parquet(out/"input_events.parquet")
    for i,condition in enumerate(["E1","E2","E3"]):
        sample=inp[(inp.seed==seed)&(inp.condition_id==condition)&(inp.input_slot<8)]
        axes[1].scatter(sample.time_ms,sample.input_slot+i*10,s=3,color=COLORS[i],label=condition)
    axes[1].set(xlabel="Neural simulation time (ms)",ylabel="Input cell (first 8 per level)",title="Hypothetical Poisson drive · separate encoder")
    axes[1].legend(fontsize=8)
    _save(fig,figures/"02_stimulus_and_inputs")

    # Display a small set of strongest verified <=2-hop paths; full edges/IDs are exported separately.
    g=circuit.graph()
    candidates=[]
    for source in circuit.cells(role="input"):
        for target in circuit.cells(role="output"):
            if g.has_edge(source,target):
                candidates.append((g[source][target]["weight"],[source,target]))
            for middle in set(g.successors(source)) & set(g.predecessors(target)):
                candidates.append((min(g[source][middle]["weight"],g[middle][target]["weight"]),[source,middle,target]))
    selected=set(circuit.cells(role="output"))
    cell_types=circuit.neurons.set_index("root_id").cell_type.to_dict()
    for input_type in ["LPLC2","LC4"]:
        typed_paths=[item for item in candidates if cell_types[item[1][0]]==input_type]
        for _,path in sorted(typed_paths,key=lambda x:(-x[0],x[1]))[:3]:
            selected.update(path)
    if len(selected)<=len(circuit.cells(role="output")):
        selected.update(circuit.cells(role="input")[:4])
    sub=g.subgraph(selected)
    roles=circuit.neurons.set_index("root_id").role.to_dict()
    types=circuit.neurons.set_index("root_id").cell_type.to_dict()
    pos={}
    for x,role in enumerate(["input","intermediate","output"]):
        nodes=sorted(n for n in selected if roles[n]==role)
        pos.update({n:(x,float(y)) for n,y in zip(nodes,np.linspace(0,1,max(len(nodes),1)))})
    fig,ax=plt.subplots(figsize=(14,7))
    nx.draw_networkx_nodes(sub,pos,ax=ax,node_size=520,node_color=[COLORS[["input","intermediate","output"].index(roles[n])] for n in sub],alpha=.9)
    nx.draw_networkx_edges(sub,pos,ax=ax,arrows=True,arrowsize=17,alpha=.45,width=[.5+min(3,sub[a][b]["weight"]/50) for a,b in sub.edges],connectionstyle="arc3,rad=0.06")
    labels={n:f"{types[n] or 'untyped'}\n{n}" for n in sub}
    nx.draw_networkx_labels(sub,{n:(p[0],p[1]+.045) for n,p in pos.items()},labels=labels,font_size=7,ax=ax)
    edge_labels={(a,b):int(sub[a][b]["weight"]) for a,b in sub.edges}
    nx.draw_networkx_edge_labels(sub,pos,edge_labels=edge_labels,font_size=6,ax=ax,rotate=False)
    ax.set_title(f"{circuit.metadata['connectome_version']} · strongest verified paths (subset)")
    ax.set_axis_off()
    ax.margins(.2)
    _save(fig,figures/"03_verified_circuit")

    fig,ax=plt.subplots(figsize=(10,4.8))
    for i,condition in enumerate(["E0","E1","E2","E3","E4"]):
        group=curves[curves.condition_id.eq(condition)].groupby("time_ms").rate_hz
        mean,sd=group.mean(),group.std().fillna(0)
        ax.plot(mean.index,mean,label=condition,color=COLORS[i])
        ax.fill_between(mean.index,(mean-sd).clip(lower=0),mean+sd,color=COLORS[i],alpha=.12)
    start=config["simulation"]["baseline_ms"]
    end=start+config["simulation"]["stimulus_ms"]
    ax.axvspan(start,end,color="#8d9eae",alpha=.10)
    ax.set(xlabel="Simulation time (ms)",ylabel="Mean GF/DNp06 firing rate (Hz)",title="Model output · mean ± computational seed SD")
    ax.legend(ncol=5,fontsize=8)
    _save(fig,figures/"04_output_timecourses")

    fig,ax=plt.subplots(figsize=(9,5))
    conditions=["E3","E5","E6","E7","E4"]
    groups=[metrics[metrics.condition_id.eq(c)].delta_rate_hz.to_numpy() for c in conditions]
    ax.bar(range(len(groups)),[x.mean() for x in groups],color=COLORS[:5],alpha=.8)
    for i,values in enumerate(groups):
        jitter=np.random.default_rng(i).uniform(-.13,.13,len(values))
        ax.scatter(i+jitter,values,s=11,color="#263749",alpha=.4)
    ax.set_xticks(range(5),["E3\nIntact","E5\nLPLC2 blocked","E6\nLC4 blocked","E7\nRandom blocked","E4\nRandom input"])
    ax.set(ylabel="Output Δ firing rate (Hz)",title="Virtual connection removal · paired seed experiments")
    _save(fig,figures/"05_silencing")

    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,row in contrasts.iterrows():
        ax.plot([row.computational_ci95_low,row.computational_ci95_high],[i,i],color=COLORS[0],lw=3)
        ax.scatter(row.mean_delta_hz,i,s=40,color=COLORS[1],zorder=3)
    ax.axvline(0,color="#8796a4",lw=1)
    ax.set_yticks(range(len(contrasts)),contrasts.contrast)
    ax.set(xlabel="Paired output difference (Hz)",title="95% bootstrap interval · computational variability only")
    ax.invert_yaxis()
    _save(fig,figures/"06_paired_contrasts")

    man=json.loads((out/"manifest.json").read_text())
    stats={"version":circuit.metadata["connectome_version"],"n":len(circuit.ids),"edges":len(circuit.edges),
           "inputs":len(circuit.cells(role="input")),"outputs":len(circuit.cells(role="output")),
           "seeds":int(metrics.seed.nunique()),"articles":len(scores)}
    limits=[f"뉴스는 {news_description}이며, 점수는 {score_description}입니다. 사람 평가와의 일치도는 별도 검증이 필요합니다.",
            "연결은 실제 FlyWire v783 데이터이지만 활동은 부분 회로 LIF 모델의 계산 결과입니다." if stats['version']=='FAFB_v783' else "연결과 뉴런 ID 모두 소프트웨어 검증용 가상 데이터입니다.",
            "시각 looming과 신경 Poisson 입력은 별개입니다. 생리 반응으로 보정된 tuning curve는 아직 없습니다.",
            "E4는 외부 입력 이벤트 수를 정확히 맞추지만 감각 세포 정체성까지 맞춘 대조군은 아닙니다.",
            "E7은 동일 수 뉴런의 연결 정도를 근사 매칭합니다. 잔여 차이는 metrics.csv에 기록합니다.",
            "시드 반복은 계산 변동성입니다. 생물학적 개체 수, 인간 스트레스, 호르몬 반응의 증거가 아닙니다.",
            "L/M/H의 출력 증가는 입력 함수에서 비롯될 수 있습니다. E9a/E9b 일치는 T 단독 변환의 설계 점검입니다.",
            "공개 뇌 활동 실측 및 인간 생리 자료와의 독립 검증은 아직 수행하지 않았습니다."]
    actual=summary.set_index("condition_id")
    e3=float(actual.loc['E3','delta_mean_hz'])
    e5=float(actual.loc['E5','delta_mean_hz'])
    text=f"E3의 출력 평균 변화는 {e3:.2f} Hz, LPLC2 연결을 제거한 E5는 {e5:.2f} Hz였습니다. 이 값은 선택한 부분 회로와 가상 입력 설정에 한정됩니다."
    html_summary=summary.round(4).to_html(index=False,classes="results",border=0,na_rep="미정의")
    html_contrasts=contrasts.round(4).to_html(index=False,classes="results",border=0,na_rep="미정의")
    images=[("01_news_scores","뉴스의 부정성과 위협성", f"{news_description}와 {score_description}. 동일 좌표의 기사는 겹쳐 보일 수 있습니다."),
            ("02_stimulus_and_inputs","자극과 입력의 분리","물리 영상은 초기·최종 각도와 전체 제시 시간을 맞춥니다. 신경 입력의 시간축·발화율은 별도 가정입니다."),
            ("03_verified_circuit","확인한 연결 경로","강한 경로 일부만 표시합니다. 전체 뉴런 ID와 경로는 별도 파일로 제공합니다."),
            ("04_output_timecourses","출력의 시간 변화","음영은 시드 표준편차이며 생물학적 신뢰구간이 아닙니다."),
            ("05_silencing","가상 회로 억제와 대조","시냅스 입출력 제거를 사용한 계산 처치입니다."),
            ("06_paired_contrasts","대조 조건별 차이","같은 시드의 차이를 비교합니다. 95% 구간은 계산 반복의 부트스트랩입니다.")]
    figure_html="".join(f'<section><h2>{title}</h2><p>{caption}</p><a href="figures/{name}.svg"><img src="figures/{name}.png" alt="{title}" loading="lazy"></a></section>' for name,title,caption in images)
    links=["metrics.csv","condition_summary.csv","paired_contrasts.csv","article_responses.csv","cell_metrics.parquet","input_events.parquet","cell_selections.json","manifest.json","resolved_config.json"]
    downloads=" ".join(f'<a href="{f}">{f}</a>' for f in links if (out/f).exists())
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>뉴스 × FlyWire 연구 실행 보고서</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#eff3f7;color:#233347;font:16px/1.75 system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:48px 24px}}.eyebrow{{letter-spacing:.13em;font-size:12px;color:#357d85}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.25;letter-spacing:-.04em;max-width:920px}}h2{{font-size:23px}}.lead{{color:#526476;max-width:850px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:30px 0}}.card,section{{background:white;border:1px solid #dce4ec;border-radius:14px;padding:24px}}.card strong{{display:block;font-size:27px}}.card span{{font-size:13px;color:#637489}}section{{margin:20px 0}}img{{width:100%;height:auto;display:block}}.notice{{border-left:4px solid #c08029;background:#fff8e9;padding:18px 22px}}.overflow{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #e4ebf0;text-align:right}}th{{background:#f2f6fa}}a{{color:#246b98}}.downloads a{{display:inline-block;margin:5px 12px 5px 0}}footer{{font-size:12px;color:#64758a}}li{{margin:8px 0}}</style>
<main><div class="eyebrow">COMPUTATIONAL RESEARCH · REPRODUCIBLE RUN</div><h1>뉴스의 위협성에서<br>초파리 회로 모델의 반응까지</h1><p class="lead">부정성(N)과 위협성(T)을 분리하고, 가상 입력·연결 구조·대조 조건에 따른 모델 출력을 비교합니다.</p>
<div class="notice"><strong>연구 단계: 계산 파일럿</strong><br>{html.escape(news_description)} · {html.escape(score_description)} · {'실제 연결 데이터' if stats['version']=='FAFB_v783' else '가상 연결 데이터'} · 모형 활동. 사람·초파리의 실제 스트레스 반응을 측정한 결과가 아닙니다.</div>
<div class="cards">{''.join(f'<div class="card"><strong>{value:,}</strong><span>{label}</span></div>' for label,value in [('뉴런',stats['n']),('방향성 연결',stats['edges']),('입력 세포',stats['inputs']),('출력 세포',stats['outputs']),('계산 시드',stats['seeds']),('가상 기사' if synthetic else '실제 기사',stats['articles'])])}</div>
<section><h2>이번 실행에서 관찰한 값</h2><p>{text}</p><p>버전: <code>{html.escape(stats['version'])}</code> · 엔진: <code>{man['engine']}</code></p></section>
{figure_html}<section><h2>조건별 요약</h2><div class="overflow">{html_summary}</div></section><section><h2>시드별 대응 비교</h2><div class="overflow">{html_contrasts}</div></section>
<section><h2>해석 범위와 남은 검증</h2><ul>{''.join(f'<li>{s}</li>' for s in limits)}</ul></section><section class="downloads"><h2>원자료와 재현 설정</h2>{downloads}</section>
<footer>생성 시각: {html.escape(man['created_utc'])} · 데이터 해시와 패키지 버전은 manifest.json에 저장됩니다. 그림은 PNG와 SVG로 제공됩니다.</footer></main></html>'''
    (out/"report.html").write_text(document,encoding="utf-8")
    report_md=f"# 계산 파일럿 결과\n\n버전: {stats['version']}. 뉴런 {stats['n']}개, 연결 {stats['edges']}개, 시드 {stats['seeds']}개.\n\n{text}\n\n"+"\n".join(f"- {s}" for s in limits)+"\n\n결과 테이블: condition_summary.csv / paired_contrasts.csv\n"
    (out/"report.md").write_text(report_md,encoding="utf-8")
    return out/"report.html"
