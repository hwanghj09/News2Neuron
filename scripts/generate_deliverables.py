"""Generate manuscript, poster and exploration notebook from stored measured outputs."""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from news_flywire.provenance import write_json, manifest

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'results/v783'
summary=pd.read_csv(run/'condition_summary.csv')
contrasts=pd.read_csv(run/'paired_contrasts.csv')
stats=summary.set_index('condition_id')
up=ROOT/'external/Drosophila_brain_model'
repro=json.loads((ROOT/'results/upstream_reproduction_30/reproduction.json').read_text())
sensitivity=pd.read_csv(ROOT/'results/sensitivity/summary.csv')
e3,e5,e7=[float(stats.loc[c,'delta_mean_hz']) for c in ['E3','E5','E7']]
reduction=(e3-e5)/e3*100 if e3 else 0

# Check the stored reference rather than claiming identical results.
notebook=json.loads((up/'example.ipynb').read_text())
namespace={}
exec(next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('neu_sugar =')),namespace)
archived=pd.read_parquet(up/'results/example/sugarR.parquet')
fresh=pd.read_parquet(ROOT/'results/upstream_reproduction_30/sugarR_reproduction.parquet')
def input_rate(df):
    return float(df[df.flywire_id.isin(namespace['neu_sugar'])].groupby('flywire_id').size().reindex(namespace['neu_sugar'],fill_value=0).mean()/df.trial.nunique())
comparison={'current_configured_input_hz':150,'current_observed_input_hz':input_rate(fresh),
    'archived_observed_input_hz':input_rate(archived),
    'current_spikes':len(fresh),'archived_spikes':len(archived),
    'interpretation':'Original code execution confirmed. Archived output has higher observed input rate; parameter-equivalent numerical replication of that archive is not established.',
    'environment_difference':'Current Python 3.12 / Brian2 2.10.1, original environment Python 3.10 / Brian2 2.5.1'}
write_json(ROOT/'results/upstream_reproduction_30/archived_comparison.json',comparison)

condition_table='| 조건 | Δ rate 평균 Hz | 시드 SD Hz |\n|---|---:|---:|\n'+''.join(f"| {r.condition_id} | {r.delta_mean_hz:.3f} | {r.delta_sd_hz:.3f} |\n" for r in summary.itertuples())
sensitivity_table='| 설정 | 뉴런 | 연결 | E3 Δ rate Hz |\n|---|---:|---:|---:|\n'+''.join(f"| {r.setting} | {r.neurons:,} | {r.edges:,} | {r.delta_mean_hz:.3f} |\n" for r in sensitivity[sensitivity.condition_id.eq('E3')].itertuples())
manuscript=f'''# 뉴스의 위협성 정량화에 기반한 자극 변환과 FlyWire 부분 회로 신경 반응: 계산 파일럿

## 초록

한국어 뉴스의 부정성(N)과 위협성(T)을 분리하여 평가하고, T를 가상 Poisson 입력에 연결하는 재현 가능한 계산 파이프라인을 구현했다. 직접 작성한 가상 요약 24개에 규칙 baseline을 적용했다. 실제 FAFB v783 연결과 공개 세포형 주석에서 LPLC2/LC4→GF/DNp06의 최대 3-hop 경로를 추출하고, 1,367개 뉴런·29,377개 연결의 유도 부분 회로에서 11조건×30시드 실험을 수행했다. 높은 입력 조건 E3의 평균 출력 변화는 {e3:.3f} Hz였고, LPLC2 입출력 연결을 제거한 E5는 {e5:.3f} Hz였다. 이 차이는 생리학적 실증이 아니라 가상 입력과 선택된 회로 구조에 조건부인 모델 결과다. 사람 평가, 생리 tuning curve 및 독립 영상 자료 검증은 수행하지 않았다.

## 1. 서론

부정적 뉴스는 인간의 정서·정신생리적 반응과 관련될 수 있으나, 부정성·즉각적 위협·각성·호르몬은 서로 다른 구성개념이다[1,2]. FlyWire는 신경 연결을 제공하는 구조 자료다[3,4]. Shiu 등의 LIF 모델은 연결에 기초한 계산 실험을 가능하게 하지만, 검증된 감각 회로의 성능이 뉴스 의미나 시각 위협 회로로 자동 이전되지는 않는다[5]. 본 파일럿의 목적은 텍스트 변수와 감각 회로 사이의 가정을 기록하고 대조 조건을 실행할 수 있는 도구를 만드는 것이다.

## 2. 방법

### 2.1 기사와 점수

실제 언론 기사를 수집했다고 주장하지 않는다. 가상 기사 24개를 재난·경제·사회·건강·중립·긍정 범주에 4개씩 작성했다. 같은 태풍 사례는 event_group을 공유한다. 단어별 최대 앵커를 사용하는 투명한 N/T/A 규칙 baseline을 사용하고 원척도 0~4를 4로 나눴다. 이 방식은 문맥·부정문·표현 방식에 취약하다. LLM JSONL 가져오기와 독립 3인 평가·사건 단위 보류 분할·서열형 α·Spearman·MAE·군집 bootstrap은 구현했으나 실제 평가 자료가 없어 성능 수치는 산출하지 않았다.

### 2.2 물리 영상과 신경 입력

주 변환은 T 단독이다. 1/3, 2/3 경계를 사용하여 30/80/150 Hz의 세 단계에 대응시킨다. 이 발화율은 생리 자료에서 추정한 tuning curve가 아니다. 별도의 물리 looming 함수는 θ=2 arctan(ℓ/d), ℓ=0.01 m, 초기 2도·최종 80도, ℓ/|v|=80/40/20 ms를 기록한다. 등속 접근의 서로 다른 지속시간을 맞추기 위해 초기 원을 유지하는 시간을 조절한다. 물리 영상의 시간축과 300 ms 신경 입력은 보정된 대응 관계가 없다.

### 2.3 연결과 LIF

Shiu 저장소 commit `{repro['upstream_commit']}`의 v783 연결 자료와 Schlegel 주석 commit `a83b2776d60d5764cef36b927f5f9679c16c47a2`를 사용했다. GF는 주석의 DNp01/Giant Fiber를 통해 확인했다. LPLC2 210개와 LC4 104개를 입력으로, GF 2개와 DNp06 2개를 출력으로 지정했다. 최소 시냅스 수 5에서 최대 3-edge 경로에 속하는 노드를 선택하고 그 사이의 연결을 모두 보존했다. 외부 회로에서 들어오는 입력은 제외된다.

별도 축소 LIF 엔진은 원본의 v_rest=v_reset=−52 mV, v_threshold=−45 mV, τ_m=20 ms, τ_syn=5 ms, 불응기 2.2 ms, synapse당 0.275 mV, 입력 점프 68.75 mV를 사용했다. subthreshold 방정식은 해석적으로 갱신한다. 기본 dt=.5 ms에서 요청 지연 1.8 ms는 2.0 ms로 양자화한다. Brian2의 상태 갱신→역치→시냅스 이벤트→재설정 순서를 사용한다. dt=.1 ms의 고정 입력 3뉴런 회로에서 독립 Brian2 실행과 발화 수·시점을 비교하는 테스트를 수행했다. 이것이 전뇌 또는 생물학적 검증을 대체하지는 않는다.

### 2.4 대조와 지표

기저/자극/회복은 100/300/100 ms이고, 주 지표는 출력 4개 뉴런의 평균 발화율 차이다. E0 무입력, E1/E2/E3 세 단계, E4 같은 외부 이벤트를 임의의 다른 세포로 전달, E5 LPLC2 연결 제거, E6 LC4 연결 제거, E7 동등 수·연결 정도 근사 매칭 세포 제거, E8 N 단독 입력, E9a/b 고정 T에서 N 변경을 수행했다. 각 조건은 같은 30개 계산 시드를 공유한다. E4의 세포는 감각 세포와 매칭되지 않았고 E7은 역할·자극 노출을 통제하지 못한다. 억제는 입출력 시냅스를 제거하며, 원본 Shiu 함수의 outgoing만 제거하는 구현과 다르다.

시드 평균·SD와 대응 차이의 bootstrap 95% 구간을 보고한다. 이 구간은 계산 변동성이며 생물학적 표본의 추론이나 p값이 아니다. 잠복기·recruitment·AUC·recovery의 정의는 preregistration.md에 기록했다.

## 3. 결과

{condition_table}

E3 대비 E5 평균 출력은 {reduction:.1f}% 감소했다. E7의 평균은 {e7:.3f} Hz였다. 표적 입력의 연결 제거와 다른 세포 제거는 직접 자극 여부에서 차이가 있으므로 이 비교만으로 생물학적 회로 특이성을 입증하지 않는다. E9a/E9b의 동일 출력은 T 단독 인코더의 설계상 불변성을 확인한다. E8도 높은 입력률을 사용하므로 E3와 같은 결과이며 뉴스 의미를 신경 회로가 이해했다는 근거가 아니다.

### 3.1 민감도

각 설정은 3개 계산 시드만 사용한 파일럿이다. 시냅스 임계를 바꾸면 경로에 포함되는 노드 자체가 달라지므로 아래 차이는 연결 가중치와 회로 경계가 함께 달라진 결과다.

{sensitivity_table}

### 3.2 원본 전뇌 코드 실행

변경하지 않은 Shiu `example.ipynb`의 먹이 감각 입력 21개를 사용하여 v630에서 1초×30회 실행했다. 현재 소스의 기본 입력률은 150 Hz다. {repro['n_spikes']:,}개 스파이크와 {repro['active_neurons']}개 활성 뉴런을 기록했고 실행 시간은 {repro['elapsed_seconds']:.1f}초였다. 저장소 제공 참조 출력은 {repro['reference_spikes']:,}개 스파이크·{repro['reference_active_neurons']}개 활성 뉴런이었다. 직접 입력 세포의 관측 발화율은 현재 실행 {comparison['current_observed_input_hz']:.2f} Hz, 참조 {comparison['archived_observed_input_hz']:.2f} Hz였다. 참조 출력과 입력률이 다르고 원 논문 환경(Python3.10/Brian2.5.1)과 현재 환경(Python3.12/Brian2.10.1)도 다르다. 따라서 이는 원본 코드 실행 확인이며 참조 출력의 수치적 동등 재현은 확립하지 않았다.

### 3.3 독립 검증 상태

DANDI 001869/0.260727.1644와 2026년 분석 저장소의 존재·메타데이터를 확인했다[7]. 조회한 10개 NWB 자산은 파일당 약 145GB였다. 원영상·칼슘 시계열을 재분석하지 않았다. 현재 구현은 정리된 세포형별 반응 방향 CSV를 받아 불일치와 미확인 타입을 남길 수 있다. 실제 자료가 입력되지 않아 일치율·시간 상관은 보고하지 않는다. 사람 평가와 생리 측정도 수행하지 않았다.

## 4. 논의와 한계

이 결과는 실행 가능한 계산 비교의 예다. 높은 입력률에 높은 출력이 나타나는 것은 모델 설계에서 예상될 수 있다. 가상 입력을 받는 LPLC2 연결을 제거했을 때 변화가 나타나는 것도 독립적인 생물학적 발견이 아니다. 다른 실험 개체·행동 상태의 영상 자료와의 비교, 인코더의 생리 보정, 감각 정체성 및 자극 노출이 일치하는 대조군이 필요하다. 한 개체의 구조, 부분 회로의 경계, 전달물질 예측 부호, 수용체 정보 부재, 시간 이산화가 결과에 영향을 준다.

본 연구는 초파리의 뉴스 이해, 실제 스트레스 호르몬, 사람의 불안이나 생리 반응을 검증하지 않았다. 실제 기사·독립 라벨·모델 정확도·실측 검증을 확보한 후에야 원제안서의 본 가설을 검정할 수 있다.

## 5. 재현성과 산출물

실행 설정·원자료 해시·코드 해시·의존성·모든 시드의 입력 및 스파이크를 보존했다. `results/v783/report.html`, `condition_summary.csv`, `paired_contrasts.csv`, `spikes/`와 `results/sensitivity/summary.csv`가 근거 자료다. 재실행 명령과 자료 출처는 README 및 연구계획서에 있다. 현재 문서는 계산 파일럿 원고이며 심사·출판된 논문이 아니다.

## 참고문헌

1. Marin et al. (2012). PLOS ONE. https://doi.org/10.1371/journal.pone.0047189
2. Soroka et al. (2019). PNAS. https://doi.org/10.1073/pnas.1908369116
3. Dorkenwald et al. (2024). Nature. https://doi.org/10.1038/s41586-024-07558-y
4. Schlegel et al. (2024). Nature. https://doi.org/10.1038/s41586-024-07686-5
5. Shiu et al. (2024). Nature. https://doi.org/10.1038/s41586-024-07763-9
6. Kim et al. (2023). Current Biology. https://doi.org/10.1016/j.cub.2022.12.014
7. Zucker-Scharff et al. (2026). Current Biology / DANDI. https://doi.org/10.48324/dandi.001869/0.260727.1644
'''
(ROOT/'docs/manuscript.md').write_text(manuscript,encoding='utf-8')

font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font=FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties(family='sans-serif')
fig=plt.figure(figsize=(16.5,11.7),facecolor='#f1f5f8')
def text(x,y,s,size=14,weight='normal',color='#24374a'):
    fig.text(x,y,s,fontsize=size,fontproperties=font,weight=weight,color=color,va='top')
text(.045,.96,'뉴스의 위협성에서 초파리 회로 모델의 반응까지',25,'bold')
text(.045,.905,'실제 FAFB v783 연결 · 가상 뉴스 24개 · 부분 회로 LIF · 11조건 × 30시드의 계산 파일럿',15)
text(.045,.855,'1,367개 뉴런     29,377개 연결     314개 입력     4개 출력',20,'bold','#1b7e80')
for x,name,title in [(.035,'01_news_scores','N과 T 분리 예측'),(.355,'04_output_timecourses','입력 단계와 출력 변화'),(.675,'05_silencing','가상 연결 제거와 대조')]:
    ax=fig.add_axes([x,.39,.29,.39])
    ax.imshow(plt.imread(run/f'figures/{name}.png'))
    ax.axis('off')
    text(x,.80,title,17,'bold')
text(.045,.355,'관찰한 결과',19,'bold')
text(.045,.31,f'E3 Δ rate {e3:.2f} Hz → LPLC2 연결 제거 E5 {e5:.2f} Hz ({reduction:.1f}% 감소)\n무작위 억제 E7 {e7:.2f} Hz · 동일 T의 E9a/E9b는 같은 출력',16)
text(.045,.225,'해석 범위',19,'bold')
text(.045,.18,'연결은 실제 자료, 활동은 가상 입력에 대한 계산 결과다. 입력률 증가나 입력 연결 제거 효과는 설계에서 예상될 수 있다.\n뉴스는 규칙 baseline으로 평가한 가상 예시다. 사람 평가·생리 인코더 보정·독립 신경 영상 검증은 아직 수행하지 않았다.\nE4는 감각 세포 매칭이 아니며, E7도 역할·직접 자극 여부가 완전히 일치하지 않는다. 시드 수는 생물학적 표본 수가 아니다.',13)
text(.045,.065,'재현: README.md · configs/ · results/v783/manifest.json · 모든 조건·시드의 원시 스파이크 제공',11,color='#637587')
fig.savefig(ROOT/'results/research_poster.pdf',bbox_inches='tight')
fig.savefig(ROOT/'results/research_poster.png',dpi=130,bbox_inches='tight')
plt.close(fig)

cells=[
 {'cell_type':'markdown','metadata':{},'source':['# 뉴스 × FlyWire 계산 파일럿 탐색\n','실제 연결과 모형 활동을 구별합니다. 가상 기사 및 규칙 예측이며 사람 평가 성능은 미검증입니다.']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
  'from pathlib import Path\n','import pandas as pd\n','import matplotlib.pyplot as plt\n',
  'ROOT = Path.cwd() if (Path.cwd()/"pyproject.toml").exists() else Path.cwd().parent\n',
  'run = ROOT / "results/v783"\n','summary = pd.read_csv(run/"condition_summary.csv")\n','summary\n']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
  'pd.read_csv(run/"paired_contrasts.csv")\n']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
  'curves = pd.read_csv(run/"output_curves.csv")\n',
  'curves.groupby(["condition_id", "time_ms"]).rate_hz.mean().unstack(0).plot(figsize=(12,4))\n',
  'plt.ylabel("Mean output firing rate (Hz)")\n','plt.xlabel("Simulation time (ms)")\n']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
  'neurons = pd.read_csv(run/"circuit/neurons.csv", dtype={"root_id":str})\n',
  'neurons.groupby(["role","cell_type"]).size().sort_values(ascending=False).head(20)\n']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
  'pd.read_csv(ROOT/"results/sensitivity/summary.csv").query("condition_id == \'E3\'")\n']},
 {'cell_type':'markdown','metadata':{},'source':['## 남은 검증\n','실제 뉴스 240건, 독립 평가자, LLM 평가, 생리 보정 곡선, 실제 영상과 사람 반응 검증은 별도 자료가 필요합니다.']}
]
for i,cell in enumerate(cells):
    cell['id']=f'cell-{i+1}'
write_json(ROOT/'notebooks/01_explore_results.ipynb',{'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5})
write_json(ROOT/'results/deliverables_manifest.json',manifest([run/'condition_summary.csv',run/'paired_contrasts.csv',ROOT/'results/sensitivity/summary.csv',ROOT/'docs/manuscript.md',ROOT/'results/research_poster.pdf']))
print('Generated manuscript, poster and notebook from stored results.')
