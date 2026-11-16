"""Build, package and verify the final computational-pilot deliverables."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

from news_flywire.provenance import write_json,sha256
from news_flywire.risk_review_updates import load_review_catalogs,input_key
from news_flywire.risk_assessment import AXES,danger_index

OUT=Path('results/final_release')


def build(out):
 validation=json.loads((out/'validation/summary.json').read_text())
 if not validation['passed'] or not validation['full_private_audits'] or len(validation['checks'])!=8:raise ValueError('All eight full validation checks required for this sealed final report')
 if '140 passed' not in (out/'validation/tests.log').read_text():raise ValueError('Final test count differs from report')
 for r in validation['checks']:
  if sha256(r['log'])!=r['log_sha256']:raise ValueError('Validation log changed')
 data=pd.read_parquet(out/'scores/news_risk_scores.parquet')
 csv=pd.read_csv(out/'scores/news_risk_scores.csv')
 assert len(data)==8128 and data.news_id.is_unique and data.session_reviewed.sum()==1074
 assert csv.news_id.tolist()==data.news_id.tolist() and np.allclose(csv.danger_score,data.danger_score)
 catalog=load_review_catalogs();assert len(catalog)==1074
 for r in data[data.session_reviewed].itertuples():
  review=catalog[input_key(r.title,r.assessment_body_sha256)]
  assert np.isclose(r.danger_score,review['danger_score'])
  assert all(np.isclose(getattr(r,k)*4,review[k]) for k in AXES)
 reviews=out/'reviews';reviews.mkdir(exist_ok=True)
 records=[{**{k:v for k,v in r.items() if k!='_catalog_sha256'},'source_catalog_sha256':r['_catalog_sha256']} for r in catalog.values()]
 (reviews/'review_catalog.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 write_json(reviews/'manifest.json',dict(real_direct_reviews=1074,matching='exact title and cleaned body SHA256',source_manifest_sha256=sha256('data/risk/review_catalogs.json')))
 old=pd.read_parquet('data/risk/20261110_followup/news_risk_scores.parquet')
 assert np.array_equal(old.model_danger_score,data.model_danger_score)
 assert np.array_equal(old.assessment_research_eligible,data.assessment_research_eligible)
 bridge=json.loads(Path('results/article_bridge_v1/summary.json').read_text())
 control=json.loads(Path('results/control_matching_v1/summary.json').read_text())
 assert bridge['fresh_simulations']==845 and control['new_simulations']==120 and control['reused_parent_simulations']==80
 expected={('R_intact','GF'):166.34375,('R_intact','DNp06'):5.625,('ACh314_approx','GF'):139.3125,('ACh314_approx','DNp06'):104.6875,('common178_target','GF'):104.53125,('common178_target','DNp06'):21.28125,('common178_control','GF'):73.34375,('common178_control','DNp06'):122.46875}
 actual={(r['condition'],r['cell_type']):r['mean_delta_hz'] for r in control['output_results']}
 assert all(np.isclose(actual[k],v) for k,v in expected.items())
 summary=dict(status='completed_computational_pilot',data_window_UTC=['2026-08-12','2026-11-09'],
  real_articles=len(data),publishers=int(data.source.nunique()),research_eligible=int(data.assessment_research_eligible.sum()),
  direct_AI_reviews=int(data.session_reviewed.sum()),model_only_articles=int((~data.session_reviewed).sum()),
  deployed_model='v5',independent_human_raters=0,physiological_validation=False,
  bridge_new_simulations=bridge['fresh_simulations'],control_new_simulations=control['new_simulations'],
  total_new_linked_simulations=bridge['fresh_simulations']+control['new_simulations'],control_reused_trials=control['reused_parent_simulations'],
  tests_passed=140,validation_checks=len(validation['checks']),all_validation_passed=True,
  conclusion='Reproducible computational pipeline complete. No proof of perfect news accuracy, biological circuit specificity, fly news comprehension or human stress responses.',
  prior_research_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 write_json(out/'summary.json',summary)
 write_json(out/'release_integrity.json',dict(csv_parquet_equal=True,all_1074_direct_indices_and_axes_checked=True,
  unreviewed_model_scores_unchanged=True,research_eligibility_unchanged=True,source_catalog_sha256=sha256('data/risk/review_catalogs.json')))
 (out/'index.html').write_text(HTML)
 print(out/'index.html')


def package(out):
 if (out/'release_manifest.json').exists():raise ValueError('Final package already sealed')
 if not (out/'final_report.pdf').exists():raise ValueError('Render final_report.pdf first')
 folders=[out,Path('results/article_bridge_v1'),Path('results/control_matching_v1'),Path('docs')]
 files=[]
 for folder in folders:
  files += [p for p in folder.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in ['bundle_contents.json','release_manifest.json']]
 files=sorted(set(files+[Path('README.md')]))
 contents={str(p):sha256(p) for p in files}
 archive=out/'News2Neuron_final_results.zip'
 note='''News2Neuron final results bundle\n\nOpen results/final_release/index.html for the final conclusion and downloads.\nIncluded: final score tables, PDF, reports, methods, validation logs, and raw article/control neural trials.\nFull source, training models and earlier experiments: https://github.com/hwanghj09/News2Neuron\nPublisher full text and private feature caches are not included.\nBUNDLE_MANIFEST.json records SHA-256 for every packaged file except the manifest itself and this note.\n'''
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,str(p))
  z.writestr('BUNDLE_MANIFEST.json',json.dumps(contents,ensure_ascii=False,indent=2))
  z.writestr('BUNDLE_README.txt',note)
 write_json(out/'bundle_contents.json',contents)
 write_json(out/'release_manifest.json',dict(status='sealed',archive=str(archive),archive_sha256=sha256(archive),
  files={str(p):sha256(p) for p in [*files,archive,out/'bundle_contents.json']},
  finalizer_source_sha256=sha256(__file__),no_new_labels_or_neural_outcome_tuning_at_finalization=True,
  scope='Completed computational pilot; external biological and independent human validation unperformed.'))
 print('Sealed final results package:',archive)


def verify(out):
 m=json.loads((out/'release_manifest.json').read_text())
 for p,digest in m['files'].items():
  if sha256(p)!=digest:raise ValueError('Final release file changed: '+p)
 import hashlib
 with zipfile.ZipFile(m['archive']) as z:
  contents=json.loads(z.read('BUNDLE_MANIFEST.json'))
  for p,digest in contents.items():
   if hashlib.sha256(z.read(p)).hexdigest()!=digest:raise ValueError('Archive content changed: '+p)
 print('Verified final release hashes and',len(contents),'archived files.')


HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>News2Neuron · 최종 보고서</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f5;color:#183740;font:16px/1.75 "Noto Sans CJK KR",system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:35px 22px}h1{font-size:clamp(29px,4vw,43px);line-height:1.3}h2{font-size:24px}.note{background:#fff0d5;padding:18px;border-radius:10px}section{background:white;border:1px solid #dbe5e7;border-radius:12px;padding:23px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:15px;border:1px solid #dbe5e7;border-radius:10px;background:white}.card b{display:block;font-size:28px;color:#176d63}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dfe8ea;text-align:left;vertical-align:top}a{color:#146c63}img{max-width:100%;height:auto}.links{display:flex;flex-wrap:wrap;gap:15px}small{color:#59767e}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:20px 12px}.cards{grid-template-columns:1fr 1fr}section{padding:15px}}@media print{body{background:white;font-size:10pt}main{padding:0;max-width:none}h1{font-size:25pt}h2{font-size:15pt}section{padding:12px;margin:12px 0;border-radius:0}table,.cards,.note{break-inside:avoid}h2{break-after:avoid}a{color:#146c63;text-decoration:none}.downloads{display:none}img{max-height:85mm;object-fit:contain}small{font-size:9pt}}
</style><main><small>NEWS2NEURON · FINAL COMPUTATIONAL PILOT · 2026-11-16</small><h1>실제 뉴스에서 가상 회로 반응까지.<br>최종 결과와 해석.</h1>
<p>실제 한국어 뉴스 평가, 가상 자극 변환, 실제 FlyWire 연결 구조를 사용하는 부분 회로 실험과 보강 대조를 구현·실행하고 결과를 고정했습니다.</p>
<div class="cards"><div class="card">실제 뉴스<b>8,128건</b>8개 매체 · 연구용 8,050건</div><div class="card">직접 AI 판정<b>1,074건</b>나머지 7,054건은 v5 예측</div><div class="card">새 회로 계산<b>965회</b>기사별 845 + 보강 120</div><div class="card">전체 테스트<b>140개</b>최종 검증 8종 통과</div></div>
<section><h2>최종 결론</h2><p><strong>재현 가능한 계산 파이프라인은 완성했습니다. 연결 구조와 입력·대조군 구성이 모델 출력에 큰 영향을 주었습니다. 그러나 뉴스 평가의 완벽한 정확도와 위협 회로의 생물학적 특이성, 초파리의 뉴스 이해 또는 사람의 스트레스 반응은 입증하지 못했습니다.</strong></p><p class="note">이번 산출물은 계산 파일럿으로 종료합니다. 출력 Hz는 정의한 가상 입력에 대한 LIF 모델 반응이며 실제 신경 활동·호르몬·불안의 측정값이 아닙니다. 미수행 실증 항목은 검증 완료로 표시하지 않습니다.</p></section>
<section><h2>1. 뉴스 평가: 직접 판정과 모델 예측 분리</h2><p>2026-08-12~11-09 UTC 발행 기사입니다. 10,000페이지 확인, 본문·날짜 통과 8,156건, 중복 28건 제거로 8,128건을 구성했습니다. 현재 날짜에 맞춰 새로 수집한 자료는 아닙니다. 본문·날짜·출처·중복·사건 가족·원자료 해시를 기록했습니다.</p><p>부정성 N과 잔존 위협 T를 구분하고 S/L/I/X로 잔존 위해 지수 R을 계산합니다. R은 연구 기준식이며 위험 확률이 아닙니다. 1,074건의 직접 AI 판정은 제목과 정제 본문 해시가 같은 버전에만 재사용합니다. 다른 기사는 v5 모델 출력으로 표시합니다.</p><p>후속 v6~v10 후보는 일부 지표 개선에도 채택 조건을 충족하지 못해 기본 모델로 승격하지 않았습니다. 판정자는 같은 AI 한 명이며 독립 사람 정답 정확도는 미검증입니다. 합성 테스트와 실제 뉴스도 분리했습니다.</p></section>
<section><h2>2. 기사별 새 회로 실험</h2><p>8,128건의 자극 파라미터를 만들고, 직접 AI 판정이 있는 실제 기사 24건에 7조건×5시드 840회와 무입력 5회를 새로 계산했습니다. 초기 세 단계 조회 결과를 기사에 다시 붙인 계산이 아닙니다. 연구 제외 78건에는 신경 입력률을 부여하지 않았습니다.</p><p>실제 FAFB v783 연결의 1,367개 뉴런·29,377개 방향성 연결을 사용했습니다. 가상 원판과 입력의 제시 시간은 400ms로 맞췄습니다. 평균 0~150Hz 및 시간별 입력 함수는 설계 가정이며 생리 보정된 시각 인코더는 아닙니다.</p><div class="scroll"><table><thead><tr><th>잔존 위해 기사군: 동일 R 이벤트</th><th>모델 출력 ΔHz</th></tr></thead><tbody><tr><td>정상 연결</td><td>85.98</td></tr><tr><td>LPLC2 입출력 연결 제거</td><td>23.84</td></tr><tr><td>LC4 입출력 연결 제거</td><td>80.58</td></tr><tr><td>연결 정도 근사 매칭 대조 제거</td><td>85.98</td></tr></tbody></table></div><p>LPLC2 제거에서 약 72% 감소했지만, 다른 뉴런 제거와의 직접 자극 여부 차이·부분 회로 경계 등의 조건을 고려해야 합니다. 이 차이만으로 생물학적 회로 특이성을 결론 내리지 않습니다.</p><img src="../article_bridge_v1/figures/article_controls.png" alt="기사별 입력과 회로 대조 결과"></section>
<section><h2>3. 대조군 보강으로 달라진 해석</h2><p>같은 기사 8건에서 전달물질·출력 경로·연결 특징을 보강한 새 계산 120회를 수행했습니다. 이전 조건 80회는 재사용으로 따로 기록했고 새 계산 수에 포함하지 않았습니다.</p><div class="scroll"><table><thead><tr><th>314개 입력 조건</th><th>GF ΔHz</th><th>DNp06 ΔHz</th></tr></thead><tbody><tr><td>원래 표적 입력</td><td>166.34</td><td>5.62</td></tr><tr><td>원래 무작위 입력</td><td>11.28</td><td>93.31</td></tr><tr><td>전달물질·경로·연결 특징 근사 대조</td><td>139.31</td><td>104.69</td></tr></tbody></table></div><p>GF의 표적-대조 차이는 <strong>155.06→27.03Hz</strong>로 줄었습니다. 대조군 구성에 민감한 차이였으며 여러 매칭 요인을 함께 바꿨으므로 단일 원인으로 분해하지 못했습니다.</p><p>이 부분 회로에서는 같은 hop 쌍·예측 전달물질로 314개 전부를 맞출 수 없었습니다. 정확한 층 매칭이 가능한 공통 178개에서 표적/대조는 GF <strong>104.53/73.34Hz</strong>, DNp06 <strong>21.28/122.47Hz</strong>였습니다. DNp06의 반대 반응은 남았습니다. 178개 부분집합을 314개 전체로 일반화하지 않습니다.</p><img src="../control_matching_v1/matched_controls.png" alt="314개 근사 대조와 178개 공통 부분집합을 분리한 출력 세포형별 비교"></section>
<section><h2>4. 검증 완료와 미수행 항목</h2><p>전체 테스트 140개, 저장 원응답에서 뉴스 8,128건 재추출, 초기 330회 원자료, v8~v10 판정·봉인·예측, 기사별 845회와 대조 보강 120회의 감사가 통과했습니다. 이벤트 배열과 스파이크 기반 지표를 재계산하고 대표 네트워크 실행의 스파이크 시점·뉴런 ID를 정확히 재현했습니다. 재현성 검사를 모델 정확도나 생물학적 타당성으로 바꾸어 해석하지 않았습니다.</p><div class="scroll"><table><thead><tr><th>항목</th><th>최종 상태</th></tr></thead><tbody><tr><td>데이터·코드·가상 자극·모델 대조·원자료</td><td>구현·실행·재현 확인</td></tr><tr><td>독립 사람 기준 뉴스 정확도</td><td>미검증</td></tr><tr><td>생리 보정된 시각 입력 함수</td><td>미검증</td></tr><tr><td>실제 초파리 영상/칼슘 시계열 비교</td><td>메타데이터 확인·연동 기능 구현, 원시계열 재분석 미수행</td></tr><tr><td>사람의 생리·호르몬 반응</td><td>미수행</td></tr></tbody></table></div></section>
<section class="downloads"><h2>최종 자료</h2><div class="links"><a href="final_report.pdf">최종 보고서 PDF</a><a href="News2Neuron_final_results.zip">최종 결과 묶음 ZIP</a><a href="scores/news_risk_scores.csv">최신 8,128건 CSV</a><a href="scores/news_risk_scores.parquet">최신 점수 Parquet</a><a href="reviews/review_catalog.jsonl">1,074건 직접 판정 원장</a><a href="../../docs/final-report.md">상세 최종 보고서</a><a href="../article_bridge_v1/report.html">기사별 인터랙티브 실험</a><a href="../control_matching_v1/report.html">보강 대조 보고서</a><a href="validation/summary.json">최종 검증 기록</a><a href="release_manifest.json">배포 해시 원장</a></div><p>ZIP은 최종 점수·보고서·방법 문서·검증 기록·새 두 회로 실험 원자료를 포함합니다. 전체 소스·학습 모델과 과거 실험은 News2Neuron 저장소에서 제공합니다. 원문 전문과 개인 특징 캐시는 포함하지 않았습니다.</p></section><footer><small>계산 시드·뉴런·의도적으로 고른 기사 집합을 생물학적 독립 표본으로 세지 않았습니다. 최종 문서는 계산 연구 보고서이며 심사·출판된 실증 논문이 아닙니다.</small></footer></main></html>'''


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['build','package','verify']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
 {'build':build,'package':package,'verify':verify}[a.stage](a.out)
