# News2Neuron

뉴스 위협성 × FlyWire 계산 연구

첨부 연구계획서의 계산 파일럿을 실제로 실행하는 Python 프로젝트입니다. **뉴스 N/T/A 평가 → 위협성 기반 입력 → 실제 FAFB v783 부분 회로 → LIF → 대조실험·분석·보고서**를 연결했습니다.

## 먼저 볼 결과

- [실행 결과 보고서](results/v783/report.html): 6개 분석 그림과 원자료 링크
- [연구 결과 원고](docs/manuscript.md): 방법, 실제 계산 결과, 해석의 범위
- [발표 포스터](results/research_poster.pdf)
- [실행·검증 상태](docs/implementation-status.md)
- [원래 연구계획서](docs/research-plan.md)

보고서는 인터넷 연결 없이 열 수 있습니다. 모든 그림은 PNG/SVG로 저장했습니다.

## 현재 들어 있는 것

| 항목 | 제공 범위 |
|---|---|
| 뉴스 점수 | 직접 작성한 가상 기사 24개, N/T/A 규칙 baseline, 엄격한 LLM JSONL 가져오기 |
| 사람 평가 | 3인 빈 평가 양식, 기준서, 사건 단위 고정 분할, 중위값·서열형 α·Spearman·MAE·군집 bootstrap |
| 실제 연결 | FAFB v783에서 확인한 LPLC2 210개, LC4 104개, GF 2개, DNp06 2개 |
| 회로 | 최대 3-hop 입력→출력 경로의 노드 1,367개, 유도 방향성 연결 29,377개 |
| LIF | Shiu 상수를 사용하는 별도 부분 회로 엔진, 명시적 Poisson 입력·입출력 연결 제거 |
| 실험 | E0~E9, E9a/b 포함 11조건 × 30시드 = 330회 |
| 원본 예제 | 변경하지 않은 Shiu/Brian2 전뇌 v630 먹이 감각 예제를 1초 × 30회 실행 |
| 민감도 | 연결 임계 1/3/5, 입력률 .75/1/1.25배, dt .1/.5 ms를 각 3시드에서 비교 |
| 실측 검증 | DANDI 메타데이터·자산 목록 확인, 정리된 세포형별 반응 방향 CSV 비교 기능 |

실제 기사 240개, 사람 평가·LLM 정확도, 생리 보정된 인코더, 대용량 NWB 시간계열 재분석, 사람 대상 실험은 아직 확보·수행하지 않았습니다. 이를 데이터나 검증된 성과로 생성하지 않았습니다.

## 실행

Python 3.11 이상이 필요합니다. 아래 명령은 이 폴더에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research,test]'

# 인터넷 없이 가상 연결로 전체 코드 흐름 점검
news-flywire demo --seeds 3 --out results/my_demo

# 준비된 실제 v783 부분 회로와 가상 기사 점수로 재실행
news-flywire run --seeds 30 --out results/my_v783

# 핵심 계산·스키마·분할·대조군·Brian2 수치 비교 검사
python -m pytest -q
```

이 작업 환경에는 `.venv`가 이미 설치되어 있습니다. 바로 실행하려면 `.venv/bin/news-flywire`를 사용합니다. 실행 결과가 있는 폴더는 덮어쓰지 않으므로 새 `--out`을 지정합니다.

새 컴퓨터나 압축파일에서 원본 대용량 자료까지 복원하려면:

```bash
python scripts/fetch_sources.py
news-flywire extract --out data/derived/my_v783
python scripts/reproduce_upstream.py --trials 30 --out results/my_upstream
python scripts/run_sensitivity.py --seeds 3 --max-neurons 15000 --out results/my_sensitivity
```

`fetch_sources.py`는 고정한 원본 저장소(약 370MB checkout)와 주석을 받습니다. DANDI 파일은 개당 약 145GB로 확인되어 메타데이터만 조회합니다. 정리된 부분 회로와 주 실행 결과는 압축파일에 포함합니다.

## 자신의 기사로 연구하기

`data/sample_news.csv`와 같은 컬럼의 CSV를 준비합니다. 제목·요약만으로 개인적 관련성이나 실제 스트레스를 추정하지 않습니다.

```bash
news-flywire prepare --news data/my_news.csv --out data/labels/my_study
news-flywire score --news data/my_news.csv --out data/derived/my_scores.csv
news-flywire run --scores data/derived/my_scores.csv --out results/my_news_run

# 독립 평가자 3명 이상이 실제로 작성한 뒤 수행
news-flywire evaluate --scores data/derived/my_scores.csv \
  --labels data/labels/my_study/ratings_completed.csv \
  --splits data/labels/my_study/splits.csv --out results/my_evaluation
```

LLM은 [고정 프롬프트와 JSON 형식](docs/llm-prompt.md)을 사용해 결과를 가져옵니다. API 키나 연결된 LLM이 없는 상태에서 LLM 실행을 가장하지 않습니다. 규칙 baseline은 문맥·부정문·제목과 본문의 불일치에 취약하며 성능을 검증하지 않았습니다.

실측 자료를 정리한 경우:

```bash
news-flywire validate --observed data/raw/observed_type_responses.csv \
  --run results/v783 --out results/measured_comparison.csv
```

형식과 비교 가능 범위는 [데이터 형식](docs/data-formats.md)을 따릅니다. 지원하지 않는 매핑과 불일치는 결과에 남습니다.

## 해석과 재현

원본 Shiu 예제는 v630이며 위협 회로는 v783입니다. root ID를 섞지 않습니다. 원본 코드는 그대로 두고 별도 스크립트에서 명시적인 반복·시간·시드를 설정합니다. 원본 저장 결과와 현재 기본 입력률 사이의 차이는 [재현 기록](results/upstream_reproduction_30/reproduction.json)과 연구 원고에 적었습니다.

부분 회로 엔진의 통합 순서는 Brian2의 state update → threshold → synaptic events → reset에 맞추었습니다. dt=.1 ms의 작은 고정 입력 회로에서 독립 Brian2 적분 결과와 발화 수·시점을 비교하는 테스트를 포함합니다. 이것이 전체 뇌 또는 시각 위협 회로의 생리학적 검증을 뜻하지는 않습니다.

`configs/`와 실행 폴더의 `resolved_config.json`, `manifest.json`, `input_manifest.json`에 설정·시드·데이터 해시·코드 해시·패키지 버전을 보존합니다. 모든 조건·시드의 스파이크, 외부 입력 이벤트, 세포별 발화율과 선택한 뉴런을 저장합니다. 정확한 패키지 버전은 `requirements-lock.txt`에 있습니다.

입력 증가에 따른 출력 증가, 표적 입력 뉴런 제거에 따른 출력 감소는 이 모델 설계에서 예상 가능한 결과입니다. E4의 무작위 세포는 감각 세포 매칭이 아니며 E7도 자극 여부·기능·연결 정도를 완전히 맞추지 않습니다. 회로 특이성이나 실제 신경 반응을 입증한 결과로 해석하지 않습니다.

## 파일 구조

```text
configs/                   점수·자극·커넥톰 설정
data/sample_news.csv       직접 작성한 가상 기사
data/labels/               빈 독립평가 양식·고정 분할
data/derived/flywire_v783/ 실제 연결·주석·경로·출처
docs/                     계획·기준서·방법·한계·원고
src/news_flywire/          CLI·점수·입력·그래프·LIF·평가·보고서
scripts/                  자료 복원·원본 실행·민감도·발표물 생성
notebooks/                결과 탐색 노트북
results/v783/             330회 계산 결과·그림·보고서
results/sensitivity/      설정별 민감도 실행
results/upstream_reproduction_30/ 원본 전뇌 실행 결과
tests/                    연구상 중요한 불변조건·수치 검증
```

연구·데이터 인용은 [계획서 참고문헌](docs/research-plan.md)과 각 데이터의 출처를 따릅니다. Shiu 원본의 MIT 라이선스는 `docs/upstream-LICENSE`에 보존했습니다. 이 저장소의 실행은 공개 연결 구조를 이용한 계산 분석이며, 초파리가 뉴스를 이해하거나 인간의 스트레스 호르몬이 증가한다는 주장을 하지 않습니다.
