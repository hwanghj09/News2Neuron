# 입력과 결과 형식

`news-flywire`는 저장소 루트에서 실행한다. 뉴런 root ID는 항상 문자열로 보존한다. 자바스크립트 숫자형으로 변환하면 큰 ID의 정밀도가 손실될 수 있다.

| 파일 | 필수 컬럼 / 역할 |
|---|---|
| 뉴스 CSV | news_id, event_group, title, summary, category, data_source. 권장 source, url, published_date |
| 사람 평가 CSV | news_id, rater_id, N, T, A, notes. N/T/A는 원척도 0~4 정수. 기사당 3명 이상 |
| 모델 점수 CSV | 뉴스 메타데이터 + N/T/A 정규화 0~1, scorer, rationale, score_source |
| 실측 비교 CSV | cell_type, response_delta, modality, source, alignment_level. alignment_level=cell_type만 방향 비교 지원 |
| neurons.csv | root_id, cell_type, role, version, 주석·연결 정도 |
| edges.parquet | pre, post, synapses, signed_synapses |
| metrics.csv | condition_id, seed, connectome_version, 입력 점수·빈도, 발화율·잠복기·AUC·recruitment·recovery |
| article_responses.csv | news_id/event_group와 해당 L/M/H 조건의 시드별 출력. 같은 단계의 기사는 같은 결과를 공유 |
| cell_metrics.parquet | 모든 조건·시드·뉴런의 기저/자극/Δ rate |
| input_events.parquet | 모든 외부 입력 이벤트의 step, time_ms, input_slot, root_id, condition_id, seed |
| spikes/{condition}/seed*.parquet | 모든 시드·조건의 원시 스파이크. 첫 시드는 실행 폴더 최상위에도 제공 |
| cell_selections.json | 조건별 입력·출력·차단 root ID 및 입력 파라미터 |
| *_manifest.json | 원자료 해시, 소프트웨어 버전, 시드·출처·한계 |

실측 CSV는 이미 정리된 반응 변화량을 입력한다. 칼슘 신호와 스파이크 발화율은 단위가 다르므로 방향 비교만 수행하며 진폭 MAE나 일대일 뉴런 매칭을 주장하지 않는다. 아직 지원하지 않는 영역 매핑과 미확인 타입은 삭제하지 않고 comparison_status로 남긴다.

```csv
cell_type,response_delta,modality,source,alignment_level
```

이 헤더만으로는 실측 자료가 되지 않는다. 출처와 반응 값은 독립적으로 확보해야 한다. 현재 DANDI 자료의 자산 목록과 메타데이터만 `data/derived/public_sources/`에 기록되어 있다. 원영상/NWB 재분석, 시간 정렬·상관 검증은 미구현 확장 항목이다.
