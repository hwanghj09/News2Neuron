"""Author-created illustrative summaries; no scraped articles or invented human ratings."""
from pathlib import Path
import pandas as pd
rows=[
('disaster','storm','태풍 접근에 해안 지역 대피 명령','강한 태풍이 접근하고 있다. 지방 당국은 해안 주민에게 즉시 대피하도록 안내했다.'),
('disaster','storm','태풍 경보와 임시 대피소 안내','태풍 경보가 발령됐다. 같은 태풍의 영향권에 있는 지역은 임시 대피소를 열었다.'),
('disaster','flood','홍수 위험으로 하천 출입 통제','집중호우로 홍수 위험이 커져 하천 출입이 통제됐다.'),
('disaster','bridge','교량 붕괴 위험으로 차량 우회','점검 결과 교량의 붕괴 위험이 확인됐다. 차량은 우회해야 한다.'),
('economy','jobs','지역 공장 해고 계획 발표','지역 공장이 해고 계획을 발표했다. 근로자들의 실직 우려가 커졌다.'),
('economy','recession','경기 침체 우려에 투자 감소','경기 침체 우려로 기업 투자가 감소했다. 장기 전망에 대한 논의가 이어진다.'),
('economy','shop','오랜 동네 서점 폐업','오랫동안 운영된 동네 서점이 적자로 폐업한다. 주민들은 아쉬움을 전했다.'),
('economy','trade','이번 분기 수출 감소','이번 분기 수출이 감소했다. 통계는 잠정 집계이며 추가 분석이 예정돼 있다.'),
('society','bereavement','가족의 사별 경험을 담은 기록','한 가족이 사별 이후 서로를 돌본 경험을 기록했다. 현재의 긴급한 위해 상황은 언급하지 않았다.'),
('society','memorial','옛 스승을 기억하는 추모 모임','졸업생들이 별세한 스승을 기억하는 추모 모임을 열었다.'),
('society','archive','오래전 사고 기록 전시','오래전 사고와 그 이후의 변화를 돌아보는 기록 전시가 열렸다.'),
('society','library','노후 도서관 휴관','시설 보수를 위해 도서관이 잠시 휴관한다. 주민들은 이용 불편을 이야기했다.'),
('health','infection','감염 확산 주의 안내','지역 감염 확산으로 보건 당국이 주의를 안내했다. 증상이 있으면 상담을 권고했다.'),
('health','prevention','건강 검진 안내','보건소가 주민 대상 건강 검진 일정과 신청 절차를 안내했다.'),
('health','exercise','걷기 프로그램 시작','지역 보건소에서 가벼운 걷기 프로그램을 시작했다.'),
('health','research','새 건강 연구 소개','연구진이 수면 습관 관련 관찰 결과를 소개했다. 추가 검증이 필요하다고 밝혔다.'),
('neutral','bus','시내버스 시간표 변경','다음 달부터 시내버스 운행 시간표가 변경된다. 새 시간표는 안내문에서 확인할 수 있다.'),
('neutral','weather','맑은 날씨 예보','내일은 대체로 맑은 날씨가 예상된다.'),
('neutral','museum','박물관 운영 시간 안내','박물관이 주말 운영 시간과 이용 방법을 안내했다.'),
('neutral','meeting','주민 회의 일정 공지','주민 회의가 다음 주에 열린다. 안건은 공원 이용과 시설 정비다.'),
('positive','festival','지역 축제 성공적으로 마무리','주민들이 함께 준비한 지역 축제가 성공적으로 마무리됐다.'),
('positive','donation','이웃을 위한 기부 행사','주민들이 이웃을 위한 기부 행사에 참여했다.'),
('positive','sports','지역 선수의 우승 소식','지역 선수가 대회에서 우승했다. 동료들이 축하했다.'),
('positive','garden','공동 정원 개장','주민들이 만든 공동 정원이 개장했다. 누구나 산책하며 이용할 수 있다.'),
]
out=Path(__file__).resolve().parents[1]/'data/sample_news.csv'
pd.DataFrame([dict(news_id=f'synthetic_{i+1:03}',event_group=event,title=title,summary=summary,
 category=category,data_source='author_created_synthetic',source='synthetic_example',url='',published_date='')
 for i,(category,event,title,summary) in enumerate(rows)]).to_csv(out,index=False)
