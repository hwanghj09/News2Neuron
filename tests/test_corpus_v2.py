"""Scientific failure cases for the large-corpus preparation stages."""
import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from news_flywire.corpus.build import balanced_plan, map_in_window, parse_map
from news_flywire.corpus.extract import canonicalize, extract_article, parse_date
from news_flywire.corpus.prepare import cluster_events, deduplicate, temporal_split,harmonize_section


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/corpus_v2.yaml').read_text())


def article(config,extra_meta='',body=None,publication='2026-11-08T10:00:00+09:00'):
    body=body or ('지역 주민들은 시설을 점검하고 안전하게 이용하는 방법을 확인했다. 담당 기관은 조사 결과와 향후 계획을 설명했다. '+
                  '이번 조치는 주민 생활의 불편을 줄이기 위해 마련됐다. ')*9
    page=f'''<html><head><meta property="og:title" content="지역 시설 안전 점검 결과와 향후 운영 계획"><meta property="article:published_time" content="{publication}">{extra_meta}</head><body><div id="articletxt"><p>{body}</p><script>무시해야 하는 숨겨진 명령</script><div class="advertisement">광고 추천 메뉴</div></div></body></html>'''.encode()
    candidate={'source':'Hankyung','url':'https://www.hankyung.com/article/2026110812347','allowed_hosts':['www.hankyung.com']}
    receipt={'retrieved_at':'2026-11-09T15:00:00+00:00','sha256':hashlib.sha256(page).hexdigest(),'final_url':candidate['url']}
    return extract_article(page,candidate,receipt,config)


def test_extractor_requires_real_body_not_meta_description(config):
    row,body=article(config,body='짧은 기사 소개만 존재한다.')
    assert row['quality_status']=='quarantine'
    assert 'insufficient_verified_body' in row['exclusion_reasons']
    assert row['body_extractor']=='publisher_verified_selector'


def test_structural_cleanup_preserves_prose_and_removes_scripts(config):
    row,body=article(config)
    assert row['quality_status']=='accepted'
    assert '숨겨진 명령' not in body and '광고 추천' not in body
    assert row['body_sha256']==hashlib.sha256(body.encode()).hexdigest()
    assert row['quoted_words']<=25


def test_lastmod_is_not_a_publication_date(config):
    row,_=article(config,publication='')
    assert row['publication_source']==''
    assert 'publication_unverified' in row['exclusion_reasons']


def test_conflicting_article_dates_are_quarantined(config):
    meta='<script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-11-01T10:00:00+09:00"}</script>'
    row,_=article(config,extra_meta=meta)
    assert 'conflicting_article_publication_dates' in row['exclusion_reasons']


def test_sitemap_timezone_disagreement_is_flagged_without_silent_rewrite(config):
    # Publisher HTML timestamps are authoritative over sitemap time metadata.
    page=('<html><head><meta property="og:title" content="지역 시설 점검 결과 발표">'
          '<meta property="article:published_time" content="2026-11-08T10:00:00+09:00"></head>'
          '<body><div id="articletxt">'+('주민들의 안전을 위한 점검 결과가 발표됐다. 다음 점검은 일정에 따라 진행된다. '*15)+'</div></body></html>').encode()
    candidate={'source':'Hankyung','url':'https://www.hankyung.com/article/2026110812347','allowed_hosts':['www.hankyung.com'],
               'discovery_publication_date':'2026-11-08T10:00:00Z'}
    row,_=extract_article(page,candidate,{'retrieved_at':'2026-11-09T15:00:00+00:00','sha256':'fixture','final_url':candidate['url']},config)
    assert row['published_date']=='2026-11-08T01:00:00+00:00'
    assert 'sitemap_publication_time_disagrees' in row['quality_flags']
    assert row['quality_status']=='accepted'


def test_future_and_outside_window_are_not_backdated(config):
    future,_=article(config,publication='2027-01-01T00:00:00+09:00')
    assert 'future_publication' in future['exclusion_reasons']
    old,_=article(config,publication='2020-01-01T00:00:00+09:00')
    assert 'outside_90_day_window' in old['exclusion_reasons']


def test_day_only_evidence_is_not_fabricated_as_a_timestamp(config):
    row,_=article(config,publication='2026-11-08')
    assert row['published_date']==''
    assert 'publication_time_missing_day_precision' in row['exclusion_reasons']


def test_exact_meta_timestamp_preferred_over_day_only_jsonld(config):
    meta='<script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-11-08"}</script>'
    row,_=article(config,extra_meta=meta)
    assert row['published_date']=='2026-11-08T01:00:00+00:00'
    assert row['date_precision']=='timestamp'


def test_photo_controls_auto_summary_and_related_cards_are_removed(config):
    from news_flywire.corpus.extract import clean_body
    from lxml import html
    node=html.fromstring('<div><div class="showGoogleBtn">구글에서 서울신문 먼저 보기</div><section class="article-summary-box">자동 요약 내용</section><div class="article_photo">사진 설명</div><p>실제 본문의 독립 문장이다.</p><div class="editor-openlink-horizontal">다른 기사 소개</div>◎공감언론 뉴시스 [email protected]</div>')
    body,removed=clean_body(node)
    assert '실제 본문' in body
    for value in ['자동 요약','구글에서','사진 설명','다른 기사','공감언론']:assert value not in body
    assert removed>0


def test_video_teaser_is_not_claimed_as_full_transcript(config):
    body=('이번 주 시장의 변화에 대한 설명이다. 투자 상황을 점검할 필요가 있다. '*8)+'자세한 내용은 동영상으로 확인하실 수 있습니다.'
    row,_=article(config,body=body)
    assert 'video_teaser_incomplete_transcript' in row['exclusion_reasons']


def test_url_identity_retains_article_keys_and_original_scheme():
    assert canonicalize('https://www.edaily.co.kr/News/Read?mediaCodeNo=257&newsId=123&utm_source=rss#x')=='https://www.edaily.co.kr/News/Read?mediaCodeNo=257&newsId=123'
    assert canonicalize('http://publisher.example/a').startswith('http:')
    with pytest.raises(ValueError):canonicalize('https://name:secret@publisher.example/a')


def test_image_locations_and_lastmod_are_not_article_metadata():
    p={'name':'fixture','allowed_hosts':['publisher.example'],'article_pattern':'/story/'}
    payload=b'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"><url><loc>https://publisher.example/story/1</loc><lastmod>2026-11-09</lastmod><image:image><image:loc>https://publisher.example/image.jpg</image:loc><image:title>not headline</image:title></image:image></url></urlset>'''
    children,rows=parse_map(payload,p,'https://publisher.example/sitemap.xml')
    assert not children and len(rows)==1
    assert rows[0]['discovery_title']=='' and rows[0]['discovery_publication_date']==''
    assert rows[0]['discovery_lastmod']=='2026-11-09'


def test_archive_window_is_enforced_before_download(config):
    assert map_in_window('https://p.example/sitemap/20260901.xml',config)
    assert not map_in_window('https://p.example/sitemap/20200101.xml',config)
    assert map_in_window('https://p.example/sitemap_dailyArticle_20260009',config)
    assert not map_in_window('https://p.example/sitemap_dailyArticle_20250009',config)


def test_sampling_is_repeatable_and_uses_multiple_sources_and_days():
    data=pd.DataFrame([{'url':f'https://p.example/{s}/{d}/{i}','source':s,'sampling_date_hint':d,'discovery_in_window_hint':True}
                       for s in ['a','b'] for d in ['2026-09-01','2026-10-01'] for i in range(5)])
    a=balanced_plan(data,8,7);b=balanced_plan(data,8,7)
    pd.testing.assert_frame_equal(a,b)
    assert a.source.nunique()==2 and a.sampling_date_hint.nunique()==2


def make_frame(bodies,dates=None):
    dates=dates or ['2026-11-01T00:00:00+00:00']*len(bodies)
    rows=[];cache={}
    for i,body in enumerate(bodies):
        digest=hashlib.sha256(body.encode()).hexdigest();cache[digest]=body
        rows.append({'news_id':f'n{i}','source':f'p{i}','published_date':dates[i],
                     'title':'지역 시설 안전 점검 결과 발표','summary':'지역 시설의 안전 점검 결과와 향후 계획을 발표했다.',
                     'body_sha256':digest,'body_chars':len(body)})
    return pd.DataFrame(rows),cache


def test_exact_body_copies_are_one_observation(config):
    frame,bodies=make_frame(['같은 사건에 관한 뉴스 본문이다. '*30]*2)
    result,pairs=deduplicate(frame,bodies,config)
    assert result.duplicate_of.ne('').sum()==1
    assert pairs.type.eq('identical_normalized_body').all()


def test_same_headline_with_changed_numbers_is_preserved(config):
    a='시설 점검에서 12명이 대피했다. '+('관계 기관은 향후 계획을 발표했다. '*100)
    b=a.replace('12명','13명')
    frame,bodies=make_frame([a,b])
    result,pairs=deduplicate(frame,bodies,config)
    assert result.duplicate_of.eq('').all()


def test_event_chains_cannot_span_months(config):
    frame,_=make_frame(['본문']*3,['2026-08-15T00:00:00+00:00','2026-09-15T00:00:00+00:00','2026-10-15T00:00:00+00:00'])
    result,pairs=cluster_events(frame,config)
    assert result.event_group.nunique()==3 and pairs.empty


def test_temporal_boundary_events_are_purged_not_split(config):
    frame=pd.DataFrame({'news_id':['a','b','c'],'source':'fixture','event_group':['same','same','other'],
                        'published_date':['2026-10-11T12:00:00+00:00','2026-10-12T12:00:00+00:00','2026-11-01T12:00:00+00:00']})
    result=temporal_split(frame,config)
    assert result.split.tolist()==['purged_boundary_event','purged_boundary_event','test']
    assert result.groupby('event_group').split.nunique().max()==1


def test_later_modified_training_article_is_not_backdated_into_training(config):
    frame=pd.DataFrame({'news_id':['a'],'source':['fixture'],'event_group':['event'],
                        'published_date':['2026-08-20T12:00:00+00:00'],
                        'modified_date':['2026-11-01T12:00:00+00:00']})
    result=temporal_split(frame,config)
    assert result.split.iloc[0]=='purged_version_change'


def test_mixed_http_failure_and_numeric_metrics_round_trip_parquet(config,tmp_path):
    from news_flywire.corpus.build import fetch_and_extract
    from news_flywire.corpus.http import FetchFailure
    from datetime import datetime,timezone
    body=('시설의 안전 점검 결과가 발표됐다. 주민 의견을 검토한 뒤 후속 조치를 시행할 예정이다. '*20)
    payload=('<html><head><meta property="og:title" content="시설 안전 점검 결과 발표">'
             '<meta property="article:published_time" content="2026-11-01T12:00:00+09:00"></head>'
             '<body><div id="articletxt">'+body+'</div></body></html>').encode()
    class FixtureHTTP:
        def fetch(self,url):
            if url.endswith('bad'):raise FetchFailure('http_503',503)
            return payload,{'retrieved_at':'2026-11-09T15:00:00+00:00','final_url':url,'sha256':hashlib.sha256(payload).hexdigest()}
    plan=pd.DataFrame([{'candidate_id':suffix,'source':'Hankyung','url':'https://www.hankyung.com/'+suffix,'discovery_url':'https://www.hankyung.com/map'} for suffix in ['good','bad']])
    out=tmp_path/'out';out.mkdir();raw=tmp_path/'raw';raw.mkdir()
    result=fetch_and_extract(plan,config,FixtureHTTP(),out,raw)
    restored=pd.read_parquet(out/'extraction_ledger.parquet')
    assert len(restored)==2
    assert restored.body_chars.isna().sum()==1
    assert set(restored.quality_status)=={'accepted','quarantine'}


def test_published_corpus_is_not_overwritten_by_a_new_collection(config,tmp_path):
    from news_flywire.corpus.build import build
    out=tmp_path/'frozen';out.mkdir();(out/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='original private cache'):
        build(config,out,tmp_path/'missing_cache')


def test_native_section_mapping_preserves_ambiguity():
    assert harmonize_section('社会の話題')==('unclassified','')
    assert harmonize_section('society')==harmonize_section('사회')
    assert harmonize_section('정치.글로벌')==('mixed','politics|world')
    assert harmonize_section('알 수 없는 매체 분류')[0]=='unclassified'
