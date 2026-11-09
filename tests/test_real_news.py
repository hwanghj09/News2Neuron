"""Explicit fixtures for feed and article provenance boundaries; no human labels."""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from news_flywire.news_ingestion import parse_feed, canonical_url, clean_text, group_events
from news_flywire.news_analysis import validate_real_news, analyze_news


NOW=datetime(2026,11,9,16,tzinfo=timezone.utc)


def feed_item(title="강풍에 해상 운항 주의",date="Mon, 09 Nov 2026 12:00:00 +0900",description=None):
    description=description or '<p>기상 당국은 바람이 강해질 수 있다며 해상 운항에 주의를 당부했다. 추가 원문은 복사하지 않는다.</p>'
    return f'''<rss><channel><item><title>{title}</title><link>https://publisher.example/story?id=1&amp;utm_source=rss</link><description><![CDATA[{description}]]></description><pubDate>{date}</pubDate></item></channel></rss>'''.encode()


def test_feed_records_preserve_dates_source_and_short_scope():
    rows,excluded=parse_feed(feed_item(),"fixture_publisher","society","https://publisher.example/rss",NOW,max_quote_words=12)
    assert not excluded and len(rows)==1
    row=rows[0]
    assert row['published_date']=='2026-11-09T03:00:00+00:00'
    assert row['url']=='https://publisher.example/story?id=1'
    assert row['data_source']=='real_publisher_rss'
    assert row['input_scope'].endswith('not_full_article')
    assert row['quoted_words']<=12
    assert '<p>' not in row['summary']


def test_unknown_or_future_publication_is_not_fabricated():
    rows,excluded=parse_feed(feed_item(date='Wed, 11 Nov 2026 12:00:00 +0900'),'fixture','society','https://publisher.example/rss',NOW)
    assert not rows and excluded['future_publication']==1
    rows,excluded=parse_feed(feed_item(date='unknown'),'fixture','society','https://publisher.example/rss',NOW)
    assert not rows and excluded['invalid_date_or_url']==1


def test_full_encoded_article_does_not_substitute_for_missing_excerpt():
    payload=b'''<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><item><title>fixture title</title><link>https://publisher.example/a</link><pubDate>Mon, 09 Nov 2026 12:00:00 +0900</pubDate><content:encoded>full article must not be used</content:encoded></item></channel></rss>'''
    rows,excluded=parse_feed(payload,'fixture','society','https://publisher.example/rss',NOW)
    assert not rows and excluded['missing_title_description_or_date']==1


def test_html_and_canonical_url_do_not_execute_content():
    assert clean_text('<script>bad instructions</script><p>실제 제목</p>')=='실제 제목'
    assert canonical_url('https://www.mbn.co.kr/pages/news/newsView.php?category=mbn00003&news_seq_no=123&plink=RSSREADER').endswith('?category=mbn00003&news_seq_no=123')
    with pytest.raises(ValueError):canonical_url('javascript:alert(1)')


def test_near_duplicate_updates_share_one_group():
    news=pd.DataFrame({'news_id':['a','b','c'],'title':['해상 강풍으로 항로 운항 중단','해상 강풍으로 항로 운항 중단','신작 공연 일정 공개'],
                       'summary':['기상 악화로 운항에 차질','기상 악화로 운항에 차질','공연이 다음 주 열린다']})
    grouped,pairs=group_events(news)
    assert grouped.event_group[0]==grouped.event_group[1]
    assert grouped.event_group[2]!=grouped.event_group[0]
    assert len(pairs)==1


def make_news():
    row=parse_feed(feed_item(),'fixture','society','https://publisher.example/rss',NOW)[0][0]
    row['event_group']='event_fixture'
    return pd.DataFrame([row])


def test_real_analysis_rejects_synthetic_mixture_and_bad_dates():
    news=make_news()
    validate_real_news(news)
    news.loc[0,'data_source']='synthetic_fixture'
    with pytest.raises(ValueError,match='synthetic'):validate_real_news(news)
    news=make_news()
    news.loc[0,'published_date']='2027-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='Publication'):validate_real_news(news)


def test_real_article_linkage_uses_existing_level_and_preserves_rationale(tmp_path):
    news=make_news()
    news_path=tmp_path/'news.csv'
    news.to_csv(news_path,index=False)
    scores=news.copy()
    scores[['N','T','A']]=[.5,.75,.5]
    scores['scorer']='fixture_ai'
    scores['score_source']='llm_prediction'
    scores['rationale']='구체적인 강풍 경고이며 자료는 테스트 fixture다'
    scores_path=tmp_path/'scores.csv'
    scores.to_csv(scores_path,index=False)
    run=tmp_path/'simulation'
    run.mkdir()
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/stimulation_v1.yaml').read_text())
    (run/'resolved_config.json').write_text(json.dumps(c))
    (run/'manifest.json').write_text('{}')
    pd.DataFrame([dict(condition_id=condition,seed=seed,delta_rate_hz=value) for condition,value in [('E1',10),('E2',20),('E3',30)] for seed in [1,2]]).to_csv(run/'metrics.csv',index=False)
    out=tmp_path/'report'
    analyze_news(news_path,scores_path,out,run)
    linked=pd.read_csv(out/'article_responses.csv')
    assert len(linked)==2 and linked.condition_id.eq('E3').all()
    assert linked.delta_rate_hz.eq(30).all() and linked.shared_level_simulation.all()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['new_simulation_trials']==0 and not summary['human_accuracy_evaluated']
    article=pd.read_csv(out/'article_scores.csv')
    assert article.rationale.iloc[0]==scores.rationale.iloc[0]
    assert 'href="https://publisher.example/story?id=1"' in (out/'report.html').read_text()
    with pytest.raises(ValueError,match='already exists'):analyze_news(news_path,scores_path,out,run)


def test_analysis_rejects_ratings_for_modified_text(tmp_path):
    news=make_news()
    news_path=tmp_path/'news.csv'
    news.to_csv(news_path,index=False)
    scores=news.copy()
    scores['summary']='평가 이후 임의로 바뀐 발췌문'
    scores_path=tmp_path/'scores.csv'
    scores.to_csv(scores_path,index=False)
    with pytest.raises(ValueError,match='metadata: summary'):
        analyze_news(news_path,scores_path,tmp_path/'report',tmp_path/'unused')


def test_default_score_command_uses_real_news(tmp_path,monkeypatch):
    from news_flywire.cli import main
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    output=tmp_path/'default_scores.csv'
    main(['score','--out',str(output)])
    scored=pd.read_csv(output)
    assert len(scored)==240
    assert scored.data_source.eq('real_publisher_rss').all()
    assert scored.url.str.startswith('https://').all()
    assert not scored.news_id.str.startswith('synthetic_').any()
