from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from lxml import html

KST = timezone(timedelta(hours=9))
SELECTORS = {
    "Hankyung": "//*[@id='articletxt']",
    "Edaily": "//div[contains(concat(' ',normalize-space(@class),' '),' news_body ')]",
    "Fnnews": "//div[contains(concat(' ',normalize-space(@class),' '),' article-view__body ')] | //*[@id='article_content']",
    "Seoul": "//*[@id='articleContent']//*[contains(concat(' ',normalize-space(@class),' '),' viewContent ')] | //*[@id='atic_txt1']",
    "Newsis": "//div[contains(concat(' ',normalize-space(@class),' '),' viewer ')]",
    "SBS": "//div[contains(concat(' ',normalize-space(@class),' '),' text_area ')]",
    "Khan": "//*[@id='articleBody']",
}


def normalize(value):
    return re.sub(r"[ \t\r\f\v]+"," ",unicodedata.normalize("NFKC",value).replace("\u200b","").replace("\ufeff","")).strip()


def canonicalize(url):
    p=urlsplit(url)
    if p.scheme not in {"http","https"} or not p.hostname or p.username or p.password:
        raise ValueError("invalid_article_url")
    tracking={"plink","ref","from","fbclid","gclid","ncid","nclick","utm_source","utm_medium","utm_campaign","utm_content","utm_term"}
    query=sorted((k,v) for k,v in parse_qsl(p.query,keep_blank_values=True) if k.lower() not in tracking and not k.lower().startswith("utm_"))
    return urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path,urlencode(query),""))


def parse_date(value):
    if not value:
        return None,"missing",False
    raw=normalize(str(value))
    match=re.match(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})(?:[. ]*([0-2]?\d):([0-5]\d)(?::([0-5]\d))?)?",raw)
    try:
        dt=datetime.fromisoformat(raw.replace("Z","+00:00"))
    except ValueError:
        if not match:
            return None,"invalid",False
        try:
            groups=match.groups()
            dt=datetime(*[int(x) for x in groups[:3]],*[int(x or 0) for x in groups[3:]])
        except ValueError:
            return None,"invalid",False
    precision="timestamp" if re.search(r"\d:\d",raw) else "day"
    inferred=dt.tzinfo is None
    if inferred:
        dt=dt.replace(tzinfo=KST)
    return dt.astimezone(timezone.utc),precision,inferred


def _article_json(tree):
    result=[]
    def walk(value):
        if isinstance(value,list):
            for row in value:walk(row)
        elif isinstance(value,dict):
            types=value.get("@type",[])
            if isinstance(types,str):types=[types]
            if any(t in {"NewsArticle","Article","ReportageNewsArticle","AnalysisNewsArticle"} for t in types):result.append(value)
            if "@graph" in value:walk(value["@graph"])
    for text in tree.xpath('//script[@type="application/ld+json"]/text()'):
        try:walk(json.loads(text))
        except (ValueError,TypeError):pass
    return result


def clean_body(element):
    node=copy.deepcopy(element)
    removed=0
    for bad in node.xpath('.//script|.//style|.//nav|.//aside|.//form|.//button|.//iframe|.//figure|.//figcaption|.//table|.//noscript'):
        removed+=len(bad.text_content());bad.drop_tree()
    for bad in node.xpath('.//*[@class or @id]'):
        marker=(bad.get("class","")+" "+bad.get("id","")).lower()
        blocked=re.search(r"(?:^|[ _-])(?:ad|ads|advertisement|related|recommend|share|social|copyright|byline)(?:$|[ _-])",marker)
        blocked=blocked or any(term in marker for term in ["article_photo","art_photo","v_photoarea","editor-openlink","article-summary-box","showgooglebtn","hidetext"])
        if blocked and any(ancestor is node for ancestor in bad.iterancestors()):
            removed+=len(bad.text_content());bad.drop_tree()
    for br in node.xpath('.//br'):
        br.tail="\n"+(br.tail or "")
    for p in node.xpath('.//p'):
        p.tail="\n"+(p.tail or "")
    lines=[]
    for raw in node.text_content().splitlines():
        line=normalize(raw)
        if not line:continue
        boilerplate=re.match(r"^(?:ⓒ|©|Copyright|[☞◎]공감언론|무단\s*(?:전재|복제).*금지|기사\s*제보|구독\s*하기|관련\s*기사|지금\s*구독)",line,re.I)
        boilerplate=boilerplate or line in {"구글에서 서울신문 먼저 보기","세줄 요약","이미지 확대","이미지 확대 보기","닫기"}
        boilerplate=boilerplate or re.match(r"^(?:[가-힣]{2,5}\s*기자\s*)?[\w.+-]+@[\w.-]+\.[a-z]{2,}$",line,re.I)
        boilerplate=boilerplate or re.match(r"^(?:\[email protected\]\s*)?[가-힣]{2,5}\s*기자$",line)
        boilerplate=boilerplate or re.match(r"^#(?:[가-힣\w]{1,30})$|^\d{4}-\d{2}-\d{2}\s+[A-Z]\d{1,3}면$|^\(사진=.*\)$",line)
        boilerplate=boilerplate or (line.startswith("▲") and any(t in line for t in ["사진","모습","포스터","이미지"]))
        if boilerplate:
            removed+=len(line)
            continue
        lines.append(line)
    return "\n".join(lines),removed


def extract_article(payload,candidate,receipt,config):
    text=None
    for encoding in ["utf-8","cp949"]:
        try:text=payload.decode(encoding);break
        except UnicodeDecodeError:pass
    if text is None:text=payload.decode("utf-8",errors="replace")
    tree=html.fromstring(text)
    meta={}
    for node in tree.xpath('//meta[@content]'):
        key=node.get("property") or node.get("name") or node.get("itemprop")
        if key:meta.setdefault(key,node.get("content",""))
    data=_article_json(tree)
    title=normalize(meta.get("og:title") or (data[0].get("headline","") if data else "") or " ".join(tree.xpath('//h1[1]//text()')))
    body="";extractor="none";removed=0;next_data={}
    if candidate["source"]=="News1":
        scripts=tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
        try:
            next_data=json.loads(scripts[0])["props"]["pageProps"]["articleView"]
            fragments=[]
            for part in next_data.get("contentArrange",[]):
                if part.get("type")=="text":
                    fragments.append(clean_body(html.fragment_fromstring(part.get("content",""),create_parent="div"))[0])
            body="\n".join(fragments)
            extractor="publisher_structured_next_data"
        except (ValueError,KeyError,IndexError,TypeError):pass
    if not body and candidate["source"] in SELECTORS:
        nodes=tree.xpath(SELECTORS[candidate["source"]])
        if nodes:
            body,removed=clean_body(max(nodes,key=lambda e:len(e.text_content())))
            extractor="publisher_verified_selector"
    if not body:
        for item in data:
            if isinstance(item.get("articleBody"),str):
                body,removed=clean_body(html.fragment_fromstring(item["articleBody"],create_parent="div"))
                extractor="structured_articleBody"
                break
    canonical=receipt.get("final_url",candidate["url"])
    links=tree.xpath('//link[contains(translate(@rel,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"canonical")]/@href')
    if links:
        proposed=links[0]
        if proposed.startswith("https://") and urlsplit(proposed).hostname in candidate["allowed_hosts"]:
            canonical=proposed
    canonical=canonicalize(canonical)
    evidence=[]
    for item in data:
        if item.get("datePublished"):evidence.append(("jsonld_datePublished",item["datePublished"]))
    for key in ["article:published_time","datePublished","date","pubdate"]:
        if meta.get(key):evidence.append(("meta_"+key,meta[key]))
    if next_data.get("published_time"):evidence.append(("publisher_next_data",next_data["published_time"]))
    if candidate.get("discovery_publication_date"):evidence.append(("publisher_news_sitemap",candidate["discovery_publication_date"]))
    parsed=[(source,raw,*parse_date(raw)) for source,raw in evidence]
    parsed=[row for row in parsed if row[2] is not None]
    article_evidence=[r for r in parsed if r[0]!="publisher_news_sitemap"]
    exact_article_evidence=[r for r in article_evidence if r[3]=="timestamp"]
    exact_map_evidence=[r for r in parsed if r[0]=="publisher_news_sitemap" and r[3]=="timestamp"]
    chosen=next(iter(exact_article_evidence or exact_map_evidence or article_evidence or parsed),None)
    flags=[];reasons=[]
    if chosen:
        published=chosen[2]
        if chosen[3]=="day":reasons.append("publication_time_missing_day_precision")
        if chosen[4]:flags.append("publication_timezone_inferred_KST")
        if published>datetime.fromisoformat(receipt["retrieved_at"])+timedelta(minutes=5):reasons.append("future_publication")
        if not config["start_date"]<=published.date().isoformat()<=config["end_date"]:reasons.append("outside_90_day_window")
        for source,raw,date,precision,inferred in parsed:
            if source==chosen[0] and raw==chosen[1]:continue
            delta=abs((date-published).total_seconds())
            if source=="publisher_news_sitemap" and delta>300:
                flags.append("sitemap_publication_time_disagrees")
            elif source!="publisher_news_sitemap" and precision=="timestamp" and chosen[3]=="timestamp" and delta>3600:
                reasons.append("conflicting_article_publication_dates")
    else:reasons.append("publication_unverified")
    modified_evidence=[]
    for item in data:
        if item.get("dateModified"):modified_evidence.append(("jsonld_dateModified",item["dateModified"]))
    for key in ["article:modified_time","dateModified"]:
        if meta.get(key):modified_evidence.append(("meta_"+key,meta[key]))
    if next_data.get("modified_time"):modified_evidence.append(("publisher_next_data",next_data["modified_time"]))
    modified=None;modified_source=""
    for source,value in modified_evidence:
        dt,precision,inferred=parse_date(value)
        if dt is not None and precision=="timestamp":modified,modified_source=dt,source;break
    if modified is None:flags.append("modification_time_unavailable")
    else:
        if modified>datetime.fromisoformat(receipt["retrieved_at"])+timedelta(minutes=5):reasons.append("future_modification_time")
        if chosen and modified<chosen[2]-timedelta(hours=1):flags.append("modification_precedes_publication")
    letters=re.findall(r"[A-Za-z가-힣]",body)
    korean=sum("가"<=c<="힣" for c in letters)/max(1,len(letters))
    sentences=len(re.findall(r"(?:다[.!?]|요[.!?]|[.!?]\s)",body))
    if not title or len(title)<8:reasons.append("invalid_headline")
    if len(title.split())>config["max_excerpt_words_including_title"]:reasons.append("headline_exceeds_public_excerpt_budget")
    if len(body)<config["min_body_chars"]:reasons.append("insufficient_verified_body")
    if len(body)>config["max_body_chars"]:reasons.append("abnormally_long_body")
    if korean<config["min_korean_letter_ratio"]:reasons.append("non_korean_or_encoding_failure")
    if sentences<config["min_sentences"]:reasons.append("insufficient_prose_sentences")
    if any(term in title for term in ["중간순위","[전적]","클로징","이 시각 헤드라인","주요뉴스","오늘의 운세","별자리 운세"]):reasons.append("table_roundup_or_nonarticle")
    if re.search(r"(?:회원|구독자)\s*전용|로그인\s*후\s*(?:이용|확인)|구독하시면.*전문",body[:500]):reasons.append("access_limited_body")
    if len(body)<1500 and re.search(r"자세한\s*내용은\s*(?:동영상|영상)|내용은\s*영상으로",body):reasons.append("video_teaser_incomplete_transcript")
    if re.search(r"브랜드뉴스|협찬|원고료를|유료광고|ADVERTORIAL",title+" "+body[-500:],re.I):flags.append("possible_sponsored_content")
    if any(word in title for word in ["사설","칼럼","기고","시론","세상읽기"]):flags.append("opinion_genre")
    discovered=candidate.get("discovery_title","")
    if discovered and SequenceMatcher(None,normalize(discovered),title).ratio()<.55:flags.append("headline_changed_since_discovery")
    url_dates=re.findall(r"(20\d{2})/?(\d{2})/?(\d{2})",urlsplit(candidate["url"]).path)
    if chosen and url_dates:
        try:
            url_date=datetime(*map(int,url_dates[0])).date()
            if abs((url_date-chosen[2].astimezone(KST).date()).days)>2:flags.append("url_date_differs_from_publication")
        except ValueError:pass
    budget=max(0,config["max_excerpt_words_including_title"]-len(title.split()))
    excerpt=" ".join(body.split()[:budget])
    body_hash=hashlib.sha256(body.encode()).hexdigest()
    section=meta.get("article:section") or (data[0].get("articleSection","") if data else "") or next_data.get("upper_category_nm","")
    if isinstance(section,list):section=" / ".join(map(str,section))
    row={"news_id":"corpus_"+hashlib.sha256(canonical.encode()).hexdigest()[:20],"source":candidate["source"],
        "url":canonical,"requested_url":candidate["url"],"title":title,"summary":excerpt,
        "published_date":chosen[2].isoformat() if chosen and chosen[3]=="timestamp" else "","publication_source":chosen[0] if chosen else "",
        "publication_raw":str(chosen[1]) if chosen else "","date_precision":chosen[3] if chosen else "unknown",
        "date_evidence_json":json.dumps([{ "source":s,"raw":str(raw),"parsed_utc":d.isoformat() if p=="timestamp" else None,"precision":p,"timezone_inferred":i} for s,raw,d,p,i in parsed],ensure_ascii=False),
        "collected_at":receipt["retrieved_at"],"html_sha256":receipt["sha256"],"body_sha256":body_hash,
        "modified_date":modified.isoformat() if modified else "","modification_source":modified_source,
        "version_observed_at":receipt["retrieved_at"],
        "body_chars":len(body),"korean_letter_ratio":round(korean,4),"sentence_count":sentences,
        "boilerplate_chars_removed":removed,"body_extractor":extractor,"category":str(section or "unclassified"),
        "quality_flags":"|".join(sorted(set(flags))),"exclusion_reasons":"|".join(sorted(set(reasons))),
        "quality_status":"accepted" if not reasons else "quarantine","data_source":"real_publisher_article_verified",
        "body_scope":"published_prose_without_media_tables_related_cards_or_site_controls",
        "input_scope":"full_body_verified_locally_public_release_short_excerpt","quoted_words":len(title.split())+len(excerpt.split()),
        "wire_mention_at_lead":"|".join(w for w in ["연합뉴스","뉴스1","뉴시스"] if w in body[:250]),
        "wire_mention_status":"string_mention_only_not_verified_original_publisher"}
    return row,body
