"""Collect short, dated records from publishers' public RSS feeds."""
from __future__ import annotations

import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .provenance import manifest, sha256, write_json


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    parser = _Text()
    parser.feed(html.unescape(value or ""))
    text = " ".join(parser.parts)
    text = re.sub(r"【\s*(?:앵커멘트|기자)\s*】", " ", text)
    text = re.sub(r"^\([^)]{1,45}=\s*[^)]{1,20}\)\s*[^=]{0,45}=\s*", "", text)
    return " ".join(text.split())


def canonical_url(value):
    p = urlsplit(value.strip())
    if p.scheme not in {"http", "https"} or not p.netloc:
        raise ValueError("RSS item does not contain an absolute HTTP(S) article URL")
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_") and k.lower() not in {"plink", "ref", "from"}]
    return urlunsplit(("https", p.netloc.lower(), p.path, urlencode(query), ""))


def parse_feed(payload, source, category, feed_url, collected_at, max_quote_words=25):
    root = ET.fromstring(payload)
    rows, excluded = [], Counter()
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title"))
        description = clean_text(item.findtext("description"))
        # Never copy content:encoded: some publishers put full articles there.
        date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
        if not title or not description or not date:
            excluded["missing_title_description_or_date"] += 1
            continue
        try:
            try:
                published = parsedate_to_datetime(date)
            except (TypeError, ValueError):
                published = datetime.fromisoformat(date.replace("Z", "+00:00"))
            if published.tzinfo is None:
                raise ValueError("Timezone missing")
            published = published.astimezone(timezone.utc)
            url = canonical_url(item.findtext("link") or "")
        except (TypeError, ValueError):
            excluded["invalid_date_or_url"] += 1
            continue
        if published > collected_at:
            excluded["future_publication"] += 1
            continue
        budget = max_quote_words - len(title.split())
        if budget < 4:
            excluded["headline_exceeds_excerpt_budget"] += 1
            continue
        summary = " ".join(description.split()[:budget])
        # Omit graphic/sexual/self-harm content in this low-intensity pilot.
        if any(term in title + " " + summary for term in ["성폭행", "성추행", "강간", "성관계", "자해", "극단적 선택", "투신", "시신 훼손"]):
            excluded["pilot_content_exclusion"] += 1
            continue
        if len(summary) < 12:
            excluded["insufficient_excerpt"] += 1
            continue
        key = hashlib.sha256(url.encode()).hexdigest()[:14]
        rows.append({"news_id": f"real_{key}", "event_group": "", "title": title,
            "summary": summary, "category": category, "data_source": "real_publisher_rss",
            "source": source, "url": url, "published_date": published.isoformat(),
            "collected_at": collected_at.isoformat(), "feed_url": feed_url,
            "publisher_category": clean_text(item.findtext("category")) or category,
            "input_scope": "publisher_headline_and_short_rss_excerpt_not_full_article",
            "summary_source": "publisher_rss_excerpt", "quoted_words":len(title.split())+len(summary.split())})
    return rows, dict(excluded)


def group_events(news, threshold=0.52):
    """Conservative heuristic; pair table is exported for manual audit."""
    out = news.copy().reset_index(drop=True)
    normalized = out.title.map(lambda s: re.sub(r"\W+", "", s.casefold()))
    text = (out.title + " " + out.summary).tolist()
    x = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1).fit_transform(text)
    similarities = cosine_similarity(x)
    parent = list(range(len(out)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pairs = []
    for i in range(len(out)):
        for j in range(i+1,len(out)):
            if normalized[i] == normalized[j] or similarities[i,j] >= threshold:
                parent[find(j)] = find(i)
                pairs.append({"news_id_a":out.news_id[i],"news_id_b":out.news_id[j],
                              "similarity":float(similarities[i,j]),"status":"automatic_candidate"})
    groups = defaultdict(list)
    for i in range(len(out)):
        groups[find(i)].append(out.news_id[i])
    labels = {root: "event_"+hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:12] for root,ids in groups.items()}
    out["event_group"] = [labels[find(i)] for i in range(len(out))]
    return out,pd.DataFrame(pairs,columns=["news_id_a","news_id_b","similarity","status"])


def collect_news(config, out_dir, raw_dir="data/raw/news_rss"):
    out, raw = Path(out_dir), Path(raw_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Collection output exists; choose a fresh output directory")
    out.mkdir(parents=True,exist_ok=True)
    raw.mkdir(parents=True,exist_ok=True)
    now = datetime.now(timezone.utc)
    all_rows, logs, raw_files = [], [], []
    for feed in config["feeds"]:
        try:
            request = urllib.request.Request(feed["url"],headers={"User-Agent":"News2Neuron/0.1 (noncommercial public RSS research)"})
            with urllib.request.urlopen(request,timeout=25) as response:
                payload = response.read(4_000_001)
            if len(payload)>4_000_000:
                raise ValueError("Feed exceeds 4 MB collection limit")
            name = hashlib.sha256(feed["url"].encode()).hexdigest()[:16]+"_"+hashlib.sha256(payload).hexdigest()[:16]+".xml"
            path = raw/name
            path.write_bytes(payload)
            rows,excluded = parse_feed(payload,feed["source"],feed["category"],feed["url"],now,config["max_quote_words"])
            all_rows.extend(rows)
            raw_files.append(path)
            logs.append({**feed,"status":"ok","items_retained":len(rows),"excluded":excluded,"sha256":sha256(path)})
        except Exception as error:
            logs.append({**feed,"status":"failed","error_type":type(error).__name__,"error":str(error)[:200]})
        print(f"{feed['source']} / {feed['category']}: {logs[-1]['status']}",flush=True)
    write_json(out/"collection_log.json",logs)
    if not all_rows:
        raise ValueError("No dated publisher news obtained; no synthetic fallback is permitted")
    candidates=pd.DataFrame(all_rows).drop_duplicates("url").drop_duplicates("title").sort_values(["published_date","news_id"],ascending=[False,True])
    selected=[]
    shortages={}
    for category in config["categories"]:
        part=candidates[candidates.category.eq(category)]
        queues={source:frame.to_dict("records") for source,frame in part.groupby("source",sort=True)}
        count=0
        while count<config["per_category"] and any(queues.values()):
            for source in sorted(queues):
                if queues[source] and count<config["per_category"]:
                    selected.append(queues[source].pop(0))
                    count+=1
        if count<config["per_category"]:
            shortages[category]=count
    if shortages:
        write_json(out/"shortages.json",shortages)
        raise ValueError(f"Real-news category quotas not met: {shortages}; no fabricated rows added")
    news,pairs=group_events(pd.DataFrame(selected),config["event_similarity_threshold"])
    news.to_csv(out/"news.csv",index=False)
    pairs.to_csv(out/"similar_event_pairs.csv",index=False)
    write_json(out/"manifest.json",manifest(raw_files,collection_utc=now.isoformat(),
        dataset_type="real_news_convenience_pilot",n_articles=len(news),n_events=int(news.event_group.nunique()),
        categories=news.category.value_counts().to_dict(),sources=news.source.value_counts().to_dict(),
        published_min=news.published_date.min(),published_max=news.published_date.max(),
        max_quoted_words_per_article=config["max_quote_words"],config=config,
        news_csv_sha256=sha256(out/"news.csv"),limitations=[
            "Convenience sample of available RSS feeds, not representative of Korean news",
            "Publisher sections are not negativity/neutrality labels; health/lifestyle sections may include culture",
            "Title plus short publisher excerpt only; complete articles not evaluated",
            "Automatic event grouping requires manual audit before a confirmatory study",
            "No independent human ground-truth ratings or measured physiological outcomes"]))
    return news
