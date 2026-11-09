from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from .extract import canonicalize, extract_article, parse_date
from .http import CachedHTTP, FetchFailure
from ..provenance import write_json


def parse_map(payload,publisher,map_url):
    if payload.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as f:
            payload=f.read(60_000_001)
        if len(payload)>60_000_000:raise ValueError("decompressed_sitemap_size_limit")
    tree=ET.fromstring(payload)
    children=[];rows=[]
    if tree.tag.split("}")[-1]=="sitemapindex":
        return [e.findtext("{*}loc") for e in tree if e.findtext("{*}loc")],[]
    if tree.tag.split("}")[-1]=="rss":
        entries=tree.findall(".//item")
        for e in entries:
            rows.append({"url":e.findtext("link"),"discovery_title":e.findtext("title") or "",
                         "discovery_publication_date":e.findtext("pubDate") or "","discovery_lastmod":""})
    else:
        for e in tree:
            url=e.findtext("{*}loc")
            if not url:continue
            # Do not confuse image:loc, video:loc or sitemap lastmod with article metadata.
            news=next((x for x in e if x.tag.split("}")[-1]=="news"),None)
            title=news.findtext("{*}title") if news is not None else ""
            date=news.findtext("{*}publication_date") if news is not None else ""
            rows.append({"url":url,"discovery_title":title or "","discovery_publication_date":date or "",
                         "discovery_lastmod":e.findtext("{*}lastmod") or ""})
    valid=[]
    for row in rows:
        if urlsplit(row["url"]).hostname not in publisher["allowed_hosts"]:continue
        if not re.search(publisher["article_pattern"],row["url"],re.I):continue
        row.update(source=publisher["name"],discovery_url=map_url)
        valid.append(row)
    return children,valid


def map_in_window(url,config):
    start=datetime.fromisoformat(config["start_date"])-timedelta(days=1)
    end=datetime.fromisoformat(config["end_date"])+timedelta(days=1)
    daily=re.search(r"(20\d{2}[01]\d[0-3]\d)",url)
    if daily:
        try:return start<=datetime.strptime(daily[1],"%Y%m%d")<=end
        except ValueError:pass
    # Seoul's archive encodes months as YYYY00MM.
    month=re.search(r"(20\d{2})00(0[1-9]|1[0-2])",url)
    if not month:month=re.search(r"(20\d{2})-?(0[1-9]|1[0-2])",url)
    if month:
        ym=month[1]+month[2]
        return start.strftime("%Y%m")<=ym<=end.strftime("%Y%m")
    return True


def discover(config,http,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    found=[];logs=[]
    def publisher_job(publisher):
        queue=[(url,0) for url in publisher["roots"]];visited=set();local=[];activity=[]
        while queue:
            url,depth=queue.pop(0)
            if url in visited or depth>3 or not map_in_window(url,config):continue
            visited.add(url)
            try:
                payload,receipt=http.fetch(url,xml=True)
                children,rows=parse_map(payload,publisher,url)
                local.extend(rows)
                queue.extend((u,depth+1) for u in children if map_in_window(u,config))
                activity.append({"source":publisher["name"],"url":url,"status":"parsed","article_candidates":len(rows),
                                 "children":len(children),"response_sha256":receipt["sha256"]})
                if len(visited)%15==0:print(f"Discovery {publisher['name']}: {len(visited)} maps, {len(local)} candidate rows",flush=True)
            except (FetchFailure,ValueError,ET.ParseError) as e:
                activity.append({"source":publisher["name"],"url":url,"status":"failed","reason":str(e)})
        print(f"Discovery {publisher['name']} complete: {len(local)} rows",flush=True)
        return local,activity
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for task in as_completed([pool.submit(publisher_job,p) for p in config["publishers"]]):
            rows,activity=task.result();found.extend(rows);logs.extend(activity)
    write_json(out/"discovery_log.json",logs)
    if not found:raise ValueError("No publisher article candidates obtained")
    frame=pd.DataFrame(found)
    frame["url"]=frame.url.map(canonicalize)
    frame=frame.sort_values(["source","url","discovery_url"]).drop_duplicates("url").reset_index(drop=True)
    frame["candidate_id"]=frame.url.map(lambda u:"candidate_"+hashlib.sha256(u.encode()).hexdigest()[:20])
    # URL dates are used only for discovery sampling, never as verified publication dates.
    def hint(row):
        date,_,_=parse_date(row.discovery_publication_date)
        if date is not None:return date.date().isoformat()
        m=re.search(r"(20\d{2})/?([01]\d)/?([0-3]\d)",urlsplit(row.url).path)
        if m:
            try:return datetime(*map(int,m.groups())).date().isoformat()
            except ValueError:pass
        return "unknown"
    frame["sampling_date_hint"]=[hint(r) for r in frame.itertuples()]
    start=(datetime.fromisoformat(config["start_date"])-timedelta(days=2)).date().isoformat()
    end=(datetime.fromisoformat(config["end_date"])+timedelta(days=2)).date().isoformat()
    frame["discovery_in_window_hint"]=frame.sampling_date_hint.eq("unknown") | frame.sampling_date_hint.between(start,end)
    frame.to_parquet(out/"discovered_candidates.parquet",index=False)
    return frame


def balanced_plan(candidates,limit,seed):
    """Round-robin across publishers and days, deterministic within a stratum."""
    eligible=candidates[candidates.discovery_in_window_hint].copy()
    eligible["_rank"]=eligible.url.map(lambda u:hashlib.sha256(f"{seed}:{u}".encode()).hexdigest())
    queues={}
    for source,part in eligible.groupby("source",sort=True):
        days={day:group.sort_values("_rank").drop(columns="_rank").to_dict("records") for day,group in part.groupby("sampling_date_hint",sort=True)}
        ordered=[]
        while any(days.values()):
            for day in sorted(days):
                if days[day]:ordered.append(days[day].pop(0))
        queues[source]=ordered
    plan=[]
    while len(plan)<limit and any(queues.values()):
        for source in sorted(queues):
            if queues[source] and len(plan)<limit:plan.append(queues[source].pop(0))
    return pd.DataFrame(plan)


def fetch_and_extract(plan,config,http,out,raw):
    out,raw=Path(out),Path(raw)
    state=raw/"extraction_results.jsonl"
    existing={}
    if state.exists():
        for line in state.read_text().splitlines():
            item=json.loads(line);existing[item["candidate_id"]]=item
    publishers={p["name"]:p for p in config["publishers"]}
    todo=[row for row in plan.to_dict("records") if row["candidate_id"] not in existing]
    counts=Counter(item.get("quality_status") for item in existing.values())
    def job(candidate):
        candidate={**candidate,"allowed_hosts":publishers[candidate["source"]]["allowed_hosts"]}
        try:
            payload,receipt=http.fetch(candidate["url"])
            row,body=extract_article(payload,candidate,receipt,config)
            row.update(candidate_id=candidate["candidate_id"],discovery_url=candidate["discovery_url"])
            body_path=raw/"bodies"/(row["body_sha256"]+".txt.gz")
            body_path.parent.mkdir(parents=True,exist_ok=True)
            body_path.write_bytes(gzip.compress(body.encode(),mtime=0))
            return row
        except Exception as error:
            return {"candidate_id":candidate["candidate_id"],"source":candidate["source"],"requested_url":candidate["url"],
                    "url":candidate["url"],"quality_status":"quarantine","exclusion_reasons":str(error) if isinstance(error,FetchFailure) else type(error).__name__}
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool, state.open("a",encoding="utf-8") as f:
        tasks=[pool.submit(job,row) for row in todo]
        for task in as_completed(tasks):
            row=task.result();existing[row["candidate_id"]]=row
            f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
            counts[row["quality_status"]]+=1
            if sum(counts.values())%100==0:
                print(f"Fetched {sum(counts.values())}/{len(plan)}; accepted {counts['accepted']}; quarantine {counts['quarantine']}",flush=True)
                write_json(out/"progress.json",dict(counts))
    rows=[existing[row.candidate_id] for row in plan.itertuples()]
    frame=pd.DataFrame(rows).fillna("")
    for column in ["body_chars","korean_letter_ratio","sentence_count","boilerplate_chars_removed","quoted_words"]:
        if column in frame:frame[column]=pd.to_numeric(frame[column],errors="coerce")
    frame.to_parquet(out/"extraction_ledger.parquet",index=False)
    return frame


def build(config,out_dir,raw_dir):
    out,raw=Path(out_dir),Path(raw_dir)
    out.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists() and not (raw/"extraction_results.jsonl").exists():
        raise ValueError("Published derived corpus exists without its original private cache; choose fresh --out and --raw directories")
    frozen=out/"resolved_config.json"
    if frozen.exists() and json.loads(frozen.read_text())!=config:
        previous=json.loads(frozen.read_text())
        changed={key for key in set(previous)|set(config) if previous.get(key)!=config.get(key)}
        if changed!={"fetch_target"} or config["fetch_target"]<=previous["fetch_target"] or config["fetch_target"]>config["candidate_target"]:
            raise ValueError("Config differs from this run's frozen settings; use a new run directory")
        history=out/"acquisition_budget_revisions.json"
        revisions=json.loads(history.read_text()) if history.exists() else []
        revisions.append({"previous_fetch_target":previous["fetch_target"],"new_fetch_target":config["fetch_target"],
                          "reason":"Increase acquisition only; source/window/quality/scoring thresholds unchanged", "at_utc":datetime.now(timezone.utc).isoformat()})
        write_json(history,revisions)
    write_json(frozen,config)
    http=CachedHTTP(config,raw/"http")
    discovery=out/"discovered_candidates.parquet"
    candidates=pd.read_parquet(discovery) if discovery.exists() else discover(config,http,out)
    plan=balanced_plan(candidates,min(config["candidate_target"],len(candidates)),config["seed"])
    plan.to_parquet(out/"candidate_sample.parquet",index=False)
    fetching=balanced_plan(plan,min(config["fetch_target"],len(plan)),config["seed"]+1)
    fetching.to_parquet(out/"fetch_plan.parquet",index=False)
    frame=fetch_and_extract(fetching,config,http,out,raw)
    print(f"Extraction complete: {len(frame)} pages, {frame.quality_status.eq('accepted').sum()} preliminary accepted",flush=True)
    return frame
