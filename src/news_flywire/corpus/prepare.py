from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .extract import canonicalize
from ..news_scoring import rule_score
from ..provenance import manifest, sha256, write_json


def normalized_title(title):
    title=re.sub(r"\((?:종합|종합\d*보|\d보|상보|종합2보)\)","",title)
    title=re.sub(r"\[(?:속보|종합|단독|영상|포토|사진)\]","",title)
    return re.sub(r"[^가-힣a-z0-9]","",title.casefold())


def normalized_body(text):
    text=re.sub(r"^[\[(][^\])]{0,80}(?:뉴스1|뉴시스|연합뉴스)[^\])]{0,40}[\])]", "",text)
    return re.sub(r"[^가-힣a-z0-9]","",text.casefold())


def harmonize_section(value):
    """Metadata-only topic tags; preserve ambiguous/multiple native sections."""
    tokens=set(re.split(r"[\s./ㆍ·:>]+",str(value).casefold()))
    vocab={
        "politics":{"politics","정치","국방","외교","북한","정치외교","안보","국방외교"},
        "economy":{"economy","business","finance","경제","산업","기업","증권","금융","부동산","재테크","유통"},
        "society":{"society","사회","교육","사건사고","법조","노동","환경"},
        "world":{"world","international","global","국제","세계","글로벌","해외"},
        "culture":{"culture","entertainment","문화","연예","방송","예술","생활","라이프","패션"},
        "sports":{"sports","sport","스포츠","야구","축구","농구","골프"},
        "health":{"health","medical","건강","의료","의학","보건","바이오"},
        "technology":{"technology","tech","it","science","과학","기술","테크","정보통신"},
        "regional":{"regional","local","지방","지역","전국","수도권","서울","경기","인천","부산","대구","대전","울산","강원","충청","전라","경상","제주"},
        "opinion":{"opinion","오피니언","사설","칼럼","시론"}}
    tags=sorted(topic for topic,words in vocab.items() if tokens&words)
    return (tags[0] if len(tags)==1 else "mixed" if tags else "unclassified"),"|".join(tags)


def _simhash(text):
    tokens=re.findall(r"[가-힣A-Za-z0-9]{2,}",text.casefold())
    counts=Counter(tokens)
    if not counts:return 0
    hashes=np.array([int.from_bytes(hashlib.blake2b(word.encode(),digest_size=8).digest(),"little") for word in counts],dtype=np.uint64)
    weights=np.array([min(3,n) for n in counts.values()],dtype=np.int32)
    bits=((hashes[:,None] >> np.arange(64,dtype=np.uint64)) & 1).astype(np.int32)*2-1
    signs=(bits*weights[:,None]).sum(axis=0)>=0
    return sum(int(value)<<i for i,value in enumerate(signs))


def _shingles(text):
    text=normalized_body(text)
    return {text[i:i+7] for i in range(max(0,len(text)-6))}


def deduplicate(frame,bodies,config):
    """Different IDs and same title alone are insufficient to declare duplicates."""
    out=frame.copy().reset_index(drop=True)
    out["duplicate_of"]="";out["duplicate_type"]=""
    parent=list(range(len(out)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=[];seen={};url_seen={}
    normalized=[normalized_body(bodies[row.body_sha256]) for row in out.itertuples()]
    hashes=[hashlib.sha256(body.encode()).hexdigest() for body in normalized]
    if "url" in out:
        for i,url in enumerate(out.url):
            if url in url_seen:
                j=url_seen[url];parent[find(i)]=find(j)
                pairs.append((j,i,"same_canonical_article_versions",1.,1. if hashes[i]==hashes[j] else 0.))
            else:url_seen[url]=i
    for i,digest in enumerate(hashes):
        if digest in seen:
            j=seen[digest];parent[find(i)]=find(j)
            pairs.append((j,i,"identical_normalized_body",1.,1.))
        else:seen[digest]=i
    signatures=[_simhash(bodies[row.body_sha256]) for row in out.itertuples()]
    bands=defaultdict(list);compared=set();shingles={}
    dates=pd.to_datetime(out.published_date,utc=True,format="ISO8601")
    titles=[normalized_title(t) for t in out.title]
    for i,value in enumerate(signatures):
        candidates=set()
        for band in range(8):candidates.update(bands[band,(value>>(band*8))&255])
        for j in candidates:
            if find(i)==find(j) or (j,i) in compared:continue
            compared.add((j,i))
            if abs((dates[i]-dates[j]).total_seconds())>72*3600:continue
            ratio=len(normalized[i])/max(1,len(normalized[j]))
            if not .80<ratio<1.25:continue
            title_similarity=SequenceMatcher(None,titles[i],titles[j]).ratio()
            if title_similarity<config["duplicate_title_similarity"]:continue
            if (value^signatures[j]).bit_count()>10:continue
            for k in [i,j]:
                if k not in shingles:shingles[k]=_shingles(bodies[out.body_sha256[k]])
            overlap=len(shingles[i]&shingles[j])/max(1,len(shingles[i]|shingles[j]))
            if overlap>=config["duplicate_body_similarity"]:
                numbers_i=set(re.findall(r"\d+(?:[.,]\d+)*",bodies[out.body_sha256[i]]))
                numbers_j=set(re.findall(r"\d+(?:[.,]\d+)*",bodies[out.body_sha256[j]]))
                if numbers_i!=numbers_j:
                    pairs.append((j,i,"numeric_change_preserved_for_review",title_similarity,overlap))
                    continue
                parent[find(i)]=find(j)
                pairs.append((j,i,"near_identical_body_and_headline",title_similarity,overlap))
        for band in range(8):bands[band,(value>>(band*8))&255].append(i)
    groups=defaultdict(list)
    for i in range(len(out)):groups[find(i)].append(i)
    for group in groups.values():
        if len(group)==1:continue
        if "url" in out and out.loc[group,"url"].nunique()==1 and "collected_at" in out:
            representative=sorted(group,key=lambda i:out.collected_at[i])[0]
        else:
            representative=sorted(group,key=lambda i:(-int(out.body_chars[i]),dates[i],out.news_id[i]))[0]
        for i in group:
            if i!=representative:
                out.loc[i,"duplicate_of"]=out.news_id[representative]
                out.loc[i,"duplicate_type"]="canonical_article_version" if "url" in out and out.url[i]==out.url[representative] else ("exact_body" if hashes[i]==hashes[representative] else "near_body_copy")
    table=pd.DataFrame([{"news_id_a":out.news_id[i],"news_id_b":out.news_id[j],"type":kind,
                         "title_similarity":ts,"body_similarity":bs} for i,j,kind,ts,bs in pairs],
                       columns=["news_id_a","news_id_b","type","title_similarity","body_similarity"])
    return out,table


def cluster_events(frame,config):
    """Bounded-time nearest-neighbor candidates; never allocates an N×N matrix."""
    out=frame.copy().reset_index(drop=True)
    if len(out)==0:return out,pd.DataFrame()
    x=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=80000,min_df=1,dtype=np.float32).fit_transform(out.title+" "+out.summary)
    nearest=NearestNeighbors(n_neighbors=min(12,len(out)),metric="cosine",algorithm="brute",n_jobs=1).fit(x)
    dates=pd.to_datetime(out.published_date,utc=True,format="ISO8601")
    parent=list(range(len(out)));low=list(dates);high=list(dates)
    tokens=[set(re.findall(r"[가-힣A-Za-z0-9]{2,}",title)) for title in out.title]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=[]
    for start in range(0,len(out),128):
        distances,indices=nearest.kneighbors(x[start:start+128])
        for local,(ds,js) in enumerate(zip(distances,indices)):
            i=start+local
            for d,j in zip(ds,js):
                j=int(j)
                if j<=i or 1-float(d)<config["event_similarity"]:continue
                if abs((dates[i]-dates[j]).total_seconds())>config["event_max_span_hours"]*3600:continue
                jaccard=len(tokens[i]&tokens[j])/max(1,len(tokens[i]|tokens[j]))
                if jaccard<config["event_min_token_jaccard"]:continue
                a,b=find(i),find(j)
                if max(high[a],high[b])-min(low[a],low[b])>pd.Timedelta(hours=config["event_max_span_hours"]):continue
                if a!=b:
                    parent[b]=a;low[a]=min(low[a],low[b]);high[a]=max(high[a],high[b])
                pairs.append({"news_id_a":out.news_id[i],"news_id_b":out.news_id[j],"similarity":1-float(d),
                              "headline_token_jaccard":jaccard,"status":"heuristic_candidate_not_human_validated"})
    groups=defaultdict(list)
    for i in range(len(out)):groups[find(i)].append(out.news_id[i])
    names={root:"event_"+hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16] for root,ids in groups.items()}
    out["event_group"]=[names[find(i)] for i in range(len(out))]
    out["event_group_status"]="automated_candidate_unvalidated"
    return out,pd.DataFrame(pairs,columns=["news_id_a","news_id_b","similarity","headline_token_jaccard","status"])


def temporal_split(frame,config):
    out=frame[["news_id","event_group","source","published_date"]].copy()
    day=pd.to_datetime(out.published_date,utc=True,format="ISO8601").dt.strftime("%Y-%m-%d")
    out["split"]=np.select([day.le(config["train_end"]),day.le(config["validation_end"])],["train","validation"],default="test")
    if "modified_date" in frame:
        modified=pd.to_datetime(frame.modified_date.replace("",pd.NA),utc=True,errors="coerce",format="ISO8601").dt.strftime("%Y-%m-%d")
        modified_split=np.select([modified.le(config["train_end"]),modified.le(config["validation_end"])],["train","validation"],default="test")
        out.loc[modified.notna() & out.split.ne(modified_split),"split"]="purged_version_change"
    cross=out.groupby("event_group").split.nunique()
    out.loc[out.event_group.isin(cross[cross>1].index),"split"]="purged_boundary_event"
    assert out.groupby("event_group").split.nunique().max()<=1
    return out


def prepare(config,out_dir,raw_dir):
    out,raw=Path(out_dir),Path(raw_dir)
    ledger=pd.read_parquet(out/"extraction_ledger.parquet").fillna("")
    accepted=ledger[ledger.quality_status.eq("accepted")].copy()
    if accepted.empty:raise ValueError("No body-verified article passed extraction gates")
    accepted["body_chars"]=pd.to_numeric(accepted.body_chars)
    bodies={h:gzip.decompress((raw/"bodies"/(h+".txt.gz")).read_bytes()).decode() for h in accepted.body_sha256.unique()}
    for digest,body in bodies.items():
        assert hashlib.sha256(body.encode()).hexdigest()==digest
    dedup,pairs=deduplicate(accepted,bodies,config)
    dedup.to_parquet(out/"deduplication_ledger.parquet",index=False)
    pairs.to_csv(out/"duplicate_pairs.csv",index=False)
    clean=dedup[dedup.duplicate_of.eq("")].copy()
    clean,event_pairs=cluster_events(clean,config)
    splits=temporal_split(clean,config)
    legacy=Path("data/real_news/2026-11-09/news.csv")
    clean["known_legacy_pilot_overlap"]=False
    if legacy.exists():
        previous=pd.read_csv(legacy)
        known=set(previous.url.map(canonicalize))
        hits=dedup[dedup.url.map(canonicalize).isin(known)]
        known_ids=set(hits.news_id)|set(hits.loc[hits.duplicate_of.ne(""),"duplicate_of"])
        known_groups=set(clean.loc[clean.news_id.isin(known_ids),"event_group"])
        clean["known_legacy_pilot_overlap"]=clean.event_group.isin(known_groups)
        splits.loc[splits.event_group.isin(known_groups),"split"]="excluded_previously_scored_pilot"
    clean=clean.merge(splits[["news_id","split"]],on="news_id",validate="one_to_one")
    topics=[harmonize_section(value) for value in clean.category]
    clean["publisher_category"]=clean.category
    clean["normalized_topic"]=[t[0] for t in topics]
    clean["topic_tags"]=[t[1] for t in topics]
    clean["topic_status"]="publisher_metadata_mapping_not_content_validated"
    # Preserve all quality-passing records, but cap a separate research subset.
    selected_ids=set(clean.loc[clean.split.isin(["train","validation","test"]) & ~clean.quality_flags.str.contains("possible_sponsored_content"),"news_id"])
    while selected_ids:
        selected=clean[clean.news_id.isin(selected_ids)]
        counts=selected.source.value_counts()
        majority=counts.index[0]
        cap=int(config["max_research_source_share"]*len(selected))
        if counts.iloc[0]<=cap:break
        excess=int(counts.iloc[0]-cap)
        drop=selected[selected.source.eq(majority)].sample(n=excess,random_state=config["seed"]).news_id
        selected_ids.difference_update(drop)
    clean["research_subset_selected"]=clean.news_id.isin(selected_ids)
    clean.to_parquet(out/"news.parquet",index=False)
    clean.to_csv(out/"news.csv",index=False)
    splits.to_csv(out/"splits.csv",index=False)
    event_pairs.to_csv(out/"event_candidate_pairs.csv",index=False)
    # Provisional full-body lexical scores are diagnostic outputs, not semantic AI labels.
    provisional=[]
    for row in clean.itertuples():
        score=rule_score(row.title+" "+bodies[row.body_sha256])
        provisional.append({"news_id":row.news_id,"event_group":row.event_group,"split":row.split,
            **score,"input_scope":"verified_full_body_private","label_status":"unvalidated_rule_baseline_not_ground_truth"})
    pd.DataFrame(provisional).to_csv(out/"provisional_rule_scores.csv",index=False)
    selected=[]
    # Stratified queue for actual independent QC/labeling, with blank judgments.
    for source,part in clean.groupby("source"):
        selected.append(part.sample(n=min(8,len(part)),random_state=config["seed"]).assign(qc_set="accepted"))
    for source,part in ledger[ledger.quality_status.eq("quarantine")].groupby("source"):
        selected.append(part.sample(n=min(4,len(part)),random_state=config["seed"]).assign(qc_set="quarantine"))
    duplicates=dedup[dedup.duplicate_of.ne("")]
    if len(duplicates):selected.append(duplicates.sample(n=min(24,len(duplicates)),random_state=config["seed"]).assign(qc_set="duplicate_removed"))
    review=pd.concat(selected).fillna("").sort_values(["source","published_date"])
    review=review.reindex(columns=["candidate_id","news_id","source","url","title","summary","published_date","body_chars","body_extractor","quality_flags","event_group","qc_set","exclusion_reasons","duplicate_of"]).fillna("").copy()
    for field in ["reviewer_id","title_body_consistent","date_correct","extraction_complete","event_group_correct","notes"]:review[field]=""
    review.to_csv(out/"independent_qc_template.csv",index=False)
    label=clean.loc[clean.research_subset_selected,["news_id","event_group","source","url","title","summary","split"]].copy()
    label=label.drop_duplicates("event_group").sample(n=min(240,label.event_group.nunique()),random_state=config["seed"])
    label["rater_id"]="";label[["N","T","A"]]=""
    label.to_csv(out/"human_annotation_candidates.csv",index=False)
    for rater in range(1,4):
        sheet=label.copy();sheet["rater_id"]=f"rater_{rater}"
        sheet.to_csv(out/f"blind_rater_{rater}_template.csv",index=False)
    reasons=Counter(reason for value in ledger.exclusion_reasons for reason in str(value).split("|") if reason)
    flags=Counter(flag for value in clean.quality_flags for flag in str(value).split("|") if flag)
    source_dates=clean.assign(day=pd.to_datetime(clean.published_date,utc=True,format="ISO8601").dt.strftime("%Y-%m-%d")).groupby(["source","day"]).size().rename("articles").reset_index()
    source_dates.to_csv(out/"source_day_coverage.csv",index=False)
    clean.groupby(["source","normalized_topic"]).size().rename("articles").reset_index().to_csv(out/"source_topic_coverage.csv",index=False)
    groups=clean.groupby("event_group").agg(articles=("news_id","count"),sources=("source","nunique"),first=("published_date","min"),last=("published_date","max"))
    groups.to_csv(out/"event_group_summary.csv")
    discovery=pd.read_parquet(out/"discovered_candidates.parquet")
    summary={"configured_start":config["start_date"],"configured_end":config["end_date"],
       "discovered_unique_urls":len(discovery),"discovery_date_hint_in_window":int(discovery.discovery_in_window_hint.sum()),
       "sampled_candidates":len(pd.read_parquet(out/"candidate_sample.parquet")),"pages_attempted":len(ledger),
       "body_verified_before_deduplication":len(accepted),"duplicates_removed":int(dedup.duplicate_of.ne("").sum()),
       "accepted_unique_articles":len(clean),"publishers":clean.source.value_counts().to_dict(),
       "research_subset_articles":int(clean.research_subset_selected.sum()),
       "research_subset_publisher_counts":clean[clean.research_subset_selected].source.value_counts().to_dict(),
       "research_subset_max_source_share":float(clean[clean.research_subset_selected].source.value_counts(normalize=True).max()) if selected_ids else None,
       "publication_min":clean.published_date.min(),"publication_max":clean.published_date.max(),
       "covered_utc_days":int(pd.to_datetime(clean.published_date,utc=True).dt.date.nunique()),
       "partitions":clean.split.value_counts().to_dict(),"event_groups":int(clean.event_group.nunique()),
       "known_legacy_pilot_overlap_articles":int(clean.known_legacy_pilot_overlap.sum()),
       "normalized_topics":clean.normalized_topic.value_counts().to_dict(),
       "topic_mapping_scope":"Publisher metadata only; mixed and unclassified retained; not emotion or threat labels",
       "event_groups_human_validated":0,"independent_human_ratings":0,"qc_template_articles":len(review),
       "quarantine_reasons":dict(reasons),"quality_flags":dict(flags),
       "goals":{"candidate_goal":config["candidate_target"],"unique_article_goal":config["accepted_goal"],
                "accepted_goal_met":len(clean)>=config["accepted_goal"],"publisher_goal_met":clean.source.nunique()>=config["min_publishers"]},
       "scoring_scope":"Full-body keyword baseline only. Existing 240-item session judgments are not transferred or reused as truth.",
       "limitations":["Publisher-time coverage is uneven and reported explicitly","Event clusters are automated candidates and may miss or overmerge events",
                      "Full bodies are private caches; public releases contain limited excerpts and non-reversible hashes",
                      "Rule scores are unvalidated diagnostic values, not labels or human physiological outcomes"]}
    write_json(out/"summary.json",summary)
    files=[out/name for name in ["resolved_config.json","discovered_candidates.parquet","extraction_ledger.parquet","news.parquet","splits.csv","provisional_rule_scores.csv"]]
    if legacy.exists():files.append(legacy)
    code=list(Path(__file__).parent.glob("*.py"))
    write_json(out/"manifest.json",manifest(files+code,config_version=config["version"],data_source="official_publisher_pages",raw_cache=str(raw),
       publication_dates="Article JSON-LD/meta first; publisher sitemap only as declared fallback; lastmod and URL dates never promoted to publication time",
       input_hashes_verified=True,independent_human_accuracy_evaluated=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary
