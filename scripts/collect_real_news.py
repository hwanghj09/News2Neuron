"""Fetch a dated, source-balanced pilot corpus without a synthetic fallback."""
import argparse
from pathlib import Path
import yaml
from news_flywire.news_ingestion import collect_news

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/news_sources_v1.yaml')
p.add_argument('--out',default='data/real_news/2026-11-09')
a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text())
news=collect_news(c,a.out)
print(news.groupby(['category','source']).size().to_string())
print(f'{len(news)} actual articles → {a.out}/news.csv')
