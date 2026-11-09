import argparse
from pathlib import Path
import yaml
from news_flywire.corpus.build import build

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/corpus_v2.yaml')
p.add_argument('--out',default='data/corpus/20261109_v2')
p.add_argument('--raw',default='data/raw/corpus/20261109_v2')
a=p.parse_args()
config=yaml.safe_load(Path(a.config).read_text())
build(config,a.out,a.raw)
