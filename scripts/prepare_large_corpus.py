import argparse
import json
from pathlib import Path
from news_flywire.corpus.prepare import prepare
from news_flywire.corpus.report import report

p=argparse.ArgumentParser()
p.add_argument('--out',default='data/corpus/20261109_v2')
p.add_argument('--raw',default='data/raw/corpus/20261109_v2')
a=p.parse_args()
c=json.loads((Path(a.out)/'resolved_config.json').read_text())
prepare(c,a.out,a.raw)
print(report(a.out))
