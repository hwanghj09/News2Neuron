"""Fetch pinned public sources; preserve checksums; never download the 145 GB movies."""
from pathlib import Path
import json
import subprocess
import urllib.request
import yaml

from news_flywire.provenance import write_json, manifest, git_commit

ROOT=Path(__file__).resolve().parents[1]
config=yaml.safe_load((ROOT/'configs/connectome_v783.yaml').read_text())
target=ROOT/'external/Drosophila_brain_model'
if not target.exists():
    subprocess.run(['git','clone','--no-checkout',config['upstream_repository'],str(target)],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',config['upstream_commit']],check=True)
if git_commit(target)!=config['upstream_commit']:
    raise SystemExit('Upstream checkout differs from pinned commit. Use a separate checkout at '+config['upstream_commit'])
annotation=ROOT/config['annotations']
annotation.parent.mkdir(parents=True,exist_ok=True)
url=f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{config['annotations_commit']}/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
temporary=annotation.with_suffix('.download')
with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as f:
    while block:=response.read(1024*1024):
        f.write(block)
temporary.replace(annotation)
write_json(ROOT/'data/derived/source_manifest.json',manifest([annotation,target/'model.py',target/'Connectivity_783.parquet'],
    annotation_url=url,annotation_commit=config['annotations_commit'],upstream_commit=git_commit(target)))
for key,url in {
    'dandi_version':'https://api.dandiarchive.org/api/dandisets/001869/versions/0.260727.1644/',
    'dandi_assets':'https://api.dandiarchive.org/api/dandisets/001869/versions/0.260727.1644/assets/?page_size=10',
}.items():
    with urllib.request.urlopen(url,timeout=30) as response:
        write_json(ROOT/f'data/derived/public_sources/{key}.json',json.load(response))
print('Pinned code/connectivity and annotations ready; DANDI metadata only.')
