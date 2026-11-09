"""Download only pinned public model artifacts; no account/API key required."""
import argparse
import urllib.request
from pathlib import Path

from news_flywire.provenance import sha256, write_json
from news_flywire.risk_features import MODEL_ID, ONNX_SHA256, REVISION, TOKENIZER_SHA256

p=argparse.ArgumentParser()
p.add_argument('--out',default='external/multilingual-e5-small')
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
files={'onnx/model_qint8_avx512_vnni.onnx':ONNX_SHA256,'tokenizer.json':TOKENIZER_SHA256}
for name,digest in files.items():
    target=out/Path(name).name
    if target.exists() and sha256(target)==digest:
        continue
    partial=target.with_suffix(target.suffix+'.partial')
    urllib.request.urlretrieve(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}',partial)
    if sha256(partial)!=digest:
        raise ValueError('Downloaded model artifact hash mismatch: '+name)
    partial.replace(target)
write_json(out/'download_manifest.json',{'model_id':MODEL_ID,'revision':REVISION,'files':{Path(k).name:v for k,v in files.items()}})
print(out)
