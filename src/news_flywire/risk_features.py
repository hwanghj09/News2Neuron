"""Frozen multilingual features from every token of the cached article prose.

The encoder is a retrieval model, not a news danger classifier. Labels and
supervised fitting live in risk_assessment.py. No article instructions execute.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256, write_json

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
ONNX_SHA256 = "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"
TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
PROBES = [
    "현재 화재와 붕괴, 홍수와 태풍이 발생 중이고 사람들이 죽거나 다칠 위험으로 즉시 대피해야 한다.",
    "전쟁과 무력 충돌, 테러와 핵무기, 군사적 공격이 계속되어 민간인의 생명과 안전이 위협받는다.",
    "전염병과 유해 물질, 감염과 질병이 확산되어 사람들의 건강에 중대한 위해가 발생할 수 있다.",
    "실직과 해고, 경기 침체와 파산, 금융 사기와 경제적 손실로 생계가 위협받고 있다.",
    "성폭력과 살인, 아동 학대와 가정 폭력, 교통사고와 산업 재해로 심각한 피해가 발생했다.",
    "오래전 사건을 회고하거나 고인을 추모한다. 이미 불이 꺼지고 구조가 끝나 추가 위험은 없다.",
    "위험 예방과 안전 교육, 질병 치료와 기술 개발, 지원 정책으로 피해를 막으려 한다.",
    "축제와 우승, 문화와 예술, 여행과 신제품, 기업의 성과와 기부를 소개하는 좋은 소식이다.",
]


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, model_dir, max_length=256):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        weights = model_dir / "model_qint8_avx512_vnni.onnx"
        tokenizer = model_dir / "tokenizer.json"
        if sha256(weights) != ONNX_SHA256 or sha256(tokenizer) != TOKENIZER_SHA256:
            raise ValueError("Encoder weights/tokenizer differ from the pinned version")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=max_length, stride=32)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights), options, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.session.get_inputs()}

    def encode_chunks(self, chunks):
        ids = np.array([c.ids for c in chunks], dtype=np.int64)
        mask = np.array([c.attention_mask for c in chunks], dtype=np.int64)
        inputs = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.names:
            inputs["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, inputs)[0]
        return normalize((hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]).astype(np.float32)

    def single(self, texts):
        return self.encode_chunks(self.tokenizer.encode_batch(texts))


def build_features(corpus, raw, out, cache, model_dir):
    corpus, raw, out, cache = map(Path, (corpus, raw, out, cache))
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    news = pd.read_parquet(corpus / "news.parquet")
    snapshot = {"news_parquet_sha256": sha256(corpus / "news.parquet"), "model_id": MODEL_ID,
                "revision": REVISION, "weights_sha256": ONNX_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
                "max_tokens": 256, "overlap_tokens": 32, "probes": PROBES,
                "feature_version": "all_body_windows_v1"}
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    guard = cache / "feature_run_config.json"
    if guard.exists():
        if json.loads(guard.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Partial feature cache belongs to a different frozen configuration")
    elif (cache / "articles").exists():
        raise ValueError("Existing partial cache has no frozen configuration receipt")
    else:
        write_json(guard, {**snapshot, "fingerprint":fingerprint})
    if (cache / "features_manifest.json").exists():
        old = json.loads((cache / "features_manifest.json").read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Use a new feature cache: frozen input or encoder settings changed")
    encoder = Encoder(model_dir)
    probes = encoder.single(["query: " + text for text in PROBES])
    if (cache / "titles.npy").exists():
        title_features = np.load(cache / "titles.npy")
        if title_features.shape != (len(news), 384):
            raise ValueError("Title cache shape differs from the frozen corpus")
    else:
        title_features = []
        for start in range(0, len(news), 16):
            title_features.extend(encoder.single(["passage: " + str(t) for t in news.title[start:start + 16]]))
            if start % 800 == 0:
                print(f"Encoded article titles {min(start+16,len(news))}/{len(news)}", flush=True)
        title_features = np.asarray(title_features, dtype=np.float32)
        np.save(cache / "titles.npy", title_features)
    features, evidence, counts = [], [], []
    for i, row in enumerate(news.itertuples()):
        path = cache / "articles" / (row.body_sha256 + ".npz")
        text_path = cache / "evidence" / (row.body_sha256 + ".json")
        if path.exists() and text_path.exists():
            item = np.load(path)
            feature = item["features"]
            record = json.loads(text_path.read_text())
        else:
            body = gzip.decompress((raw / "bodies" / (row.body_sha256 + ".txt.gz")).read_bytes()).decode()
            if hashlib.sha256(body.encode()).hexdigest() != row.body_sha256:
                raise ValueError("Body hash mismatch: " + row.news_id)
            # Tokenizer overflow windows cover the complete body, including late
            # negations/corrections; there is no head-only truncation.
            text = "passage: " + row.title + "\n" + body
            encoded = encoder.tokenizer.encode(text)
            chunks = [encoded] + list(encoded.overflowing)
            vectors = np.concatenate([encoder.encode_chunks(chunks[j:j + 16]) for j in range(0, len(chunks), 16)])
            similarities = vectors @ probes.T
            salience = similarities[:, :5].max(axis=1) - similarities[:, 5:].max(axis=1)
            selected = int(np.argmax(salience))
            mean = normalize(vectors.mean(axis=0))
            feature = np.concatenate([mean, vectors[selected], similarities.max(axis=0), similarities.mean(axis=0)]).astype(np.float32)
            offsets = [offset for offset in chunks[selected].offsets if offset[1] > offset[0]]
            start, end = min(a for a, _ in offsets), max(b for _, b in offsets)
            record = {"news_id": row.news_id, "body_sha256": row.body_sha256,
                      "lead": body[:900], "salient_window": text[start:end],
                      "salient_char_start": start, "salient_char_end": end,
                      "chunks": len(chunks), "input_chars": len(text),
                      "all_body_tokens_processed": True}
            path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, features=feature)
            write_json(text_path, record)
        features.append(feature)
        evidence.append(record)
        counts.append(record["chunks"])
        if (i + 1) % 100 == 0:
            print(f"Encoded complete body {i+1}/{len(news)}; windows={sum(counts)}", flush=True)
    x = np.concatenate([title_features, np.array(features)], axis=1)
    np.save(cache / "features.npy", x)
    news[["news_id", "body_sha256", "title", "source", "split", "event_group"]].to_csv(cache / "feature_rows.csv", index=False)
    write_json(cache / "features_manifest.json", {**snapshot, "fingerprint": fingerprint,
               "features_sha256": sha256(cache / "features.npy"), "shape": list(x.shape),
               "articles": len(news), "windows": sum(counts), "complete_body_coverage": True})
    write_json(out / "encoder_manifest.json", json.loads((cache / "features_manifest.json").read_text()))
    print(f"Saved {x.shape} features, {sum(counts)} body windows", flush=True)
