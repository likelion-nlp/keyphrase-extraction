"""신규 논문에 대해 전체 시스템(B0~B4, P1/P2, P5/P6)을 예측하고 정답 없이 자동 채점한다.

채점 지표 (reference-free):
  ① 검색 효용   — 키프레이즈 1개를 검색어로 썼을 때 원 논문이 top-10에 나오는가 (MRR, Hit@10)
  ② 특이성 IDF  — 일반어인가 문서 특이 표현인가 (KP20k 10만편 문서빈도 기반)
  ③ 근거성      — absent 구절이 초록 문장으로 뒷받침되는가 / 환각 의심률
  ④ 다양성      — stem 중복률, 의미 중복도(semantic redundancy)

실행: python scripts/scorecard_all_systems.py
산출: outputs/metrics/scorecard_all_systems.csv
      outputs/predictions/newpapers_allsystems_keyphrases.csv  (시스템별 top-10 long 포맷)
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data import build_source, load_kp20k, plain_doc_text
from src.diversity import mmr_select
from src.extraction import KeyBertExtractor, TfidfExtractor
from src.generation import generate_keyphrases
from src.metrics import semantic_redundancy, stem_duplicate_ratio
from src.preprocessing import classify_prmu, stem_tokens, tokenize
from src.reranking import SemanticScorer, fuse_scores, merge_candidates
from src.utils import set_seed

OUT = PROJECT_ROOT / "outputs"
CK = OUT / "checkpoints"
TOP_K, MMR_LAMBDA, HALLU_TH = 10, 0.5, 0.30
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


set_seed(42)
papers = pd.read_csv(PROJECT_ROOT / "cs_papers_20232024.csv")
ids = papers.arxiv_id.tolist()
titles = papers.title.tolist()
texts = [plain_doc_text(t, a) for t, a in zip(papers.title, papers.abstract)]
sources = [build_source(t, a) for t, a in zip(papers.title, papers.abstract)]
stem_cache = [stem_tokens(t) for t in texts]
log(f"신규 논문 {len(ids)}편")

scorer = SemanticScorer(device="cuda")
preds: dict[str, list[list[str]]] = {}

# ---------------------------------------------------------------- B0 TF-IDF
log("B0 TF-IDF")
train_fit = load_kp20k(["train"], subset_sizes={"train": 50_000}, seed=42)["train"]
tfidf = TfidfExtractor(max_features=300_000)
tfidf.fit([plain_doc_text(r["title"], r["abstract"]) for r in train_fit])
preds["B0 TF-IDF"] = [[p for p, _ in doc] for doc in tfidf.extract_batch(texts, top_n=TOP_K)]

# ---------------------------------------------------------------- B1/B2 KeyBERT
log("B1/B2 KeyBERT (±MMR)")
extractor = KeyBertExtractor(device="cuda")
kb30 = extractor.extract_batch(texts, top_n=30)          # 후보 pool용
preds["B1 KeyBERT"] = [[p for p, _ in doc[:TOP_K]] for doc in kb30]
kb_mmr = extractor.extract_batch(texts, top_n=TOP_K, use_mmr=True, diversity=0.5)
preds["B2 KeyBERT+MMR"] = [[p for p, _ in doc] for doc in kb_mmr]

# ---------------------------------------------------------------- B3/B4 생성 모델
def gen_preds(ckpt_name: str, strategy: str, top_k: int | None = TOP_K):
    tok = AutoTokenizer.from_pretrained(str(CK / ckpt_name))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(CK / ckpt_name)).to("cuda").eval()
    out = generate_keyphrases(model, tok, sources, strategy=strategy,
                              max_source_length=384, batch_size=4)
    del model
    torch.cuda.empty_cache()
    return out


log("B3 BART-base (full, beam5)")
preds["B3 BART"] = [[g["text"] for g in doc[:TOP_K]] for doc in gen_preds("bart_base_full", "beam5")]

log("B4 KeyBART (full, beam5)")
preds["B4 KeyBART"] = [[g["text"] for g in doc[:TOP_K]] for doc in gen_preds("keybart_full", "beam5")]

log("후보 pool용 KeyBART beam10")
gen10 = gen_preds("keybart_full", "beam10")

# ---------------------------------------------------------------- P1/P2 fusion
log("P1/P2 Hybrid fusion (±MMR)")
fused_docs, cand_embs = [], []
for i in range(len(ids)):
    merged = merge_candidates(kb30[i], gen10[i])
    ranked = fuse_scores(merged, texts[i], titles[i], scorer)
    fused_docs.append(ranked)
    cand_embs.append(scorer.encode([c["phrase"] for c in ranked]) if ranked else None)


def select(score_key_docs, lam):
    out = []
    for d, doc in enumerate(fused_docs):
        if not doc:
            out.append([]); continue
        sc = np.asarray(score_key_docs[d], dtype=float)
        lo, hi = sc.min(), sc.max()
        rel = (sc - lo) / (hi - lo) if hi > lo else np.full_like(sc, 0.5)
        if lam is None:
            order = list(np.argsort(-rel)[:TOP_K])
        else:
            order = mmr_select(cand_embs[d], rel.tolist(), top_k=TOP_K, lambda_=lam)
        out.append([doc[j]["phrase"] for j in order])
    return out


fusion_scores = [[c["final_score"] for c in doc] for doc in fused_docs]
preds["P1 Hybrid fusion"] = select(fusion_scores, None)
preds["P2 Hybrid fusion+MMR"] = select(fusion_scores, MMR_LAMBDA)

# ---------------------------------------------------------------- P5/P6 pairwise CE
@torch.no_grad()
def ce_scores(tag: str):
    tok = AutoTokenizer.from_pretrained(str(CK / f"reranker_{tag}"))
    model = AutoModelForSequenceClassification.from_pretrained(str(CK / f"reranker_{tag}")).to("cuda").eval()
    out = []
    for i, doc in enumerate(fused_docs):
        if not doc:
            out.append([]); continue
        ph = [c["phrase"] for c in doc]
        enc = tok([texts[i]] * len(ph), ph, truncation="only_first", max_length=384,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy().tolist())
    del model
    torch.cuda.empty_cache()
    return out


for tag, label in [("unweighted", "P5 Pairwise 무가중"), ("prmu_w", "P6 Pairwise PRMU-가중")]:
    if (CK / f"reranker_{tag}" / "config.json").exists():
        log(label)
        preds[label] = select(ce_scores(tag), None)   # CE는 MMR 없이 (실험 결과)

# ================================================================ 채점 준비
log("채점 준비 — 검색 코퍼스(신규 50 + distractor 20,000)")
distract = load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=7)["test"]
corpus_texts = texts + [plain_doc_text(r["title"], r["abstract"]) for r in distract]
corpus_emb = scorer.encode(corpus_texts, batch_size=256)

log("특이성 — KP20k train 100,000편 문서빈도")
df_train = load_kp20k(["train"], subset_sizes={"train": 100_000}, seed=1)["train"]
N_DF = len(df_train)
dfc: Counter = Counter()
for r in df_train:
    dfc.update(set(tokenize(plain_doc_text(r["title"], r["abstract"]))))


def idf(t: str) -> float:
    return math.log((N_DF + 1) / (dfc.get(t, 0) + 1))


def specificity(kps: list[str]) -> float:
    vals = [float(np.mean([idf(t) for t in tokenize(kp)])) for kp in kps if tokenize(kp)]
    return float(np.mean(vals)) if vals else 0.0


_SENT = re.compile(r"(?<=[.!?])\s+")
sent_embs = [scorer.encode([s for s in _SENT.split(t) if len(s.split()) >= 4]) for t in texts]


def score_system(name: str, docs: list[list[str]]) -> dict:
    # ① 검색: 구절 1개 = 검색어
    queries, targets = [], []
    for i, kps in enumerate(docs):
        for kp in kps[:TOP_K]:
            queries.append(kp); targets.append(i)
    q_emb = scorer.encode(queries, batch_size=256)
    sim = q_emb @ corpus_emb.T
    ranks = np.array([int((row > row[t]).sum()) + 1 for row, t in zip(sim, targets)])

    # ②③④
    specs, dups, reds, faiths, hallus, n_abs, n_tot = [], [], [], [], [], 0, 0
    for i, kps in enumerate(docs):
        if not kps:
            continue
        specs.append(specificity(kps))
        dups.append(stem_duplicate_ratio(kps))
        emb = scorer.encode(kps)
        reds.append(semantic_redundancy(emb))
        absent = [k for k in kps if classify_prmu(k, "", stem_cache[i]) != "P"]
        n_abs += len(absent); n_tot += len(kps)
        if absent and sent_embs[i] is not None and len(sent_embs[i]):
            ms = (scorer.encode(absent) @ sent_embs[i].T).max(axis=1)
            faiths.append(float(ms.mean())); hallus.append(float((ms < HALLU_TH).mean()))
    return {
        "시스템": name,
        "구절검색 MRR": round(float(np.mean(1 / ranks)), 4),
        "구절 Hit@10": round(float(np.mean(ranks <= 10)), 4),
        "구절 평균순위": round(float(np.mean(ranks)), 1),
        "특이성(IDF)": round(float(np.mean(specs)), 3),
        "absent 비율": round(n_abs / max(1, n_tot), 3),
        "근거성(absent)": round(float(np.mean(faiths)), 4) if faiths else None,
        "환각의심률": round(float(np.mean(hallus)), 4) if hallus else None,
        "stem중복률": round(float(np.mean(dups)), 4),
        "의미중복도": round(float(np.mean(reds)), 4),
    }


log("채점 시작")
rows = [score_system(n, d) for n, d in preds.items()]

# 참조점: 논문 제목 자체를 검색어로 (상한선)
q = scorer.encode(titles, batch_size=64)
sim = q @ corpus_emb.T
r_title = np.array([int((row > row[i]).sum()) + 1 for i, row in enumerate(sim)])
rows.append({"시스템": "[참조] 논문 제목", "구절검색 MRR": round(float(np.mean(1 / r_title)), 4),
             "구절 Hit@10": round(float(np.mean(r_title <= 10)), 4),
             "구절 평균순위": round(float(np.mean(r_title)), 1)})

res = pd.DataFrame(rows)
res.to_csv(OUT / "metrics" / "scorecard_all_systems.csv", index=False, encoding="utf-8-sig")

# 시스템별 예측도 long 포맷으로 저장
long_rows = []
for name, docs in preds.items():
    for i, kps in enumerate(docs):
        for rank, kp in enumerate(kps, 1):
            tag = classify_prmu(kp, "", stem_cache[i])
            long_rows.append({"시스템": name, "arxiv_id": ids[i], "title": titles[i],
                              "rank": rank, "keyphrase": kp,
                              "type": "present" if tag == "P" else "absent", "prmu": tag})
pd.DataFrame(long_rows).to_csv(OUT / "predictions" / "newpapers_allsystems_keyphrases.csv",
                               index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print()
print(res.to_string(index=False))
log("저장: outputs/metrics/scorecard_all_systems.csv + newpapers_allsystems_keyphrases.csv")
