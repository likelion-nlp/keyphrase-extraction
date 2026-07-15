"""정답 없이 키프레이즈 품질을 자동 채점한다 (reference-free scorecard).

정답(gold)이 없는 신규 논문에도 쓸 수 있는 세 가지 지표:

  ① 검색 효용 (Retrieval Utility) — 키프레이즈를 '검색어'로 썼을 때 원 논문이 다시 찾아지는가?
     키프레이즈의 존재 이유가 색인·검색이므로, 이것이 가장 본질적인 채점이다.
     지표: MRR, Hit@1, Hit@5 (distractor 문서를 섞은 코퍼스에서)
     KP20k에서는 gold를 검색어로 쓴 값이 '천장'이 되어 캘리브레이션이 된다.

  ② 특이성 (Specificity, IDF) — 이 논문에만 해당하는 표현인가, 아무 CS 논문에나 붙는 일반어인가?
     KP20k train 코퍼스의 document frequency로 IDF를 계산해 평균낸다. 높을수록 문서 특이적.

  ③ 근거성 (Faithfulness) — absent 키프레이즈가 초록 내용에 실제로 뒷받침되는가?
     각 absent 구절과 초록 문장들의 최대 코사인 유사도. 낮으면 환각(hallucination) 의심.

실행: python scripts/reference_free_scorecard.py
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

from src.data import load_kp20k, plain_doc_text
from src.preprocessing import normalize_phrase, tokenize
from src.reranking import SemanticScorer

OUT = PROJECT_ROOT / "outputs"
T0 = time.time()
HALLU_THRESHOLD = 0.30      # 초록 문장과의 최대 유사도가 이보다 낮으면 환각 의심


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- 데이터
papers = pd.read_csv(PROJECT_ROOT / "cs_papers_20232024.csv")
doc_text = {r.arxiv_id: plain_doc_text(r.title, r.abstract) for r in papers.itertuples()}
doc_ids = list(doc_text)

SYSTEMS = {
    "Fusion (P2)": "newpapers_cs_papers_20232024_keyphrases.csv",
    "Pairwise (P6)": "newpapers_cs_papers_20232024_pairwise_keyphrases.csv",
}
preds: dict[str, dict[str, list[str]]] = {}
for name, fn in SYSTEMS.items():
    df = pd.read_csv(OUT / "predictions" / fn, encoding="utf-8-sig")
    preds[name] = {aid: g.sort_values("rank").keyphrase.tolist()
                   for aid, g in df.groupby("arxiv_id")}
    log(f"{name}: {len(preds[name])} 문서 로드")

scorer = SemanticScorer(device="cuda")

# ---------------------------------------------------------------- ① 검색 효용
# 코퍼스: 신규 논문 50편 + KP20k test 20,000편(distractor) = 20,050편
# (2,000편 코퍼스에서는 모든 시스템이 만점이라 변별이 안 됨 → distractor를 10배로)
log("① 검색 효용 — 코퍼스 구축 (신규 50 + distractor 20,000)")
distract = load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=7)["test"]
corpus_texts = [doc_text[i] for i in doc_ids] + [
    plain_doc_text(r["title"], r["abstract"]) for r in distract
]
corpus_emb = scorer.encode(corpus_texts, batch_size=256)
target_pos = {aid: i for i, aid in enumerate(doc_ids)}
log(f"  코퍼스 {len(corpus_texts):,}편 임베딩 완료")


def _ranks(queries: list[str], targets: list[int]) -> np.ndarray:
    q_emb = scorer.encode(queries, batch_size=256)
    sim = q_emb @ corpus_emb.T
    return np.array([int((row > row[t]).sum()) + 1 for row, t in zip(sim, targets)])


def retrieval_scores(kps_by_doc: dict[str, list[str]], top_k: int) -> dict[str, float]:
    """top_k개를 이어붙인 '요약 검색어'로 원 논문을 찾는다."""
    queries, targets = [], []
    for aid, kps in kps_by_doc.items():
        if aid in target_pos and kps:
            queries.append(", ".join(kps[:top_k]))
            targets.append(target_pos[aid])
    r = _ranks(queries, targets)
    return {"MRR": float(np.mean(1.0 / r)), "Hit@1": float(np.mean(r == 1)),
            "Hit@5": float(np.mean(r <= 5)), "평균순위": float(np.mean(r))}


def single_kp_retrieval(kps_by_doc: dict[str, list[str]], top_k: int = 10) -> dict[str, float]:
    """키프레이즈 '하나'를 검색어로 썼을 때 원 논문이 나오는가 — 색인어로서의 변별력.

    구절 단위로 순위를 재고, Hit@10(상위 10등 안에 원 논문이 있는 비율)과 MRR을 평균낸다.
    """
    queries, targets = [], []
    for aid, kps in kps_by_doc.items():
        if aid not in target_pos:
            continue
        for kp in kps[:top_k]:
            queries.append(kp)
            targets.append(target_pos[aid])
    r = _ranks(queries, targets)
    return {"구절MRR": float(np.mean(1.0 / r)), "구절Hit@10": float(np.mean(r <= 10)),
            "구절Hit@100": float(np.mean(r <= 100)), "구절평균순위": float(np.mean(r))}


# ---------------------------------------------------------------- ② 특이성 (IDF)
log("② 특이성 — KP20k train 100,000편으로 document frequency 계산")
train = load_kp20k(["train"], subset_sizes={"train": 100_000}, seed=1)["train"]
N_DF = len(train)
df_counter: Counter = Counter()
for r in train:
    toks = set(tokenize(plain_doc_text(r["title"], r["abstract"])))
    df_counter.update(toks)
log(f"  어휘 {len(df_counter):,}개")


def idf(token: str) -> float:
    return math.log((N_DF + 1) / (df_counter.get(token, 0) + 1))


def specificity(kps: list[str]) -> float:
    """구절의 평균 IDF (구절 내 토큰 IDF의 평균). 높을수록 문서 특이적."""
    vals = []
    for kp in kps:
        toks = tokenize(kp)
        if toks:
            vals.append(float(np.mean([idf(t) for t in toks])))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------- ③ 근거성
_SENT = re.compile(r"(?<=[.!?])\s+")


def faithfulness(aid: str, kps: list[str], absent_only: list[str]) -> tuple[float, float]:
    """absent 구절이 초록 문장들과 얼마나 유사한가 → (평균 지지도, 환각 의심 비율)."""
    if not absent_only:
        return float("nan"), float("nan")
    sents = [s for s in _SENT.split(doc_text[aid]) if len(s.split()) >= 4]
    if not sents:
        return float("nan"), float("nan")
    s_emb = scorer.encode(sents)
    k_emb = scorer.encode(absent_only)
    max_sim = (k_emb @ s_emb.T).max(axis=1)
    return float(max_sim.mean()), float((max_sim < HALLU_THRESHOLD).mean())


# ---------------------------------------------------------------- 실행
long_files = {
    "Fusion (P2)": "newpapers_cs_papers_20232024_keyphrases.csv",
    "Pairwise (P6)": "newpapers_cs_papers_20232024_pairwise_keyphrases.csv",
}
rows = []
for name, kps_by_doc in preds.items():
    log(f"채점 중: {name}")
    r5 = retrieval_scores(kps_by_doc, 5)
    sk = single_kp_retrieval(kps_by_doc, 10)

    df = pd.read_csv(OUT / "predictions" / long_files[name], encoding="utf-8-sig")
    spec = float(np.mean([specificity(k) for k in kps_by_doc.values()]))
    faith, hallu = [], []
    for aid, g in df.groupby("arxiv_id"):
        absent = g[g.type == "absent"].keyphrase.tolist()
        f, h = faithfulness(aid, g.keyphrase.tolist(), absent)
        if not math.isnan(f):
            faith.append(f); hallu.append(h)
    rows.append({
        "시스템": name,
        "요약검색 MRR(top5)": round(r5["MRR"], 4),
        "구절검색 MRR": round(sk["구절MRR"], 4),
        "구절 Hit@10": round(sk["구절Hit@10"], 4),
        "구절 Hit@100": round(sk["구절Hit@100"], 4),
        "구절 평균순위": round(sk["구절평균순위"], 1),
        "특이성(IDF)": round(spec, 3),
        "근거성(absent 지지도)": round(float(np.mean(faith)), 4),
        "환각의심률": round(float(np.mean(hallu)), 4),
    })

# 참조점: 제목을 검색어로 (상한선)
title_q = {aid: [papers[papers.arxiv_id == aid].title.iloc[0]] for aid in doc_ids}
rt = single_kp_retrieval(title_q, 1)
rows.append({"시스템": "[참조] 논문 제목 1개를 검색어로",
             "요약검색 MRR(top5)": None,
             "구절검색 MRR": round(rt["구절MRR"], 4),
             "구절 Hit@10": round(rt["구절Hit@10"], 4),
             "구절 Hit@100": round(rt["구절Hit@100"], 4),
             "구절 평균순위": round(rt["구절평균순위"], 1),
             "특이성(IDF)": None, "근거성(absent 지지도)": None, "환각의심률": None})

res = pd.DataFrame(rows)
path = OUT / "metrics" / "reference_free_scorecard.csv"
res.to_csv(path, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print()
print(res.to_string(index=False))
log(f"저장: {path}")
