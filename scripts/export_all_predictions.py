"""모든 시스템(B0~B4, P1~P6)의 KP20k test 예측을 두 가지 CSV로 일괄 저장한다.

    kp20k_test_<RUN>_full.csv             문서별 1행: pred_top10([O]=정답), gold, F1@5, tp@10
    kp20k_test_<RUN>_full_keyphrases.csv  구절별 1행: rank, keyphrase, type, prmu, score, source, is_correct

예측이 저장되지 않은 시스템(P5/P6의 MMR 변형 등)은 저장된 후보 pool에서 재구성한다.
실행: python scripts/export_all_predictions.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data import load_kp20k, plain_doc_text
from src.diversity import mmr_select
from src.metrics import evaluate_corpus, precision_recall_f1_at_k
from src.preprocessing import classify_prmu, normalize_phrase, stem_tokens
from src.utils import ExperimentLogger, load_jsonl

OUT = PROJECT_ROOT / "outputs"
PRED = OUT / "predictions"
TOP_K = 10
MMR_LAMBDA = 0.5
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- 공통 데이터
log("test 20,000 + 후보 pool 로드")
test = list(load_kp20k(["test"], subset_sizes={"test": 20_000}, seed=42)["test"])
saved = load_jsonl(OUT / "candidates" / "fused_candidates_full.jsonl")
assert [r["id"] for r in test] == [r["id"] for r in saved]

texts = [plain_doc_text(r["title"], r["abstract"]) for r in test]
titles = [r["title"] for r in test]
ids = [r["id"] for r in test]
golds = [r["keyphrases"] for r in test]
prmus = [r["prmu"] for r in test]
stem_cache = [stem_tokens(t) for t in texts]
gold_norms = [{normalize_phrase(g) for g in gl} for gl in golds]

cand_phrases = [[c["phrase"] for c in r["candidates"]] for r in saved]
cand_fusion = [[float(c["final_score"]) for c in r["candidates"]] for r in saved]
cand_sources = [{c["phrase"]: "+".join(sorted(c.get("sources", []))) for c in r["candidates"]}
                for r in saved]
n_pairs = sum(len(c) for c in cand_phrases)
log(f"후보 {n_pairs:,}개 (문서당 {n_pairs / len(test):.1f})")


# ---------------------------------------------------------------- 점수·선택 유틸
def mmr_or_topk(scores_docs, embs, lam):
    """점수 리스트 → top-10 인덱스 (lam=None이면 점수 순, 아니면 MMR)."""
    out = []
    for d, sc in enumerate(scores_docs):
        if not sc:
            out.append([]); continue
        arr = np.asarray(sc, dtype=float)
        lo, hi = arr.min(), arr.max()
        rel = (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)
        if lam is None:
            order = list(np.argsort(-rel)[:TOP_K])
        else:
            order = mmr_select(embs[d], rel.tolist(), top_k=TOP_K, lambda_=lam)
        out.append([int(j) for j in order])
    return out


@torch.no_grad()
def ce_scores(ckpt_name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ck = OUT / "checkpoints" / ckpt_name
    tok = AutoTokenizer.from_pretrained(str(ck))
    model = AutoModelForSequenceClassification.from_pretrained(str(ck)).to("cuda").eval()
    flat = [(i, p) for i, doc in enumerate(cand_phrases) for p in doc]
    sc = np.zeros(len(flat), dtype=np.float32)
    B = 384
    for s in range(0, len(flat), B):
        ch = flat[s : s + B]
        enc = tok([texts[i] for i, _ in ch], [p for _, p in ch], truncation="only_first",
                  max_length=384, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc[s : s + len(ch)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (s // B) % 400 == 0:
            log(f"  CE({ckpt_name}) {s:,}/{len(flat):,}")
    del model
    torch.cuda.empty_cache()
    per_doc, k = [], 0
    for doc in cand_phrases:
        per_doc.append(sc[k : k + len(doc)].tolist())
        k += len(doc)
    return per_doc


# ---------------------------------------------------------------- 저장 함수
def export(run: str, label: str, preds, scores_lookup=None):
    """preds: 문서별 phrase 리스트. scores_lookup: 문서별 {phrase: score} (없으면 공란)."""
    # ① 문서별 1행
    doc_rows, long_rows = [], []
    for i in range(len(test)):
        pr = preds[i]
        marked = ["[O] " + p if normalize_phrase(p) in gold_norms[i] else p for p in pr]
        m5 = precision_recall_f1_at_k(pr, golds[i], 5)
        m10 = precision_recall_f1_at_k(pr, golds[i], 10)
        doc_rows.append({
            "id": ids[i], "title": titles[i], "gold": "; ".join(golds[i]),
            "gold_prmu": "".join(prmus[i]), "pred_top10": "; ".join(marked),
            "tp@10": m10["tp"], "F1@5": round(m5["f1"], 4), "R@10": round(m10["recall"], 4),
        })
        for rank, p in enumerate(pr, 1):
            tag = classify_prmu(p, "", doc_stems=stem_cache[i])
            long_rows.append({
                "id": ids[i], "title": titles[i], "rank": rank, "keyphrase": p,
                "type": "present" if tag == "P" else "absent", "prmu": tag,
                "score": round(float(scores_lookup[i].get(p, 0.0)), 4) if scores_lookup else "",
                "source": cand_sources[i].get(p, ""),
                "is_correct": int(normalize_phrase(p) in gold_norms[i]),
                "gold_all": "; ".join(golds[i]),
            })
    pd.DataFrame(doc_rows).to_csv(PRED / f"kp20k_test_{run}_full.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long_rows).to_csv(PRED / f"kp20k_test_{run}_full_keyphrases.csv", index=False, encoding="utf-8-sig")
    macro = evaluate_corpus(preds, golds, all_prmu=prmus, all_doc_texts=texts)
    log(f"✓ {run:<22} F1@5={macro['F1@5']:.4f} A-R@10={macro.get('absent_R@10', 0):.4f} "
        f"({label})")
    return {"run": run, "system": label, "F1@5": round(macro["F1@5"], 4),
            "F1@M": round(macro["F1@M"], 4),
            "present_F1@5": round(macro.get("present_F1@5", 0), 4),
            "absent_R@10": round(macro.get("absent_R@10", 0), 4),
            "recall_U": round(macro.get("recall_U", 0), 4),
            "nDCG@10": round(macro["nDCG@10"], 4)}


# ---------------------------------------------------------------- 실행
summary = []

# 베이스라인 (저장된 예측 그대로; score/source 없음)
BASELINES = [
    ("B0_tfidf", "TF-IDF", "B0_tfidf_full.jsonl"),
    ("B1_keybert", "KeyBERT", "B1_keybert_full.jsonl"),
    ("B2_keybert_mmr", "KeyBERT+MMR", "B2_keybert_mmr_full.jsonl"),
    ("B3_bart", "BART-base (full)", "B3_bart_beam5_full.jsonl"),
    ("B4_keybart", "KeyBART (full)", "B4_keybart_beam5_full.jsonl"),
]
for run, label, fn in BASELINES:
    path = PRED / fn
    if not path.exists():
        log(f"skip {run}: {fn} 없음"); continue
    rows = {r["id"]: r["pred"] for r in load_jsonl(path)}
    summary.append(export(run, label, [rows.get(i, [])[:TOP_K] for i in ids]))

# 임베딩 (MMR용)
from src.reranking import SemanticScorer

scorer = SemanticScorer(device="cuda")
log("후보 임베딩 계산")
embs = [scorer.encode(doc) if doc else None for doc in cand_phrases]

# P1: fusion, MMR 없음 / P2: fusion + MMR
fusion_lookup = [dict(zip(p, s)) for p, s in zip(cand_phrases, cand_fusion)]
for run, label, lam in [("P1_hybrid_fusion", "Hybrid fusion (no MMR)", None),
                        ("P2_hybrid_fusion_mmr", "Hybrid fusion + MMR (제안·정밀도)", MMR_LAMBDA)]:
    idx = mmr_or_topk(cand_fusion, embs, lam)
    preds = [[cand_phrases[d][j] for j in o] for d, o in enumerate(idx)]
    summary.append(export(run, label, preds, fusion_lookup))

# P5/P6: pairwise CE (무가중 / PRMU-가중) × (MMR 없음 / MMR)
for tag, base_run, label in [("unweighted", "P5_pairwise", "Pairwise CE 무가중"),
                             ("prmu_w", "P6_pairwise_prmu", "Pairwise CE PRMU-가중 (제안·커버리지)")]:
    ck = OUT / "checkpoints" / f"reranker_{tag}"
    if not (ck / "config.json").exists():
        log(f"skip {base_run}: 체크포인트 없음"); continue
    sc = ce_scores(f"reranker_{tag}")
    lookup = [dict(zip(p, s)) for p, s in zip(cand_phrases, sc)]
    for lam, suffix, note in [(None, "", ""), (MMR_LAMBDA, "_mmr", " + MMR")]:
        idx = mmr_or_topk(sc, embs, lam)
        preds = [[cand_phrases[d][j] for j in o] for d, o in enumerate(idx)]
        summary.append(export(base_run + suffix, label + note, preds, lookup))

df = pd.DataFrame(summary)
df.to_csv(OUT / "metrics" / "all_systems_summary.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print()
print(df.to_string(index=False))
log(f"완료 — CSV {2 * len(summary)}개 + 요약표 저장")
print("\n참고: P4(absent 부스트)는 최적 boost=0이 선택되어 P2와 동일한 출력이므로 별도 파일을 만들지 않았습니다.")
