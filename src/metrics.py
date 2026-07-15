"""평가 모듈: exact-match F1@K, F1@M, MAP, nDCG, PRMU recall, 다양성, 의미 기반 지표.

핵심 원칙 (마스터 플랜 8절):
- 모든 모델이 이 모듈 하나로 평가된다.
- 예측이 K개보다 적으면 부족분을 오답으로 간주해 분모 K를 유지한다.
- 기본 보고는 문서별 지표의 macro average.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

from .preprocessing import (
    classify_prmu,
    normalize_phrase,
    stem_tokens,
    unique_normalized,
)

# ---------------------------------------------------------------------------
# 문서 단위 exact-match 지표
# ---------------------------------------------------------------------------

def precision_recall_f1_at_k(
    pred_phrases: Sequence[str], gold_phrases: Sequence[str], k: int
) -> dict[str, float]:
    """Top-K exact-match P/R/F1 (stemming 정규화 후 비교).

    예측이 K개 미만이면 부족분은 오답 취급 (분모 K 유지) — 플랜 8.2절.
    """
    pred = unique_normalized(list(pred_phrases))[:k]
    gold = set(unique_normalized(list(gold_phrases)))
    tp = len(set(pred) & gold)
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}


def f1_at_k(pred_phrases: Sequence[str], gold_phrases: Sequence[str], k: int) -> dict[str, float]:
    """precision_recall_f1_at_k의 별칭 (플랜 9절 골격과 동일 시그니처)."""
    return precision_recall_f1_at_k(pred_phrases, gold_phrases, k)


def f1_at_m(pred_phrases: Sequence[str], gold_phrases: Sequence[str]) -> dict[str, float]:
    """K = 해당 문서의 gold 키프레이즈 수(M)로 두는 F1@M."""
    gold = unique_normalized(list(gold_phrases))
    return precision_recall_f1_at_k(pred_phrases, gold_phrases, k=max(1, len(gold)))


def average_precision_at_k(
    pred_phrases: Sequence[str], gold_phrases: Sequence[str], k: int
) -> float:
    """AP@K (binary relevance). MAP@K는 문서별 AP@K의 평균."""
    pred = unique_normalized(list(pred_phrases))[:k]
    gold = set(unique_normalized(list(gold_phrases)))
    if not gold:
        return 0.0
    hits, score = 0, 0.0
    for i, p in enumerate(pred, start=1):
        if p in gold:
            hits += 1
            score += hits / i
    return score / min(len(gold), k)


def ndcg_at_k(pred_phrases: Sequence[str], gold_phrases: Sequence[str], k: int) -> float:
    """binary nDCG@K. 모든 gold의 relevance=1 가정 (플랜 8.5절 주의사항 참고)."""
    pred = unique_normalized(list(pred_phrases))[:k]
    gold = set(unique_normalized(list(gold_phrases)))
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(pred, start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# 다양성 지표 (플랜 8.6절)
# ---------------------------------------------------------------------------

def exact_duplicate_ratio(pred_phrases: Sequence[str]) -> float:
    """1 - unique/total. 소문자·공백 정리만 적용한 표면 중복."""
    if not pred_phrases:
        return 0.0
    surface = [" ".join(p.lower().split()) for p in pred_phrases]
    return 1.0 - len(set(surface)) / len(surface)


def stem_duplicate_ratio(pred_phrases: Sequence[str]) -> float:
    """stemming 정규화 후 중복 비율 (neural network vs neural networks 포착)."""
    if not pred_phrases:
        return 0.0
    norms = [normalize_phrase(p) for p in pred_phrases]
    norms = [n for n in norms if n]
    if not norms:
        return 0.0
    return 1.0 - len(set(norms)) / len(norms)


def semantic_redundancy(embeddings: np.ndarray) -> float:
    """예측 임베딩 쌍별 cosine similarity 평균 (대각 제외). 낮을수록 다양."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    mask = ~np.eye(n, dtype=bool)
    return float(sim[mask].mean())


# ---------------------------------------------------------------------------
# 의미 기반 F1 (플랜 8.7절 최소 구현)
# ---------------------------------------------------------------------------

def semantic_match_f1(
    pred_embeddings: np.ndarray,
    gold_embeddings: np.ndarray,
    threshold: float = 0.7,
) -> dict[str, float]:
    """SBERT 임베딩 기반 greedy 1:1 매칭 F1.

    cosine >= threshold인 (pred, gold) 쌍을 유사도 내림차순으로 greedy 매칭한다.
    exact match가 놓치는 동의어·표기 변형을 보완한다.
    """
    n_pred, n_gold = pred_embeddings.shape[0], gold_embeddings.shape[0]
    if n_pred == 0 or n_gold == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0}
    p = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-12)
    g = gold_embeddings / (np.linalg.norm(gold_embeddings, axis=1, keepdims=True) + 1e-12)
    sim = p @ g.T
    pairs = [
        (sim[i, j], i, j) for i in range(n_pred) for j in range(n_gold) if sim[i, j] >= threshold
    ]
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches += 1
    precision = matches / n_pred
    recall = matches / n_gold
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": matches}


# ---------------------------------------------------------------------------
# 코퍼스 수준 종합 평가 (macro average)
# ---------------------------------------------------------------------------

def evaluate_document(
    pred: Sequence[str],
    gold: Sequence[str],
    gold_prmu: Sequence[str] | None = None,
    doc_text: str | None = None,
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """문서 1건의 모든 exact 지표를 계산한다."""
    out: dict[str, float] = {}
    for k in ks:
        r = precision_recall_f1_at_k(pred, gold, k)
        out[f"P@{k}"] = r["precision"]
        out[f"R@{k}"] = r["recall"]
        out[f"F1@{k}"] = r["f1"]
        out[f"MAP@{k}"] = average_precision_at_k(pred, gold, k)
        out[f"nDCG@{k}"] = ndcg_at_k(pred, gold, k)
    out["F1@M"] = f1_at_m(pred, gold)["f1"]
    out["dup_ratio"] = stem_duplicate_ratio(pred)
    out["num_pred"] = float(len(pred))

    # Present/Absent 분리 (gold는 공식 prmu, 예측은 동일 stemming 규칙로 분류)
    if gold_prmu is not None and doc_text is not None:
        doc_stems = stem_tokens(doc_text)
        gold_present = [kp for kp, t in zip(gold, gold_prmu) if t == "P"]
        gold_absent = [kp for kp, t in zip(gold, gold_prmu) if t != "P"]
        pred_present, pred_absent = [], []
        for p in pred:
            tag = classify_prmu(p, "", doc_stems=doc_stems)
            (pred_present if tag == "P" else pred_absent).append(p)

        if gold_present:
            r5 = precision_recall_f1_at_k(pred_present, gold_present, 5)
            out["present_F1@5"] = r5["f1"]
            out["present_R@5"] = r5["recall"]
            out["present_F1@M"] = f1_at_m(pred_present, gold_present)["f1"]
        if gold_absent:
            out["absent_R@5"] = precision_recall_f1_at_k(pred_absent, gold_absent, 5)["recall"]
            out["absent_R@10"] = precision_recall_f1_at_k(pred_absent, gold_absent, 10)["recall"]
            out["absent_F1@5"] = precision_recall_f1_at_k(pred_absent, gold_absent, 5)["f1"]

        # PRMU 유형별 recall (전체 예측 기준, 플랜 8.4절)
        pred_norms = set(unique_normalized(list(pred)))
        type_total: Counter = Counter()
        type_hit: Counter = Counter()
        for kp, t in zip(gold, gold_prmu):
            type_total[t] += 1
            if normalize_phrase(kp) in pred_norms:
                type_hit[t] += 1
        for t in "PRMU":
            if type_total[t]:
                out[f"recall_{t}"] = type_hit[t] / type_total[t]
    return out


def evaluate_corpus(
    all_preds: Sequence[Sequence[str]],
    all_golds: Sequence[Sequence[str]],
    all_prmu: Sequence[Sequence[str]] | None = None,
    all_doc_texts: Sequence[str] | None = None,
    ks: Sequence[int] = (5, 10),
    return_per_doc: bool = False,
) -> dict[str, Any]:
    """코퍼스 전체 macro average 평가. 모든 실험 결과 보고에 이 함수를 사용한다."""
    assert len(all_preds) == len(all_golds), "pred/gold 문서 수 불일치"
    per_doc: list[dict[str, float]] = []
    for i in range(len(all_preds)):
        prmu = all_prmu[i] if all_prmu is not None else None
        doc_text = all_doc_texts[i] if all_doc_texts is not None else None
        per_doc.append(evaluate_document(all_preds[i], all_golds[i], prmu, doc_text, ks))
    keys = sorted({k for d in per_doc for k in d})
    macro = {}
    for key in keys:
        vals = [d[key] for d in per_doc if key in d]
        macro[key] = float(np.mean(vals)) if vals else 0.0
    macro["num_docs"] = len(per_doc)

    # Micro F1@5 (보조 지표, 플랜 8.8절)
    for k in ks:
        tp_total, pred_total, gold_total = 0, 0, 0
        for pred, gold in zip(all_preds, all_golds):
            p = unique_normalized(list(pred))[:k]
            g = set(unique_normalized(list(gold)))
            tp_total += len(set(p) & g)
            pred_total += k
            gold_total += len(g)
        micro_p = tp_total / pred_total if pred_total else 0.0
        micro_r = tp_total / gold_total if gold_total else 0.0
        macro[f"micro_F1@{k}"] = (
            2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0
        )
    if return_per_doc:
        return {"macro": macro, "per_doc": per_doc}
    return macro


def candidate_recall(
    all_candidates: Sequence[Sequence[str]],
    all_golds: Sequence[Sequence[str]],
    ks: Sequence[int] = (20, 50, 100),
) -> dict[str, float]:
    """후보 pool이 gold를 얼마나 포함하는지 (플랜 11.2절).

    reranker 성능의 상한선: 후보에 없는 정답은 어떤 ranker도 올릴 수 없다.
    """
    out = {}
    for k in ks:
        recalls = []
        for cands, gold in zip(all_candidates, all_golds):
            g = set(unique_normalized(list(gold)))
            if not g:
                continue
            c = set(unique_normalized(list(cands))[:k])
            recalls.append(len(c & g) / len(g))
        out[f"cand_recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
    return out
