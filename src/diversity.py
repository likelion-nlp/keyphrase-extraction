"""다양성 모듈: MMR 중복 제거 (플랜 7.7절).

MMR(c) = λ·Rel(c, doc) − (1−λ)·max_{s∈선택됨} Sim(c, s)

- λ 높음(→1): 관련도 우선 (중복 허용 가능성 증가)
- λ 낮음(→0): 다양성 우선 (관련도 손실 가능성 증가)
- 권장 탐색: λ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, 기본 0.7
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def mmr_select(
    cand_embeddings: np.ndarray,
    relevance_scores: Sequence[float],
    top_k: int = 10,
    lambda_: float = 0.7,
) -> list[int]:
    """MMR로 후보 인덱스를 순서대로 선택한다.

    Args:
        cand_embeddings: (N, d) 후보 임베딩 (정규화 여부 무관 — 내부 정규화).
        relevance_scores: 후보별 관련도 점수 (reranker/score fusion 결과, 0~1 권장).
        top_k: 선택할 개수.
        lambda_: 관련도-다양성 균형 계수.

    Returns:
        선택된 후보 인덱스 리스트 (선택 순서 = 최종 순위).
    """
    n = cand_embeddings.shape[0]
    if n == 0:
        return []
    top_k = min(top_k, n)
    emb = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-12)
    sim = emb @ emb.T
    rel = np.asarray(relevance_scores, dtype=np.float64)

    selected: list[int] = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(selected)
    while len(selected) < top_k and remaining:
        rem = np.array(sorted(remaining))
        max_sim_to_selected = sim[np.ix_(rem, selected)].max(axis=1)
        mmr = lambda_ * rel[rem] - (1.0 - lambda_) * max_sim_to_selected
        pick = int(rem[np.argmax(mmr)])
        selected.append(pick)
        remaining.discard(pick)
    return selected


def mmr_select_candidates(
    candidates: list[dict[str, Any]],
    scorer,
    top_k: int = 10,
    lambda_: float = 0.7,
    score_key: str = "final_score",
) -> list[dict[str, Any]]:
    """score fusion/reranking을 마친 후보 레코드 리스트에 MMR을 적용한다.

    scorer: reranking.SemanticScorer (encode 메서드 필요, GPU).
    관련도는 후보 집합 내 min-max 정규화된 score_key 값을 사용한다.
    """
    if not candidates:
        return []
    phrases = [c["phrase"] for c in candidates]
    emb = scorer.encode(phrases)
    scores = [c.get(score_key, 0.0) for c in candidates]
    lo, hi = min(scores), max(scores)
    rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]
    order = mmr_select(emb, rel, top_k=top_k, lambda_=lambda_)
    out = []
    for rank, idx in enumerate(order, start=1):
        rec = dict(candidates[idx])
        rec["mmr_rank"] = rank
        out.append(rec)
    return out
