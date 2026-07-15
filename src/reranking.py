"""재정렬 모듈: 점수 정규화·결합(score fusion)과 Cross-Encoder reranker.

플랜 7.5절: S(c) = w_gen·S_gen + w_ext·S_ext + w_sem·S_sem + w_title·S_title
- 모든 점수는 후보 집합 내에서 min-max 정규화 후 결합한다.
- reranker는 '중요도'가 아니라 '관련/비관련'을 배운다 (플랜 7.6절 해석 주의).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .preprocessing import normalize_phrase, stem_tokens

DEFAULT_WEIGHTS = {"gen": 0.35, "ext": 0.20, "sem": 0.35, "title": 0.10}


def minmax_normalize(values: Sequence[float | None]) -> list[float]:
    """None은 해당 소스가 만들지 않은 후보 → 정규화 후 0점 처리."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    if hi - lo < 1e-12:
        return [0.5 if v is not None else 0.0 for v in values]
    return [((v - lo) / (hi - lo)) if v is not None else 0.0 for v in values]


class SemanticScorer:
    """SBERT 임베딩으로 문서-후보 유사도, 후보 간 유사도를 계산한다 (GPU)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def doc_candidate_similarity(self, doc_text: str, candidates: Sequence[str]) -> np.ndarray:
        embs = self.encode([doc_text] + list(candidates))
        return embs[1:] @ embs[0]


def title_overlap_score(candidate: str, title: str) -> float:
    """title 토큰(stem)과의 겹침 비율. title 등장 전문용어에 보조 가점 (플랜 7.5절)."""
    cand_stems = set(stem_tokens(candidate))
    title_stems = set(stem_tokens(title))
    if not cand_stems or not title_stems:
        return 0.0
    return len(cand_stems & title_stems) / len(cand_stems)


def merge_candidates(
    ext_scored: Sequence[tuple[str, float]],
    gen_scored: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """추출·생성 후보를 정규형 기준으로 병합한다 (C = C_ext ∪ C_gen).

    반환 레코드: {"phrase", "sources", "ext_score", "gen_score"}
    같은 정규형이 양쪽에서 나오면 sources에 둘 다 기록된다 (하이브리드 근거 제공).
    """
    merged: dict[str, dict[str, Any]] = {}
    for phrase, score in ext_scored:
        key = normalize_phrase(phrase)
        if not key:
            continue
        rec = merged.setdefault(
            key, {"phrase": phrase, "sources": [], "ext_score": None, "gen_score": None}
        )
        rec["sources"].append("extractive") if "extractive" not in rec["sources"] else None
        rec["ext_score"] = max(score, rec["ext_score"]) if rec["ext_score"] is not None else score
    for item in gen_scored:
        phrase, score = item["text"], item.get("gen_score")
        key = normalize_phrase(phrase)
        if not key:
            continue
        rec = merged.setdefault(
            key, {"phrase": phrase, "sources": [], "ext_score": None, "gen_score": None}
        )
        rec["sources"].append("generative") if "generative" not in rec["sources"] else None
        if score is not None:
            rec["gen_score"] = max(score, rec["gen_score"]) if rec["gen_score"] is not None else score
    return list(merged.values())


def fuse_scores(
    candidates: list[dict[str, Any]],
    doc_text: str,
    title: str,
    scorer: SemanticScorer,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """min-max 정규화 후 가중 결합 점수를 계산해 내림차순 정렬한다.

    입력 candidates는 merge_candidates의 출력 형식이어야 한다.
    """
    if not candidates:
        return []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    phrases = [c["phrase"] for c in candidates]
    sem_raw = scorer.doc_candidate_similarity(doc_text, phrases)
    gen_norm = minmax_normalize([c["gen_score"] for c in candidates])
    ext_norm = minmax_normalize([c["ext_score"] for c in candidates])
    sem_norm = minmax_normalize(list(sem_raw))
    title_scores = [title_overlap_score(p, title) for p in phrases]
    for i, c in enumerate(candidates):
        c["sem_score"] = float(sem_raw[i])
        c["title_score"] = float(title_scores[i])
        # float() 필수: numpy 스칼라가 섞이면 JSON 저장/재로딩 시 타입이 흔들린다
        c["final_score"] = float(
            w["gen"] * gen_norm[i]
            + w["ext"] * ext_norm[i]
            + w["sem"] * sem_norm[i]
            + w["title"] * title_scores[i]
        )
    return sorted(candidates, key=lambda c: -c["final_score"])


# ---------------------------------------------------------------------------
# Cross-Encoder reranker (확장 실험, 플랜 7.6절)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """문서-후보 쌍의 관련도를 직접 채점하는 Cross-Encoder (GPU).

    기본 모델은 MS-MARCO로 학습된 것이므로 zero-shot으로도 쓸 수 있고,
    KP20k 정답/negative로 fine-tuning하면 learned reranker가 된다.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 384,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, doc_text: str, candidates: Sequence[str], batch_size: int = 64) -> np.ndarray:
        pairs = [(doc_text, c) for c in candidates]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size))

    def rerank(
        self, doc_text: str, candidates: list[dict[str, Any]], batch_size: int = 64
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scores = self.score(doc_text, [c["phrase"] for c in candidates], batch_size)
        norm = minmax_normalize(list(scores))
        for c, raw, s in zip(candidates, scores, norm):
            c["ce_score"] = float(raw)
            c["final_score"] = float(s)
        return sorted(candidates, key=lambda c: -c["final_score"])


def build_reranker_training_pairs(
    rows: Sequence[dict],
    candidate_lists: Sequence[Sequence[str]],
    num_random_negatives: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Cross-Encoder 학습 데이터 구성 (플랜 7.6절).

    - positive: gold keyphrase
    - hard negative: 모델이 만들었지만 gold와 불일치한 후보
    - random negative: 다른 문서의 gold keyphrase
    """
    import random

    rng = random.Random(seed)
    all_gold_flat = [kp for row in rows for kp in row["keyphrases"]]
    pairs = []
    for row, cands in zip(rows, candidate_lists):
        doc_text = f"{row['title']}. {row['abstract']}"
        gold_norms = {normalize_phrase(kp) for kp in row["keyphrases"]}
        for kp in row["keyphrases"]:
            pairs.append({"doc": doc_text, "candidate": kp, "label": 1})
        hard = [c for c in cands if normalize_phrase(c) not in gold_norms]
        for c in hard[: len(row["keyphrases"])]:
            pairs.append({"doc": doc_text, "candidate": c, "label": 0})
        for _ in range(num_random_negatives):
            neg = rng.choice(all_gold_flat)
            if normalize_phrase(neg) not in gold_norms:
                pairs.append({"doc": doc_text, "candidate": neg, "label": 0})
    rng.shuffle(pairs)
    return pairs
