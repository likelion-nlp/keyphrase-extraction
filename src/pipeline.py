"""GERD 하이브리드 파이프라인: Generate–Extract–Rerank–Diversify (플랜 7절).

Title+Abstract
  ├─ Extractive 후보 (KeyBERT)      ─┐
  └─ Generative 후보 (KeyBART/BART) ─┴→ 후보 통합·정규화
                                        → Score fusion (또는 Cross-Encoder)
                                        → MMR 중복 제거
                                        → Ranked Present+Absent Top-K
"""
from __future__ import annotations

from typing import Any

from .data import build_source, plain_doc_text
from .diversity import mmr_select_candidates
from .extraction import KeyBertExtractor
from .generation import generate_keyphrases
from .preprocessing import classify_prmu, clean_candidates, stem_tokens
from .reranking import CrossEncoderReranker, SemanticScorer, fuse_scores, merge_candidates


class KeyphrasePipeline:
    """추출기 + 생성기 + 재정렬 + MMR을 하나로 묶은 최종 시스템."""

    def __init__(
        self,
        generator_model=None,
        generator_tokenizer=None,
        extractor: KeyBertExtractor | None = None,
        scorer: SemanticScorer | None = None,
        cross_encoder: CrossEncoderReranker | None = None,
        fusion_weights: dict[str, float] | None = None,
        decoding_strategy: str = "beam10",
        max_source_length: int = 384,
    ):
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.extractor = extractor
        self.scorer = scorer or SemanticScorer()
        self.cross_encoder = cross_encoder
        self.fusion_weights = fusion_weights
        self.decoding_strategy = decoding_strategy
        self.max_source_length = max_source_length

    def predict_keyphrases(
        self,
        title: str,
        abstract: str,
        top_k: int = 5,
        n_extract: int = 30,
        n_generate: int = 30,
        mmr_lambda: float = 0.7,
        use_mmr: bool = True,
    ) -> list[dict[str, Any]]:
        """플랜 Notebook 11의 최종 인터페이스.

        Returns:
            [{"phrase", "score", "source", "type", ...}, ...] 관련도순 Top-K.
        """
        doc_text = plain_doc_text(title, abstract)
        doc_stems = stem_tokens(doc_text)

        # E: Extract
        ext_scored: list[tuple[str, float]] = []
        if self.extractor is not None:
            ext_scored = self.extractor.extract(doc_text, top_n=n_extract)
            kept = set(clean_candidates([p for p, _ in ext_scored]))
            ext_scored = [(p, s) for p, s in ext_scored if p in kept] or ext_scored

        # G: Generate
        gen_scored: list[dict[str, Any]] = []
        if self.generator_model is not None:
            source = build_source(title, abstract)
            gen_results = generate_keyphrases(
                self.generator_model,
                self.generator_tokenizer,
                [source],
                strategy=self.decoding_strategy,
                max_source_length=self.max_source_length,
            )
            gen_scored = gen_results[0][:n_generate]

        # 통합·정규화
        candidates = merge_candidates(ext_scored, gen_scored)
        if not candidates:
            return []

        # R: Rerank (Cross-Encoder가 있으면 그것을, 없으면 score fusion)
        if self.cross_encoder is not None:
            ranked = self.cross_encoder.rerank(doc_text, candidates)
        else:
            ranked = fuse_scores(candidates, doc_text, title, self.scorer, self.fusion_weights)

        # D: Diversify
        if use_mmr:
            selected = mmr_select_candidates(ranked, self.scorer, top_k=top_k, lambda_=mmr_lambda)
        else:
            selected = ranked[:top_k]

        out = []
        for rec in selected:
            prmu = classify_prmu(rec["phrase"], "", doc_stems=doc_stems)
            out.append(
                {
                    "phrase": rec["phrase"],
                    "score": round(float(rec.get("final_score", 0.0)), 4),
                    "source": rec.get("sources", []),
                    "type": "present" if prmu == "P" else "absent",
                    "prmu": prmu,
                }
            )
        return out
