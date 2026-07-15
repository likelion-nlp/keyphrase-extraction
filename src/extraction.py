"""추출형 베이스라인 모듈: TF-IDF n-gram 후보와 KeyBERT (GPU 임베딩).

- TF-IDF: 가장 단순한 통계 기반 하한선 (플랜 6.2절)
- KeyBERT: 의미 임베딩 기반 강한 추출 베이스라인 (플랜 6.4절)
둘 다 원문에 있는 표현만 뽑을 수 있으므로 U(Unseen)는 구조적으로 불가능하다.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# TF-IDF 베이스라인
# ---------------------------------------------------------------------------

class TfidfExtractor:
    """corpus DF 기반 TF-IDF n-gram 추출기.

    53만 문서 전체에 3-gram vocabulary를 만들면 메모리가 크므로,
    `fit_texts`는 train subset을 쓰고 `max_features`로 제한한다 (플랜 Notebook 04).
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 3),
        max_features: int = 300_000,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=True,
            lowercase=True,
        )
        self._feature_names: np.ndarray | None = None

    def fit(self, texts: Sequence[str]):
        self.vectorizer.fit(texts)
        self._feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self

    def extract(self, text: str, top_n: int = 20) -> list[tuple[str, float]]:
        """단일 문서에서 TF-IDF 상위 n-gram을 (phrase, score)로 반환."""
        assert self._feature_names is not None, "fit()을 먼저 호출하세요"
        vec = self.vectorizer.transform([text])
        if vec.nnz == 0:
            return []
        coo = vec.tocoo()
        order = np.argsort(-coo.data)[:top_n]
        return [(str(self._feature_names[coo.col[i]]), float(coo.data[i])) for i in order]

    def extract_batch(self, texts: Sequence[str], top_n: int = 20) -> list[list[tuple[str, float]]]:
        assert self._feature_names is not None, "fit()을 먼저 호출하세요"
        mat = self.vectorizer.transform(texts).tocsr()
        out = []
        for i in range(mat.shape[0]):
            row = mat.getrow(i).tocoo()
            order = np.argsort(-row.data)[:top_n]
            out.append(
                [(str(self._feature_names[row.col[j]]), float(row.data[j])) for j in order]
            )
        return out


# ---------------------------------------------------------------------------
# KeyBERT 베이스라인 (GPU)
# ---------------------------------------------------------------------------

class KeyBertExtractor:
    """KeyBERT 래퍼. SentenceTransformer를 GPU에 올려 배치 처리한다."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        import torch
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_model = SentenceTransformer(model_name, device=device)
        self.model = KeyBERT(model=self.st_model)
        self.device = device

    def extract(
        self,
        text: str,
        top_n: int = 10,
        ngram_range: tuple[int, int] = (1, 3),
        use_mmr: bool = False,
        diversity: float = 0.5,
    ) -> list[tuple[str, float]]:
        return self.model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
        )

    def extract_batch(
        self,
        texts: Sequence[str],
        top_n: int = 10,
        ngram_range: tuple[int, int] = (1, 3),
        use_mmr: bool = False,
        diversity: float = 0.5,
    ) -> list[list[tuple[str, float]]]:
        """여러 문서를 한 번에 처리 (KeyBERT는 list 입력 시 내부 배치 임베딩 사용)."""
        results = self.model.extract_keywords(
            list(texts),
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
        )
        # 단일 문서 입력이면 list[tuple]이 오므로 통일
        if results and isinstance(results[0], tuple):
            results = [results]
        return results


def candidates_only(scored: list[tuple[str, float]]) -> list[str]:
    return [p for p, _ in scored]
