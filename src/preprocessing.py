"""전처리·정규화 모듈: 모든 모델과 평가가 공유하는 단일 규칙.

마스터 플랜의 핵심 원칙:
- 모든 모델이 동일한 normalize/stemming 규칙을 사용해야 공정한 비교가 된다.
- 하이픈 포함 단어(`graph-based`)는 쪼개지 않는다 (공식 데이터 카드의 spaCy 커스텀 규칙과 정렬).
- PRMU 재분류는 (1) 공식 라벨 검증, (2) "예측" 키프레이즈 분류에만 사용한다.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from functools import lru_cache

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# 하이픈으로 연결된 영숫자 시퀀스를 한 토큰으로 유지한다.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

KP_SEP = "<kp_sep>"
PRESENT_TOKEN = "<present>"
ABSENT_TOKEN = "<absent>"
TITLE_TOKEN = "<title>"
ABSTRACT_TOKEN = "<abstract>"

SPECIAL_TOKENS = [TITLE_TOKEN, ABSTRACT_TOKEN, PRESENT_TOKEN, ABSENT_TOKEN, KP_SEP]


@lru_cache(maxsize=200_000)
def stem_token(token: str) -> str:
    return _stemmer.stem(token)


def tokenize(text: str) -> list[str]:
    """소문자화 후 영숫자(하이픈 포함) 토큰만 추출한다."""
    return TOKEN_RE.findall(text.lower())


def stem_tokens(text: str) -> list[str]:
    return [stem_token(tok) for tok in tokenize(text)]


def normalize_phrase(phrase: str) -> str:
    """평가·중복제거에 사용하는 정규형: lowercase → 토큰화 → Porter stem → 공백 join."""
    if not phrase:
        return ""
    phrase = phrase.lower().strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return " ".join(stem_token(tok) for tok in TOKEN_RE.findall(phrase))


def unique_normalized(phrases: list[str]) -> list[str]:
    """정규형 기준 중복 제거 후 정규형 리스트를 순서 유지하며 반환한다."""
    seen: OrderedDict[str, str] = OrderedDict()
    for phrase in phrases:
        norm = normalize_phrase(phrase)
        if norm and norm not in seen:
            seen[norm] = phrase.strip()
    return list(seen.keys())


def dedupe_keep_original(phrases: list[str]) -> list[str]:
    """정규형 기준 중복을 제거하되 원본 표기를 유지한다 (표시용)."""
    seen: OrderedDict[str, str] = OrderedDict()
    for phrase in phrases:
        norm = normalize_phrase(phrase)
        if norm and norm not in seen:
            seen[norm] = phrase.strip()
    return list(seen.values())


def _contains_contiguous(doc_stems: list[str], phrase_stems: list[str]) -> bool:
    """phrase_stems가 doc_stems에 같은 순서로 연속 등장하는지 확인한다."""
    n, m = len(doc_stems), len(phrase_stems)
    if m == 0 or m > n:
        return False
    first = phrase_stems[0]
    for i in range(n - m + 1):
        if doc_stems[i] == first and doc_stems[i : i + m] == phrase_stems:
            return True
    return False


def is_present(phrase: str, doc_text: str, doc_stems: list[str] | None = None) -> bool:
    """예측 키프레이즈가 title+abstract에 연속(같은 순서로) 등장하는지 판정한다.

    Present/Absent 분리 평가에서 '예측' 쪽 분류에 사용한다.
    """
    if doc_stems is None:
        doc_stems = stem_tokens(doc_text)
    return _contains_contiguous(doc_stems, stem_tokens(phrase))


def classify_prmu(phrase: str, doc_text: str, doc_stems: list[str] | None = None) -> str:
    """키프레이즈를 P/R/M/U로 분류한다 (Boudin & Gallina, NAACL 2021 정의).

    - P (Present):   구성 stem이 문서에 같은 순서로 연속 등장
    - R (Reordered): 구성 stem이 모두 문서에 있으나 연속·동일 순서가 아님
    - M (Mixed):     일부 stem만 문서에 존재
    - U (Unseen):    어떤 stem도 문서에 없음

    주의: 기본 실험에서는 데이터셋이 제공하는 공식 `prmu` 필드를 사용하고,
    이 함수는 (1) 공식 라벨 재현 검증, (2) 모델 '예측'의 분류에만 쓴다.
    """
    if doc_stems is None:
        doc_stems = stem_tokens(doc_text)
    phrase_stems = stem_tokens(phrase)
    if not phrase_stems:
        return "U"
    if _contains_contiguous(doc_stems, phrase_stems):
        return "P"
    doc_set = set(doc_stems)
    in_doc = [s in doc_set for s in phrase_stems]
    if all(in_doc):
        return "R"
    if any(in_doc):
        return "M"
    return "U"


def split_present_absent(
    phrases: list[str], doc_text: str, doc_stems: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """예측 리스트를 (present, absent)로 나눈다. 순서는 유지된다."""
    if doc_stems is None:
        doc_stems = stem_tokens(doc_text)
    present, absent = [], []
    for p in phrases:
        (present if is_present(p, doc_text, doc_stems) else absent).append(p)
    return present, absent


# ---------------------------------------------------------------------------
# 생성 출력 파싱 (마스터 플랜 7.4절 후보 정규화 파이프라인)
# ---------------------------------------------------------------------------

_SPECIAL_RE = re.compile(
    "|".join(re.escape(t) for t in [PRESENT_TOKEN, ABSENT_TOKEN, TITLE_TOKEN, ABSTRACT_TOKEN])
)


def parse_generated_sequence(text: str, fallback_seps: tuple[str, ...] = (";", ",")) -> list[str]:
    """모델이 생성한 원시 시퀀스를 phrase 리스트로 분해한다.

    `<present>`/`<absent>` 경계 토큰은 제거하고 `<kp_sep>` 기준으로 나눈다.
    `<kp_sep>`가 전혀 없으면(스페셜 토큰 미학습 모델 등) 세미콜론/쉼표로 fallback.
    """
    if not text:
        return []
    text = _SPECIAL_RE.sub(f" {KP_SEP} ", text)
    if KP_SEP in text:
        parts = text.split(KP_SEP)
    else:
        parts = [text]
        for sep in fallback_seps:
            if sep in text:
                parts = text.split(sep)
                break
    return [p.strip() for p in parts if p.strip()]


def clean_candidates(
    candidates: list[str],
    min_words: int = 1,
    max_words: int = 5,
    max_chars: int = 100,
    source_text: str | None = None,
) -> list[str]:
    """후보 정규화 파이프라인 (플랜 7.4절). 원본 표기를 유지한 채 필터링한다.

    1. 공백 정리 / 특수 토큰 제거
    2. 빈 문자열·<unk> 포함 출력 제거
    3. 단어 수 필터 (기본 1~5 단어)
    4. 입력 전체 복사 등 비정상적으로 긴 출력 제거
    5. stem 정규형 기준 중복 제거
    """
    out: list[str] = []
    for cand in candidates:
        if not cand:
            continue
        cand = _SPECIAL_RE.sub(" ", cand).replace(KP_SEP, " ")
        cand = re.sub(r"\s+", " ", cand).strip(" .,;:")
        if not cand or "<unk>" in cand.lower():
            continue
        if len(cand) > max_chars:
            continue
        n_words = len(cand.split())
        if not (min_words <= n_words <= max_words):
            continue
        if source_text and len(cand) > 0.8 * len(source_text):
            continue
        out.append(cand)
    return dedupe_keep_original(out)
