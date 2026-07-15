"""KP20k 데이터 로딩·변환·감사(audit) 모듈.

공식 데이터셋: https://huggingface.co/datasets/taln-ls2n/kp20k
필드: id / title / abstract / keyphrases / prmu

주의: 이 저장소는 스크립트 기반(kp20k.py)이라 datasets>=3.0에서
`load_dataset("taln-ls2n/kp20k")`가 실패한다. 따라서 JSONL 파일을
hf_hub_download로 받아 `load_dataset("json", ...)`으로 읽는다.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .preprocessing import (
    ABSENT_TOKEN,
    ABSTRACT_TOKEN,
    KP_SEP,
    PRESENT_TOKEN,
    TITLE_TOKEN,
    classify_prmu,
    stem_tokens,
)

KP20K_REPO = "taln-ls2n/kp20k"
_SPLIT_FILES = {"train": "train.json", "validation": "validation.json", "test": "test.json"}


def load_kp20k(
    splits: list[str] | None = None,
    subset_sizes: dict[str, int] | None = None,
    seed: int = 42,
):
    """KP20k를 DatasetDict로 로드한다.

    Args:
        splits: 로드할 split 목록. None이면 train/validation/test 전부.
        subset_sizes: split별 샘플 수 제한 (예: {"train": 10000, "test": 2000}).
            지정 시 seed 고정 shuffle 후 앞에서 자른다 (단계적 확장 전략, 플랜 13절).
        seed: subset 샘플링 시드.

    Returns:
        datasets.DatasetDict
    """
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import hf_hub_download

    splits = splits or list(_SPLIT_FILES)
    data_files = {
        s: hf_hub_download(KP20K_REPO, _SPLIT_FILES[s], repo_type="dataset") for s in splits
    }
    ds = load_dataset("json", data_files=data_files)
    if subset_sizes:
        out = {}
        for s in splits:
            n = subset_sizes.get(s)
            if n and n < len(ds[s]):
                out[s] = ds[s].shuffle(seed=seed).select(range(n))
            else:
                out[s] = ds[s]
        ds = DatasetDict(out)
    return ds


# ---------------------------------------------------------------------------
# 입력/타깃 문자열 구성 (플랜 2.3 수정1, 4.4 문제2·3)
# ---------------------------------------------------------------------------

def build_source(title: str, abstract: str, input_mode: str = "title_abstract") -> str:
    """입력 인코딩. 기본은 `<title> {title} <abstract> {abstract}` (E4 ablation 지원)."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if input_mode == "title_abstract":
        return f"{TITLE_TOKEN} {title} {ABSTRACT_TOKEN} {abstract}"
    if input_mode == "abstract_only":
        return f"{ABSTRACT_TOKEN} {abstract}"
    if input_mode == "title_only":
        return f"{TITLE_TOKEN} {title}"
    raise ValueError(f"unknown input_mode: {input_mode}")


def plain_doc_text(title: str, abstract: str) -> str:
    """present 판정·임베딩용 순수 텍스트 (스페셜 토큰 없음)."""
    return f"{(title or '').strip()}. {(abstract or '').strip()}"


def split_present_absent_gold(keyphrases: list[str], prmu: list[str]) -> tuple[list[str], list[str]]:
    """공식 prmu 라벨로 gold를 (present=P, absent=R/M/U)로 나눈다 (엄격 기준)."""
    present = [kp for kp, tag in zip(keyphrases, prmu) if tag == "P"]
    absent = [kp for kp, tag in zip(keyphrases, prmu) if tag != "P"]
    return present, absent


def build_target(
    keyphrases: list[str],
    prmu: list[str] | None = None,
    target_format: str = "present_absent",
    shuffle_seed: int | None = None,
) -> str:
    """One2Seq 타깃 문자열을 만든다.

    - "present_absent": `<present> p1 <kp_sep> p2 <absent> a1 <kp_sep> a2` (권장)
    - "plain":          `kp1 <kp_sep> kp2 <kp_sep> kp3`
    - shuffle_seed 지정 시 phrase 순서를 섞는다 (E5 순서 편향 ablation).
    """
    kps = list(keyphrases)
    tags = list(prmu) if prmu is not None else None
    if shuffle_seed is not None:
        import random

        rng = random.Random(shuffle_seed)
        idx = list(range(len(kps)))
        rng.shuffle(idx)
        kps = [kps[i] for i in idx]
        if tags:
            tags = [tags[i] for i in idx]

    if target_format == "plain" or tags is None:
        return f" {KP_SEP} ".join(kps)
    if target_format == "present_absent":
        present = [kp for kp, tag in zip(kps, tags) if tag == "P"]
        absent = [kp for kp, tag in zip(kps, tags) if tag != "P"]
        parts = []
        if present:
            parts.append(f"{PRESENT_TOKEN} " + f" {KP_SEP} ".join(present))
        if absent:
            parts.append(f"{ABSENT_TOKEN} " + f" {KP_SEP} ".join(absent))
        return " ".join(parts) if parts else PRESENT_TOKEN
    raise ValueError(f"unknown target_format: {target_format}")


def build_example(
    row: dict,
    input_mode: str = "title_abstract",
    target_format: str = "present_absent",
    shuffle_seed: int | None = None,
) -> dict:
    """HF dataset row → {"source", "target"} (Seq2Seq 학습용)."""
    return {
        "source": build_source(row["title"], row["abstract"], input_mode),
        "target": build_target(row["keyphrases"], row.get("prmu"), target_format, shuffle_seed),
    }


# ---------------------------------------------------------------------------
# 데이터 감사 (Notebook 01)
# ---------------------------------------------------------------------------

def audit_split(ds, split_name: str, max_docs: int | None = None) -> dict[str, Any]:
    """플랜 Notebook 01의 검사 항목을 계산한다."""
    import itertools

    n = len(ds) if max_docs is None else min(max_docs, len(ds))
    ids: Counter = Counter()
    titles: Counter = Counter()
    stats = {
        "split": split_name,
        "num_docs": len(ds),
        "audited_docs": n,
        "null_or_empty_title": 0,
        "null_or_empty_abstract": 0,
        "empty_keyphrases": 0,
        "len_mismatch_keyphrases_prmu": 0,
        "empty_string_inside_keyphrases": 0,
        "duplicate_keyphrases_within_doc": 0,
        "non_ascii_docs": 0,
    }
    kp_counts = []
    for row in itertools.islice(ds, n):
        title, abstract = row.get("title"), row.get("abstract")
        kps, prmu = row.get("keyphrases") or [], row.get("prmu") or []
        ids[row.get("id")] += 1
        if title:
            titles[title.strip().lower()] += 1
        if not title or not str(title).strip():
            stats["null_or_empty_title"] += 1
        if not abstract or not str(abstract).strip():
            stats["null_or_empty_abstract"] += 1
        if not kps:
            stats["empty_keyphrases"] += 1
        if len(kps) != len(prmu):
            stats["len_mismatch_keyphrases_prmu"] += 1
        if any(not kp or not kp.strip() for kp in kps):
            stats["empty_string_inside_keyphrases"] += 1
        if len(set(kp.strip().lower() for kp in kps)) < len(kps):
            stats["duplicate_keyphrases_within_doc"] += 1
        text = f"{title} {abstract}"
        if any(ord(c) > 127 for c in text[:2000]):
            stats["non_ascii_docs"] += 1
        kp_counts.append(len(kps))
    stats["duplicate_ids"] = sum(1 for c in ids.values() if c > 1)
    stats["duplicate_titles"] = sum(1 for c in titles.values() if c > 1)
    stats["avg_keyphrases_per_doc"] = round(sum(kp_counts) / max(1, len(kp_counts)), 3)
    return stats


def official_prmu_stats(ds, max_docs: int | None = None) -> dict[str, float]:
    """공식 prmu 필드 기반 P/R/M/U 비율(%)을 계산한다."""
    import itertools

    counter: Counter = Counter()
    n = len(ds) if max_docs is None else min(max_docs, len(ds))
    for row in itertools.islice(ds, n):
        counter.update(row["prmu"])
    total = sum(counter.values()) or 1
    return {tag: round(100.0 * counter.get(tag, 0) / total, 2) for tag in "PRMU"}


def custom_prmu_stats(ds, max_docs: int | None = None) -> tuple[dict[str, float], list[dict]]:
    """팀 자체 classify_prmu 함수로 재계산한 통계와, 공식 라벨과 불일치한 사례를 반환한다.

    플랜 2.3 수정3: 공식 통계와 차이가 나는 원인을 추적하기 위한 함수.
    """
    import itertools

    counter: Counter = Counter()
    disagreements: list[dict] = []
    n = len(ds) if max_docs is None else min(max_docs, len(ds))
    for row in itertools.islice(ds, n):
        doc_stems = stem_tokens(plain_doc_text(row["title"], row["abstract"]))
        for kp, official in zip(row["keyphrases"], row["prmu"]):
            ours = classify_prmu(kp, "", doc_stems=doc_stems)
            counter[ours] += 1
            if ours != official and len(disagreements) < 200:
                disagreements.append(
                    {"id": row.get("id"), "keyphrase": kp, "official": official, "ours": ours}
                )
    total = sum(counter.values()) or 1
    ratios = {tag: round(100.0 * counter.get(tag, 0) / total, 2) for tag in "PRMU"}
    return ratios, disagreements
