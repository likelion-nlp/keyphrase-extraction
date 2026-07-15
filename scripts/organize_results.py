"""GitHub 배포용 결과 정리 — outputs/predictions 의 불투명한 파일명을
모델명·기법이 드러나는 이름으로 results/ 폴더에 복사한다 (원본 비파괴).

파이프라인 순서대로 번호(00~41)를 붙여 폴더가 스스로 목차가 되게 한다.
    results/per_document/   ← 문서별 (full)   : 30_reranker_scibert.csv ...
    results/per_keyphrase/  ← 구절별 (keyphrases)
    results/metrics/        ← 통합 지표표
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions"
METRICS = ROOT / "outputs" / "metrics"
RESULTS = ROOT / "results"

# (정렬번호, 깨끗한 이름, 원본 run_id stem)  — 원본: kp20k_test_{stem}_full(.csv/_keyphrases.csv)
MAP = [
    ("00", "baseline_tfidf",          "B0_tfidf"),
    ("01", "baseline_keybert",        "B1_keybert"),
    ("02", "baseline_keybert_mmr",    "B2_keybert_mmr"),
    ("03", "baseline_bart_gen",       "B3_bart"),
    ("04", "baseline_keybart_gen",    "B4_keybart"),
    ("10", "hybrid_fusion",           "P1_hybrid_fusion"),
    ("11", "hybrid_fusion_mmr",       "P2_hybrid_fusion_mmr"),
    ("20", "pairwise_unweighted",     "P5_pairwise"),
    ("21", "pairwise_prmu_weighted",  "P6_pairwise_prmu"),
    ("30", "reranker_scibert",        "P7_scibert"),
    ("31", "reranker_specter2",       "P8"),
    ("32", "reranker_gte_modernbert", "P10"),
    ("33", "reranker_qwen3_0.6b",     "P11"),
    ("34", "reranker_embeddinggemma", "P12"),
    ("35", "reranker_cs_roberta",     "P15"),
    ("40", "ensemble_rrf_selected",   "P13_ensemble"),
    ("41", "ensemble_team_spec",      "P16_team_ensemble"),
]

# per_keyphrase(구절별, ~45MB/파일)도 전 모델 복사 (사용자 요청 — 전량 보존).
COPY_ALL_KEYPHRASE = True

METRIC_COPIES = [
    ("all_models_comparison.csv",       "grand_comparison_all_models.csv"),
    ("baselines_detailed.csv",          "results_full_dataset.csv"),
    ("pairwise_vs_baseline.csv",        "pairwise_vs_baseline_comparison.csv"),
    ("reference_free_scorecard.csv",    "scorecard_all_systems.csv"),
]


def main() -> None:
    doc_dir = RESULTS / "per_document"
    kp_dir = RESULTS / "per_keyphrase"
    met_dir = RESULTS / "metrics"
    for d in (doc_dir, kp_dir, met_dir):
        d.mkdir(parents=True, exist_ok=True)

    missing, copied = [], 0
    for num, clean, stem in MAP:
        src_full = PRED / f"kp20k_test_{stem}_full.csv"
        src_kp = PRED / f"kp20k_test_{stem}_full_keyphrases.csv"
        if src_full.exists():
            shutil.copy2(src_full, doc_dir / f"{num}_{clean}.csv"); copied += 1
        else:
            missing.append(src_full.name)
        if COPY_ALL_KEYPHRASE:
            if src_kp.exists():
                shutil.copy2(src_kp, kp_dir / f"{num}_{clean}.csv"); copied += 1
            else:
                missing.append(src_kp.name)
        print(f"  {num} {clean:<28} <- {stem}")

    for dst, src in METRIC_COPIES:
        s = METRICS / src
        if s.exists():
            shutil.copy2(s, met_dir / dst); copied += 1
        else:
            missing.append(src)

    print(f"\n복사 완료: {copied}개 파일 → {RESULTS}")
    if missing:
        print(f"누락(원본 없음): {missing}")


if __name__ == "__main__":
    main()
