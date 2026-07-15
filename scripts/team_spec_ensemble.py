"""P16 — 팀 사양 앙상블 (팀원 요청 스펙 그대로).

스펙: 지정 5개 모델 SPECTER2(P8) / DeBERTa-v3(P9) / gte-reranker-ModernBERT(P10) /
      Qwen3-Reranker-0.6B(P11) / EmbeddingGemma(P12)가 각자 keyphrase 10개 추출
      → 합집합에서 중복 제거 → RRF(1/(60+rank))로 리랭킹 → 최종 top-10.

P13(검증 기반 조합 선택, 후보 풀 전체 RRF)과의 차이:
  - 멤버가 팀 지정 5개로 고정 (조합 탐색 없음)
  - 최종 후보를 "각 멤버 top-10의 합집합"으로 제한 (풀 전체가 아님)
  ※ P9는 bf16 발산으로 체크포인트가 없어 4개로 실행된다 (로그에 기록).

실행: python scripts/team_spec_ensemble.py   (저장된 scores/*.npz 재사용 — GPU 불필요)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

from encoder_tournament import SCORES, evaluate_and_record, load_test_assets, log

TEAM_MEMBERS = ["P8", "P9", "P10", "P11", "P12"]  # 팀 지정 순서
TOP_PER_MEMBER = 10


def main() -> None:
    test, texts, recs_list, _, sources_map = load_test_assets()

    member_scores: dict[str, list[np.ndarray]] = {}
    for m in TEAM_MEMBERS:
        f = SCORES / f"{m}.npz"
        if not f.exists():
            log(f"멤버 {m}: 점수 파일 없음 (학습 실패) — 제외하고 진행")
            continue
        d = np.load(f)
        member_scores[m] = np.split(d["scores"], np.cumsum(d["lens"])[:-1])
    log(f"앙상블 멤버 {len(member_scores)}/{len(TEAM_MEMBERS)}: {list(member_scores)}")

    scores_docs = []
    union_sizes = []
    for i, recs in enumerate(recs_list):
        n = len(recs)
        if n == 0:
            scores_docs.append([])
            continue
        rrf = np.zeros(n)
        union: set[int] = set()
        for sc in member_scores.values():
            order = np.argsort(-sc[i])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            rrf += 1.0 / (60 + ranks)
            union.update(order[:TOP_PER_MEMBER].tolist())
        union_sizes.append(len(union))
        final = np.full(n, -1.0)  # 합집합 밖 후보는 선택 불가
        idx = sorted(union)
        final[idx] = rrf[idx]
        scores_docs.append(final.tolist())

    log(f"문서당 합집합 후보 평균 {np.mean(union_sizes):.1f}개 (각 멤버 top-{TOP_PER_MEMBER} 중복 제거)")
    evaluate_and_record(
        "P16_team_ensemble",
        f"팀사양 RRF-union10({'+'.join(member_scores)})",
        "team-spec: member top10 union → RRF",
        test, texts, recs_list, scores_docs, sources_map)
    log("완료")


if __name__ == "__main__":
    main()
