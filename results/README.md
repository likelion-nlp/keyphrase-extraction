# results/ — 모델 예측 결과와 지표

KP20k test **20,000편**에 대한 전 모델의 예측과 집계 지표. 파일명이 곧 기법이고,
앞 번호는 파이프라인 발전 순서다. 모델별 코드·기법 매핑은 [`../MODEL_CATALOG.md`](../MODEL_CATALOG.md).

## 폴더

| 폴더 | 단위 | 수록 | 주요 열 |
|---|---|---|---|
| `predictions_per_document/` | 문서 1개 = 1행 (20,000행) | 전 모델 17개 | id, title, gold, pred_top10(`[O]`=정답), F1@5, R@10 |
| `predictions_per_keyphrase/` | 예측 구절 1개 = 1행 | 전 모델 17개 (~708MB) | rank, keyphrase, type, prmu, score, source, is_correct |
| `metrics/` | 집계 지표표·원장 | — | 전 모델 비교 |

열 상세: [`../docs/CSV_COLUMN_DICTIONARY.md`](../docs/CSV_COLUMN_DICTIONARY.md).

## 번호 = 발전 단계

| 번호 | 단계 | 모델 |
|---|---|---|
| `00~04` | 베이스라인 | tfidf, keybert, keybert_mmr, bart_gen, keybart_gen |
| `10~11` | Hybrid 융합 (비학습) | hybrid_fusion, **hybrid_fusion_mmr** |
| `20~21` | Pairwise 리랭커 | pairwise_unweighted, pairwise_prmu_weighted |
| `30` | **메인 단일 모델** | **reranker_scibert (P7)** |
| `31~35` | 채점표 재료 (앙상블 멤버 — 메인 비교 제외) | specter2, gte_modernbert, qwen3_0.6b, embeddinggemma, cs_roberta |
| `40~41` | **채점표 (앙상블, 정답 기준)** | rrf_selected, **team_spec** |

> 메인 모델 비교는 **베이스라인(00~21) + P7(30)**. `31~35`(토너먼트 단일 모델)와 `40~41`(앙상블)은
> **채점표 및 그 재료**로, gold-F1 경쟁이 아니라 P7·베이스라인을 채점하는 기준이다.

## 지표 핵심 파일 (metrics/)

- `grand_comparison_all_models.csv` — **전 17개 모델 통합 비교표** (여기부터 보면 됨)
- `experiments.csv` — 모든 실험 원장 (run_id 기준 누적)
- `ensemble_selection.json` — 앙상블 조합 탐색 기록
- `author_patterns.json` — 저자 키프레이즈 선정 패턴 통계

## 용량

`predictions_per_keyphrase/`는 총 ~708MB(파일당 45MB)로 일반 git에 포함된다.
저장소가 다소 커서 클론이 느릴 수 있다. `predictions_per_document/`는 파일당 ~7.6MB.
