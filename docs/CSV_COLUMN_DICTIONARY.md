# CSV 컬럼 사전 — 산출물 파일별 지표 의미 해설

> 실제 저장된 CSV들의 컬럼을 기준으로 정리 (2026-07-14 헤더 실측).
> 수식이 필요하면 [METRICS.md](METRICS.md) 참조 — 이 문서는 "이 숫자를 어떻게 읽어야 하는가"에 집중한다.

---

## 1. `outputs/metrics/experiments.csv` — 실험 원장 (모든 실험이 여기 누적)

한 행 = 실험 1회. `run_id`가 같으면 최신 결과로 덮어쓴다.

### 식별·설정 컬럼

| 컬럼 | 의미 |
|---|---|
| `run_id` | 실험 고유 이름 (B0~B4 베이스라인, P1~P15 제안 계열, N1 외부 노트북). `_full` 접미사 = 전체 데이터(test 20,000) 조건 |
| `model` | 시스템의 사람용 이름 |
| `input` | 입력 구성 — 전부 `T+A`(제목+초록). ablation 대비용 |
| `num_docs` | 평가 문서 수 (2,000 = dev 조건 / 20,000 = full). **다르면 직접 비교 금지** |
| `decoder` | 생성 후보를 만든 디코딩 (Beam5/Beam10 등) |
| `reranker` | 랭킹 방식 (ScoreFusion / CE-* / SciBERT+aux / RRF ensemble …) |
| `MMR` | 다양화 적용 여부와 λ |
| `seed`, `train_subset`, `epochs`, `logged_at` | 재현 정보 |
| `tfidf_fit`, `boost` | 해당 실험에만 쓰인 부가 설정 (없으면 공란) |

### 품질 컬럼 (전부 0~1, 높을수록 좋음 — `dup_ratio`만 낮을수록 좋음)

| 컬럼 | 읽는 법 |
|---|---|
| `F1@5`, `F1@10` | **주 지표.** 상위 5/10개와 정답의 겹침 (정밀도·재현율 조화평균). 예측이 K개보다 적어도 분모는 K 고정 — "적게 내서 점수 올리기" 차단 |
| `F1@M` | K를 그 문서의 정답 개수로 둔 F1 — 문서마다 정답 수(평균 5.3개)가 다른 것을 보정 |
| `present_F1@5` | **원문에 등장하는** 정답만 대상으로 한 F1@5 — 추출 능력 |
| `absent_R@10` | **원문에 없는** 정답(R/M/U) 중 상위 10개 안에서 회수한 비율 — 생성 능력. 절대값이 작은 게 정상(최고 0.05대), 시스템 간 배율로 비교 |
| `recall_P/R/M/U` | 정답 유형별 회수율. P(원문 연속 등장)→U(단어 자체가 없음) 순으로 어려움. **주의: K 잘림 없이 예측 리스트 전체 기준이라 리스트 길이가 다른 시스템 간 절대 비교는 부적합** — 공정 비교는 F1@K/R@K로 |
| `MAP@10` | 정답을 만날 때마다의 정밀도 평균 — "헛걸음 없이 얼마나 빨리 정답을 다 모으나". 순위 한두 칸에 민감 |
| `nDCG@10` | 위 순위일수록 큰 보상(로그 감쇠)을 이상적 배치 대비 정규화 — "좋은 자리에 정답을 앉혔나". ⚠️ 정답에 중요도 등급이 없어 binary — "정답이 위에 있나"까지만 해석 |
| `dup_ratio` | 어간(stem) 기준 중복 출력 비율. 0이면 표면 중복 없음 |
| `num_pred` | 문서당 평균 출력 개수 (top-10 시스템은 10.0) |

---

## 2. `results_full_dataset.csv` — 본 실험 메인 표 (7개 시스템 × 전 지표)

원장에서 full 조건만 추려 P/R을 보강한 보고서용 표.

| 추가 컬럼 | 의미 |
|---|---|
| `P@5`, `P@10` | 상위 K개 중 정답 비율 (정밀도) |
| `R@5`, `R@10` | 전체 정답 중 상위 K개가 회수한 비율 (재현율). **모든 시스템이 같은 K로 잘려 유형별 recall보다 공정한 비교** |
| `absent_R@5` | absent 정답의 상위 5개 기준 재현율 |
| `train_data` | 학습 데이터 규모 표기 ("530,809 (full)" / "-(무학습)") |
| `avg_preds_per_doc` | num_pred와 동일 |

---

## 3. `pairwise_vs_baseline_comparison.csv` — P2 기준 상대 비교

| 특수 컬럼 | 읽는 법 |
|---|---|
| `설명` | 각 시스템이 뭔지 한 줄 (한글) |
| `F1@5_vs_P2` | **P2(fusion+MMR) 대비 증감.** 음수 = 정확도 희생 |
| `absent_R@10_vs_P2`, `recall_U_vs_P2` | 양수 = absent 개선. **이 두 열과 F1 열의 부호가 반대인 것이 pairwise 실험의 핵심 trade-off** |

---

## 4. `scorecard_all_systems.csv` — 무정답(reference-free) 채점 (신규 논문 50편)

정답이 없는 문서에서 키프레이즈의 "쓸모"를 재는 대체 지표들.

| 컬럼 | 의미와 해석 |
|---|---|
| `구절검색 MRR` | 키프레이즈 **1개**를 검색어로 2만 편 코퍼스에서 원 논문을 찾을 때 순위의 역수 평균. 1.0 = 항상 1등. **색인어로서의 변별력** |
| `구절 Hit@10` | 그 검색에서 원 논문이 10등 안에 든 비율 |
| `구절 평균순위` | 낮을수록 좋음 (2만 편 중 몇 등인가) |
| `특이성(IDF)` | 구절 토큰의 역문서빈도 평균 — **높으면 그 논문만의 표현, 낮으면 아무 논문에나 붙는 일반어** (`performance` 류). KP20k 10만 편 기준 |
| `absent 비율` | 출력 중 원문에 없는 구절의 비율 |
| `근거성(absent)` | absent 구절과 초록 문장의 최대 의미 유사도 평균 — 낮으면 내용과 동떨어진 생성 |
| `환각의심률` | absent 구절 중 근거성 < 0.30인 비율 — **낮을수록 안전.** 생성 모델의 리스크 지표 |
| `stem중복률` / `의미중복도` | 표면 중복 / 임베딩 쌍별 코사인 평균(표면이 달라도 뜻이 겹치는 중복). 의미중복도는 너무 낮아도 "서로 무관한 출력"일 수 있어 관련성과 함께 봄 |

⚠️ 이 표의 순위는 exact-F1 순위와 **역전**된다 (KeyBERT가 검색 1위) — 지표가 재는 것이 다르기 때문이며, 그 역전 자체가 본 연구의 발견이다.

---

## 5. `predictions/kp20k_test_{run}_full.csv` — 문서별 예측 (20,000행)

| 컬럼 | 의미 |
|---|---|
| `gold`, `gold_prmu` | 정답 목록과 유형 문자열 (예: `PPRMU` = 5개 정답의 각 유형) |
| `pred_top10` | 예측 10개 — **정답 맞힌 것 앞에 `[O]`** 표시 (눈 검수용) |
| `tp@10` | 10개 중 정답 개수 |
| `F1@5`, `R@10` | 이 문서 한 편의 점수 (표 전체 평균 = 원장의 macro 값) |

## 6. `predictions/kp20k_test_{run}_full_keyphrases.csv` — 구절별 예측 (200,000행, 분석용)

| 컬럼 | 의미 |
|---|---|
| `rank` | 그 문서 안 최종 순위 (1~10) |
| `type` / `prmu` | present/absent 이분 / P·R·M·U 세분 (예측 구절을 원문 대조로 분류) |
| `score` | 그 시스템의 랭킹 점수. ⚠️ **시스템마다 스케일이 다름** (fusion 0~1, CE·SciBERT는 logit) — 파일 간 직접 비교 금지, 파일 안 상대 비교만 |
| `source` | 후보 출처 — `extractive` / `generative` / `extractive+generative`(**합의** — 적중률이 단독의 1.5~50배로 가장 신뢰) |
| `is_correct` | 1 = 정답 (피벗 집계용) |
| `gold_all` | 그 문서의 정답 전체 |

## 7. `newpapers_*_summary.csv` — 외부 논문(정답 없음) 예측 요약

| 컬럼 | 의미 |
|---|---|
| `keyphrases_top10` | 최종 출력 (정오 표시 없음 — gold가 없으므로) |
| `n_present`/`n_absent` | 출력 구성비 |
| `doc_pred_cosine` | 출력과 문서의 의미 유사도 — gold 없이 보는 관련성 proxy |
| `stem_dup_ratio`/`semantic_redundancy` | §4와 동일 |

## 8. 보조 파일

| 파일 | 컬럼 요지 |
|---|---|
| `results_candidate_recall_full.csv` | `cand_recall@20/50/100` — 후보 pool 상위 K가 정답을 덮는 비율. **어떤 랭커도 못 넘는 상한선.** `candidate_source` = KeyBERT/Generator/Union(인터리브) |
| `results_mmr_sweep_full.csv` | `mmr_setting`(no_mmr, λ값)별 F1과 중복도 — λ 선택 근거. F1 유지 + 중복 감소 지점이 최적 |

---

## 통독 가이드 — 두 잣대와 파일 매핑

| 잣대 | "좋다"의 의미 | 파일 | 1위 |
|---|---|---|---|
| ① exact-F1 | 저자 키워드 재현 | experiments / results_full_dataset | **P7**(메인 단일 모델) |
| ② reference-free | 새 문서에서의 쓸모 (특이성·무환각) | scorecard_all_systems | 용도별 상이 |
| ③ 채점표(앙상블) 재현 | 앙상블 채점표 랭킹을 얼마나 따르나 | ensemble_reference_comparison_summary | **P7** (vs 베이스라인) |

> 메인 비교는 베이스라인 vs P7. 토너먼트 단일 모델(P8~P15)과 앙상블은 **채점표 및 그 재료**라
> 메인 순위에서 제외한다. grand_comparison엔 이들 데이터가 남지만 경쟁 순위로는 다루지 않는다.

같은 시스템이 잣대마다 순위가 달라지는 것이 정상이며, **어느 CSV를 근거로 쓸지는 "무엇에 쓸 시스템인가"가 결정한다.**

> `ensemble_reference_comparison_summary.csv`: 앙상블 랭킹을 기준으로 P7·KeyBART의 nDCG@10/RBO@10
> 평균, P7 우세 문서 비율, Wilcoxon p. 해설: [`ENSEMBLE_REFERENCE_COMPARISON.md`](ENSEMBLE_REFERENCE_COMPARISON.md).
