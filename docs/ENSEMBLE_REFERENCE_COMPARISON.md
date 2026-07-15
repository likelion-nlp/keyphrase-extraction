# 앙상블을 기준으로 한 P7 vs 베이스라인 비교

> 노트북: [`../notebooks/05_ensemble_reference_comparison.ipynb`](../notebooks/05_ensemble_reference_comparison.ipynb)
> 요약 데이터: [`../results/metrics/ensemble_reference_comparison_summary.csv`](../results/metrics/ensemble_reference_comparison_summary.csv)

## 채점표를 구성한 앙상블 모델 (4개)

채점표는 아래 **4개 리랭커의 RRF 합의**로 만든다. 각각 혈통이 다른 인코더이며, 모두 P7과 동일한
레시피(aux 3피처 + PRMU 가중 pairwise)로 학습했다 — 인코더만 다르다.

| 멤버 | 모델 (HuggingFace) | 혈통·특화 | 풀링 |
|---|---|---|---|
| P8 | `allenai/specter2_base` | 과학 논문 인용 신호 (110M) | CLS |
| P10 | `Alibaba-NLP/gte-reranker-modernbert-base` | 리랭킹 전용 (2026, 149M) | CLS |
| P11 | `Qwen/Qwen3-Reranker-0.6B` | LLM 디코더 혈통 (596M) | last-token |
| P12 | `google/embeddinggemma-300m` | Gemma 임베딩 (303M) | mean |

> 정의 위치: `scripts/encoder_tournament.py`의 `ROSTER` dict. 팀 지정 5개 중 DeBERTa-v3(P9)는
> bf16 발산으로 탈락해 4개로 구성됐다. **P7은 채점표에 넣지 않는다** — 채점 대상이 P7이므로
> 포함하면 순환이 된다.

## 앙상블 = 채점표 (정답 기준)

위 4-모델 앙상블("LLM 앙상블")은 **gold-F1을 두고 겨루는 참가 모델이 아니라
다른 모델을 채점하는 채점표**다. 여러 리랭커의 합의(consensus) 랭킹을 정답으로 두고 물었다:

> **어떤 단일 모델이 앙상블의 판단을 가장 잘 재현하는가?** — P7(우리 Hybrid 리랭커) vs KeyBART(생성 베이스라인).

즉 앙상블은 단순한 "챔피언 산출물"이 아니라, **P7과 베이스라인을 견주는 잣대**로 쓰였다.

## 방법

문서마다 세 모델의 top-10 키프레이즈 랭킹을 비교한다 (앙상블 / P7 / KeyBART).

1. **1:1 매칭**: 앙상블 top-10과 후보 모델 top-10 사이에 SentenceTransformer 임베딩 유사도
   행렬을 만들고 `linear_sum_assignment`(헝가리안)로 총 유사도 최대 매칭. 유사도 < 0.75면 미매칭
   (같은 개념이 아니라고 판단).
2. **nDCG@10**: 앙상블 순위를 관련도로 사용 — 앙상블 1위 키워드의 관련도가 가장 높다
   ($rel_i = 11 - \text{rank}_{앙상블}(i)$). 후보가 앙상블 상위어를 자기 상위에 두면 높은 점수.
3. **RBO@10** (Rank-Biased Overlap, $p=0.9$): 상위일수록 크게 가중한 순위 겹침.
4. **Wilcoxon signed-rank test**: 문서별 P7 점수 vs KeyBART 점수의 차이가 유의한지.

## 결과 — P7이 앙상블 합의를 압도적으로 잘 재현

| 지표 | P7 (Hybrid) | KeyBART (베이스라인) | P7 우세 문서 비율 | Wilcoxon p |
|---|---:|---:|---:|---:|
| **nDCG@10** | **0.898** | 0.784 | **85.2%** | ≈ 0 |
| **RBO@10** | **0.452** | 0.325 | **85.1%** | ≈ 0 |

- P7은 앙상블 순서를 nDCG 0.90 수준으로 따라가고, **문서의 85%에서 KeyBART보다 앙상블에 가깝다.**
- 두 지표 모두 Wilcoxon p ≈ 0 — 차이가 통계적으로 확고하다 (우연이 아니다).

## 해석 — P7 = 앙상블의 배포용 단일 프록시

앙상블은 4개 모델을 돌려야 하지만, **P7 단일 모델이 앙상블의 판단을 사실상 재현**한다.
반면 생성 베이스라인(KeyBART)은 앙상블 합의와 크게 어긋난다. 이는 두 가지를 뒷받침한다:

1. **P7 방법론(인코더 + aux + PRMU 가중 pairwise)의 타당성** — 여러 인코더를 앙상블해도
   그 합의는 결국 P7 하나가 내는 랭킹과 가깝다 (인코더 토너먼트의 "인코더 불감" 결론과 정합:
   [`ENCODER_TOURNAMENT_STATUS.md`](ENCODER_TOURNAMENT_STATUS.md)).
2. **배포 선택지**: 앙상블 품질이 필요하되 4모델 추론이 부담되면 **P7 단일 모델로 대체 가능**.

> 정리: 앙상블은 gold-F1 경쟁 모델이 아니라 **채점표(정답 기준)**다. 이 채점표로 채점하면
> **P7이 앙상블 합의를 가장 잘 재현하는 단일 모델**로서 베이스라인을 압도한다.
> (앙상블 자체의 gold-F1 수치 P16 0.3011 등은 `grand_comparison`에 데이터로만 남아 있다.)

## 재현

노트북 05는 팀의 랭킹 비교 시트(`Keyphrase_최종`에서 내보낸 CSV: 문서별 LLM/Hybrid/KeyBART top-10)를
입력으로 받는다. 해당 CSV는 저장소에 포함하지 않았으나, 노트북에는 원 실행 결과가 임베드돼 있고
요약 수치는 위 CSV로 제공된다.
