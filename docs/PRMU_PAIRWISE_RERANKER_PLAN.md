# PRMU-가중 Pairwise Reranker 파이프라인 계획

> 작성일: 2026-07-13 · 전제: 전체 데이터셋 실험 완료 상태 (`results_full_dataset.csv`)
> 목표: **학습형 pairwise 랭킹**으로 하이브리드 파이프라인의 선택(selection) 단계를 교체하고,
> **PRMU 타입 가중 loss**가 absent(특히 U) recall을 exact F1 손실 없이 끌어올리는지 검증한다.

---

## 0. 왜 이 실험인가 — 실측 근거

전체 데이터 실험에서 확인된 두 가지 사실이 출발점이다.

| 실측 | 수치 | 의미 |
|---|---|---|
| 후보 pool의 U 커버리지 | recall_U ≈ **0.058** (P1 후보 전체) | 생성 단계의 상한선 |
| 최종 top-10에 살아남은 U | recall_U ≈ **0.019** (P2) | 선택 단계에서 **2/3 유실** |

현재 선택기(score fusion)는 문서 유사도(w_sem=0.35)를 크게 반영하는데, absent 후보는
정의상 원문과 표면이 달라 유사도 점수가 낮게 나온다 → **absent를 체계적으로 밀어낸다.**
이 유실분(0.019→0.058)이 학습형 reranker가 노릴 수 있는 headroom이다.

**주의(개념)**: PRMU는 중요도 척도가 아니라 표면 중첩 분류다 (P→R→M→U는 추출 난이도 순서).
여기서 가중치는 "U가 더 중요해서"가 아니라 **"U는 더 어렵고 현재 선택기가 구조적으로 불리하게
취급하므로 학습 신호를 보정한다"**는 난이도 보정 논리다.

---

## 1. 문제 정식화

문서 $x$와 후보 집합 $C = C_{ext} \cup C_{gen}$이 주어질 때, 스코어 함수 $s_\theta(x, c)$를 학습한다.

**Pairwise margin ranking loss + PRMU 타입 가중:**

$$L = \sum_{(c^+,\, c^-)} w_{type(c^+)} \cdot \max\big(0,\ m - (s_\theta(x, c^+) - s_\theta(x, c^-))\big)$$

- $c^+$: gold keyphrase (공식 `prmu` 필드로 타입 태깅)
- $c^-$: negative (아래 §3.2)
- $w$: 타입별 가중치, $w_U \ge w_M \ge w_R \ge w_P$ (기본 후보 3:2:1.5:1, validation에서 탐색)
- $m$: margin (기본 1.0)

**Pairwise를 쓰는 이유**: KP20k gold는 "1등, 2등" 점수가 없는 **집합**이다.
절대 점수 회귀(pointwise)보다 "gold > non-gold" 상대 비교가 라벨 구조와 맞는다.
Listwise는 구현 복잡도 대비 이득이 불확실해 확장 항목으로 둔다.

---

## 2. 전체 파이프라인

```text
                        [기존 — 재사용]                            [신규 — 이번 실험]

Title+Abstract ──► KeyBERT top30 ─┐
                                  ├─► merge_candidates ──► ① AbsentQuota (비학습 baseline)
               ──► KeyBART_full   │                        ② PairwiseReranker (무가중)
                   beam10 pool  ──┘                        ③ PairwiseReranker (PRMU 가중)
                                                                     │
                                                           MMR (λ는 validation 재선택)
                                                                     │
                                                           Top-K + 평가 (동일 evaluator)
```

①은 ②③의 대조군: "규칙 기반 개입만으로 얼마나 회복되는가"를 먼저 재고,
학습형이 그 이상을 하는지 본다.

---

## 3. 단계별 계획

### Phase 0 — Absent 쿼터 baseline (비학습, ~30분)

저장된 `fused_candidates_full.jsonl`을 그대로 사용. 재학습·재생성 없음.

- **방법 A (쿼터)**: top-K 선택 시 최소 $m_a$개는 absent-분류 후보 중 최고점에서 강제 선발
- **방법 B (부스트)**: absent-분류 후보의 fusion 점수에 $+\alpha$ 보정 후 통상 선택
- 탐색: $m_a \in \{1, 2, 3\}$, $\alpha \in \{0.05, 0.1, 0.2\}$ — **validation 후보로 선택**, test로 보고
- 산출물: `outputs/metrics/absent_quota_tradeoff.csv` (F1@5 vs absent_R@10 trade-off 곡선)

### Phase 1 — Reranker 학습 데이터 구축 (~1시간, GPU)

**누수 방지 원칙 (마스터 플랜 7.6)**: 생성기는 train split로만 학습됐다.
→ reranker 학습 문서는 **validation split**에서 뽑는다 (생성기가 못 본 문서).

| 항목 | 설계 |
|---|---|
| 문서 | validation에서 10,000건 (λ 선택에 쓴 500건과 겹쳐도 무방 — 둘 다 개발용) |
| 후보 | KeyBART_full beam10 (batch 4, 증분 저장 재사용) + KeyBERT top30 → merge |
| Positive | 해당 문서의 gold 전체, 공식 `prmu`로 타입 태깅 |
| Hard negative | 후보 중 gold와 정규형 불일치인 것 (문서당 gold 수만큼) |
| Random negative | 다른 문서의 gold에서 문서당 2개 |
| 쌍 구성 | 같은 문서 안에서 (positive, negative) 조합 — 문서당 ~15쌍, 총 ~15만 쌍 |
| 저장 | `outputs/reranker/train_pairs.jsonl` — {doc, pos, pos_type, neg} |

기존 코드 재사용: `src/reranking.build_reranker_training_pairs`를 타입 태깅 지원으로 확장.

### Phase 2 — 학습 (~30분/run, GPU)

| 항목 | 설계 |
|---|---|
| 인코더 | `cross-encoder/ms-marco-MiniLM-L-6-v2` (기본) / `allenai/scibert_scivocab_uncased` (비교, 시간 시) |
| 입력 | `[CLS] title+abstract [SEP] candidate` (max 384) |
| 헤드 | 스칼라 점수 1개 (기존 CrossEncoder 구조 그대로) |
| Loss | margin ranking (m=1.0), 같은 배치에 (pos, neg) 쌍을 나란히 넣어 차이 계산 |
| 가중치 조건 | ② $w=(1,1,1,1)$ 무가중 / ③ $w=(1,1.5,2,3)$ 기본 / 그리드 $(1,2,3,5)$, $(1,1,2,2)$ |
| 옵티마이저 | AdamW, lr 2e-5, 1~2 epoch, batch 32쌍, bf16 |
| 체크포인트 | validation holdout(학습 미사용 1,000건)의 F1@5 기준 best |
| 산출물 | `outputs/checkpoints/reranker_{unweighted,prmu_w}/` |

구현 메모: sentence-transformers의 CrossEncoder.fit은 BCE 계열이라 margin ranking에
맞지 않음 → `AutoModelForSequenceClassification(num_labels=1)` + 커스텀 학습 루프
(~80줄, `scripts/train_reranker.py`)로 작성. 추론은 기존 `CrossEncoderReranker` 재사용.

### Phase 3 — 평가·Ablation (~1시간, GPU)

test 20,000건의 저장된 `fused_candidates_full.jsonl` 후보를 각 조건으로 재랭킹.
**모든 조건에서 후보·evaluator 동일** — 차이는 선택기뿐.

| 조건 | run_id |
|---|---|
| 기존 fusion (대조군, 이미 있음) | `P1/P2_hybrid_*_full` |
| Phase 0 쿼터/부스트 최적 | `P4_absent_quota_full` |
| Pairwise 무가중 | `P5_pairwise_full` |
| Pairwise PRMU-가중 | `P6_pairwise_prmu_full` |
| (각각) +MMR λ 재선택 | `*_mmr` |

**판정 지표** (전부 기존 `experiments.csv`에 자동 누적):

1. 주장 검증: `absent_R@10`, `recall_U`, `recall_M` — ③ > ② > fusion 이면 가설 성립
2. 비용 확인: `F1@5`, `present_F1@5` — 하락 폭이 오차 수준이어야 "공짜 개선"
3. 랭킹 품질: `MAP@10`, `nDCG@10`
4. Trade-off 곡선: 가중치 세기별 (F1@5, absent_R@10) 산점도 → 보고서 그림

### Phase 4 — 통합·문서화 (~30분)

- 최고 조건을 `KeyphrasePipeline`의 reranker 슬롯에 연결 (`configs/hybrid.yaml`에 옵션 추가)
- 노트북 09(reranking)에 결과 섹션 추가, `results_full_dataset.csv` 갱신
- 실패해도 보고 가치 있음: "PRMU 가중은 selection 병목(5.8% 상한) 안에서만 작동한다"는
  부정적 결과 + candidate recall 분석이 남는다

---

## 4. 일정·자원 (RTX 5080 기준)

| Phase | 내용 | 예상 시간 |
|---|---|---|
| 0 | absent 쿼터 baseline | 0.5h |
| 1 | validation 10k 후보 생성 + 쌍 구축 | 1.0h |
| 2 | 학습 ②③ (+가중 그리드 2회) | 1.5h |
| 3 | test 재랭킹 4조건 + MMR + 평가 | 1.0h |
| 4 | 통합·문서화 | 0.5h |
| **합계** | | **~4.5h** (백그라운드 실행 가능) |

주의: 생성 배치 4 유지 + 증분 저장 (VRAM 공유 환경 WDDM spill 대응 — 기존 `stage_hybrid` 방식).

---

## 5. 성공 기준과 리스크

**성공 기준** (test 20,000, 기존 P2 대비):

- [ ] absent_R@10: 0.022 → **0.035+** (쿼터) / **0.040+** (PRMU-가중)
- [ ] recall_U: 0.019 → **0.035+**
- [ ] F1@5 하락 ≤ 0.005 (0.2560 → 0.251 이상 유지)
- [ ] 무가중 대비 가중 조건의 absent 지표 우위 (ablation 핵심)

**리스크와 대응**:

| 리스크 | 대응 |
|---|---|
| 상한선이 낮음 (pool U 5.8%) | 개선 폭을 상한 대비 %로 보고: "유실분의 x%를 회수" |
| U positive가 희소해 가중해도 신호 부족 | absent 문서 oversampling 병행 |
| cross-encoder가 fusion보다 전반적으로 약함 (10k 실험에서 zero-shot이 열세였음) | fine-tuning이 전제 — zero-shot 결과와 혼동 금지 |
| 가중치가 hallucination 성향 후보까지 올림 | 오류 분석에서 hallucination proxy(U-예측 & 문서 cos<0.25) 비율 추적 |

---

## 6. 산출물 요약

```text
scripts/train_reranker.py            # Phase 1~2: 쌍 구축 + pairwise 학습
scripts/eval_reranker.py             # Phase 0, 3: 쿼터/재랭킹/ablation
outputs/reranker/train_pairs.jsonl
outputs/checkpoints/reranker_unweighted/ , reranker_prmu_w/
outputs/metrics/absent_quota_tradeoff.csv
outputs/metrics/experiments.csv      # P4~P6 행 자동 누적 (기존 체계)
docs/PRMU_PAIRWISE_RERANKER_PLAN.md  # 본 문서
```

## 7. 연구 스토리 한 줄

> 후보 pool은 U 정답의 5.8%를 이미 담고 있지만 표면 유사도 기반 선택이 그 2/3를 버린다.
> 우리는 PRMU 타입을 난이도 가중으로 쓰는 pairwise reranker로 이 유실을 회수하고,
> exact F1을 유지하면서 absent recall을 개선할 수 있는지를 ablation으로 검증한다.
