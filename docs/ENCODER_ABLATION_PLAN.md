# 인코더 토너먼트 계획 — P7 방법론 × 특화 인코더 4종 (P8~P11)

> 작성일: 2026-07-14 · 전제: P7(SciBERT+aux) 완료, F1@5 0.2850 현 챔피언
> 질문: **P7의 성능은 SciBERT라서인가, 방법론(aux+PRMU-가중 pairwise) 덕인가?**
> 방법: 레시피는 P7 그대로 고정, **인코더만** 성격이 다른 특화 모델 4종으로 교체해
> 도메인 특화 / 검색 특화 / 현대 아키텍처 / LLM 계열의 효과를 분리한다.

---

## 0. 통제 변수 — "인코더 외에는 전부 P7과 동일"

| 고정되는 것 | 값 |
|---|---|
| 후보 pool | KeyBERT top30 ∪ keybart_full beam10 (`fused_candidates_full.jsonl` 재사용) |
| 학습 데이터 | validation 10,000편, 학습쌍 ~105,000개 (P7과 동일 seed·동일 쌍 구성) |
| aux 피처 | gen_score/5.0, has_gen, is_present (3개) |
| 헤드 | Linear(hidden+3, 128) → ReLU → Dropout(0.2) → Linear(128,1) |
| 손실 | margin ranking (m=0.3) + PRMU 가중 1 : 1.5 : 2 : 2.5 |
| 학습 | AdamW lr 2e-5, batch 16, 2 epochs, max_len 256, bf16 |
| 출력 | **top-10** (K=10, MMR 없음 — P7 실측에서 MMR이 해로움 확인) |
| 평가 | test 20,000, 동일 evaluator, `experiments.csv` 누적 |

이 통제 덕에 P7~P11 다섯 결과의 차이는 **인코더(+풀링)에만** 귀속된다.

---

## 1. 인코더 로스터 — 서로 다른 "특화 축" 4종 (2026-07 웹 검증 반영판)

> 2026-07 기준 리랭커 벤치마크 확인 결과를 반영해 초안에서 P10·P11을 상향 교체했다.
> 확인된 사실: ModernBERT 계열 리랭커(gte-reranker-modernbert-base)가 최상위권을
> 8배 작은 크기로 달성, LLM 혈통에서는 Qwen3-Reranker 시리즈가 표준.
> Cohere/Voyage 등 상용 최상위는 API 전용이라 fine-tuning 실험에 사용 불가.

| run_id | 모델 | 크기 | 특화 축 | 풀링 | 선정 이유 |
|---|---|---|---|---|---|
| P7 (기준) | `allenai/scibert_scivocab_uncased` | 110M | 과학 어휘 (2019) | [CLS] | 현 챔피언 |
| **P8** | `allenai/specter2_base` | 110M | **과학 문서 표현** — 인용 관계로 학습 | [CLS] | "과학 도메인" 축의 신형. SciBERT의 상위 호환인지 검증 |
| **P9** | `microsoft/deberta-v3-base` | 86M(+임베딩) | **클래식 범용 최강** — disentangled attention | [CLS] | 도메인 특화 없이 순수 인코더 성능만 좋으면 되는지 검증 |
| **P10** | `Alibaba-NLP/gte-reranker-modernbert-base` | 149M | **2026 리랭킹 최상위권** — ModernBERT 백본 + 리랭커 사전학습 | [CLS] | 현세대 아키텍처 + 이미 랭킹 과제로 사전학습된 최강 소형 |
| **P11** | `Qwen/Qwen3-Reranker-0.6B` | 600M | **LLM 혈통 리랭커 SOTA** (2025~26 표준) | 마지막 토큰 / yes-logit | "Gemma 같은 LLM 계열" 축 — 랭킹 특화로는 Gemma보다 검증된 선택 |

> **Gemma 관련 결정**: 초안의 `google/embeddinggemma-300m`은 bi-encoder 임베딩 특화라
> cross-encoder 백본으로는 오프라벨 사용이 된다. "LLM 혈통" 축의 취지를 살리면서
> 랭킹 과제로 검증된 Qwen3-Reranker-0.6B로 교체. **Gemma 자체를 꼭 넣고 싶으면
> P12로 EmbeddingGemma를 추가 실행** (계획서 §2의 CLI 한 줄이면 됨).

> 로스터 설계 논리: P8은 "같은 축(과학)의 신형", P9는 "축 없는 클래식 강자",
> P10은 "현 벤치마크 1위권 소형", P11은 "LLM 혈통 SOTA 계열". 네 방향이 겹치지 않아
> 어떤 축이 이기든 해석이 깔끔하다.

### 공정성 주의 2가지
1. **크기 혼재** (86M~600M): "인코더 종류" 실험이지 동일 크기 실험이 아님 — 파라미터 수 병기
   + 성능-크기 산점도 필수.
2. **사전학습 이점 혼재**: P10·P11은 이미 리랭킹으로 사전학습된 모델이라 P7~P9(일반 인코더)보다
   유리한 출발점 — 이것도 "축"의 일부로 명시 보고 (랭킹 사전학습의 가치 자체가 측정 대상).

---

## 2. 구현 — 기존 스크립트 일반화 (새 코드 최소)

`scripts/scibert_hybrid_ranker.py` → `--encoder` / `--pooling` 인자를 추가해 일반화:

```text
python scripts/scibert_hybrid_ranker.py --stage all --encoder allenai/specter2_base                  --run-id P8
python scripts/scibert_hybrid_ranker.py --stage all --encoder microsoft/deberta-v3-base              --run-id P9
python scripts/scibert_hybrid_ranker.py --stage all --encoder Alibaba-NLP/gte-reranker-modernbert-base --run-id P10
python scripts/scibert_hybrid_ranker.py --stage all --encoder Qwen/Qwen3-Reranker-0.6B               --run-id P11 --pooling last
# (선택) Gemma 축 추가:
python scripts/scibert_hybrid_ranker.py --stage all --encoder google/embeddinggemma-300m             --run-id P12 --pooling mean
```

수정 포인트 4곳:
1. `ENCODER` 상수 → CLI 인자화, 체크포인트 디렉터리를 `reranker_{run_id}`로
2. **풀링 어댑터**: `[CLS]` 기본 / `--pooling mean`(마스크 가중 평균, Gemma류) /
   `--pooling last`(마지막 유효 토큰, Qwen3 디코더 혈통 표준)
3. hidden size 자동 감지 (이미 `config.hidden_size` 사용 중 — 확인만)
4. P11(600M)은 학습 시 batch 8 + accumulation 2로 시작 (effective 16 유지),
   시퀀스 분류 헤드가 이미 있는 모델(P10)은 헤드를 우리 MLP로 교체하지 않고
   **백본만 추출**해 동일 구조 적용 (공정성)

**Phase 0 스모크(필수)**: 각 모델을 32쌍 × 5 step만 돌려 로드·풀링·VRAM 확인 후 본 실행.
(EmbeddingGemma는 tokenizer/gating 요구사항이 다를 수 있어 여기서 걸러낸다)

---

## 3. 실행 계획과 비용 (RTX 5080 16GB)

P7 실측: 학습 21분 (110M, 10.5만 쌍 2 epochs) + test 20,000 스코어링·평가 11분.

| Phase | 내용 | 예상 |
|---|---|---|
| 0 | 4종 로드 스모크 (5 step) — 게이트/풀링/VRAM 확인 | 20분 |
| 1 | P8 (110M) 학습+평가 | ~35분 |
| 2 | P9 (86M) 〃 | ~30분 |
| 3 | P10 (149M) 〃 | ~45분 |
| 4 | P11 (600M) 〃 — batch 8 + accum 2, bf16 | ~2시간 |
| 5 | 통합표·Rank CI·CSV 내보내기 | 30분 |
| **6** | **P13 최종 앙상블** (아래 §4.5) — 점수 재사용이라 산수만 | **30분** |
| **합계** | | **약 5~5.5시간** (백그라운드 순차 실행) |

**앙상블 대비 구현 메모**: Phase 1~4의 eval 단계에서 **전 후보 점수를
`outputs/reranker/scores/{run_id}.npz`로 저장**한다 (top-10만이 아니라 후보 73만 개 전부).
P7 점수도 동일 형식으로 1회 재채점(11분)해 저장 → Phase 6은 저장된 점수의 산술 결합만.

VRAM: 최대 P11 600M — bf16 가중치 1.2GB + AdamW 상태 ~4.8GB + 활성값(batch 8×256) ≈ 9~11GB.
16GB에서 가능하나 여유가 적으므로 **다른 GPU 앱 종료 후 실행** (WDDM spill 교훈).
OOM 시 batch 4 + accum 4로 자동 강등 (effective batch 유지).

---

## 4. 평가와 산출물

### 4.1 1차 판정 — KP20k exact (자동 누적)
- `experiments.csv`에 P8~P11 행 추가 → F1@5/@10/@M, present/absent, PRMU recall, MAP, nDCG
- **Rank CI** (Holm 보정, 2차 노트북 방식 재사용): P7~P11 다섯 랭커의 순위 신뢰구간
  — "P7과 P8이 통계적으로 구분되는가"까지 답한다

### 4.2 예측 CSV (기존 포맷 그대로)
- `kp20k_test_{P8..P11}_full.csv` (문서별) + `_keyphrases.csv` (구절별, rank/score/source/is_correct)
- rank별 적중률 곡선 5종 겹쳐 그리기 — "앞이 뾰족하고 뒤가 두꺼운" P7 패턴의 재현 여부

### 4.3 2차 판정 — 실용 지표 연결 (승자만)
- exact 상위 2개 인코더를 **신규 논문 스코어카드**(검색효용·특이성·환각률)에
  투입 — "exact 챔피언 ≠ 실용 챔피언" 역전 재점검

### 4.5 Phase 6 — P13 최종 앙상블

**멤버 풀**: P7(기존 챔피언) + P8~P11 중 상위 + **P2 fusion(비학습, 오류 패턴이 가장 다름)**.
학습형 랭커끼리는 같은 학습쌍·같은 loss로 훈련돼 오류가 상관될 수 있으므로,
다양성 담당인 P2 포함이 핵심이다 (rank 곡선 상보성: fusion=뾰족한 머리, CE류=두꺼운 꼬리).

**결합 방식 2종**:
1. **RRF** (주력): $score(c) = \sum_{m \in members} 1/(60 + rank_m(c))$ — 점수 스케일 무관, 보정 불필요
2. 정규화 가중합 (보조): min-max 후 가중치 그리드

**선택 규약 (test 편향 방지)**:
- 멤버 조합·가중치는 **validation 잔여분 1,000편** (셔플 10,000~11,000번째 —
  랭커 학습 10,000편과 불겹침)에서 F1@5로 선택
- test 20,000에는 선택된 구성 **1개만** 적용해 `P13_ensemble_full`로 기록

**기대치 관리**: P7이 이미 fusion 신호를 aux 피처로 흡수했으므로 P7+P2 이득은 제한적일 수
있다. +0.005 미만이면 "P7이 앙상블 이득을 이미 내재화했다"는 결론 — 그 자체로 발견.

### 4.4 분석 축 (보고서 그림 3장)
1. **성능 vs 크기 산점도** — 특화가 크기를 이기는가
2. **도메인 축 대결**: SciBERT(2019 과학) vs SPECTER2(신형 과학) vs DeBERTa(무특화)
   → "과학 특화가 정말 유효한가, 그냥 좋은 인코더면 충분한가"
3. **absent 지표 재현성**: 다섯 인코더 모두에서 absent_R@10이 fusion보다 높으면
   → "absent 개선은 인코더가 아니라 **학습형 랭킹 방법론**의 성질" 확정 (P5/P6/P7에서 3회 재현됨)

---

## 5. 성공 기준

- [ ] 4종 전부 완주 + `experiments.csv` 기록 (부분 실패 시 해당 모델만 제외하고 보고)
- [ ] Rank CI에서 1위 그룹 식별 (단독 1위든 동률 그룹이든)
- [ ] P7 대비 ±0.005 이내면 "인코더 불감(insensitive)" 선언 → 방법론 기여로 결론
- [ ] 어느 인코더든 F1@5 ≥ 0.29 달성 시 신기록 → 최종 제안 시스템 교체 검토

## 6. 리스크

| 리스크 | 대응 |
|---|---|
| EmbeddingGemma 게이트/라이선스 동의 필요 가능성 | Phase 0에서 확인, 실패 시 대체: `BAAI/bge-base-en-v1.5` (검색 특화 축 유지) |
| ModernBERT가 구버전 transformers 요구와 충돌 | 우리는 5.13 — 지원됨. Phase 0에서 검증 |
| P11 mean pooling이 [CLS] 대비 불리한 비교라는 반론 | 풀링은 모델별 표준 관행을 따른 것임을 명시 (Gemma에 [CLS]를 강요하는 게 오히려 불공정) |
| 5개 비교의 다중검정 | Rank CI가 Holm 보정을 내장 — 그대로 사용 |

## 6.5 추가 분과 — 과학 도메인 특화 확장 (P14~P16, 앙상블 강화용)

> 배경: 앙상블의 이득은 멤버의 "다양성"에서 온다. 같은 레시피로 학습해도
> **사전학습 코퍼스가 다르면** 오류 패턴이 갈라진다 — 도메인 특화 모델을 늘려
> 앙상블 pool을 두텁게 한다. (모델 ID는 2026-07-14 HF hub 실재 확인 완료)

| run_id | 모델 | 특징 | 비고 |
|---|---|---|---|
| **P14** | `KISTI-AI/scideberta-cs` | DeBERTa를 **CS 논문**으로 도메인 적응 | KP20k와 도메인 일치. **P9(무특화 DeBERTa)와 짝 비교 = DAPT 효과 측정** (단 v2/v3 세대 차이는 명시) |
| **P15** | `allenai/cs_roberta_base` | RoBERTa **CS 도메인** DAPT (Don't Stop Pretraining, 2020) | KP20k와 도메인 일치 |
| P16 (선택) | `malteos/scincl` | 인용 이웃 대조학습 (SPECTER 계열 경쟁 모델) | P8과 같은 축의 다른 방법 |

실행: 본 토너먼트 체인 완료 후 `--stage run --run-id P14/P15` 추가 →
**ensemble 단계 재실행** (조합 탐색이 자동으로 확장 pool {P2,P7,P8,P9,P10,P11,P14,P15}에서
2~4개 조합 ~154개를 dev 1,000편으로 검색). 추가 비용: 모델당 ~35분 + 앙상블 재탐색 ~30분.

**기대**: 과학 특화 3~4종(P7 SciBERT, P8 SPECTER2, P14, P15)은 어휘·코퍼스가 서로 달라
(scivocab vs S2ORC-CS vs 인용신호) 상관이 낮은 편 — "같은 도메인, 다른 사전학습"의
앙상블 이득을 측정하는 부속 실험이 된다.

## 7. 한 줄 요약

> P7의 레시피를 그대로 두고 인코더만 과학특화·범용최강·현세대·Gemma계열로 갈아끼워,
> **챔피언의 힘이 인코더에서 오는지 방법론에서 오는지**를 top-10 기준 동일 채점표 위에서 판정한다.
